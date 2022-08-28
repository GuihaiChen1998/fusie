import math

import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from ordered_set import OrderedSet
from functools import partial
from helper import *
from tqdm import tqdm
import pysnooper
from datetime import datetime


def get_samples_idxes(sub, pred, label, obj, device):
	row_range = torch.arange(sub.shape[0], device=device)
	target_pred = pred[row_range, obj]  # shape (len(b_range),) 按照提供的行idxes和列idxes取pred中对应位置的元素
	pred = torch.where(label.byte(), torch.zeros_like(pred), pred)  # 通过用0来mask原pred中真实标签的位置 从而获取新的pred
	pred[row_range, obj] = target_pred
	samples_idxes = torch.argsort(pred, dim=1, descending=True)

	return samples_idxes


def func(x, sub, rel, sr2o_all, n_neg):
	neg_obj_temp = []
	# for ele in x[1]:
	# 	flags = torch.tensor((sub[x[0]].cpu().item(), rel[x[0]].cpu().item(), ele.cpu().item())) == torch.tensor(
	# 		all_triples)
	# 	flag = flags[:, 0] & flags[:, 1] & flags[:, 2]
	# 	if not flag.any().item():
	# 		neg_obj_temp.append(ele.cpu().item())
	# 		if len(neg_obj_temp) == n_neg:
	# 			break

	item, max_neg_num = x[1], 50
	# 设个max_neg_num是为了有序集合运算加快速度，毕竟只需看前面部分置信度高的，即看max_neg_num位置前面的，从中取need_neg_num即可
	
	assert n_neg <= max_neg_num, "问题所在：所需的负样本数量大于为加快集合运算而设置的max_neg_num"
	# For DEBUG
	
	result = OrderedSet(item.cpu().numpy()[:max_neg_num]) - OrderedSet(sr2o_all[(sub[x[0]].item(), rel[x[0]].item())])
	neg_obj_temp = list(result)[:n_neg]

	return neg_obj_temp

class BinaryCrossEntropyLoss(torch.nn.Module):
	"""This class implements :class:`torch.nn.Module` interface."""

	def __init__(self, p):
		super().__init__()
		self.p = p
		self.sig = torch.nn.Sigmoid()
		self.loss = torch.nn.BCELoss(reduction='mean')

	def forward(self, positive_triplets, negative_triplets):
		"""
        Parameters
        ----------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the true triplets as returned by the `forward` methods
            of the models.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the negative triplets as returned by the `forward`
            methods of the models.
        Returns
        -------
        loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
            Loss of the form :math:`-\\eta \\cdot \\log(f(h,r,t)) +
            (1-\\eta) \\cdot \\log(1 - f(h,r,t))` where :math:`f(h,r,t)`
            is the score of the fact and :math:`\\eta` is either 1 or
            0 if the fact is true or false.
        """
		if self.p.lbl_smooth != 0.0:
			return self.loss(self.sig(positive_triplets),
						 (1-self.p.lbl_smooth)*torch.ones_like(positive_triplets) + self.p.lbl_smooth/self.p.num_rel) + \
				   self.loss(self.sig(negative_triplets),
						 torch.zeros_like(negative_triplets) + self.p.lbl_smooth/self.p.num_rel)
		else:
			return self.loss(self.sig(positive_triplets),
							 torch.ones_like(positive_triplets)) + \
				   self.loss(self.sig(negative_triplets),
							 torch.zeros_like(negative_triplets))

class SeparableConv2d(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size:tuple, stride, padding):
		super(SeparableConv2d, self).__init__()
		self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
		self.pointwise = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=1, padding=0)

	def forward(self, x):
		return self.pointwise(self.conv1(x))


class Way2(torch.nn.Module):
	def __init__(self, params, chequer_perm_3vec):
		super(Way2, self).__init__()
		self.params = params
		self.chequer_perm_3vec = chequer_perm_3vec

		# -*-*-*- 深度可分离卷积 -*-*-*-
		self.way2_cnn = torch.nn.Sequential(
			SeparableConv2d(2, 32, kernel_size=(3, 5), stride=1, padding=0),  # shape (bs,32,18,28) (128,32,8,26)
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			SeparableConv2d(32, 32, kernel_size=(3, 5), stride=1, padding=0),  # shape (bs,64,16,26) (128,64,6,22)
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 64, kernel_size=(3, 5), stride=1, padding=0),  # shape (bs,128,12,22) (128,128,4,18)
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(0.2),
			SeparableConv2d(64, 64, kernel_size=(3, 5), stride=1, padding=0),  # shape (bs,128,8,18) (128,128,2,14)
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
		)


		self.way2_fc = torch.nn.Linear(64*12*14, 1)    # 64*8*18 64*2*14,

	def forward(self, batchsize, sub_emb, rel_emb, obj_emb, hid_feature):
		"""

		Parameters
		----------
		batchsize:
		sub_emb: 一个batch的sub的embedding shape (batchsize, embed_dim)
		rel_emb: 一个batch的rel的embedding shape (batchsize, embed_dim)
		obj_emb: 一个batch的obj的embedding shapeshape (batchsize,embed_dim)

		Returns
		-------

		"""
		comb_emb_hrt = torch.cat([sub_emb, rel_emb, obj_emb], dim=1)  # shape (256, 600)
		# print("comb_emb: ", comb_emb)
		chequer_perm_hrt = comb_emb_hrt[:, self.chequer_perm_3vec]  # shape是(batchsize, perm数，600)
		integrate_inp = chequer_perm_hrt.reshape(batchsize, self.params.perm_3vec, self.params.k_h, -1)   # shape (256, 1, 20, 30)
		# print("inp: \n", integrate_inp)
		print(integrate_inp.shape)

		comb_hid_feat_t = torch.cat([hid_feature, obj_emb], dim=1).reshape(batchsize, 1, self.params.k_h, -1)   # shape (256,1,20,20)
		pad_size = (integrate_inp.shape[-1] - comb_hid_feat_t.shape[-1])/2
		assert math.modf(pad_size)[0]==0, "Error: padding两端不能整除"
		comb_hid_feat_t = F.pad(comb_hid_feat_t, pad=(int(pad_size),int(pad_size),0,0), mode='constant',value=0)

		inp = torch.cat([integrate_inp, comb_hid_feat_t], dim=1)
		x = self.way2_cnn(inp)   # shape (bs,128,8,18)
		x = x.flatten(1)                   # shape (bs, 128*8*18)
		x = self.way2_fc(x)                # shape (batchsize, 1)
		x = F.sigmoid(x)

		return x




class BPRLoss(torch.nn.Module):
	"""
    Bayesian Personalized Ranking Loss: 让正例得分与负例得分之差越大越好\n
    .. math:: 原公式： L(\\theta) = \\sum\\ln{\\sigma(r_ui-r_uj)}
    .. math:: 修改后： L(\\theta) = \\sum\\ln{(1+e^{-(r_ui-r_uj)})}
    """

	def __init__(self, reduction='mean'):
		super(BPRLoss, self).__init__()
		self.reduction = reduction
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, pos_preds, neg_preds):
		distance = pos_preds - neg_preds
		loss = torch.sum(torch.log(1e-6+(1 + torch.exp(-distance))))  # 把原公式的sigmoid(distance)部分修改成1+exp(-distance)
		if self.reduction == 'mean':
			loss = loss.mean()

		#         print('loss:', loss)
		return loss


def dynamic_weighted_binary_crossentropy_withlogits(l, y_pred, y_true, alpha=0.5):
	def loss(y_pred, y_true):
		w_neg = torch.sum(y_true).item() / l
		w_pos = 1 - w_neg
		r = 2 * w_neg * w_pos
		w_neg /= r
		w_pos /= r

		b_ce = F.binary_cross_entropy_with_logits(y_pred, y_true)
		w_b_ce = b_ce * y_true * w_pos + b_ce * (1 - y_true) * w_neg
		return torch.mean(w_b_ce) * alpha + torch.mean(b_ce) * (1 - alpha)

	return loss(y_pred, y_true)

class KGML(nn.Module):
	def __init__(self, params):
		super(KGML, self).__init__()
		self.params = params
		self.k_s = (5, 5)
		self.KGML_cnn = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.k_s, padding=0),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Dropout(0.2),
			# nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.k_s, padding=0),
			SeparableConv2d(in_channels=32, out_channels=64, kernel_size=self.k_s, padding=0, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Dropout(0.1),
			# nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.k_s, padding=0),
			SeparableConv2d(in_channels=64, out_channels=128, kernel_size=self.k_s, padding=0, stride=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
		)
		filtered_shape = (18, 8)
		self.KGML_fc = nn.Linear(128*filtered_shape[0]*filtered_shape[1], self.params.num_rel)
		# self.sig = nn.Sigmoid()

	def forward(self, sub_emb, obj_emb, label):
		x = torch.cat([sub_emb, sub_emb*obj_emb, obj_emb], dim=1)        # shape (-,600)   # TODO: 外面记得转cuda
		x = x.view(-1, 1, 30, 20)
		x = self.KGML_cnn(x)
		x = x.flatten(1)
		x = self.KGML_fc(x)
		# x = self.sig(x)

		loss = dynamic_weighted_binary_crossentropy_withlogits(self.params.num_rel, x, label)

		return x, loss


"""
这里用一个正例和一个负例

"""
class Way3(torch.nn.Module):
	def __init__(self, parmas, ent_embed, rel_embed):
		super(Way3, self).__init__()
		self.params = parmas
		self.ent_embed = ent_embed
		self.rel_embed = rel_embed
		self.kernelsize = (5,5)
		# self.stride = (2,2)

		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(1, 32, self.kernelsize, stride=1, padding=0),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 32, self.kernelsize, stride=1, padding=0),
			# torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 32, self.kernelsize, stride=1, padding=0),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 64, self.kernelsize, stride=1, padding=0),
			# torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			SeparableConv2d(64, 64, self.kernelsize, stride=1, padding=0),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			# SeparableConv2d(64, 64, self.kernelsize, stride=1, padding=0),
			# # torch.nn.BatchNorm2d(64),
			# torch.nn.ReLU(),
		)

		self.filtered_shape = (self.params.embed_dim-5*self.kernelsize[0]+5,
							   self.params.embed_dim-5*self.kernelsize[1]+5)
		self.way3_fc = torch.nn.Linear(64*self.filtered_shape[0]*self.filtered_shape[1], 1)

	def forward(self, ent_target_emb, obj_emb):
		"""

		Parameters
		----------
		ent_target_emb: 头实体和关系经过交互学习得到的target embedding
		obj_emb: 尾实体emb, shape (batchsize, embed_dim)

		Returns
		-------

		"""
		inp = torch.bmm(obj_emb.unsqueeze(2), ent_target_emb.unsqueeze(1))   # shape (bs, embed_dim, embed_dim)
		stack_inp = inp.unsqueeze(1)      # shape (bs, 1, 200, 200)


		x = self.cnn(stack_inp)      # shape (bs, 64, 176, 176)
		x = x.flatten(1)
		x = self.way3_fc(x)          # shape (bs, 1)
		score = x.squeeze(1)             # shape (bs,)
		score = F.sigmoid(score)

		return score




class InteractE(torch.nn.Module):
	"""
	Proposed method in the paper. Refer Section 6 of the paper for mode details 

	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model
	
	Returns
	-------
	The InteractE model instance
		
	"""
	def __init__(self, params, chequer_perm, checquer_perm_3vec):
		super(InteractE, self).__init__()
		self.p                  = params
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.ent_embed		= torch.nn.Embedding(self.p.num_ent,   self.p.embed_dim, padding_idx=None).to(self.device); xavier_normal_(self.ent_embed.weight)
		# shape(实体数，embedding数)即(40943,200);这里相当于给的是lookup table
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None).to(self.device); xavier_normal_(self.rel_embed.weight)
		# shape(关系数*2，embedding数)即(22,200)
		self.bceloss		= torch.nn.BCELoss()

		self.bprloss = BPRLoss()

		self.way2_bceloss = BinaryCrossEntropyLoss(self.p)

		self.inp_drop		= torch.nn.Dropout(self.p.inp_drop)
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.feature_map_drop	= torch.nn.Dropout2d(self.p.feat_drop)
		self.bn0		= torch.nn.BatchNorm2d(self.p.perm).to(self.device)

		flat_sz_h 		= self.p.k_h
		flat_sz_w 		= 2*self.p.k_w
		self.padding 		= 0

		self.bn1 		= torch.nn.BatchNorm2d(self.p.num_filt*self.p.perm).to(self.device)
		self.flat_sz 		= flat_sz_h * flat_sz_w * self.p.num_filt*self.p.perm

		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim).to(self.device)
		self.fc 		= torch.nn.Linear(self.flat_sz, self.p.embed_dim).to(self.device)
		self.chequer_perm	= chequer_perm       # shape是(perm数, 400)即(4, 400)
		self.chequer_perm_3vec = checquer_perm_3vec   # perm取1即可, shape (1, 600)

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent).to(self.device)))
		# self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz,  self.p.ker_sz).to(self.device)));
		# xavier_normal_(self.conv_filt)

		if self.p.perm >=4:
			self.kernelsizes = [17,15,13,11]
			# self.conv_filt_1, self.conv_filt_2, self.conv_filt_3, self.conv_filt_4 = None, None, None, None
			# self.conv_filts = [self.conv_filt_1, self.conv_filt_2, self.conv_filt_3, self.conv_filt_4]
		elif self.p.perm == 3:
			self.kernelsizes = [17,15,13]
			# self.conv_filt_1, self.conv_filt_2, self.conv_filt_3 = None, None, None
			# self.conv_filts = [self.conv_filt_1, self.conv_filt_2, self.conv_filt_3]
		elif self.p.perm == 2:
			self.kernelsizes = [17,15]
			# self.conv_filt_1, self.conv_filt_2 = None, None
			# self.conv_filts = [self.conv_filt_1, self.conv_filt_2]
		else:
			self.kernelsizes = [17]
			# self.conv_filt_1 = None
			# self.conv_filts = [self.conv_filt_1]

		self.conv_filts = []
		for i in range(len(self.kernelsizes)):
			self.register_parameter('conv_filt'+'_'+str(i+1), Parameter(torch.zeros(self.p.num_filt, 1, self.kernelsizes[i],  self.kernelsizes[i]).to(self.device)))
			if i==0:
				xavier_normal_(self.conv_filt_1)
				self.conv_filts.append(self.conv_filt_1)
			elif i==1:
				xavier_normal_(self.conv_filt_2)
				self.conv_filts.append(self.conv_filt_2)
			elif i==2:
				xavier_normal_(self.conv_filt_3)
				self.conv_filts.append(self.conv_filt_3)
			else:
				xavier_normal_(self.conv_filt_4)
				self.conv_filts.append(self.conv_filt_4)

		# self.way3 = Way3(self.p, self.ent_embed, self.rel_embed).to(self.device)
		# self.way2 = Way2(self.p, self.chequer_perm_3vec).to(self.device)

		self.kgml = KGML(self.p).to(self.device)
		self.way2 = Way2(self.p, self.chequer_perm_3vec).to(self.device)

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0];      # TODO: 这俩句根本没用到
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def way3_loss(self, pos_score, neg_score):
		loss = self.bprloss(pos_score, neg_score)
		return loss

	def way2_loss(self, pos_trips_pred, neg_trips_pred):
		loss = self.way2_bceloss(pos_trips_pred, neg_trips_pred)
		return loss

	def circular_padding_chw(self, batch, padding):
		upper_pad	= batch[..., -padding:, :]
		lower_pad	= batch[..., :padding, :]
		temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

		left_pad	= temp[..., -padding:]
		right_pad	= temp[..., :padding]
		padded		= torch.cat([left_pad, temp, right_pad], dim=3)
		return padded

	def divide_group(self, nchannels, kernel_sizes):
		"""
		:param: nchannels: num of in_channels
		        kernel_sizes: list, e.g.: [3,7,11,15]. And the length should >= self.p.perm
		"""
		assert len(kernel_sizes) >= self.p.perm
		self.nchannels = nchannels
		self.groups = len(kernel_sizes)

		self.split_channels = [nchannels // self.groups for _ in range(self.groups)]
		self.split_channels[0] += nchannels - sum(self.split_channels)

		"""
		self.layers = []
		for i in range(self.groups):
			self.layers.append(nn.Conv2d(in_channels=self.split_channels[i],out_channels=self.split_channels[i],
									 kernel_size=kernel_sizes[i], stride=stride,padding=int(kernel_sizes[i]//2), groups=self.split_channels[i]))
		"""

		return self.split_channels

	def forward(self, sub, rel, neg_ents, label=None, is_train:bool=True, sr2o_all=None, so2r=None, strategy='one_to_x', step=None):
		bs = sub.shape[0]
		# sub和rel都是id，shape都是(batchsize,)即(256,)
		sub_emb		= self.ent_embed(sub)  # shape是(batchsize, embedding_dim)即(256, 200)
		rel_emb		= self.rel_embed(rel)  # shape是(batchsize, embedding_dim)即(256, 200)
		if is_train:
			pass
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=1)  # shape是(batchsize, embedding*2)即(256, 400)
		chequer_perm	= comb_emb[:, self.chequer_perm]  # shape是(batchsize, perm数，400)即(256,4,400)
		stack_inp	= chequer_perm.reshape((-1, self.p.perm, 2*self.p.k_w, self.p.k_h))
		# shape是(batchsize, perm数，2*k_w, k_h)即(256,4,20,20)
		stack_inp	= self.bn0(stack_inp)
		x		= self.inp_drop(stack_inp)
		# x		= self.circular_padding_chw(x, self.p.ker_sz//2)   # shape是(batchsize, perm数，30, 30)即(256,4,30,30)
		# x		= F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)  # shape是(batchsize, perm数*96，20, 20)即(256,384,20,20)

		# -*-*-
		self.split_channels = self.divide_group(x.shape[1], kernel_sizes=self.kernelsizes)
		# print("split_channels: ", self.split_channels)
		split_x = torch.split(x, self.split_channels, dim=1)
		# padding
		padded_x = []
		FMs = []
		for i, item in enumerate(zip(split_x, self.kernelsizes)):
			# print("i: ", i)
			# print("item: ", item)
			x_i = self.circular_padding_chw(item[0], item[1] // 2)
			# print("x_i: ", x_i)
			padded_x.append(x_i)
			fm = F.conv2d(x_i, self.conv_filts[i], padding=self.padding)
			# print(f"i is {i}, shape of self.conv_filts_{i} is :", self.conv_filts[i].shape)
			FMs.append(fm)
		x = torch.cat(FMs, dim=1)

		x		= self.bn1(x)  # shape是(batchsize, perm数*96，20, 20)即(256,384,20,20)
		x		= F.relu(x)
		x		= self.feature_map_drop(x)
		x		= x.view(-1, self.flat_sz)   # flatten操作，shape是(batchsize, flat_sz_h*flat_sz_w*num_filt*perm)即(256,153600)
		feature		= self.fc(x)   # shape是(batchsize, embedding数)即(256,200)
		x		= self.hidden_drop(feature)
		x		= self.bn2(x)
		x		= F.relu(x)


		if strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.bias[neg_ents]

		pred	= torch.sigmoid(x)

		# -*-*-*- way 2 and way 3 (RP and TC) -*-*-*-
		way3_pos_score, way3_neg_score = 0, 0
		way3Loss = 0
		way2Loss = 0
		if is_train:
			label = (label>0.5).byte()              # shape (bs, num_ent) 训练阶段的label是经过label smooth操作的
			label_idxes = torch.nonzero(label==1)
			temp_label_dict = defaultdict(list)           # 存放本个batch中，每个样本的label
			for i in label_idxes:
				temp_label_dict[i[0].item()].append(i[1].item())
			# obj = torch.tensor(list(map(lambda x: random.choice(x[1]), temp_label_dict.items()))).to(self.device)
			obj_lists = list(map(lambda x: x[1], temp_label_dict.items()))
			obj = torch.tensor(list(map(lambda x: random.choice(x), obj_lists))).to(self.device)

			# assert torch.numel(obj) == len(sub)
			# # just for DEBUG

			# -*-*- way3 -*-*-
			r_lbs = []
			temp_sub = []
			temp_obj = []
			st2 = datetime.now()
			for idx,lst in enumerate(obj_lists):
				for o in lst:
					r_label = torch.tensor(so2r[(sub[idx].item(), o)]).unsqueeze(0)
					"""
					try:
						r_label = torch.tensor(so2r[(sub[idx].item(), o)]).unsqueeze(0)
					except KeyError:
						# tmp = torch.tensor(so2r[(o, sub[idx].item())])
						# r_label = torch.where(tmp<self.p.num_rel, tmp, tmp-self.p.num_rel).unsqueeze(0)
						# # r_label = (torch.tensor(so2r[(o, sub[idx].item())])-self.p.num_rel).unsqueeze(0)
						raise KeyError
					"""
					target = torch.zeros(r_label.shape[0], self.p.num_rel).scatter(1, r_label, 1)
					r_lbs.append(target)
					temp_sub.append(sub[idx])
					# temp_obj.append(obj[idx])
					temp_obj.append(o)
			# print("way3获取label花费时间：", datetime.now()-st2)
			r_lbs = torch.cat(r_lbs, dim=0).to(self.device)
			if self.p.lbl_smooth != 0.0:
				r_lbs = (1-self.p.lbl_smooth)*r_lbs + self.p.lbl_smooth/self.p.num_rel    # label smooth

			rp_task_sub_emb = self.ent_embed(torch.LongTensor(temp_sub).to(self.device))
			rp_task_obj_emb = self.ent_embed(torch.LongTensor(temp_obj).to(self.device))
			_, way3Loss = self.kgml(rp_task_sub_emb, rp_task_obj_emb, r_lbs)


			obj_emb = self.ent_embed(obj)       # shape (bs, embed_dim)   # 尾实体emb

			# -*-*- 获取负例 -*-*-
			samples_idxes = get_samples_idxes(sub, pred, label, obj, device=self.device).detach().data

			# -*- 获取每个正例三元组对应的一个负样本id -*-
			neg_obj_list = []
			n_neg = self.p.need_n_neg    # 待取的负例数

			# with pysnooper.snoop():
			start = datetime.now()
			neg_obj_list = list(map(partial(func, sub=sub.data, rel=rel.data, sr2o_all=sr2o_all, n_neg=n_neg), enumerate(samples_idxes)))
			# print("neg_obj_list: ", neg_obj_list)
			# print("获取负例集过程Duration: ", datetime.now()-start)

			"""
			for idx, row in tqdm(enumerate(samples_idxes)):
				neg_obj_temp = []
				n = 0
				for ele in row:
					# if ele not in temp_label_dict[idx]:   # 条件：不在label中
					# if (sub[idx].item(), rel[idx].item(), ele.item()) not in all_triples:    # 条件：不能在KG中
					flags = torch.tensor((sub[idx].cpu().item(), rel[idx].cpu().item(), ele.cpu().item())) == torch.tensor(all_triples)
					flag = flags[:,0] & flags[:,1] & flags[:,2]
					if not flag.any().item():                     # 条件：构成的负例不在KG中
						neg_obj_temp.append(ele.cpu().item())
						n += 1
						if len(neg_obj_temp) == n_neg:
							break
				neg_obj_list.append(neg_obj_temp)                 # neg_obj_list可以作为way2的负例集
			"""


			# assert len(neg_obj_list) == len(samples_idxes)
			# # just for DEBUG

			# # -*- way3的核心计算 -*-
			# neg_obj = [item[0] for item in neg_obj_list]          # 取最前面的一个负例作为way3的本个batch的负例
			# assert len(neg_obj) == len(samples_idxes)
			# neg_obj = torch.tensor(neg_obj).to(self.device)
			# neg_obj_emb = self.ent_embed(neg_obj)
			#
			# way3_pos_score = self.way3(feature, obj_emb)      # 正例得分
			# way3_neg_score = self.way3(feature, neg_obj_emb)  # 负例得分
			#
			# way3Loss = self.way3_loss(way3_pos_score, way3_neg_score)

		# -*-*-*- 走辅助之路way2 -*-*-*-
		if is_train:
			neg_obj_list = list(zip(*neg_obj_list))
			pos_out = self.way2(bs, sub_emb, rel_emb, obj_emb, feature)
			temp_losses = []
			for i in range(n_neg):
				neg_obj_i_emb = self.ent_embed(torch.tensor(neg_obj_list[i]).to(self.device))
				neg_out = self.way2(bs, sub_emb, rel_emb, neg_obj_i_emb, feature)
				temp_loss = self.way2_bceloss(pos_out, neg_out).item()
				temp_losses.append(temp_loss)

			way2Loss = torch.tensor(temp_losses).mean()

		torch.cuda.empty_cache()

		return pred, way3Loss, way2Loss
