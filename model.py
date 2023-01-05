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
	target_pred = pred[row_range, obj]  
	pred = torch.where(label.byte(), torch.zeros_like(pred), pred)  
	pred[row_range, obj] = target_pred 
	samples_idxes = torch.argsort(pred, dim=1, descending=True)

	return samples_idxes


def func(x, sub, rel, sr2o_all, n_neg):
	neg_obj_temp = []

	item, max_neg_num = x[1], 50     # feel free to set max_neg_num
	
	assert n_neg <= max_neg_num
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


class Way1(torch.nn.Module):
	"""changed convkb"""
	def __init__(self, params, chequer_perm_3vec):
		super(Way1, self).__init__()
		self.params = params
		self.chequer_perm_3vec = chequer_perm_3vec

		# -*-*-*- convolution -*-*-*-
		# self.way1_cnn = torch.nn.Sequential(
		# 	torch.nn.Conv2d(1, 32, kernel_size=(3,5), stride=1, padding=0),    
		# 	torch.nn.BatchNorm2d(32),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d(32, 32, kernel_size=(3,5), stride=1, padding=0),   
		# 	torch.nn.BatchNorm2d(32),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout(0.2),
		# 	torch.nn.Conv2d(32, 64, kernel_size=(3,5), stride=1, padding=0),  
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout2d(0.2),
		# 	torch.nn.Conv2d(64, 64, kernel_size=(3,5), stride=1, padding=0),  
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout(0.2),
		# )

		# -*-*-*- Depthwise separable convolution -*-*-*-
		self.way1_cnn = torch.nn.Sequential(
			SeparableConv2d(1, 32, kernel_size=(3, 5), stride=1, padding=0),  
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			SeparableConv2d(32, 32, kernel_size=(3, 5), stride=1, padding=0),  
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
			SeparableConv2d(32, 64, kernel_size=(3, 5), stride=1, padding=0),  
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout2d(0.2),
			SeparableConv2d(64, 64, kernel_size=(3, 5), stride=1, padding=0),  
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.2),
		)


		self.way1_fc = torch.nn.Linear(64*12*14, 1)    

	def forward(self, batchsize, sub_emb, rel_emb, obj_emb):
		"""

		Parameters
		----------
		batchsize:
		sub_emb: 
		rel_emb: 
		obj_emb:

		Returns
		-------

		"""
		comb_emb_hrt = torch.cat([sub_emb, rel_emb, obj_emb], dim=1)  
		chequer_perm_hrt = comb_emb_hrt[:, self.chequer_perm_3vec]  
		integrate_inp = chequer_perm_hrt.reshape(batchsize, self.params.perm_3vec, self.params.k_h, -1)   


		x = self.way1_cnn(integrate_inp)   
		x = x.flatten(1)                   
		x = self.way1_fc(x)                
		x = F.sigmoid(x)

		return x




class BPRLoss(torch.nn.Module):
	"""
	"""

	def __init__(self, reduction='mean'):
		super(BPRLoss, self).__init__()
		self.reduction = reduction
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, pos_preds, neg_preds):
		distance = pos_preds - neg_preds
		loss = torch.sum(torch.log(1e-6+(1 + torch.exp(-distance))))  
		if self.reduction == 'mean':
			loss = loss.mean()

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
			SeparableConv2d(in_channels=32, out_channels=64, kernel_size=self.k_s, padding=0, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Dropout(0.1),
			SeparableConv2d(in_channels=64, out_channels=128, kernel_size=self.k_s, padding=0, stride=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
		)
		filtered_shape = (18, 8)
		self.KGML_fc = nn.Linear(128*filtered_shape[0]*filtered_shape[1], self.params.num_rel)


	def forward(self, sub_emb, obj_emb, label):
		x = torch.cat([sub_emb, sub_emb*obj_emb, obj_emb], dim=1)        
		x = x.view(-1, 1, 30, 20)
		x = self.KGML_cnn(x)
		x = x.flatten(1)
		x = self.KGML_fc(x)

		loss = dynamic_weighted_binary_crossentropy_withlogits(self.params.num_rel, x, label)

		return x, loss



class Way2(torch.nn.Module):
	def __init__(self, parmas, ent_embed, rel_embed):
		super(Way2, self).__init__()
		self.params = parmas
		self.ent_embed = ent_embed
		self.rel_embed = rel_embed
		self.kernelsize = (5,5)


		# -*-*-*- convolution -*-*-*-
		# self.cnn = torch.nn.Sequential(
		# 	torch.nn.Conv2d(1, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout(0.2),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	# torch.nn.Dropout(0.2),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Dropout(0.2),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# 	torch.nn.Conv2d(64, 64, self.kernelsize, padding=0),
		# 	torch.nn.BatchNorm2d(64),
		# 	torch.nn.ReLU(),
		# )

		# -*-*-*- Depthwise separable convolution -*-*-*-
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
		)

		self.filtered_shape = (self.params.embed_dim-5*self.kernelsize[0]+5,
							   self.params.embed_dim-5*self.kernelsize[1]+5)
		self.way2_fc = torch.nn.Linear(64*self.filtered_shape[0]*self.filtered_shape[1], 1)

	def forward(self, ent_target_emb, obj_emb):
		"""
		Parameters
		----------
		ent_target_emb: 
		obj_emb: 
		Returns
		-------
		"""
		inp = torch.bmm(obj_emb.unsqueeze(2), ent_target_emb.unsqueeze(1))   
		stack_inp = inp.unsqueeze(1)      
		x = self.cnn(stack_inp)      
		x = x.flatten(1)
		x = self.way2_fc(x)          
		score = x.squeeze(1)             
		score = F.sigmoid(score)
		return score



class InteractE(torch.nn.Module):
	"""
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
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None).to(self.device); xavier_normal_(self.rel_embed.weight)
		self.bceloss		= torch.nn.BCELoss()
		self.bprloss = BPRLoss()

		self.way1_bceloss = BinaryCrossEntropyLoss(self.p)

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
		self.chequer_perm	= chequer_perm       
		self.chequer_perm_3vec = checquer_perm_3vec   

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent).to(self.device)))
		self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz,  self.p.ker_sz).to(self.device)));
		xavier_normal_(self.conv_filt)


		self.kgml = KGML(self.p).to(self.device)
		self.way1 = Way1(self.p, self.chequer_perm_3vec).to(self.device)

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0];      
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def way2_loss(self, pos_score, neg_score):
		loss = self.bprloss(pos_score, neg_score)
		return loss

	def way1_loss(self, pos_trips_pred, neg_trips_pred):
		loss = self.way1_bceloss(pos_trips_pred, neg_trips_pred)
		return loss

	def circular_padding_chw(self, batch, padding):
		upper_pad	= batch[..., -padding:, :]
		lower_pad	= batch[..., :padding, :]
		temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

		left_pad	= temp[..., -padding:]
		right_pad	= temp[..., :padding]
		padded		= torch.cat([left_pad, temp, right_pad], dim=3)
		return padded


	def forward(self, sub, rel, neg_ents, label=None, is_train:bool=True, sr2o_all=None, so2r=None, strategy='one_to_x', step=None):
		bs = sub.shape[0]
		sub_emb		= self.ent_embed(sub)  
		rel_emb		= self.rel_embed(rel)  
		if is_train:
			pass
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=1)  
		chequer_perm	= comb_emb[:, self.chequer_perm]  
		stack_inp	= chequer_perm.reshape((-1, self.p.perm, 2*self.p.k_w, self.p.k_h))

		stack_inp	= self.bn0(stack_inp)
		x		= self.inp_drop(stack_inp)
		x		= self.circular_padding_chw(x, self.p.ker_sz//2)   
		x		= F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)  


		x		= self.bn1(x)  
		x		= F.relu(x)
		x		= self.feature_map_drop(x)
		x		= x.view(-1, self.flat_sz)   
		x		= self.fc(x)   
		x		= self.hidden_drop(x)
		x		= self.bn2(x)
		x		= F.relu(x)


		if strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.bias[neg_ents]

		pred	= torch.sigmoid(x)


		way2_pos_score, way2_neg_score = 0, 0
		way2Loss = 0
		way1Loss = 0
		if is_train:
			label = (label>0.5).byte()              
			label_idxes = torch.nonzero(label==1)
			temp_label_dict = defaultdict(list)           
			for i in label_idxes:
				temp_label_dict[i[0].item()].append(i[1].item())
			obj_lists = list(map(lambda x: x[1], temp_label_dict.items()))
			obj = torch.tensor(list(map(lambda x: random.choice(x), obj_lists))).to(self.device)

			# assert torch.numel(obj) == len(sub)
			# # just for DEBUG

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
					temp_obj.append(o)
			r_lbs = torch.cat(r_lbs, dim=0).to(self.device)
			if self.p.lbl_smooth != 0.0:
				r_lbs = (1-self.p.lbl_smooth)*r_lbs + self.p.lbl_smooth/self.p.num_rel    
			rp_task_sub_emb = self.ent_embed(torch.LongTensor(temp_sub).to(self.device))
			rp_task_obj_emb = self.ent_embed(torch.LongTensor(temp_obj).to(self.device))
			_, way2Loss = self.kgml(rp_task_sub_emb, rp_task_obj_emb, r_lbs)
			obj_emb = self.ent_embed(obj)       
			samples_idxes = get_samples_idxes(sub, pred, label, obj, device=self.device).detach().data
			neg_obj_list = []
			n_neg = self.p.need_n_neg    
			start = datetime.now()
			neg_obj_list = list(map(partial(func, sub=sub.data, rel=rel.data, sr2o_all=sr2o_all, n_neg=n_neg), enumerate(samples_idxes)))


		if is_train:
			neg_obj_list = list(zip(*neg_obj_list))
			pos_out = self.way1(bs, sub_emb, rel_emb, obj_emb)
			temp_losses = []
			for i in range(n_neg):
				neg_obj_i_emb = self.ent_embed(torch.tensor(neg_obj_list[i]).to(self.device))
				neg_out = self.way1(bs, sub_emb, rel_emb, neg_obj_i_emb)
				temp_loss = self.way1_bceloss(pos_out, neg_out).item()
				temp_losses.append(temp_loss)

			way1Loss = torch.tensor(temp_losses).mean()

		torch.cuda.empty_cache()

		return pred, way2Loss, way1Loss
