Useage:

```
python Main.py --gpu 0 --num_workers 8 --data WN18RR --name train_wn18rr --batch 256 --train_strategy one_to_n --feat_drop 0.2 --hid_drop 0.3 --perm 4 --perm_3vec 1 --ker_sz 11 --lr 0.001 --model_loaded_path None --embed_dim 200 --k_h 20 --k_w 10 --need_n_neg 10
```



Note that: 

The value of some hyperparameters can be changed according to your own conditions, for example, if you use a server with a large GPU memory,  just set the `batchsize` larger; if you want to use more negative samples, just set `need_n_neg` larger.  We recommend using hyperparameter search methods such as grid search to get the optimal hyperparameters. You can even use some machine learning platforms, which usually have integrated hyperparameter search algorithm APIs and can also record training logs better, such as [comet.ml](https://www.comet.com/site/).