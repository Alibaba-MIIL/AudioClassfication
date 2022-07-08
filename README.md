# Audio classification - 
This work is done as sound classification task in Alibaba Israel, link to paper https://arxiv.org/abs/2204.11479

# Cite - 
@article{gazneli2022end,
  title={End-to-End Audio Strikes Back: Boosting Augmentations Towards An Efficient Audio Classification Network},
  author={Gazneli, Avi and Zimerman, Gadi and Ridnik, Tal and Sharir, Gilad and Noy, Asaf},
  journal={arXiv preprint arXiv:2204.11479},
  year={2022}
}

# Third party -
utils/resample.py is mainly taken from - https://github.com/danpovey/filtering/blob/master/lilfilter/resampler.py

# Network configuration - 
## EAT-S - 
emb_dim 128
nf 16
dim_feedforward 512
n_layers 4
n_head 8

## EAT-M - 
emb_dim 256
nf 32
dim_feedforward 2048
n_layers 6
n_head 16

# Supported datasets -
ESC-50
Audioset
Uraban8K
Speechcommands

# General -
## augmentations -
The augmentations contain two types of transforms -
1. label preserving (audio_augs) and label mixing - implemented on cpu during fetching sample in dataset
2. label mixing - implemented on GPU (batch_augs)

## ESC50 - 
The samples downsampled to 22.05KHz and saved as wav format. if one want to use the original samples jusst modify the esc_dataset to read the coresponding file type.

## Urban 8K - 
The samples resampled to 22.05KHz and saved as wav format. During training the sample will be zero padded in case if it is smaller than 4 seconds

## Speechcommands - 
Fs=16KHz
seq_len=16384 ~1sec

## Audioset - 
Fs=22.05KHz\
seq_len=221184 ~10sec\
Requires preprocessing\
1. convert labels to list of integers\
2. resample to 22.05KHz and compress by saving in flac or ogg format\
3. verify missing files, missing labels etc\
4. save the data in pkl file\

# training - 
## esc50 - 
Fs=22.05KHz\
seq_len = 114688 ~5sec\
python trainer.py --max_lr 3e-4 --run_name r1 --emb_dim 128  --dataset esc50 --seq_len 114688  --mix_ratio 1 --epoch_mix 12 --mix_loss bce --batch_size 128 --n_epochs 3500 --ds_factors 4 4 4 4 --amp --save_path outputs\

## audioset - 
Fs=22.05KHz\
seq_len = 221184 ~10sec\
EAT-M - (for EAT-S modify the network parameters)\
python trainer.py --max_lr 3e-4 --run_name r1 --dataset audioset --seq_len 221184 --mix_ratio 1 --epoch_mix 2 --mix_loss bce --batch_size 208 --n_epochs 250 --scheduler onecycle --ds_factors 4 4 4 4 --save_path outputs --num_workers 32 --use_balanced_sampler --multilabel --amp --data_subtype full --use_dp --loss_type bce --augs_noise none --emb_dim 256 --nf 32 --dim_feedforward 2048 --n_layers 6 --n_head 16\

## urban8k - 
Fs=22.05KHz\
seq_len = 90112 ~4sec\
python trainer.py --max_lr 3e-4 --run_name r1 --emb_dim 128  --dataset urban8k --seq_len 90112  --mix_ratio 1 --epoch_mix 12 --mix_loss bce --batch_size 128 --n_epochs 3500 --ds_factors 4 4 4 4 --amp --save_path outputs\

## speechcommands - 
Fs=16KHz\
seq_len = 16384 ~1sec\
use use_bg in case one want to add background noise given in speechcommands dataset
python trainer.py --max_lr 3e-4 --run_name r1 --emb_dim 128  --dataset esc50 --seq_len 16384  --mix_ratio 1 --epoch_mix 12 --mix_loss bce --batch_size 128 --n_epochs 1500 --ds_factors 4 4 4 --amp --save_path outputs

# inference - 
python inference.py --f_res outputs/r1