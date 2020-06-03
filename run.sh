######################################################################################################################################################
######################################################################################################################################################
#######################################################               CIFAR-FS              ##########################################################
######################################################################################################################################################
######################################################################################################################################################

# # # # self supervision
# python3 train_supervised_ssl.py \
# --tags cifarfs,may30 \
# --model resnet12_ssl \
# --model_path save/backup \
# --dataset CIFAR-FS \
# --data_root ../../Datasets/CIFAR_FS/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1 \
# --epochs 65 \
# --lr_decay_epochs 60 \
# --gamma 2.0 &



# for i in {0..0}; do
# python3 train_distillation.py \
# --tags cifarfs,gen1,may30 \
# --model_s resnet12_ssl \
# --model_t resnet12_ssl \
# --path_t save/backup/resnet12_ssl_CIFAR-FS_lr_0.05_decay_0.0005_trans_D_trial_1/model_firm-sun-394.pth \
# --model_path save/backup \
# --dataset CIFAR-FS \
# --data_root ../../Datasets/CIFAR_FS/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1 \
# --epochs 65 \
# --lr_decay_epochs 60 \
# --gamma 0.1 &
# sleep 1m
# done






# # # evaluation
# CUDA_VISIBLE_DEVICES=0 python3 eval_fewshot.py \
# --model resnet12_ssl \
# --model_path save/backup2/resnet12_ssl_toy_lr_0.05_decay_0.0005_trans_A_trial_1/model_upbeat-dew-17.pth \
# --dataset toy \
# --data_root ../../Datasets/CIFAR_FS/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1










######################################################################################################################################################
######################################################################################################################################################
#######################################################                 FC100               ##########################################################
######################################################################################################################################################
######################################################################################################################################################


# # # # GEN0
# CUDA_VISIBLE_DEVICES=1 python3 train_supervised_ssl.py \
# --model resnet12_ssl \
# --model_path save/backup \
# --tags fc100 \
# --dataset FC100 \
# --data_root ../Datasets/neurips2020/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1 \
# --epochs 65 \
# --lr_decay_epochs 60 \
# --gamma 2 &


# # # # GEN1
# for i in {0..0}; do
# python3 train_distillation5.py \
# --model_s resnet12_ssl \
# --model_t resnet12_ssl \
# --path_t save/backup/resnet12_ssl_FC100_lr_0.05_decay_0.0005_trans_D_trial_1/model_effortless-wood-315.pth \
# --tags fc100 \
# --model_path save/neurips2020 \
# --dataset FC100 \
# --data_root ../Datasets/FC100/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1 \
# --batch_size 64 \
# --epochs 8 \
# --lr_decay_epochs 3 \
# --gamma 0.2 \
# # sleep 4m
# done

















######################################################################################################################################################
######################################################################################################################################################
#######################################################                 miniImagenet               ##########################################################
######################################################################################################################################################
######################################################################################################################################################




# # # # GEN0
# python3 train_supervised_ssl.py \
# --model resnet12_ssl \
# --model_path save/neurips2020 \
# --tags miniimagenet,gen0 \
# --dataset miniImageNet \
# --data_root ../Datasets/MiniImagenet/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1 \
# --epochs 65 \
# --lr_decay_epochs 60 \
# --gamma 2.0 &


# params=( 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5 )

# # # # GEN1
# for i in {0..2}; do
# python3 train_distillation.py \
# --model_s resnet12_ssl \
# --model_t resnet12_ssl \
# --path_t save/neurips2020/resnet12_ssl_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_1/model_swift-lake-4.pth \
# --tags miniimagenet,gen1,beta \
# --model_path save/neurips2020 \
# --dataset miniImageNet \
# --data_root ../Datasets/MiniImagenet/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1 \
# --batch_size 64 \
# --epochs 8 \
# --lr_decay_epochs 5 \
# --gamma ${params[i]} &
# sleep 30m
# done
















######################################################################################################################################################
######################################################################################################################################################
#######################################################                 tieredImageNet               ##########################################################
######################################################################################################################################################
######################################################################################################################################################




# # # # GEN0
# python3 train_supervised_ssl.py \
# --model resnet12_ssl \
# --model_path save/backup \
# --tags tieredimageNet \
# --dataset tieredImageNet \
# --data_root ../Datasets/TieredImagenet/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1 \
# --epochs 60 \
# --lr_decay_epochs 30,40,50 \
# --gamma 2 




# # # # GEN1
# for i in {0..6}; do
# python3 train_distillation5.py \
# --model_s resnet12_ssl \
# --model_t resnet12_ssl \
# --path_t save/backup/resnet12_ssl_FC100_lr_0.05_decay_0.0005_trans_D_trial_1/model_effortless-wood-315.pth \
# --tags fc100 \
# --model_path save/backup \
# --dataset FC100 \
# --data_root ../../Datasets/FC100/ \
# --n_aug_support_samples 5 \
# --n_ways 5 \
# --n_shots 1 \
# --batch_size 64 \
# --epochs 8 \
# --lr_decay_epochs 3 \
# --gamma ${params[i]} \
# # sleep 4m
# done

