CUDA_VISIBLE_DEVICES=6 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_random_smile_bs50_fuse_mlp_CLS --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm-model esm2_t12_35M_UR50D.pt --esmmodel-dim 480 --repr-layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --fuse_mode mlp --use_SMILE_cls_token > with_random_smile.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name baseline_attn3 --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm-model esm2_t12_35M_UR50D.pt --esm-model-dim 480 --repr-layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_top_k_sum > baseline_attn3.out &
sleep 5s
CUDA_VISIBLE_DEVICES=2 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name baseline_attn3 --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm-model esm2_t12_35M_UR50D.pt --esm-model-dim 480 --repr-layer 12 --evaluate_freq 25 --use_extra_attention --seed 22 --batch_size 50 --use_top_k_sum > baseline_attn3.out &
sleep 5s
CUDA_VISIBLE_DEVICES=3 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name baseline_attn3 --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm-model esm2_t12_35M_UR50D.pt --esm-model-dim 480 --repr-layer 12 --evaluate_freq 25 --use_extra_attention --seed 33 --batch_size 50 --use_top_k_sum > baseline_attn3.out &



CUDA_VISIBLE_DEVICES=4 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_random_smile_bs50_fuse_mlp --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm-model esm2_t12_35M_UR50D.pt --esm-model-dim 480 --repr-layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --fuse_mode mlp  > with_random_smile.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_random_smile_bs50_fuse_mlp_CLS --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm-model esm2_t12_35M_UR50D.pt --esm-model-dim 480 --repr-layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --fuse_mode mlp --use_SMILE_cls_token > with_random_smile.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_random_smile_bs50_CLS --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm-model esm2_t12_35M_UR50D.pt --esm-model-dim 480 --repr-layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --use_SMILE_cls_token > with_random_smile.out &




CUDA_VISIBLE_DEVICES=2 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_negative_smile_random_smile_bs50_fuse_mlp_attn2 --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --fuse_mode mlp --use_negative_smile --use_top_k > with_negative_smile_attn2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_negative_smile_random_smile_bs50_fuse_mlp_CLS_attn2 --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --fuse_mode mlp --use_SMILE_cls_token --use_negative_smile --use_top_k > with_negative_smile_attn2.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_negative_smile_random_smile_bs50_CLS_attn2 --epoch 1000 --temp_esm_path moco_temp_esm_path_10_esm1/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --use_SMILE_cls_token --use_negative_smile --use_top_k > with_negative_smile_attn2.out &