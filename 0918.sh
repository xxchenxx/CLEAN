CUDA_VISIBLE_DEVICES=0 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split30 --model_name with_negative_smile_random_smile_bs50_CLS_attn2 --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --use_SMILE_cls_token --use_negative_smile --use_top_k > with_negative_smile_attn2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split30 --model_name with_negative_smile_random_smile_bs50_CLS_attn2 --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --use_SMILE_cls_token --use_negative_smile --use_top_k > with_negative_smile_attn2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split30 --model_name with_negative_smile_random_smile_bs50_CLS_attn2 --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --use_SMILE_cls_token --use_negative_smile --use_top_k > with_negative_smile_attn2.out &
