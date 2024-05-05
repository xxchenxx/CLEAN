CUDA_VISIBLE_DEVICES=2 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name base_attention_attn_lr_5e-4 --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 100 --use_v &



CUDA_VISIBLE_DEVICES=0 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name base_attention_attn_lr_5e-4_no_value --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 100 &


CUDA_VISIBLE_DEVICES=0 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name base_attention_attn_lr_1e-3_no_value --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 100 --attention_learning_rate 1e-3 &


CUDA_VISIBLE_DEVICES=1 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name base_attention_attn_lr_1e-4_no_value --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 100 --attention_learning_rate 1e-4 &

CUDA_VISIBLE_DEVICES=1 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name base_attention_attn_lr_1e-5_no_value --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 100 --attention_learning_rate 1e-5 &

CUDA_VISIBLE_DEVICES=0 nohup python -u ./train-triplet-with-backbone_moco.py  --training_data split10 --model_name base_attention_attn_lr_1e-6_no_value --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 100 --attention_learning_rate 1e-6 &
