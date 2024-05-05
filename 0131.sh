CUDA_VISIBLE_DEVICES=2 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_negative_smile_random_smile_bs50_CLS_attn2 --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --use_SMILE_cls_token --use_negative_smile --use_top_k --use_cosine_ranking_loss --remap > with_negative_smile_attn2.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u ./train-triplet-with-backbone_moco-with_smile.py  --training_data split10 --model_name with_negative_smile_random_smile_bs50_CLS_attn2_short_queue --epoch 1000 --temp_esm_path moco_path_12/ --train_esm_rate 1000 --adaptive_rate 1000 --esm_model esm2_t12_35M_UR50D.pt --esm_model_dim 480 --repr_layer 12 --evaluate_freq 25 --use_extra_attention --seed 11 --batch_size 50 --use_random_augmentation --use_SMILE_cls_token --use_negative_smile --use_top_k --use_cosine_ranking_loss --remap --queue_size 100 > with_negative_smile_attn2_short_queue.out &



