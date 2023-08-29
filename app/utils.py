import os
import torch
import logging

import numpy as np
import torch.nn as nn
import wandb 
import argparse

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    formatter = logging.Formatter('[%(asctime)s] [%(module)s.%(funcName)s.%(lineno)d] [%(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger()

def save_state_dicts(model, esm_model, query, 
                     key, optimizer, esm_optimizer, 
                     attentions_optimizer, args,
                     is_best=False, epoch=0, save_esm=True,
                     output_dir='./data/model/', output_name='model'):
    state_dicts_to_save = {}
    state_dicts_to_save['model_state_dict'] = model.state_dict()
    if save_esm:
        state_dicts_to_save['esm_model_state_dict'] = esm_model.state_dict()

    state_dicts_to_save['optimizer_state_dict'] = optimizer.state_dict()
    if save_esm:
        state_dicts_to_save['esm_optimizer_state_dict'] = esm_optimizer.state_dict()

    if args.use_extra_attention:
        state_dicts_to_save['query_state_dict'] = query.state_dict()
        state_dicts_to_save['key_state_dict'] = key.state_dict()
        state_dicts_to_save['attentions_optimizer_state_dict'] = attentions_optimizer.state_dict()
    
    if is_best:
        torch.save(state_dicts_to_save, os.path.join(
            output_dir, f'best_{output_name}_checkpoints.pth.tar'
        ))
    else:
        torch.save(state_dicts_to_save, os.path.join(
            output_dir, f'{output_name}_checkpoints.pth.tar'
        ))
    log.info(f"Saving state_dicts at epoch {epoch}")
    

def calculate_distance_matrix_for_ecs(dict_, mode='linear'):
    ecs = list(dict_.keys())
    score_matrix = np.ones((len(ecs)+ 1, len(ecs) + 1))
    for i in range(len(ecs)):
        for j in range(i, len(ecs)):
            e1 = ecs[i].strip().split(".")
            e2 = ecs[j].strip().split(".")
            k = 0
            while k <= 4:
                if k == 4:
                    break
                if e1[k] != e2[k]:
                    break
                else:
                    k += 1
            score_matrix[i, j] = k
            score_matrix[j, i] = k
    score_matrix[score_matrix >= 4] = 100000
    return score_matrix

def get_attention_modules(args, lr, device):
    if args.use_learnable_k:
        learnable_k = nn.Parameter(torch.zeros(1, args.esm_model_dim))
    else:
        learnable_k = None

    if args.use_extra_attention:
        query = nn.Linear(args.esm_model_dim, 64, bias=False).to(device)
        # nn.init.constant_(query.weight, 1e-3)
        nn.init.normal_(query.weight, std=np.sqrt(2 / (64 + args.esm_model_dim)))
        key = nn.Linear(args.esm_model_dim, 64, bias=False).to(device)
        # nn.init.constant_(key.weight, 1e-3)
        nn.init.normal_(key.weight, std=np.sqrt(2 / (64 + args.esm_model_dim)))
        if args.use_learnable_k:
            attentions_optimizer = torch.optim.Adam([{"params": query.parameters(), "lr": lr, "momentum": 0.9}, {"params": key.parameters(), "lr": lr, "momentum": 0.9}, {"params": learnable_k, "lr": lr, "momentum": 0.9}])
        else:
            attentions_optimizer = torch.optim.Adam([{"params": query.parameters(), "lr": lr, "momentum": 0.9}, {"params": key.parameters(), "lr": lr, "momentum": 0.9}])
        attentions = [query, key]
    else:
        attentions = [None, None]
        attentions_optimizer = None
    return learnable_k, attentions, attentions_optimizer

class Dummy:
    def log(self, key, step=0):
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('--esm_learning_rate', type=float, default=1e-5)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--train_esm_rate', type=int, default=100)
    parser.add_argument('--use_extra_attention', action="store_true")
    parser.add_argument('--use_top_k', action="store_true")
    parser.add_argument('--use_top_k_sum', action="store_true")
    parser.add_argument('--use_learnable_k', action="store_true")
    parser.add_argument('--use_input_as_k', action="store_true")
    parser.add_argument('--temp_esm_path', type=str, required=True)
    parser.add_argument('--distance_path', type=str, default='./data/distance_map/')
    parser.add_argument('--evaluate_freq', type=int, default=50)
    parser.add_argument('--esm-model', type=str, default='esm1b_t33_650M_UR50S.pt')
    parser.add_argument('--esm-model-dim', type=int, default=1280)
    parser.add_argument('--repr-layer', type=int, default=33)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--use_negative_smile', action="store_true")
    parser.add_argument('--use_random_augmentation', action="store_true")
    parser.add_argument('--use_weighted_loss', action="store_true")
    parser.add_argument('--use_ranking_loss', action="store_true")
    parser.add_argument('--use_SMILE_cls_token', action="store_true")
    parser.add_argument('--no_wandb', action="store_true")
    args = parser.parse_args()

    if args.wandb_name is None:
        args.wandb_name = args.model_name
        
    if args.no_wandb:
        return args, Dummy()
    else:
        run = wandb.init(project='protein_generalist', name=args.wandb_name)
        run.config.update(args)
        return args, run
