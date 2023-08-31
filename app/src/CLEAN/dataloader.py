import torch
import random
from .utils import format_esm, get_logger
import numpy as np
import pandas as pd

logger = get_logger(__name__)
def mine_hard_negative(dist_map, knn=10):
    #print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    for i, target in enumerate(ecs):
        sort_orders = sorted(
            dist_map[target].items(), key=lambda x: x[1], reverse=False)
        if sort_orders[1][1] != 0:
            freq = [1/i[1] for i in sort_orders[1:1 + knn]]
            neg_ecs = [i[0] for i in sort_orders[1:1 + knn]]
        elif sort_orders[2][1] != 0:
            freq = [1/i[1] for i in sort_orders[2:2+knn]]
            neg_ecs = [i[0] for i in sort_orders[2:2+knn]]
        elif sort_orders[3][1] != 0:
            freq = [1/i[1] for i in sort_orders[3:3+knn]]
            neg_ecs = [i[0] for i in sort_orders[3:3+knn]]
        else:
            freq = [1/i[1] for i in sort_orders[4:4+knn]]
            neg_ecs = [i[0] for i in sort_orders[4:4+knn]]

        normalized_freq = [i/sum(freq) for i in freq]
        negative[target] = {
            'weights': normalized_freq,
            'negative': neg_ecs
        }
    return negative


def mine_negative(anchor, id_ec, ec_id, mine_neg):
    anchor_ec = id_ec[anchor]
    pos_ec = random.choice(anchor_ec)
    neg_ec = mine_neg[pos_ec]['negative']
    weights = mine_neg[pos_ec]['weights']
    # print(weights)
    
    if np.any(np.isnan(weights)):
        weights = np.ones_like(weights) / len(weights)

    result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    while result_ec in anchor_ec:
        result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    neg_id = random.choice(ec_id[result_ec])
    return neg_id


def random_positive(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])
    pos = id
    if len(ec_id[pos_ec]) == 1:
        return pos + '_' + str(random.randint(0, 9))
    while pos == id:
        pos = random.choice(ec_id[pos_ec])
    return pos


class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, path='data/esm_data/'):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_neg = mine_neg
        self.path = path
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        a = torch.load(f'{self.path}/' + anchor + '.pt')
        p = torch.load(f'{self.path}/' + pos + '.pt')
        n = torch.load(f'{self.path}/' + neg + '.pt')
        return format_esm(a), format_esm(p), format_esm(n)

class Triplet_dataset_with_mine_EC_text(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        a = anchor
        p = pos
        n = neg
        return a, p, n


class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = format_esm(torch.load('./data/esm_data/' +
                       anchor + '.pt')).unsqueeze(0)
        data = [a]
        for _ in range(self.n_pos):
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            p = format_esm(torch.load('./data/esm_data/' +
                           pos + '.pt')).unsqueeze(0)
            data.append(p)
        for _ in range(self.n_neg):
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            n = format_esm(torch.load('./data/esm_data/' +
                           neg + '.pt')).unsqueeze(0)
            data.append(n)
        return torch.cat(data)


class MultiPosNeg_dataset_with_mine_EC_text(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = anchor
        data = [a]
        for _ in range(self.n_pos):
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            p = pos
            data.append(p)
        for _ in range(self.n_neg):
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            n = neg
            data.append(n)
        return torch.cat(data)

def mine_positive(anchor, id_ec, ec_id, mine_pos):
    anchor_ec = id_ec[anchor]
    positives = mine_pos[anchor_ec[0]]
    result_ec = random.choice(positives["positive"])

    pos_id = random.choice(ec_id[result_ec])
    return pos_id


class MoCo_dataset_with_mine_EC_text(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_pos, with_ec_number=False):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_pos = mine_pos
        self.with_ec_number = with_ec_number
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = mine_positive(anchor, self.id_ec, self.ec_id, self.mine_pos)
        a = anchor
        p = pos
        if self.with_ec_number:
            return a, p, anchor_ec
        else:
            return a, p


class MoCo_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_pos, path='data/esm_data/', 
                 with_ec_number=False, return_name = False, **kwargs) :
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_pos = mine_pos
        self.path = path
        self.with_ec_number = with_ec_number
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

        self.return_name = return_name

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = random_positive(anchor, self.id_ec, self.ec_id)

        if not self.return_name:
            anchor = format_esm(torch.load(f'{self.path}/' + anchor + '.pt'))
            pos = format_esm(torch.load(f'{self.path}/' + pos + '.pt'))
        
        if self.with_ec_number:
            return anchor, pos, anchor_ec
        else:
            return anchor, pos
    


class MoCo_dataset_with_mine_EC_and_SMILE(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_pos, path='data/esm_data/', 
                 with_ec_number=False, use_random_augmentation=False,
                 return_name = False, use_smile_cls_token=False, use_SMILE_cls_token=False):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_pos = mine_pos
        self.path = path
        self.with_ec_number = with_ec_number
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)
        self.smile_embed = torch.load("Rhea_tensors.pt", map_location='cpu')
        self.rhea_map = pd.read_csv("rhea2ec.tsv", sep='\t')
        self.all_ecs = self.rhea_map.ID.values
        self.negative_mode = 'random'
        self.rhea_keys = list(self.smile_embed.keys())

        if use_SMILE_cls_token:
            for key in self.smile_embed:
                self.smile_embed[key] = self.smile_embed[key][0]

        self.use_random_augmentation = use_random_augmentation
        self.return_name = return_name
        self.use_smile_cls_token = use_smile_cls_token
        self.smile_embed_shape = next(iter(self.smile_embed.items()))[1][0].shape
        self.use_SMILE_cls_token = use_SMILE_cls_token

    def _get_smile_embed(self, anchor_ec):

        rhea = self.rhea_map.loc[self.rhea_map.ID == anchor_ec, 'RHEA_ID'].values
        smile_features = []
        for i in rhea:
            for j in range(1, 4):
                try:
                    smile_features.extend(self.smile_embed[str(i + j)]) # self.smile_embed[str(i)] is a list
                except:
                    pass
        
        if len(smile_features) == 0:
            smile_features = [torch.zeros(self.smile_embed_shape)]
        if self.use_random_augmentation:
            random_index = random.choice(range(len(smile_features)))
            smile_features = smile_features[random_index].mean(0)
        else:
            smile_features = torch.cat(smile_features, 0).mean(0)

        if self.negative_mode == 'random': 
            count = 0 
            while True:
                rhea_id_neg = random.choice(self.rhea_keys)
                if rhea_id_neg not in rhea:
                    break
                count += 1
                if count > 5:
                    break
            if self.use_random_augmentation:
                index = random.choice(range(2))
                negative_smile_features = self.smile_embed[rhea_id_neg][index].mean(0)
            else:
                negative_smile_features = torch.cat(self.smile_embed[rhea_id_neg]).mean(0)

        return smile_features, negative_smile_features

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        smile_features, negative_smile_features = self._get_smile_embed(anchor_ec)
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = random_positive(anchor, self.id_ec, self.ec_id)

        if not self.return_name:
            anchor = format_esm(torch.load(f'{self.path}/' + anchor + '.pt'))
            pos = format_esm(torch.load(f'{self.path}/' + pos + '.pt'))
        
        if self.with_ec_number:
            return anchor, pos, smile_features, negative_smile_features, anchor_ec
        else:
            return anchor, pos, smile_features, negative_smile_features