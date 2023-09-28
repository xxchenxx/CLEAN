from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
from CLEAN.distance_map import *
from CLEAN.evaluate import *
import torch

eval_dataset = 'new'
id_ec_test, ec_id_dict_test = get_ec_id_dict(f'./data/{eval_dataset}.csv')

prediction = torch.load("prediction.pth.tar")

ec_dict = list(prediction.keys())
print(ec_dict)
prediction_concat = [prediction[key] for key in ec_dict]
prediction_concat = np.stack(prediction_concat, 1)
# print(prediction_concat)
assert len(id_ec_test) == prediction_concat.shape[0]
labels = []
for key in id_ec_test:
    ecs = id_ec_test[key]
    index = []
    for key in ecs:
        if key in ec_dict:
            index.append(ec_dict.index(key))
    
    label = np.zeros(len(ec_dict))
    for i in index: label[i] = 1
    labels.append(label)
# print(index)
labels = np.stack(labels)
normal_argmax = np.argmax(prediction_concat, -1)
random = np.random.rand(labels.shape[0], labels.shape[1]) >= 0.5
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, average_precision_score

print(precision_score(labels, normal_argmax, average='weighted'))
print(recall_score(labels, normal_argmax, average='weighted'))
print(f1_score(labels, normal_argmax, average='weighted'))

print(precision_score(labels, random, average='weighted'))
print(recall_score(labels, random, average='weighted'))
print(f1_score(labels, random, average='weighted'))


