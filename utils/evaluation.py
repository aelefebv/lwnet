import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import rankdata

def dice_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

def accuracy_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    num_els = actual.size
    intersection = np.logical_and(actual, predicted)
    return float(intersection.sum()) / num_els

def fast_auc(actual, predicted):
    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(r[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def ewma(data, window=5):
    # exponetially-weighted moving averages
    data = np.array(data)
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc


def evaluate(logits, labels):
    all_targets = []
    all_probs_0 = []
    all_probs_1 = []
    all_probs_2 = []

    for i in range(len(logits)):
        probs = torch.nn.Softmax(dim=0)(logits[i]).detach().cpu().numpy()
        all_probs_0.extend(probs[0].ravel())
        all_probs_1.extend(probs[1].ravel())
        all_probs_2.extend(probs[2].ravel())

        target = labels[i].numpy()

        all_targets.append(target.ravel())

    all_probs_np = np.stack([all_probs_0, all_probs_1, all_probs_2], axis=1)
    all_preds_np = np.argmax(all_probs_np, axis=1)

    all_targets_np = np.hstack(all_targets)
    all_preds_np = 1 + all_preds_np  # we are predicting only three classes and ignoring background
    all_preds_np[all_targets_np == 0] = 0

    arts_bin = all_targets_np == 2
    veins_bin = all_targets_np == 3
    print(arts_bin.shape, veins_bin.shape)
    print(all_preds_np.shape, all_targets_np.shape)

    print(all_preds_np[arts_bin].shape, all_preds_np[veins_bin].shape)
    
    all_preds_bin = np.stack([all_preds_np[arts_bin],all_preds_np[veins_bin]], axis=0)
    all_targets_bin = np.stack([np.zeros_like(arts_bin),np.ones_like(veins_bin)], axis=0)

    print(fast_auc(all_targets_bin, all_preds_bin))
    sys.exit()
    return f1_score(all_targets_np, all_preds_np, average='weighted', labels=[1, 2, 3]), \
           mcc(all_targets_np[all_targets_np != 0], all_preds_np[all_targets_np != 0])

# def evaluate(logits, labels):
#     all_targets = []
#     all_probs_0 = []
#     all_probs_1 = []
#     all_probs_2 = []
#     all_probs_3 = []
#
#     for i in range(len(logits)):
#         probs = torch.nn.Softmax(dim=0)(logits[i]).detach().cpu().numpy()
#         all_probs_0.extend(probs[0].ravel())
#         all_probs_1.extend(probs[1].ravel())
#         all_probs_2.extend(probs[2].ravel())
#         all_probs_3.extend(probs[3].ravel())
#
#         target = labels[i].numpy()
#
#         all_targets.append(target.ravel())
#
#     all_probs_np = np.stack([all_probs_0, all_probs_1, all_probs_2, all_probs_3], axis=1)
#     all_preds_np = np.argmax(all_probs_np, axis=1)
#     all_targets_np = np.hstack(all_targets)
#
#     return f1_score(all_targets_np, all_preds_np,average='weighted'), mcc(all_targets_np, all_preds_np)
#
# # def evaluate(logits, labels, n_classes, ignore_index = -100, fast=True):
# #
# #     all_probs_0 = []
# #     all_targets = []
# #
# #
# #     if n_classes == 4:
# #         all_probs_1 = []
# #         all_probs_2 = []
# #         all_probs_3 = []
# #
# #     act = torch.sigmoid if n_classes==1 else torch.nn.Softmax(dim=0)
# #
# #     for i in range(len(logits)):
# #         # prediction = act(logits[i]).detach().cpu().numpy()[-1]  # this takes last channel in multi-class, ok for 2-class
# #         # logits[i] is n_classes x h x w
# #         prob = act(logits[i]).detach().cpu().numpy()  # prob is n_classes x h x w
# #         target = labels[i].cpu().numpy()
# #
# #         if n_classes==1:
# #             all_probs_0.extend(prob.ravel())
# #         else:
# #             all_probs_0.extend(prob[0].ravel())
# #             all_probs_1.extend(prob[1].ravel())
# #             all_probs_2.extend(prob[2].ravel())
# #             all_probs_3.extend(prob[3].ravel())
# #
# #         all_targets.append(target.ravel())
# #
# #     if n_classes == 1: all_probs_np = np.hstack(all_probs_0)
# #     else: all_probs_np = np.stack([all_probs_0, all_probs_1, all_probs_2, all_probs_3], axis=1)
# #
# #     all_targets_np = np.hstack(all_targets)
# #
# #     all_probs_np = all_probs_np[all_targets_np != ignore_index]
# #     all_targets_np = all_targets_np[all_targets_np!=ignore_index]
# #
# #     if n_classes == 4:
# #         all_preds_np = np.argmax(all_probs_np, axis=1)
# #         return roc_auc_score(all_targets_np, all_probs_np, multi_class='ovo',average='weighted'), f1_score(all_targets_np, all_preds_np,average='weighted')
# #     else:
# #         all_preds_np = all_probs_np > 0.5
# #         if fast==True:
# #             return fast_auc(all_targets_np>0.5, all_probs_np), f1_score(all_targets_np>0.5, all_preds_np)
# #         else:
#             # return roc_auc_score(all_targets_np, all_probs_np), f1_score(all_targets_np, all_preds_np)