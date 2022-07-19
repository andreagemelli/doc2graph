from nis import match
from langcodes import best_match
import torch
import torchvision
import numpy as np
from scipy.optimize import linprog


def match_pred_w_gt(bbox_preds : torch.Tensor, bbox_gts : torch.Tensor):
    bbox_iou = torchvision.ops.box_iou(boxes1=bbox_preds, boxes2=bbox_gts)
    bbox_iou = bbox_iou.numpy()

    A_eq = np.zeros(shape=(bbox_iou.shape[1], bbox_iou.shape[0] * bbox_iou.shape[1]))
    for r in range(bbox_iou.shape[1]):
        A_eq[r, r::bbox_iou.shape[1]] = 1
    b_eq = np.ones(shape=A_eq.shape[0])
    A_ub = np.zeros(shape=(bbox_iou.shape[0], bbox_iou.shape[0] * bbox_iou.shape[1]))
    for r in range(bbox_iou.shape[0]):
        st = r * bbox_iou.shape[1]
        A_ub[r, st:st + bbox_iou.shape[1]] = 1
    b_ub = np.ones(shape=A_ub.shape[0])

    assignaments_score = linprog(c=-bbox_iou.reshape(-1), A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="simplex")
    assignaments_score = assignaments_score.x.reshape(bbox_iou.shape)
    assignaments_ids = assignaments_score.argmax(axis=1)

    # matched
    opt_assignaments = {}
    for idx in range(assignaments_score.shape[0]):
        if (bbox_iou[idx, assignaments_ids[idx]] > 0) and (assignaments_score[idx, assignaments_ids[idx]] > 0.9):
            opt_assignaments[idx] = assignaments_ids[idx] 
    # unmatched predictions
    false_positive = [idx for idx in range(bbox_preds.shape[0]) if idx not in opt_assignaments]
    # unmatched gts
    false_negative = [idx for idx in range(bbox_gts.shape[0]) if idx not in opt_assignaments.values()]

    return opt_assignaments, false_positive, false_negative


if __name__ == "__main__":
    bbox_gts = torch.Tensor([[3, 3, 6, 6], [7, 7, 11, 11], [10, 10, 17, 17]])
    bbox_preds = torch.Tensor([[1, 1, 4, 4], [5, 5, 7, 7], [15, 15, 20, 20], [2, 2, 4, 4]])

    print(match_pred_w_gt(bbox_preds, bbox_gts))