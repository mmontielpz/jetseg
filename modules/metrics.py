import sys
import torch


def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there are no pixels for this class, set IoU to NaN
        else:
            ious.append(float(intersection) / float(union))
    return ious


def compute_mIoU(pred, target, num_classes):
    ious = []
    for i in range(len(pred)):
        ious.append(compute_iou(pred[i], target[i], num_classes))

    ious = torch.tensor(ious, device=pred.device)
    non_nans = ~torch.isnan(ious)
    mean_iou = torch.mean(ious[non_nans])
    mean_iou = float(mean_iou)
    mean_iou *= 100
    mean_iou = round(mean_iou, 1)
    return mean_iou
