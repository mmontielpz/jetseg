import pathmagic

import sys
import torch

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

assert pathmagic


class JetLoss(torch.nn.Module):
    def __init__(self, n_classes=32,
                 pixels_per_class=None, weight=True, debug=False):
        super().__init__()

        self.n_classes = n_classes
        self.pixels_per_class = pixels_per_class
        self.weight = weight
        self.background_idx = 30

        self.eps = 1e-6
        self.debug = debug

    def compute_ioub_background(self, target, predict):
        background_mask = target[:, self.background_idx, :, :]
        predict_background_mask = predict[:, self.background_idx, :, :]
        intersection = torch.sum(background_mask * predict_background_mask, dim=(1, 2))
        union = torch.sum(background_mask + predict_background_mask, dim=(1, 2)) - intersection
        ioub = (intersection + self.eps) / (union + self.eps)
        ioub = torch.mean(ioub).cuda()
        ioub *= 10

        # clamp the loss to avoid negative values
        ioub = torch.clamp(ioub, min=0.0)

        return ioub

    def compute_precision(self, target, predict):
        tp = torch.sum(target * predict, dim=(2, 3))
        tot = torch.sum(predict, dim=(2, 3))
        fp = tot - tp
        tp = tp + self.eps
        fp = fp + self.eps
        precision = (tp + self.eps) / (tp + fp + self.eps)
        precision = torch.mean(precision).cuda()
        precision *= 100

        # clamp the loss to avoid negative values
        precision = torch.clamp(precision, min=0.0)

        return precision

    def compute_recall(self, target, predict):
        tp = torch.sum(target * predict, dim=(2, 3))
        tot = torch.sum(target, dim=(2, 3))
        fn = tot - tp
        tp = tp + self.eps
        fn = fn + self.eps
        recall = (tp + self.eps) / (fn + tp + self.eps)
        recall = torch.mean(recall).cuda()

        # clamp the loss to avoid negative values
        recall = torch.clamp(recall, min=0.0)

        return recall

    def compute_class_weights(self, target):
        total_pixels = torch.sum(target, dim=(2, 3))
        class_weights = total_pixels / torch.sum(total_pixels)
        class_weights[:, self.background_idx] = 0.0
        class_weights = torch.mean(class_weights)
        # class_weights *= 10

        return class_weights

    def forward(self, predict, target):

        # Computing recall and precision loss
        recall = self.compute_recall(target, predict)
        precision = self.compute_precision(target, predict)

        if self.weight:
            weights = self.compute_class_weights(target)
        else:
            weights = 1.0

        # Adding weights
        recall_loss = recall * weights
        prec_loss = precision * weights

        if self.debug:
            print(f'[DEBUG] Weighted Recall Loss: {recall_loss}')
            print(f'[DEBUG] Weighted Precision Loss: {prec_loss}')

        # Computing Intersection over Union for Background (IoUB)
        ioub_loss = self.compute_ioub_background(target, predict)

        # Adding more importance to background loss
        ioub_loss *= weights

        if self.debug:
            print(f'[DEBUG] Background Loss: {ioub_loss}')

        # Compute total loss (recall loss + precision loss + IoUB loss)
        loss = recall_loss + prec_loss + ioub_loss

        # Scale loss
        loss *= 10

        # clamp the loss to avoid negative values
        loss = torch.clamp(loss, min=0.0)

        # Require grad
        loss = loss.requires_grad_()

        return loss
