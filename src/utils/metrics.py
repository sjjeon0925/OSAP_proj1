"""src/utils/metrics.py"""
import numpy as np

class SegmentationMetric:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def add_batch(self, preds, gts):
        for p, g in zip(preds, gts):
            # 255 라벨(ignore)은 계산에서 제외 
            mask = (g >= 0) & (g < self.n_classes)
            self.confusion_matrix += np.bincount(
                self.n_classes * g[mask].astype(int) + p[mask],
                minlength=self.n_classes**2
            ).reshape(self.n_classes, self.n_classes)

    def mean_iou(self):
        # IoU = TP / (TP + FP + FN) [cite: 58]
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / union
        return np.nanmean(iou) # 클래스별 IoU의 평균 [cite: 56]