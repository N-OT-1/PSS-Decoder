import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from .boundary import boundary_counts, summarize_boundaries


@METRICS.register_module()
class NativeBoundaryIoUMetric(BaseMetric):
    def __init__(self, widths=(1, 2, 4), ignore_index=255, **kwargs):
        super().__init__(**kwargs)
        self.widths = tuple(widths)
        self.ignore_index = ignore_index

    def process(self, data_batch, data_samples):
        classes = len(self.dataset_meta['classes'])
        for sample in data_samples:
            truth = sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
            prediction = sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            counts, _ = boundary_counts(truth, prediction, classes, self.widths, self.ignore_index)
            self.results.append(counts)

    def compute_metrics(self, results):
        rows = summarize_boundaries(np.sum(results, axis=0), self.widths)
        return {f"BoundaryIoU_{row['width_pixels']}px": row['macro_iou'] for row in rows}
