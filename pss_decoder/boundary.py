import cv2
import numpy as np
WIDTHS = (1, 2, 4)

def inner_band(mask, width):
    binary = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=width, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return binary.astype(bool) & ~eroded.astype(bool)

def boundary_counts(gt, prediction, classes, widths=WIDTHS, ignore_index=255):
    gt, prediction = (np.asarray(gt), np.asarray(prediction))
    if gt.ndim != 2 or gt.shape != prediction.shape:
        raise ValueError('Boundary inputs must be aligned 2-D label arrays.')
    if not np.issubdtype(gt.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError('Boundary labels must contain integer class IDs.')
    if classes < 1 or not widths or len(set(widths)) != len(widths):
        raise ValueError('Expected classes and unique positive boundary widths.')
    if any((not isinstance(w, (int, np.integer)) or isinstance(w, bool) or w < 1 for w in widths)):
        raise ValueError('Boundary width must be a positive integer.')
    valid = gt != ignore_index
    if np.any((gt[valid] < 0) | (gt[valid] >= classes)):
        raise ValueError('Ground truth contains an unexpected non-void class.')
    if np.any((prediction[valid] < 0) | (prediction[valid] >= classes)):
        raise ValueError('Prediction contains an unexpected valid-region class.')
    supports = [cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=width, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool) for width in widths]
    counts = np.zeros((len(widths), classes, 2), dtype=np.int64)
    for category in range(classes):
        truth = (gt == category) & valid
        predicted = (prediction == category) & valid
        if not truth.any() and (not predicted.any()):
            continue
        for j, (width, support) in enumerate(zip(widths, supports)):
            gb = inner_band(truth, width) & support
            pb = inner_band(predicted, width) & support
            counts[j, category] = [np.count_nonzero(gb & pb), np.count_nonzero(gb | pb)]
    return (counts, np.asarray([np.count_nonzero(s) for s in supports], dtype=np.int64))

def summarize_boundaries(counts, widths=WIDTHS):
    values = np.asarray(counts)
    if values.ndim != 3 or values.shape[0] != len(widths) or values.shape[-1] != 2:
        raise ValueError('Expected width x class x (intersection, union).')
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values[..., 0] > values[..., 1]):
        raise ValueError('Invalid boundary intersection/union counts.')
    rows = []
    for width, pair in zip(widths, values):
        intersection, union = (pair[:, 0], pair[:, 1])
        iou = np.divide(100 * intersection.astype(float), union, out=np.full(len(union), np.nan), where=union > 0)
        rows.append(dict(width_pixels=int(width), class_iou=[float(v) if np.isfinite(v) else None for v in iou], macro_iou=float(iou.mean()) if np.isfinite(iou).all() else None, undefined_classes=np.flatnonzero(union == 0).tolist()))
    return rows
