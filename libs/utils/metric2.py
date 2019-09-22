# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    #print(np.unique(label_pred))
    #print(np.unique(label_true))
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        if(lt.size > 0):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Overall Acc": acc,
        "Mean Acc": acc_cls,
        "FreqW Acc": fwavacc,
        "Mean IoU": mean_iu,
    }, cls_iu

def scores_gzsl(label_trues, label_preds, n_class, seen_cls, unseen_cls):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        if(lt.size > 0):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(hist).sum() / hist.sum()
        seen_acc = np.diag(hist)[seen_cls].sum() / hist[seen_cls].sum()
        unseen_acc = np.diag(hist)[unseen_cls].sum() / hist[unseen_cls].sum()
        h_acc = 2./(1./seen_acc + 1./unseen_acc)
        if np.isnan(h_acc):
            h_acc = 0
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        seen_acc_cls = np.diag(hist)[seen_cls] / hist.sum(axis=1)[seen_cls]
        unseen_acc_cls = np.diag(hist)[unseen_cls] / hist.sum(axis=1)[unseen_cls]
        acc_cls = np.nanmean(acc_cls)
        seen_acc_cls = np.nanmean(seen_acc_cls)
        unseen_acc_cls = np.nanmean(unseen_acc_cls)
        h_acc_cls = 2./(1./seen_acc_cls + 1./unseen_acc_cls)
        if np.isnan(h_acc_cls):
            h_acc_cls = 0
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        seen_mean_iu = np.nanmean(iu[seen_cls])
        unseen_mean_iu = np.nanmean(iu[unseen_cls])
        h_mean_iu = 2./(1./seen_mean_iu + 1./unseen_mean_iu)
        if np.isnan(h_mean_iu):
            h_mean_iu = 0
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq * iu)
        fwavacc[np.isnan(fwavacc)] = 0
        seen_fwavacc = fwavacc[seen_cls].sum()
        unseen_fwavacc = fwavacc[unseen_cls].sum()
        h_fwavacc = 2./(1./seen_fwavacc + 1./unseen_fwavacc)
        if np.isnan(h_fwavacc):
            h_fwavacc = 0
        fwavacc = fwavacc.sum()
        cls_iu = dict(zip(range(n_class), iu))

    return {
        "Overall Acc": acc,
        "Overall Acc Seen": seen_acc,
        "Overall Acc Unseen": unseen_acc,
        "Overall Acc Harmonic": h_acc,
        "Mean Acc": acc_cls,
        "Mean Acc Seen": seen_acc_cls,
        "Mean Acc Unseen": unseen_acc_cls,
        "Mean Acc Harmonic": h_acc_cls,
        "FreqW Acc": fwavacc,
        "FreqW Acc Seen": seen_fwavacc,
        "FreqW Acc Unseen": unseen_fwavacc,
        "FreqW Acc Harmonic": h_fwavacc,
        "Mean IoU": mean_iu,
        "Mean IoU Seen": seen_mean_iu,
        "Mean IoU Unseen": unseen_mean_iu,
        "Mean IoU Harmonic": h_mean_iu,
    }, cls_iu

if __name__ == "__main__":
    a = [np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])]
    b = [np.array([1, 1, 4, 1, 2, 1, 1, 4, 1, 2, 1, 2, 3, 4, 5])]
    print(scores_gzsl(a,b,6,[0,1,2],[3,4,5]))



