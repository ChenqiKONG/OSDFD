import numpy as np
import torch
from scipy import interpolate
from sklearn.metrics import roc_curve, roc_auc_score, auc, average_precision_score
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.optimize import brentq

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp +fn==0) else float(tp) / float(tp +fn)
    fpr = 0 if (fp +tn==0) else float(fp) / float(fp +tn)
    acc = float(tp +tn) /dist.shape[0]
    return tpr, fpr, acc

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def FPR_FNR(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
    FPR = fp / (tn*1.0 + fp*1.0)
    FNR = fn / (fn * 1.0 + tp * 1.0)
    return FPR, FNR

def eer_auc(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    AUC = auc(fpr, tpr)
    return eer, AUC

def compute_mAP(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

def get_metrics(pred, gt, thre):
    mAP = compute_mAP(gt, pred)
    gt_bool = (gt>0.5)
    _,_,ACC = calculate_accuracy(thre, pred, gt_bool)
    FPR, FNR = FPR_FNR(thre, pred, gt_bool)
    EER, AUC = eer_auc(gt, pred) 
    return AUC, ACC, FPR, FNR, EER, mAP

# def get_metrics2(pred, gt, thre):
#     EER, AUC = eer_auc(gt, pred) 
#     return AUC, EER