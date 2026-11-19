from .utilities import require
import numpy as np

from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def classification_metrics(labels, scores, predictions):
    labels, scores, predictions = np.asarray(labels), np.asarray(scores), np.asarray(predictions)
    require(np.isfinite(scores).all(), 'Nonfinite scores')
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0,1]).ravel()
    return dict(macro_f1=float(f1_score(labels, predictions, average='macro', labels=[0,1], zero_division=0)),
                micro_f1=float(f1_score(labels, predictions, average='micro', zero_division=0)),
                binary_f1=float(f1_score(labels, predictions, zero_division=0)),
                precision=float(precision_score(labels, predictions, zero_division=0)),
                recall=float(recall_score(labels, predictions, zero_division=0)),
                accuracy=float(accuracy_score(labels, predictions)), fpr=float(fp/(fp+tn)) if fp+tn else None,
                average_precision=float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
                roc_auc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=len(labels))
