import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from IPython.display import display
from sklearn import metrics
from collections import Counter


def make_figure(y_test, error):
    atk, nrm = Counter(y_test)[1], Counter(y_test)[0]

    # make figure
    sns.set(style='whitegrid', rc={"grid.linewidth": 0.5, 'grid.linestyle': '--'})
    plt.figure(dpi=80)

    # ROC curve
    fpr, tpr, threshold = metrics.roc_curve(y_test, error, drop_intermediate=False)
    plt.plot(fpr, tpr, linestyle='-')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='_nolegend_')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(linestyle='--')
    plt.show()

    # AUC
    print("AUC:", metrics.auc(fpr, tpr))
    print()

    # performance table
    fpr_, tpr_, thr_ = fpr, tpr, threshold
    tpr, fpr, acc, rec, pre, spe, f1, thr = [list() for i in range(8)]
    for r in range(90, 100):
        r *= 0.01
        tpr.append(tpr_[np.where(tpr_ >= r)[0][0]])
        fpr.append(fpr_[np.where(tpr_ >= r)[0][0]])
        acc.append((tpr[-1] * atk + (1 - fpr[-1]) * nrm) / (atk + nrm))
        rec.append(tpr[-1])
        pre.append(tpr[-1] * atk / (tpr[-1] * atk + fpr[-1] * nrm))
        spe.append(1 - fpr[-1])
        f1.append(2 * rec[-1] * pre[-1] / (rec[-1] + pre[-1]))
        thr.append(thr_[np.where(tpr_ >= r)[0][0]])
    df = pd.DataFrame({
        'TPR': tpr, 'FPR': fpr, 'Threshold': thr, 'Accuracy': acc,
        'Specifity': spe, 'Precision': pre, 'Recall': rec, 'F1-score': f1
    })
    df = df.round(3)

    display(df)
