from __future__ import absolute_import, print_function
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn import metrics
from collections import Counter
from IPython.display import display

from model_memae import AutoencoderMem
from memae.util import get_dataset, set_seed


x_train = get_dataset(filename='train_normal.hdf5', tagname='x')
x_test = get_dataset(filename='test.hdf5', tagname='x')
y_test = get_dataset(filename='test.hdf5', tagname='y').squeeze()


atk, nrm = Counter(y_test)[1], Counter(y_test)[0]
print('Attack:', atk)
print('Normal:', nrm)

data_sampler = RandomSampler(x_train)
data_loader = DataLoader(x_train, sampler=data_sampler, batch_size=64)

model = AutoencoderMem()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.zero_grad()
set_seed()

model.train(True)

loss_fn = torch.nn.MSELoss(reduction='none')

# Train
for epoch in range(20):
    epoch_loss = []
    for step, batch in enumerate(data_loader):
        batch = batch.type(torch.float32)

        outputs = model(batch)

        output = outputs['output']
        att = outputs['att']

        loss_each = loss_fn(batch, output)
        loss_all = torch.mean(loss_each)

        loss_all.backward()
        optimizer.step()
        model.zero_grad()

        epoch_loss.append(loss_all.item())

    print("epoch {}: {}".format(epoch, sum(epoch_loss)/len(epoch_loss)))


model.train(False)


# Test
eval_sampler = SequentialSampler(x_test)
eval_dataloader = DataLoader(x_test, sampler=eval_sampler, batch_size=64)

model.eval()
error = []
for batch in eval_dataloader:
    batch = batch.type(torch.float32)

    outputs = model(batch)

    output = outputs['output']
    att = outputs['att']

    loss = loss_fn(batch, output)
    loss_mean = loss.mean(1)

    error += loss_mean.detach().tolist()

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
        'TPR' : tpr, 'FPR': fpr, 'Threshold': thr, 'Accuracy': acc,
        'Specifity': spe, 'Precision': pre, 'Recall': rec, 'F1-score': f1
    })
df = df.round(3)

display(df)
