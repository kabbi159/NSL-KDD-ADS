from __future__ import absolute_import, print_function
import torch

import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model.memae import MEMAE
from model.dagmm import DAGMM
from util.load_data import get_dataset, set_seed
from util.visualize import make_figure

# Argument Setting
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size for train and test")
parser.add_argument("--seed", default=42, type=int,
                    help="random seed for reproductability")
parser.add_argument("--lr", default=0.0001, type=float,
                    help="learning rate")
parser.add_argument("--epoch", default=5, type=int,
                    help="training epochs")
parser.add_argument("--model", default="memae", type=str,
                    help="model list = ['dagmm', 'memae']")

args = parser.parse_args()

# Fix seed
set_seed(args.seed)

# Model list
model_all = {
    'dagmm': DAGMM(),
    'memae': MEMAE()
}

# dataset path can be
x_train = get_dataset(dirname='../DAGMM-nslkdd/hdf5/', filename='train_normal.hdf5', tagname='x')
x_test = get_dataset(dirname='../DAGMM-nslkdd/hdf5/', filename='test.hdf5', tagname='x')
y_test = get_dataset(dirname='../DAGMM-nslkdd/hdf5/', filename='test.hdf5', tagname='y').squeeze()

data_sampler = RandomSampler(x_train)
data_loader = DataLoader(x_train, sampler=data_sampler, batch_size=64)

model = model_all[args.model]
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.zero_grad()

model.train(True)

# Train
for epoch in range(args.epoch):
    epoch_loss = []
    for step, batch in enumerate(data_loader):
        target = batch.type(torch.float32)

        outputs = model(target)
        loss = model.compute_loss(outputs, target)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_loss.append(loss.item())

    print("epoch {}: {}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))


model.train(False)


# Test
eval_sampler = SequentialSampler(x_test)
eval_dataloader = DataLoader(x_test, sampler=eval_sampler, batch_size=64)

model.eval()
error = []
for batch in eval_dataloader:
    target = batch.type(torch.float32)

    outputs = model(target)
    batch_error = model.compute_batch_error(outputs, target)

    error += batch_error.detach().tolist()

# visualize
make_figure(y_test, error)
