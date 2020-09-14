import torch
import torch.nn as nn

from model.memory_module import MemModule

# mean square error (MSE) is measure of the reconstruction quality (the crietrion for anomaly detection)


class MEMAE(nn.Module):
    def __init__(self):
        super(MEMAE, self).__init__()
        self.channel_num_in = 114

        self.encoder = nn.Sequential(
            nn.Linear(self.channel_num_in, 60),
            nn.BatchNorm1d(60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.BatchNorm1d(30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.Linear(10, 3),
            nn.BatchNorm1d(3),
            nn.Tanh(),
        )

        mem_dim = 120

        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=3)

        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.BatchNorm1d(30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.BatchNorm1d(60),
            nn.Tanh(),
            nn.Linear(60, self.channel_num_in),
            nn.BatchNorm1d(self.channel_num_in),
            nn.Tanh(),
        )

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att}

    def compute_loss(self, outputs, target):
        output = outputs['output']
        loss = torch.nn.MSELoss()(output, target)
        return loss

    def compute_batch_error(self, outputs, target):
        output = outputs['output']
        loss = torch.nn.MSELoss(reduction='none')(output, target)
        batch_error = loss.mean(1)
        return batch_error


if __name__ == "__main__":
    model = MEMAE()
    print(model)
