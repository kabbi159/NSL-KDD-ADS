import torch
import torch.nn as nn
import numpy as np

# mean square error (MSE) is measure of the reconstruction quality (the crietrion for anomaly detection)

# 3.2 Compression Network
# z_c : encoder로 생성된 hidden feature (reduced low-dimensional representation)
# z_r : input data와 autoencoder를 거친 output 간의 reconsturction error features
# z_r는 multi-dimensional이 될 수 있다. 여러 개의 distance metric을 사용하는 방식으로 ...
# (Euclidean distance, relative Euclidean distance, cosine similarity 등)

# z = [z_c, z_r]
# z를 estimation network에 전달한다

# 3.3 Estimation Network
# density estimation under the framework of GMM (Gaussian Mixture Model)


class DAGMM(nn.Module):
    def __init__(self):
        super(DAGMM, self).__init__()
        self.channel_num_in = 114
        self.latent_dim = 3  # 1 + 2
        self.n_gmm = 2

        self.lambda_energy = 0.1
        self.lambda_cov_diag = 0.005

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
            nn.Linear(10, 1),
            nn.BatchNorm1d(1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 10),
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
        )

        self.estimation = nn.Sequential(
            nn.Linear(self.latent_dim, 10),
            # nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, self.n_gmm),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        z_c = self.encoder(x)
        output = self.decoder(z_c)

        cos = torch.nn.functional.cosine_similarity(x, output, dim=1)  # cosine similarity
        euc = (x-output).norm(2, dim=1) / x.norm(2, dim=1)  # relative euclid distance

        z_r = torch.cat([cos.unsqueeze(-1), euc.unsqueeze(-1)], dim=1)  # [batch size, z_r dim:2]
        z = torch.cat([z_c, z_r], dim=1)  # [batch size, z_c dim + z_r dim]

        gamma = self.estimation(z)  # [batch size, n_gmm] [64, 2]
        phi = torch.sum(gamma, dim=0) / gamma.size(0)
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))  # [64, 2, 5]
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)  # [64, 2, 5, 5]

        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0)\
              / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return {'output': output, 'z': z, 'gamma': gamma, 'phi': phi, 'mu': mu, 'cov': cov}  # 여기까지 확인 완료!

    def compute_energy(self, z, phi, mu, cov, size_average=True):
        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12

        for i in range(k):
            # K x D x D
            cov_k = cov[i] + torch.eye(D)*eps
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))  # [1, 5, 5]

            det_cov.append(torch.cholesky((2*np.pi) * cov_k).diag().prod().unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov)

        # maybe avoid overflow
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max(exp_term_tmp.clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def compute_loss(self, outputs, target):
        output = outputs['output']

        energy, cov_diag = self.compute_energy(outputs['z'], outputs['phi'], outputs['mu'], outputs['cov'])

        recon_error = torch.mean((output - target) ** 2)

        loss = recon_error + self.lambda_energy * energy + self.lambda_cov_diag * cov_diag

        return loss

    def compute_batch_error(self, outputs, target):
        output = outputs['output']
        batch_error = torch.mean((output - target) ** 2, 1)

        return batch_error

# if __name__ == "__main__":
#     model = DAGMM()
#     print(model)
#     import numpy as np
#     from util.load_data import get_dataset
#     x_test = get_dataset(filename='test.hdf5', tagname='x')
#
#     # print(x_test[0])
#
#     x_tmp = x_test[0:64]
#
#     tmp = torch.tensor(np.array(x_tmp, dtype=np.float32))  # batch 2일 때 오류가 남...
#     # have cholesky_cpu: U is zero, singular U. problem
#     # https://medium.com/swlh/five-pytorch-tensor-operations-you-had-absolutely-no-idea-about-e25d3ae8db05 참고
#     print(tmp.size())
#
#     a = model(tmp)
#
#     print(a['phi'].size())  # [3] n_gmm
#     print(a['mu'].size())  # [3, 5] n_gmm z_dim
#     print(a['cov'].size())  # [3, 5, 5] n_gmm z_dim z_dim
#
#     sample_energy, cov_diag = model.compute_energy(a['z'], a['phi'], a['mu'], a['cov'])
#
#     # loss
#     # L2 norm (reconstruction error) + E(z) (sample energy)
#     # + P (패널티: 특이성 문제를 해결하기 위해서 - 공분산 행렬의 대각선 항목이 0으로 내려갈때)
#
#     lambda_energy = 0.1
#     lambda_cov_diag = 0.005
#
#     recon_error = torch.mean((tmp - a['output']) ** 2)
#
#     loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
#
#     loss.backward()
