from torch.nn import Module, Parameter
from torch.optim import SGD
from torch import randn, Tensor, mean, eye, trace, abs
from torch import sum as th_sum
from matplotlib.pyplot import matshow, gca, axes, show, figure, savefig
from typing import Tuple, Dict
from numpy import load as load_npy
from numpy import save as save_npy
from numpy.linalg import matrix_rank
from numpy import ndarray, sum, sqrt, arange, exp
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import csgraph

DS_PATH = "data/small/10p/"


def dispmat(m: ndarray, display=False, save=False, filename='M'):
    _ = figure(figsize=(M.shape[0]//M.shape[1], 2*M.shape[0]//M.shape[1]))
    matshow(m)
    gca().set_aspect('auto')
    if display:
        show()
    if save:
        savefig(filename+'.png', dpi=600)


class DMF(Module):
    def __init__(self, n, m, k):
        super().__init__()
        self.A = Parameter(0.1*eye(n, k))
        self.Z = Parameter(0.1*eye(k, k))
        self.B = Parameter(0.1*eye(k, m))

    def forward(self):
        return self.A @ (self.Z @ self.B)


# incomplete matrix
Y_np = load_npy(DS_PATH+"Y_rec_10p.npy")
Y = Tensor(Y_np)

# GT full matrix
M_np = load_npy(DS_PATH+"M_rec_10p.npy")
M = Tensor(M_np)

# Mask
Omega_np = (Y_np != 0).astype(float)
Omega = Tensor(Omega_np)

# graphs
L_row = Tensor(csgraph.laplacian(row_graph(cosine_distances(Y_np),
                                           k=10), normed=True))
L_col = Tensor(csgraph.laplacian(column_graph(Y_np.shape[1],
                                              weights=[i for i in exp(-arange(5))])))
g_row = 0.0
g_col = 0.0001

# m, n, k
n_ = Y.shape[0]
m_ = Y.shape[1]
k_ = 2*Y.shape[0]

# Model initialization
model = DMF(n=n_, m=m_, k=k_)

use_gpu = True
if use_gpu:
    Y = Y.cuda()
    M = M.cuda()
    model.cuda()
    Omega = Omega.cuda()
    L_row = L_row.cuda()
    L_col = L_col.cuda()

# Optimization parameters
num_iters = 10**5
lr = 10**(-5)

# Optimizer
opt = SGD(model.parameters(), lr=lr)

# Training loop
for iter in range(num_iters):
    opt.zero_grad()
    Y_hat = model.forward()
    train_loss = th_sum((Omega*(Y_hat - Y))**2)
    dir_row = trace(Y_hat.T @ L_row @ Y_hat)
    dir_col = trace(Y_hat @ L_col @ Y_hat.T)
    loss = train_loss + g_row * dir_row + g_col * dir_col
    loss.backward()
    opt.step()
    total_loss = th_sum((M - Y_hat)**2)
    if iter > 100:
        lr /= 2
    train_rmse = sqrt(train_loss.detach().item() / sum(Omega_np))
    total_rmse = sqrt(total_loss.detach().item() / (n_ * m_))
    if iter % 20 == 0:
        print(f"Iter {iter}: Train loss: {train_loss}, Dir row: {dir_row}, Dir col: {dir_col}"
              f" Train RMSE: {train_rmse}, Total RMSE: {total_rmse}")
        save_npy(f"Y_hat_10p_{iter}_SGMC_col.npy", Y_hat.cpu().detach().numpy())
