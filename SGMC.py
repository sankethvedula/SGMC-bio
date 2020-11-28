from torch.nn import Module, Parameter
from torch.optim import SGD
from torch import randn, Tensor, mean, eye, trace, abs
from torch import sum as th_sum
from matplotlib.pyplot import matshow, gca, axes, show, figure, savefig
from typing import Tuple, Dict
from numpy import load as load_npy
from numpy import asarray, float32, logical_and
from numpy import save as save_npy
from numpy.linalg import matrix_rank
from numpy import ndarray, sum, sqrt, arange, exp
from scipy.sparse import csgraph
from scipy.io import loadmat, savemat

# MGRNNforDTI - Drug-target interaction datasets
# e, gpcr, ic, nr
DS_PATH = "./MGRNNMforDTI/data_for_DMF/data_1_mgrnnm_e_S1.mat"


def dispmat(m: ndarray, display=False, save=False, filename='M'):
    _ = figure(figsize=(M.shape[0]//M.shape[1], 2*M.shape[0]//M.shape[1]))
    matshow(m)
    gca().set_aspect('auto')
    if display:
        show()
    if save:
        savefig(filename+'.png', dpi=600)


def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path.
        name_field, string containig the field name.
    """
    db = loadmat(path_file)
    ds = db[name_field]
    out = asarray(ds).astype(float32).T
    return out


class DMF(Module):
    def __init__(self, n, m, k):
        super().__init__()
        self.A = Parameter(0.1*eye(n, k))
        self.Z = Parameter(0.1*eye(k, k))
        self.B = Parameter(0.1*eye(k, m))

    def forward(self):
        return self.A @ (self.Z @ self.B)


# incomplete matrix
Y_np = load_matlab_file(DS_PATH, 'y2')
Y = Tensor(Y_np)

# GT full matrix
M_np = load_matlab_file(DS_PATH, 'Y')
M = Tensor(M_np)

A_row = load_matlab_file(DS_PATH, 'St')
A_col = load_matlab_file(DS_PATH, 'Sd')

# competitor
Y3 = Tensor(load_matlab_file(DS_PATH, 'y3'))

# Train mask
Omega_np = load_matlab_file(DS_PATH, 'omega_train')
Omega = Tensor(Omega_np)

# Test mask
Omega_test_np = load_matlab_file(DS_PATH, 'omega_test')
Omega_test = Tensor(Omega_test_np)

# graphs
L_row = Tensor(csgraph.laplacian(A_row, normed=True))
L_col = Tensor(csgraph.laplacian(A_col, normed=True))

g_row = 0.06
g_col = 0.06

if g_row == 0.0 and g_col == 0.0:
    method = 'DMF'
else:
    method = 'SGMC'


# m, n, k
n_ = Y.shape[0]
m_ = Y.shape[1]
k_ = int(0.5*Y.shape[0])

# Model initialization
model = DMF(n=n_, m=m_, k=k_)

use_gpu = True
if use_gpu:
    Y = Y.cuda()
    M = M.cuda()
    model.cuda()
    Omega = Omega.cuda()
    Omega_test = Omega_test.cuda()
    L_row = L_row.cuda()
    L_col = L_col.cuda()
    Y3 = Y3.cuda()

# Optimization parameters
num_iters = 20**5
lr = 10**(-4)

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
    total_loss = th_sum((Omega_test*(M - Y_hat))**2)
    comp_loss = th_sum((Omega_test*(M - Y3))**2)
    # if iter > 100:
    #     lr /= 2
    train_rmse = sqrt(train_loss.detach().item() / sum(Omega_np))
    test_rmse = sqrt(total_loss.detach().item() / sum(Omega_test_np))
    comp_rmse = sqrt(comp_loss.detach().item() / sum(Omega_test_np))
    if iter % 200 == 0:
        print(f"Iter {iter}: Train loss: {train_loss}, Dir row: {dir_row}, Dir col: {dir_col}"
              f" Train RMSE: {train_rmse}, Test RMSE: {test_rmse}, Comp RMSE: {comp_rmse}")

    if iter % 1000 == 0:
        savemat(DS_PATH[:-4]+f"_Y3_{iter}_{method}.mat", {'y3': Y_hat.cpu().detach().numpy()})
