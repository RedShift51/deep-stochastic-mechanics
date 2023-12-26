import os
import math
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusionCustom, Dataset
from torch.autograd import grad
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path
from functools import partial
from torchvision import transforms as T, utils
from accelerate import Accelerator
from multiprocessing import cpu_count
import bitsandbytes as bnb
from ema_pytorch import EMA
from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat
from random import random
import torch.nn.functional as F
from torch.autograd import grad

import numpy as np
import time



""" define a model """
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t.long())
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def divisible_by(numer, denom):
    return (numer % denom) == 0


def save(step, accelerator, ema, opt, results_folder, milestone):
    if not accelerator.is_local_main_process:
        return

    data = {
        'step': step,
        'model': accelerator.get_state_dict(model),
        'opt': opt.state_dict(),
        'ema': ema.state_dict(),
        'scaler': accelerator.scaler.state_dict() if exists(accelerator.scaler) else None,
        'version': "1.0.0"
    }

    torch.save(data, str(results_folder / f'model-{milestone}.pt'))


def load(results_folder, milestone, accelerator, model, opt):
    device = accelerator.device

    data = torch.load(str(results_folder / f'model-{milestone}.pt'), map_location=device)

    model = accelerator.unwrap_model(model)
    model.load_state_dict(data['model'])

    step = data['step']
    opt.load_state_dict(data['opt'])
    if accelerator.is_main_process:
        ema.load_state_dict(data["ema"])

    if 'version' in data:
        print(f"loading from version {data['version']}")

    if exists(accelerator.scaler) and exists(data['scaler']):
        accelerator.scaler.load_state_dict(data['scaler'])
    return step


only_sample_mode = False
objective = "pred_noise"
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False
)

diffusion = GaussianDiffusionCustom(
    model,
    image_size=256,
    timesteps=1000,  # number of steps
    sampling_timesteps=250
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

# setting training parameters
split_batches = True
mixed_precision_type = 'fp16'
amp = True
accelerator = Accelerator(
    split_batches=split_batches,
    mixed_precision=mixed_precision_type if amp else 'no'
)

channels = 3
convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(channels)
num_samples = 25
save_and_sample_every = 2500
batch_size = 4
gradient_accumulate_every = 4

train_num_steps = 300000
image_size = diffusion.image_size
max_grad_norm = 1.0

folder = '/home/alexey.buzovkin/data/jpg'
augment_horizontal_flip = True
ds = Dataset(
    folder,
    diffusion.image_size,
    augment_horizontal_flip=augment_horizontal_flip,
    convert_image_to=convert_image_to,
)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

dl = accelerator.prepare(dl)
dl = cycle(dl)

train_lr = 1e-4
adam_betas = (0.9, 0.99)
opt = bnb.optim.Adam8bit(diffusion.parameters(), lr=train_lr, betas=adam_betas)

# for logging results in a folder periodically
ema_update_every = 10,
ema_decay = 0.995
device = accelerator.device
if accelerator.is_main_process:
    ema = EMA(diffusion, beta=ema_decay, update_every=ema_update_every)
    ema.to(device)

results_folder = './results'
results_folder = Path(results_folder)
results_folder.mkdir(exist_ok=True)

""" ============== """


def get_jacobian(f, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """

    B, N = x.shape
    y = f(x)
    jacobian = list()
    for i in range(y.shape[-1]):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = grad(y,
                       x,
                       grad_outputs=v,
                       retain_graph=True,
                       create_graph=True,
                       allow_unused=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)

    jacobian = torch.stack(jacobian, dim=2).requires_grad_()

    return jacobian

N = 3 * 256 * 256

n_iter = 7000  # 7000
iter_threshhold = 400
loss_ref = np.inf
lines = [None, None, None]
losses = []
losses_newton = []
losses_sm = []
losses_init = []
N_fast = N


start = time.time()
with tqdm(range(n_iter), unit="iter") as tepoch:
    for tau in tepoch:
        if tau == 0:
            l_sm = 0
            l_nl = 0

            """ new batch initialisation """
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device).long()




            """ old batch initialisation """
            X_0 = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu,
                                                             (sig2 ** 2) * np.eye(d), batch_size)).to(device)


            X_0.requires_grad = True
            u0_val = torch.Tensor(u_0(X_0)).to(device)
            v0_val = torch.Tensor(v_0(X_0)).to(device)
            optimizer.zero_grad()
            eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), batch_size) for i in range(N + 1)]
            for i in range(1, N_fast + 1):  # iterate over time steps
                t_prev = time_splits[i - 1].clone()
                if i == 1:
                    X_prev = X_0.clone().to(device)  # X_{i-1}
                else:
                    X_prev = X_i.clone().to(device)  # X_{i-1}

                t_i = time_splits[i].clone()

                t_prev_batch = time_transform(t_prev).expand(batch_size, 1).to(device).clone()
                X_i = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                 net_v(X_prev, t_prev_batch)) \
                      + torch.Tensor(np.sqrt(h * T / (m * N)) * eps[i - 1]).to(device)

                if i > 1:
                    Xs = torch.hstack((Xs, X_i))
                else:
                    Xs = X_i.clone()

            X_0_iter0 = X_0.clone()  # collect X_0 from the initial iter
            Xs = torch.concat((X_0_iter0, Xs), axis=1)
        elif tau > iter_threshhold:
            BATCH_size = int(0.2 * batch_size)  # int(0.5* batch_size) 10

            X_0_iter = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu,
                                                                  (sig2 ** 2) * np.eye(d), BATCH_size)).to(device)
            X_0_iter.requires_grad = True
            u0_val = torch.Tensor(u_0(X_0_iter)).to(device)
            v0_val = torch.Tensor(v_0(X_0_iter)).to(device)
            optimizer.zero_grad()
            eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), BATCH_size) for i in range(N + 1)]
            # get one trajectory
            for i in range(1, N_fast + 1):
                t_prev = time_splits[i - 1].clone()
                if i == 1:
                    X_prev = X_0_iter.clone().to(device)
                else:
                    X_prev = X_i_iter.clone().to(device)

                t_i = time_splits[i].clone()

                t_prev_batch = time_transform(t_prev).expand(BATCH_size, 1).to(device)
                X_i_iter = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                      net_v(X_prev, t_prev_batch)) \
                           + torch.Tensor(np.sqrt(h * T / (m * N)) * eps[i - 1]).to(device)

                if i > 1:
                    Xs_iter = torch.hstack((Xs_iter, X_i_iter))
                else:
                    Xs_iter = X_i_iter.clone()
            Xs_iter = torch.concat((X_0_iter, Xs_iter), axis=1)

            # replace the old batch with the new one
            Xs_all = Xs_all.reshape((batch_size, (N + 1) * d))
            X_0_all = X_0_all.reshape((batch_size, d))
            Xs_all = torch.roll(Xs_all, -BATCH_size, 0)
            X_0_all = torch.roll(X_0_all, -BATCH_size, 0)
            Xs_all = torch.vstack((Xs_all[BATCH_size:].detach(), Xs_iter.detach()))
            Xs_all.requires_grad = True

            X_0_all = torch.vstack((X_0_all[BATCH_size:].detach(), X_0_iter.detach()))
            X_0_all.requires_grad = True

            Xs_all = Xs_all.reshape(batch_size * (N + 1), d)
            X_0_all = X_0_all.reshape(batch_size, d)

            time_splits_batches = time_splits.repeat(batch_size).reshape(batch_size, N + 1).reshape(
                batch_size * (N + 1), 1).to(device)
            time_splits_batches.requires_grad = True

            out_u = net_u(Xs_all, time_splits_batches)
            out_v = net_v(Xs_all, time_splits_batches)
            du_dt = torch.zeros((batch_size * (N + 1), d)).to(device)
            for i in range(d):
                vector = torch.zeros_like(out_u)
                vector[:, i] = 1

                dudt = grad(out_u, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                du_dt[:, i] = dudt[:, 0]

            dv_dt = torch.zeros((batch_size * (N + 1), d)).to(device)
            for i in range(d):
                vector = torch.zeros_like(out_u)
                vector[:, i] = 1

                dvdt = grad(out_v, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                dv_dt[:, i] = dvdt[:, 0]

            d_norm = torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_u(x, time_splits_batches), Xs_all),
                                  net_u(Xs_all, time_splits_batches)) - \
                     torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_v(x, time_splits_batches), Xs_all),
                                  net_v(Xs_all, time_splits_batches))

            dv_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                         get_jacobian(lambda x: net_v(x, time_splits_batches), x))[:,
                                            None], Xs_all)[:, :, 0]
            #             du_ddx = torch.zeros_like(dv_ddx)
            #             for j in range(d):
            #                 du_ddx[:, j] = torch.einsum("jii", get_jacobian(lambda x:
            #                                             get_jacobian(lambda x: net_u(x, time_splits_batches)[:, j][:, None], x)[:, :, 0], Xs_all))

            du_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                         get_jacobian(lambda x: net_u(x, time_splits_batches), x))[:,
                                            None],
                                  Xs_all)[:, :, 0]
            out_uv = net_v(Xs_all, time_splits_batches) * net_u(Xs_all, time_splits_batches)
            dvu_dx = grad(out_uv, Xs_all, grad_outputs=torch.ones_like(out_uv), create_graph=True)[0]
            L_sm = criterion(du_dt, -(h / (2 * m)) * dv_ddx - dvu_dx)  # / N

            L_nl = criterion(dv_dt, d_norm + (h / 2 / m) * du_ddx \
                             - V_x_i(Xs_all).to(device) / m)  # / N

            u0_val = torch.Tensor(u_0(X_0_all)).to(device)
            v0_val = torch.Tensor(v_0(X_0_all)).to(device)
            L_ic = criterion(net_u(X_0_all,
                                   time_splits[0].expand(batch_size, 1).to(device)), u0_val) \
                   + criterion(net_v(X_0_all,
                                     time_splits[0].expand(batch_size, 1).to(device)), v0_val)

            loss = (alpha * L_sm + beta * L_nl + gamma * L_ic) / 3.0
            losses.append(loss.item())
            losses_newton.append(L_nl.item())
            losses_sm.append(L_sm.item())
            losses_init.append(L_ic.item())
            tepoch.set_postfix(loss_iter=loss.item(), loss_mean=np.mean(losses[-10:]), loss_std=np.std(losses[-10:]))

            loss.backward()
            optimizer.step()

        elif tau <= iter_threshhold:
            BATCH_size = int(0.6 * batch_size)

            X_0_iter = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu,
                                                                  (sig2 ** 2) * np.eye(d), BATCH_size)).to(device)
            X_0_iter.requires_grad = True
            u0_val = torch.Tensor(u_0(X_0_iter)).to(device)
            v0_val = torch.Tensor(v_0(X_0_iter)).to(device)
            optimizer.zero_grad()
            eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), BATCH_size) for i in range(N + 1)]
            # get one trajectory
            for i in range(1, N_fast + 1):
                t_prev = time_splits[i - 1].clone()
                if i == 1:
                    X_prev = X_0_iter.clone().to(device)
                else:
                    X_prev = X_i_iter.clone().to(device)

                t_i = time_splits[i].clone()

                t_prev_batch = time_transform(t_prev).expand(BATCH_size, 1).to(device)
                X_i_iter = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                      net_v(X_prev, t_prev_batch)) \
                           + torch.Tensor(np.sqrt(h * T / (m * N)) * eps[i - 1]).to(device)

                if i > 1:
                    Xs_iter = torch.hstack((Xs_iter, X_i_iter))
                else:
                    Xs_iter = X_i_iter.clone()

            Xs_iter = torch.concat((X_0_iter, Xs_iter), axis=1)

            # replace the old batch with the new one
            if tau == 1:
                Xs_all = torch.vstack((Xs[BATCH_size:].detach(), Xs_iter.detach()))
                Xs_all.requires_grad = True

                X_0_all = torch.vstack((X_0_iter0[BATCH_size:].detach(), X_0_iter.detach()))
                X_0_all.requires_grad = True
            else:
                Xs_all = Xs_all.reshape((batch_size, (N + 1) * d))
                X_0_all = X_0_all.reshape((batch_size, d))
                Xs_all = torch.roll(Xs_all, -BATCH_size, 0)
                X_0_all = torch.roll(X_0_all, -BATCH_size, 0)
                Xs_all = torch.vstack((Xs_all[BATCH_size:].detach(), Xs_iter.detach()))
                Xs_all.requires_grad = True

                X_0_all = torch.vstack((X_0_all[BATCH_size:].detach(), X_0_iter.detach()))
                X_0_all.requires_grad = True

            Xs_all = Xs_all.reshape(batch_size * (N + 1), d)
            X_0_all = X_0_all.reshape(batch_size, d)

            time_splits_batches = time_splits.repeat(batch_size).reshape(batch_size, N + 1).reshape(
                batch_size * (N + 1), 1).to(device)
            time_splits_batches.requires_grad = True

            out_u = net_u(Xs_all, time_splits_batches)
            out_v = net_v(Xs_all, time_splits_batches)
            du_dt = torch.zeros((batch_size * (N + 1), d)).to(device)
            for i in range(d):
                vector = torch.zeros_like(out_u)
                vector[:, i] = 1

                dudt = grad(out_u, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                du_dt[:, i] = dudt[:, 0]

            dv_dt = torch.zeros((batch_size * (N + 1), d)).to(device)
            for i in range(d):
                vector = torch.zeros_like(out_u)
                vector[:, i] = 1

                dvdt = grad(out_v, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                dv_dt[:, i] = dvdt[:, 0]

            d_norm = torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_u(x, time_splits_batches), Xs_all),
                                  net_u(Xs_all, time_splits_batches)) - \
                     torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_v(x, time_splits_batches), Xs_all),
                                  net_v(Xs_all, time_splits_batches))

            dv_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                         get_jacobian(lambda x: net_v(x, time_splits_batches), x))[:,
                                            None],
                                  Xs_all)[:, :, 0]

            #             du_ddx = torch.zeros_like(dv_ddx)
            #             for j in range(d):
            #                 du_ddx[:, j] = torch.einsum("jii", get_jacobian(lambda x:
            #                                             get_jacobian(lambda x: net_u(x, time_splits_batches)[:, j][:, None],
            #                                                          x)[:, :, 0], Xs_all))

            du_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                         get_jacobian(lambda x: net_u(x, time_splits_batches), x))[:,
                                            None],
                                  Xs_all)[:, :, 0]

            out_uv = net_v(Xs_all, time_splits_batches) * net_u(Xs_all, time_splits_batches)
            dvu_dx = grad(out_uv, Xs_all, grad_outputs=torch.ones_like(out_uv), create_graph=True)[0]

            L_sm = criterion(du_dt, -(h / (2 * m)) * dv_ddx - dvu_dx)  # /N

            L_nl = criterion(dv_dt, d_norm + (h / 2 / m) * du_ddx - V_x_i(Xs_all).to(device) / m)  # /N

            u0_val = torch.Tensor(u_0(X_0_all)).to(device)
            v0_val = torch.Tensor(v_0(X_0_all)).to(device)
            L_ic = criterion(net_u(X_0_all,
                                   time_splits[0].expand(batch_size, 1).to(device)), u0_val) \
                   + criterion(net_v(X_0_all,
                                     time_splits[0].expand(batch_size, 1).to(device)), v0_val)

            loss = (alpha * L_sm + beta * L_nl + gamma * L_ic) / 3.0
            losses.append(loss.item())
            losses_newton.append(L_nl.item())
            losses_sm.append(L_sm.item())
            losses_init.append(L_ic.item())
            tepoch.set_postfix(loss_iter=loss.item(), loss_mean=np.mean(losses[-10:]), loss_std=np.std(losses[-10:]))

            loss.backward(retain_graph=True)
            optimizer.step()
        else:
            print('NO CHOICE')