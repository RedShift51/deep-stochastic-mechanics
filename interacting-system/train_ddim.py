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


def get_jacobian(y, x):
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


""" Loss calculation ======================================== """


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def q_sample(x_start, t, noise=None, model=None):
    noise = default(noise, lambda: torch.randn_like(x_start))

    return (
        extract(model.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        extract(model.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )


def predict_v(model, x_start, t, noise):
    return (
        extract(model.sqrt_alphas_cumprod, t, x_start.shape) * noise -
        extract(model.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
    )


def p_losses(x_start, t, model, objective, noise = None):
    b, c, h, w = x_start.shape

    noise = default(noise, lambda: torch.randn_like(x_start))

    # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
    offset_noise_strength = 0

    if offset_noise_strength > 0.:
        offset_noise = torch.randn(x_start.shape[:2], device="cuda")
        noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

    # noise sample

    x = q_sample(x_start=x_start, t=t, noise=noise, model=model)

    # if doing self-conditioning, 50% of the time, predict x_start from current set of times
    # and condition with unet with that
    # this technique will slow down training by 25%, but seems to lower FID significantly

    x_self_cond = None
    self_condition = False
    if self_condition and random() < 0.5:
        with torch.inference_mode():
            x_self_cond = model.model_predictions(x, t).pred_x_start
            x_self_cond.detach_()

    # predict and take gradient step

    if objective == 'pred_noise':
        target = noise
    elif objective == 'pred_x0':
        target = x_start
    elif objective == 'pred_v':
        v = predict_v(model, x_start, t, noise)
        target = v
    else:
        raise ValueError(f'unknown objective {objective}')

    """
    loss = F.mse_loss(model_out, target, reduction = 'none')
    u+v=-beta(t)*x
    v-u=-beta(t)*x + s(theta)(x,t)

    v = -beta(t)*x + s(theta)(x,t) / 2
    u = -s(theta)(x,t) / 2

    u = -x / 2
    """

    t = t.float()
    t.requires_grad_()
    x.requires_grad_()

    # time derivative
    du_dt = torch.zeros_like(x_start)
    du_dt = du_dt.view(batch_size, diffusion.image_size * diffusion.image_size * 3)
    model_out = -model.model(x, t, x_self_cond).view(batch_size, diffusion.image_size * diffusion.image_size * 3) / 2
    grad_batch = 64
    vector = torch.zeros(grad_batch, batch_size,
                              diffusion.image_size*diffusion.image_size*3).cuda()
    for i in range(diffusion.image_size * diffusion.image_size * 3 // grad_batch):
        for k in range(grad_batch):
            if i != 0:
                vector[k, :, (i-1)*grad_batch+k] = 0
            vector[k, :, i*grad_batch+k] = 1
        dudt = grad(model_out, t, grad_outputs=vector, retain_graph=True, is_grads_batched=True)[0]
        du_dt[:, grad_batch*i: grad_batch*(i+1)] = torch.transpose(dudt, 0, 1)
        print(i * grad_batch)

    # spatial derivatives
    dv_ddx = get_jacobian(
        lambda x_: torch.einsum(
            "jii",
            get_jacobian(
                lambda x_: (model.model(x, t, x_self_cond) / 2 - x).view(
                            batch_size, diffusion.image_size * diffusion.image_size * 3
                        ),
                x_
            ),
        ),
        x,
    )

    out_uv = - model_out * (-x + model_out / 2) / 2
    dvu_dx = grad(out_uv, x, grad_outputs=torch.ones_like(out_uv), create_graph=True)[0]

    h = torch.gather(model.beta_schedule, 0, t.long())
    m = 1
    L1 = model.criterion(du_dt, -(h / (2 * m)) * dv_ddx - dvu_dx) + \
         model.criterion(model.model(noise, torch.ones_like(t), x_self_cond), noise)

    loss = L1
    """
    loss = reduce(loss, 'b ... -> b', 'mean')
    """
    loss = loss * extract(model.loss_weight, t, loss.shape)
    return loss.mean()


def calculate_loss(model, img, img_size, num_timesteps, objective):
    b, c, h, w, device, img_size, = *img.shape, img.device, img_size
    assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
    t = torch.randint(0, num_timesteps, (b,), device=device).long()

    img = normalize_to_neg_one_to_one(img)
    return p_losses(img, t, model, objective)


""" ======================================== """


if __name__ == "__main__":
    only_sample_mode = False
    objective = "pred_noise"
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    )

    split_batches = True
    mixed_precision_type = 'fp16'
    amp = True
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    diffusion = GaussianDiffusionCustom(
        model,
        image_size=192,
        timesteps=1000,           # number of steps
        sampling_timesteps=250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    # setting training parameters

    channels = 3
    convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(channels)
    num_samples = 25
    save_and_sample_every = 2500
    batch_size = 1
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

    # train loop
    step = 0
    model, opt = accelerator.prepare(model, opt)
    save_best_and_latest_only = False

    device = accelerator.device

    with tqdm(initial=step, total=train_num_steps, disable=not accelerator.is_main_process) as pbar:

        while step < train_num_steps:

            total_loss = 0.

            for _ in range(gradient_accumulate_every):
                data = next(dl).to(device)

                with accelerator.autocast():
                    loss = calculate_loss(diffusion, data, diffusion.image_size, diffusion.num_timesteps, objective)
                    loss = loss / gradient_accumulate_every
                    total_loss += loss.item()

                accelerator.backward(loss)

            pbar.set_description(f'loss: {total_loss:.4f}')

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(diffusion.parameters(), max_grad_norm)

            opt.step()
            opt.zero_grad()

            accelerator.wait_for_everyone()

            step += 1
            if accelerator.is_main_process:
                ema.update()

                if step != 0 and divisible_by(step, save_and_sample_every):
                    ema.ema_model.eval()

                    with torch.inference_mode():
                        milestone = step // save_and_sample_every
                        batches = num_to_groups(num_samples, batch_size)
                        # sample_fn = ddim_sample from unet methods
                        all_images_list = list(map(lambda n: ema.ema_model.sample(batch_size=n), batches))

                    all_images = torch.cat(all_images_list, dim=0)

                    utils.save_image(all_images, str(results_folder / f'sample-{milestone}.png'),
                                     nrow=int(math.sqrt(num_samples)))

                    save(step, accelerator, ema, opt, results_folder, milestone)

            pbar.update(1)

    accelerator.print('training complete')