import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusionCustom, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
)

diffusion = GaussianDiffusionCustom(
    model,
    image_size=256,
    timesteps=1000,           # number of steps
    sampling_timesteps=250,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/home/alexey.buzovkin/data/jpg',
    train_batch_size=1,
    train_lr=8e-5,
    train_num_steps=700000,         # total training steps
    gradient_accumulate_every=1,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=True,                       # turn on mixed precision
    calculate_fid=False              # whether to calculate fid during training
)

print(trainer.calculate_fid)

trainer.train()
