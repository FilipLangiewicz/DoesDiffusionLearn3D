# Functional check with a real trained VP model: noise real images with add_noise
# (before set_timesteps), let the model predict the noise, and measure the error
# of the clean-image estimate. A wrong noise level makes the estimate bad.
import sys, torch, numpy as np
import diffusers
from diffusers import UNet2DModel, DDPMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DEISMultistepScheduler, DPMSolverSinglestepScheduler
print("diffusers from", diffusers.__file__)
torch.manual_seed(0)
unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").eval()
ref = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
cfg = dict(ref.config)
# Real images: draw clean samples is slow, so use smooth natural-like test images:
# the model's own one-step denoise of pure structure is not needed; CIFAR-like
# images come from torchvision if present, else from a fixed PNG.
from PIL import Image
imgs = []
import glob
for f in sorted(glob.glob(sys.argv[1] + "/*/*.png"))[:8]:
    im = Image.open(f).convert("RGB").resize((32, 32), Image.BICUBIC)
    imgs.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1))
x0 = torch.stack(imgs)
eps = torch.randn_like(x0)
ac = ref.alphas_cumprod
def x0_err(xt, t):
    with torch.no_grad():
        e = unet(xt, torch.full((len(xt),), t)).sample
    a = ac[t]
    x0h = (xt - (1 - a).sqrt() * e) / a.sqrt()
    return float(((x0h - x0) ** 2).mean().sqrt())
for t in (100, 300, 500):
    row = [f"t={t}", f"DDPM(ref) {x0_err(ref.add_noise(x0, eps, torch.tensor([t])), t):.3f}"]
    for S in (DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, UniPCMultistepScheduler, DEISMultistepScheduler):
        s = S.from_config(cfg)
        row.append(f"{S.__name__[:12]} {x0_err(s.add_noise(x0, eps, torch.tensor([t])), t):.3f}")
    print(" | ".join(row), flush=True)
