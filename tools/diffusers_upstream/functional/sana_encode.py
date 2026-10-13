import sys, glob, torch
from diffusers import AutoencoderDC
from dd3d.extract import preprocess
torch.set_grad_enabled(False); torch.manual_seed(0)
M = "Efficient-Large-Model/Sana_600M_512px_diffusers"
vae = AutoencoderDC.from_pretrained(M, subfolder="vae", torch_dtype=torch.float32)
files = [sorted(glob.glob(f"{sys.argv[1]}/{s}/*.png"))[i] for s in ("Sphere", "SmoothCube_v2") for i in (0, 1)]
x0 = torch.cat([vae.encode(preprocess(__import__("pathlib").Path(f), 512)).latent for f in files]) * vae.config.scaling_factor
torch.save({"x0": x0, "eps": torch.randn_like(x0), "files": files}, sys.argv[2]); print("latents", tuple(x0.shape))
