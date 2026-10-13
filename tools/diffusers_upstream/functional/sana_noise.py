import sys, torch, diffusers
from diffusers import DPMSolverMultistepScheduler
d = torch.load(sys.argv[1]); s = DPMSolverMultistepScheduler.from_pretrained("Efficient-Large-Model/Sana_600M_512px_diffusers", subfolder="scheduler")
out = {t: s.add_noise(d["x0"], d["eps"], torch.tensor([t])) for t in (200, 500, 800)}
torch.save(out, sys.argv[2]); print("noised with", diffusers.__file__)
