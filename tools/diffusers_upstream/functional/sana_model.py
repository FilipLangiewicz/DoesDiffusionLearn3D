import sys, torch
from diffusers import SanaTransformer2DModel
torch.set_grad_enabled(False); torch.set_num_threads(8)
d = torch.load(sys.argv[1]); x0, eps = d["x0"], d["eps"]
inputs = {"poprawny wzor": {t: (1 - t/1000) * x0 + t/1000 * eps for t in (200, 500, 800)},
          "diffusers bez poprawki": torch.load(sys.argv[2]), "diffusers z poprawka": torch.load(sys.argv[3])}
pc = torch.load(sys.argv[4]); emb, mask = pc["embeds"].float(), pc["mask"]
m = SanaTransformer2DModel.from_pretrained("Efficient-Large-Model/Sana_600M_512px_diffusers", subfolder="transformer", torch_dtype=torch.float32).eval()
n = len(x0)
for t in (200, 500, 800):
    row = [f"t={t}"]
    for name, inp in inputs.items():
        xt = inp[t]
        v = m(hidden_states=xt, encoder_hidden_states=emb.expand(n, -1, -1), encoder_attention_mask=mask.expand(n, -1),
              timestep=torch.full((n,), t), return_dict=False)[0]
        x0h = xt - t/1000 * v
        row.append(f"{name}: {float((x0h - x0).norm() / x0.norm()):.3f}")
    print(" | ".join(row), flush=True)
