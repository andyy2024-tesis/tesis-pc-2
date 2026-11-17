"""What the rec_all parts learned per recording: attack/fade scale, body fine-EQ range, decay per recording.
Usage: uv run python analysis/rec_use.py <run>   (a run trained with rec_all)
"""
import sys
import numpy as np
import torch
run = sys.argv[1]
sd = torch.load(f"analysis/results/e4/{run}/model.pt", map_location="cpu", weights_only=False)
sd = sd.get("model", sd) if isinstance(sd, dict) else sd
keys = [k for k in sd if "rec" in k or "eq_fine" in k or "decay_mlp" in k or "click_eq" in k or "pol_mlp" in k]
print([ (k, tuple(sd[k].shape)) for k in keys][:20])
emb = None
for k in sd:
    if k.endswith("rec_emb.weight"): emb = sd[k]
print("emb", None if emb is None else tuple(emb.shape))
if emb is not None:
    if "rec_consts.weight" in sd:
        c = emb @ sd["rec_consts.weight"].T + sd["rec_consts.bias"]
        sc = torch.exp(1.5 * torch.tanh(c))
        print("attack scale per rec:", np.round(sc[:, 0].numpy(), 2))
        print("fade scale per rec:  ", np.round(sc[:, 1].numpy(), 2))
    if "body_eq_fine.weight" in sd:
        g = emb @ sd["body_eq_fine.weight"].T + sd["body_eq_fine.bias"]
        g_db = 20 / np.log(10) * g.numpy()
        print("body fine EQ dB: per-rec range (max-min over bands):", np.round(g_db.max(1) - g_db.min(1), 1))
        print("  spread over recs per band (sd dB):", np.round(g_db.std(0), 1))

# decay per recording: the same MLP, run by hand on a grid of notes and partials
import torch.nn.functional as Fn
W0, b0, W2, b2 = (sd[k] for k in ("decay_mlp.0.weight", "decay_mlp.0.bias", "decay_mlp.2.weight", "decay_mlp.2.bias"))
def inv_softplus(y): return y + torch.log(-torch.expm1(-y))
print("\nT60 (s) of partials, rows = pitch (E2, E3, E4, E5), cols = partial freq 200/1k/3k/6k Hz; median and range over recordings")
for p in (40, 52, 64, 76):
    row = []
    for fhz in (200, 1000, 3000, 6000):
        f0 = 440 * 2 ** ((p - 69) / 12)
        if fhz < f0 * 0.99: row.append("   -    "); continue
        x = torch.cat([torch.tensor([(p - 64) / 12, np.log(fhz / 1000)]).float().expand(20, 2), emb], 1)
        h = Fn.leaky_relu(x @ W0.T + b0)
        s = Fn.softplus((h @ W2.T + b2)[:, 0] + inv_softplus(torch.tensor(1.5 + 2.0 * fhz / 1000)))
        t60 = 6.91 / s
        row.append(f"{t60.median():.2f} ({t60.min():.2f}-{t60.max():.2f})")
    print(p, " | ".join(row))
