"""Entity-based actor-critic: one network for any number of zones and AHUs.

zones --enc--> pooled per AHU (mean, max) --+--> AHU state --pooled--> building state
AHU obs --enc-----------------------------/
heads: zone (from zone, its AHU, building), AHU (AHU, building), building, value.

Inputs for a batch of B steps of ONE building:
  zone [B, V, ZONE_DIM], ahu [B, K, AHU_DIM], glob [B, GLOBAL_DIM], zone_ahu [V] (long)
"""
import numpy as np
import torch
import torch.nn as nn

from sbx.rl.features import AHU_ACT, AHU_DIM, GLOBAL_DIM, WATER_ACT, ZONE_ACT, ZONE_DIM

LOG2PI = float(np.log(2 * np.pi))


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class EntityActorCritic(nn.Module):
    def __init__(self, hidden=64, init_log_std=-1.0):
        super().__init__()
        h = hidden
        self.zone_enc = mlp(ZONE_DIM, h, h)
        self.ahu_enc = mlp(AHU_DIM, h, h)
        self.glob_enc = mlp(GLOBAL_DIM, h, h)
        self.ahu_mix = mlp(3 * h, h, h)
        self.glob_mix = mlp(3 * h, h, h)
        self.zone_head = mlp(3 * h, h, ZONE_ACT + 0)
        self.ahu_head = mlp(2 * h, h, AHU_ACT + 1)     # + run logit (direct mode)
        self.glob_head = mlp(h, h, WATER_ACT + 1)      # + hot-water run logit
        self.value_head = mlp(2 * h, h, 1)
        for head in (self.zone_head, self.ahu_head, self.glob_head):
            nn.init.zeros_(head[-1].weight)            # residual starts at exactly zero
            nn.init.zeros_(head[-1].bias)
        self.log_std_z = nn.Parameter(torch.full((ZONE_ACT,), init_log_std))
        self.log_std_a = nn.Parameter(torch.full((AHU_ACT,), init_log_std))
        self.log_std_w = nn.Parameter(torch.full((WATER_ACT,), init_log_std))

    def forward(self, zone, ahu, glob, zone_ahu):
        B, V, _ = zone.shape
        K = ahu.shape[1]
        ez = self.zone_enc(zone)                                   # [B,V,h]
        onehot = torch.zeros(K, V, device=zone.device)
        onehot[zone_ahu, torch.arange(V)] = 1.0
        count = onehot.sum(1).clamp(min=1.0)                        # [K]
        mean_z = torch.einsum("kv,bvh->bkh", onehot, ez) / count[None, :, None]
        masked = ez[:, None, :, :].expand(B, K, V, -1).masked_fill(onehot[None, :, :, None] == 0, -1e4)
        max_z = masked.max(2).values.clamp(min=-1e3)
        ha = self.ahu_mix(torch.cat([self.ahu_enc(ahu), mean_z, max_z], -1))   # [B,K,h]
        eg = self.glob_enc(glob)
        hg = self.glob_mix(torch.cat([eg, ha.mean(1), ez.mean(1)], -1))        # [B,h]
        zin = torch.cat([ez, ha[:, zone_ahu, :], hg[:, None, :].expand(B, V, -1)], -1)
        out_z = self.zone_head(zin)                                            # [B,V,2]
        out_a = self.ahu_head(torch.cat([ha, hg[:, None, :].expand(B, K, -1)], -1))  # [B,K,4]
        out_w = self.glob_head(hg)                                             # [B,3]
        value = self.value_head(torch.cat([hg, eg], -1)).squeeze(-1)           # [B]
        return {"mu_z": out_z, "mu_a": out_a[..., :AHU_ACT], "run_a": out_a[..., AHU_ACT],
                "mu_w": out_w[:, :WATER_ACT], "run_w": out_w[:, WATER_ACT], "value": value}

    # ------------------------------------------------------------ Gaussian policy over all continuous outputs
    def _gauss_logp(self, x, mu, log_std):
        return (-0.5 * ((x - mu) / log_std.exp()) ** 2 - log_std - 0.5 * LOG2PI)

    def log_prob(self, out, az, aa, aw):
        """Joint log-probability of continuous actions, one number per step."""
        lp = self._gauss_logp(az, out["mu_z"], self.log_std_z).sum((-1, -2))
        lp = lp + self._gauss_logp(aa, out["mu_a"], self.log_std_a).sum((-1, -2))
        lp = lp + self._gauss_logp(aw, out["mu_w"], self.log_std_w).sum(-1)
        return lp

    def entropy(self, V, K):
        per = lambda ls: (ls + 0.5 * (1 + LOG2PI)).sum()
        return V * per(self.log_std_z) + K * per(self.log_std_a) + per(self.log_std_w)

    @torch.no_grad()
    def act(self, zone, ahu, glob, zone_ahu, deterministic=False):
        """One step (numpy in, numpy out): sampled actions, log-prob, value."""
        t = lambda x: torch.as_tensor(x)[None]
        out = self(t(zone), t(ahu), t(glob), torch.as_tensor(zone_ahu))
        mu_z, mu_a, mu_w = out["mu_z"], out["mu_a"], out["mu_w"]
        if deterministic:
            az, aa, aw = mu_z, mu_a, mu_w
        else:
            az = mu_z + self.log_std_z.exp() * torch.randn_like(mu_z)
            aa = mu_a + self.log_std_a.exp() * torch.randn_like(mu_a)
            aw = mu_w + self.log_std_w.exp() * torch.randn_like(mu_w)
        logp = self.log_prob(out, az, aa, aw)
        n = lambda x: x[0].numpy()
        return {"az": n(az), "aa": n(aa), "aw": n(aw), "logp": float(logp[0]), "value": float(out["value"][0]),
                "run_a": n(torch.sigmoid(out["run_a"])), "run_w": n(torch.sigmoid(out["run_w"]))}
