"""TF-free nuScenes prediction metrics for GameFormer (Phase 2.2).

Computes minADE_k / minFDE_k / MissRate_k over the two jointly-predicted agents
(ego = index 0, neighbor0 = index 1), masking invalid future steps. Replaces the
Waymo/TF ``MotionMetrics`` in the upstream eval, which cannot be imported here.
"""
import torch

MISS_THRESHOLD = 2.0  # meters (standard nuScenes prediction miss-rate threshold)


def predicted_xy(outputs, level):
    """outputs[f'level_{level}_interactions'][..., :2] -> [B, N, M, T, 2]."""
    return outputs[f"level_{level}_interactions"][..., :2]


def stack_gt(ego_future, neighbor_future):
    """ego/neighbor future [B, T, 5] -> stacked [B, N=2, T, 5]."""
    return torch.stack([ego_future, neighbor_future], dim=1)


class MotionMetricsAccumulator:
    """Accumulate per-agent minADE/minFDE/miss over batches, then reduce."""

    AGENTS = ("ego", "nbr")

    def __init__(self, miss_threshold: float = MISS_THRESHOLD):
        self.thr = miss_threshold
        self._ade = {a: [] for a in self.AGENTS}
        self._fde = {a: [] for a in self.AGENTS}
        self._miss = {a: [] for a in self.AGENTS}

    @torch.no_grad()
    def update(self, pred_xy, gt):
        """pred_xy: [B, N=2, M, T, 2]; gt: [B, N=2, T, 5]."""
        pred_xy = pred_xy.detach().float().cpu()
        gt = gt.detach().float().cpu()
        gtxy = gt[..., :2]                                   # [B,N,T,2]
        valid = gtxy.abs().sum(-1) != 0                      # [B,N,T]
        dist = (pred_xy - gtxy.unsqueeze(2)).norm(dim=-1)    # [B,N,M,T]

        mask = valid.unsqueeze(2).float()                    # [B,N,1,T]
        cnt = mask.sum(-1).clamp(min=1.0)                    # [B,N,1]
        ade_per_mode = (dist * mask).sum(-1) / cnt           # [B,N,M]
        min_ade = ade_per_mode.min(dim=-1).values            # [B,N]

        B, N, T = valid.shape
        steps = torch.arange(T)
        last = torch.where(valid, steps, torch.full_like(steps, -1)).max(dim=-1).values  # [B,N]
        has = last >= 0
        last_idx = last.clamp(min=0).view(B, N, 1, 1).expand(B, N, dist.shape[2], 1)
        fde_per_mode = dist.gather(-1, last_idx).squeeze(-1)  # [B,N,M]
        min_fde = fde_per_mode.min(dim=-1).values            # [B,N]
        miss = (min_fde > self.thr).float()                  # [B,N]

        for n, agent in enumerate(self.AGENTS):
            sel = has[:, n]
            if sel.any():
                self._ade[agent].append(min_ade[sel, n])
                self._fde[agent].append(min_fde[sel, n])
                self._miss[agent].append(miss[sel, n])

    def result(self):
        out = {}
        all_ade, all_fde, all_miss = [], [], []
        for agent in self.AGENTS:
            if self._ade[agent]:
                ade = torch.cat(self._ade[agent])
                fde = torch.cat(self._fde[agent])
                miss = torch.cat(self._miss[agent])
                out[f"minADE_{agent}"] = float(ade.mean())
                out[f"minFDE_{agent}"] = float(fde.mean())
                out[f"MR_{agent}"] = float(miss.mean())
                all_ade.append(ade)
                all_fde.append(fde)
                all_miss.append(miss)
            else:
                out[f"minADE_{agent}"] = float("nan")
                out[f"minFDE_{agent}"] = float("nan")
                out[f"MR_{agent}"] = float("nan")
        if all_ade:
            out["minADE_mean"] = float(torch.cat(all_ade).mean())
            out["minFDE_mean"] = float(torch.cat(all_fde).mean())
            out["MR_mean"] = float(torch.cat(all_miss).mean())
        return out


def evaluate_all_levels(outputs, ego_future, neighbor_future, levels):
    """Convenience: one-shot metrics for each level on a single batch.

    Returns {level: metrics_dict}. For full-val eval, prefer per-level
    accumulators updated across batches.
    """
    gt = stack_gt(ego_future, neighbor_future)
    res = {}
    for k in range(levels + 1):
        acc = MotionMetricsAccumulator()
        acc.update(predicted_xy(outputs, k), gt)
        res[k] = acc.result()
    return res
