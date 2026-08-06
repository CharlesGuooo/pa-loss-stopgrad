"""nuScenes-horizon GameFormer loss (Phase 2.1).

A TF-free reimplementation of the upstream ``level_k_loss`` / ``gmm_loss`` /
``imitation_loss`` with metric indices reparameterized for the 12-step, 2 Hz
nuScenes horizon (vs. Waymo's 80-step, 10 Hz). The upstream
``utils/inter_pred_utils`` cannot be imported here because it pulls in
TensorFlow + the Waymo Open Dataset.
"""
import torch
import torch.nn.functional as F

from pa_loss_stopgrad.carriers.config import FUTURE_LEN

# Measurement steps ~ 2 s / 4 s / 6 s at 2 Hz (indices into a 12-step horizon).
DEFAULT_METRIC = [3, 7, 11] if FUTURE_LEN >= 12 else [FUTURE_LEN - 1]


def imitation_loss(trajectories, ground_truth, metric=DEFAULT_METRIC):
    # trajectories: [B, N, M, T, 2]; ground_truth: [B, N, T, 5]
    ade_distance = torch.norm(trajectories[..., :2] - ground_truth[:, :, None, :, :2], dim=-1)
    fde_distance = torch.norm(
        trajectories[:, :, :, metric, :2] - ground_truth[:, :, None, metric, :2], dim=-1
    )
    distance = fde_distance.sum(-1) + ade_distance.mean(-1)
    best_mode = torch.argmin(distance.mean(1), dim=-1)
    B, N = trajectories.shape[0], trajectories.shape[1]
    best = trajectories[
        torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]
    ].squeeze(2)
    de = F.smooth_l1_loss(best, ground_truth[:, :, :, :2], reduction="none").sum(-1)
    ade = torch.mean(de, dim=-1)
    fde = torch.sum(de[:, :, metric], dim=-1)
    loss = torch.mean((fde + ade).mean(1))
    return loss, best_mode, best


def gmm_loss(trajectories, convs, probs, ground_truth, metric=DEFAULT_METRIC):
    # trajectories: [B, N, M, T, 2]; convs: [B, N, M, T, 2]; probs: [B, M]
    distance = torch.norm(trajectories[..., :2] - ground_truth[:, :, None, :, :2], dim=-1)
    ndistance = distance.mean(-1) + distance[..., metric].sum(-1)
    best_mode = torch.argmin(ndistance.mean(1), dim=-1)
    B, N = trajectories.shape[0], trajectories.shape[1]

    best = trajectories[
        torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]
    ].squeeze(2)
    convs = convs[
        torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]
    ].squeeze(2)

    dx = best[..., 0] - ground_truth[..., 0]
    dy = best[..., 1] - ground_truth[..., 1]
    log_std_x = torch.clip(convs[..., 0], -2, 5)
    log_std_y = torch.clip(convs[..., 1], -2, 5)
    std_x, std_y = torch.exp(log_std_x), torch.exp(log_std_y)

    reg = log_std_x + log_std_y
    exp = 0.5 * ((dx ** 2) / (std_x ** 2) + (dy ** 2) / (std_y ** 2))
    loss = reg + exp
    loss = loss.mean(-1) + loss[..., metric].sum(-1)

    prob_loss = F.cross_entropy(input=probs, target=best_mode, label_smoothing=0.2)
    loss = (loss + 2 * prob_loss).mean()
    return loss, best_mode, best, convs


def level_k_loss_nuscenes(outputs, ego_future, neighbor_future, levels, gmm=True,
                          metric=DEFAULT_METRIC):
    """Sum the prediction loss over all reasoning levels (0..levels).

    outputs[f'level_{k}_interactions']: [B, N=2, M, T, 4] (mu_x, mu_y, log_sx, log_sy)
    outputs[f'level_{k}_scores']:       [B, N=2, M]
    ego_future / neighbor_future:       [B, T, 5]
    """
    loss = torch.zeros((), device=ego_future.device)
    neighbor_valid = torch.ne(neighbor_future[..., :2].sum(-1), 0)
    ego_valid = torch.ne(ego_future[..., :2].sum(-1), 0)
    future = best_mode = scores = None

    for k in range(levels + 1):
        level = outputs[f"level_{k}_interactions"]
        scores = outputs[f"level_{k}_scores"]
        traj = level[..., :2]
        ego = traj[:, 0] * ego_valid.unsqueeze(1).unsqueeze(-1)
        neighbor = traj[:, 1] * neighbor_valid.unsqueeze(1).unsqueeze(-1)
        traj = torch.stack([ego, neighbor], dim=1)
        gt_future = torch.stack([ego_future, neighbor_future], dim=1)

        if gmm:
            convs = level[..., 2:]
            gloss, best_mode, future, _ = gmm_loss(traj, convs, scores.sum(1), gt_future, metric)
            loss = loss + gloss
        else:
            il_loss, best_mode, future = imitation_loss(traj, gt_future, metric)
            sc_loss = F.cross_entropy(scores.sum(1), best_mode)
            loss = loss + 0.5 * il_loss + 2 * sc_loss

    return loss, (future, best_mode, scores)
