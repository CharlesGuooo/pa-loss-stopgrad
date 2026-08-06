# Does planning risk reshape the predictor?

Code for a stop-gradient test of planning-aware trajectory prediction.

Several planning-aware objectives route a downstream signal into prediction
training, but none establishes **where the resulting gradient lands** — whether
the predictor changed, or only the planner that consumes its forecasts. In a
jointly trained system every component moves at once, so attribution is lost by
construction.

This repository contains the instrument that settles it: two training arms that
compute the *identical* objective on identical data and differ in one operation,
whether the predicted modes are detached before entering the planning term.

```python
# src/pa_loss_stopgrad/pa_loss/pa_loss_gameformer.py:173
plan_modes = neighbor_modes.detach() if stop_gradient else neighbor_modes
```

Because the plan reference is a recorded trajectory rather than the output of a
trained planner, the only differentiable quantity in the planning term is the
predicted neighbour position. The planning gradient can therefore reach the
predictor and nothing else — which is what makes the comparison a threshold
rather than an ordinary ablation. An ablation removes a component and asks
whether performance drops, conflating the component's presence with its effect
on any particular parameter. Here the component is present in both arms and the
objective value is the same.

The test is cheap and carrier-agnostic: any planning-aware objective whose plan
reference is not itself differentiable admits it.

## What is here

```
src/pa_loss_stopgrad/
  pa_loss/      the planning-aware loss and the stop-gradient control
                three risk surrogates: Gaussian, time-to-collision, oriented-box
  carriers/     GameFormer adapter + a lightweight recurrent probe
  perception/   frozen-SparseDrive export and the adapter into the carrier
  planners/     frozen shared planner probe, fixed library planner
  eval/         displacement + safety metrics, high-conflict subset, bootstrap
scripts/        training, evaluation and statistics entry points
tests/          103 tests, no data or GPU required
figures/        paper figure generation
```

## Quick start

```bash
pip install -e ".[dev]"
git clone https://github.com/MCZhi/GameFormer.git third_party/GameFormer
cd third_party/GameFormer && git checkout fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128 && cd ../..
pytest tests/ -q                    # 103 passed, 1 skipped
python scripts/check_pa_loss_gradients.py    # the instrument, on synthetic tensors
```

Neither step needs nuScenes or a GPU.

[`REPRODUCE.md`](REPRODUCE.md) maps every number in the paper to the command
that produces it, including the results that are negative.

## Results, in one table

Under a frozen-perception protocol on nuScenes, at a displacement-controlled
working point:

| | stop-gradient | full | |
|---|---|---|---|
| collision proxy, oracle global | 0.357 | **0.223** | −37.5 % |
| collision proxy, oracle high-conflict | 0.768 | **0.577** | −24.9 % |

Four of four bootstrap intervals separate, and the separation survives three
risk surrogates, two carriers, and both oracle and perceived inputs.

**What did not work, reported as results:** mode-level risk weighting is a
carrier-dependent null; under a fixed library planner the predictor gain does
not reach the planning layer at all; and the collision *rate* remains a tie
throughout. The accuracy cost is real — mean minADE runs 1.5–1.8× the control's.

This is a controlled attribution, not a leaderboard result.

## Data and upstream code

Nothing here is a copy of an upstream repository. GameFormer publishes **no
licence**, so it is referenced and cloned rather than redistributed; SparseDrive
(MIT) and its weights come from upstream; nuScenes (CC BY-NC-SA) is downloaded
from source. See [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Citation

```bibtex
@mastersthesis{guo2026thesis,
  author  = {Guo, Xiyuan},
  title   = {Planning-Aware Trajectory Prediction: Routing Downstream Planning
             Risk into a Frozen-Perception Multimodal Predictor},
  school  = {McMaster University},
  address = {Hamilton, ON, Canada},
  type    = {{M.A.Sc.} thesis},
  year    = {2026}
}
```

A journal article condensing this work is under review; this section will be
updated when it appears.

## Licence

MIT — see [`LICENSE`](LICENSE).
