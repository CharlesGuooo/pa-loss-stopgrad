# Paper figures

Scripts are named by what they draw, not by figure number, so inserting a
figure cannot desynchronise them. Output files follow IEEE's convention of
matching the figure order: `guo3.pdf` = Fig. 3, `guo4.pdf` = Fig. 4,
`guo5.png` = Fig. 5. (Figs. 1 and 2 are TikZ, compiled inline with the paper.)

- **Fig. 3** — the accuracy/safety trade-off and the guard-cap working point.
  `python make_tradeoff.py`, reading the committed `../results/trainpath.json`.
  Runs from a fresh clone with no training: it shows both arms' six fine-tuning
  epochs, and the stop-gradient control visibly going nowhere under the same
  objective.


- **Fig. 5** — bootstrap confidence intervals for the working-point collision
  proxy. `python make_ci_plot.py` (reads the stats JSON produced by
  `scripts/build_phase2_7b_stats.py`).

- **Fig. 5** — the ghost-braking BEV pair. Rendered by the interactive
  visualization tool from cached scene data, not by re-running any model:

  ```bash
  cd <Visualization>            # python -m http.server 8000
  node export/recapture_fig4_hires.mjs
  ```

  `recapture_fig4_hires.mjs` is included here for reference. It captures frame
  token `af67f465f5994ac7bab19825336db644` (boston-seaport, command `straight`,
  high-conflict, perceived ΔL2 = 0.66 m) in the perceived domain with the
  planner layer, full vs stop-grad, decoder level 3.

  It renders in English (`?lang=en`), strips the tool chrome (`?figure=1`) and
  focuses the bounding box on the interaction (`?bboxFocus=action`). Ground
  truth is drawn dashed and the chosen plan solid, so the figure survives IEEE's
  greyscale print edition.
