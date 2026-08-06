# Paper figures

- **Fig. 3** — bootstrap confidence intervals for the working-point collision
  proxy. `python make_fig3.py` (reads the stats JSON produced by
  `scripts/build_phase2_7b_stats.py`).

- **Fig. 4** — the ghost-braking BEV pair. Rendered by the interactive
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
