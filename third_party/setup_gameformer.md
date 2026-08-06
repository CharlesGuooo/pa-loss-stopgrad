# Setting up the GameFormer checkout

GameFormer is the primary prediction carrier. It is **not** redistributed with
this repository because upstream publishes no licence (see
[`../THIRD_PARTY.md`](../THIRD_PARTY.md)), so you clone it yourself.

```bash
git clone https://github.com/MCZhi/GameFormer.git third_party/GameFormer
cd third_party/GameFormer
git checkout fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128   # the pinned commit
```

`third_party/GameFormer/` is git-ignored, so the clone never enters this
repository's history.

If you keep the checkout elsewhere, point the loader at it:

```bash
export GAMEFORMER_ROOT=/path/to/GameFormer     # Windows: set GAMEFORMER_ROOT=...
```

## Verifying it worked

```bash
pytest tests/ -q
```

Without the checkout, 9 tests fail with `FileNotFoundError: GameFormer checkout
not found` — they are the ones that build the real model. With it:

```
103 passed, 1 skipped
```

You do **not** need GameFormer's own dependencies (TensorFlow, the Waymo Open
Dataset). Only `model/GameFormer.py` and `model/modules.py` are imported; the
losses and the lane encoder are reimplemented here for the nuScenes horizon.
