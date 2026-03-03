# Machining NN Artifact Migration

Production runtime now rejects default model paths under `src/` and `models/`.
Machining NN artifacts must live under:

- `outputs/artifacts/surrogates/machining_nn/machining_surrogate.pth`

## One-time migration

```bash
python tools/migrate_machining_model.py \
  --legacy-path src/larrak2/surrogate/machining_surrogate.pth \
  --dest outputs/artifacts/surrogates/machining_nn/machining_surrogate.pth
```

Use `--move` to move instead of copy.

## Runtime behavior

- Default mode is `machining_mode="nn"` and is fail-hard if artifact is missing.
- Explicit non-production bypass is `machining_mode="analytical"`.
