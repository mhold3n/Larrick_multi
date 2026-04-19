# Vendored source manifests (audiobook / standards pipeline)

## Path policy for agents and tooling

- **Suite layout:** First-party repos live under a single parent directory, e.g.
  `.../GitHub/Larrick Engine Suite/<repo>/`. Scripts should not assume
  `.../GitHub/<repo>/` at the old depth.
- **`larrak-audio`:** This repo is **not** vendored inside Larrick Multi. JSON under
  `*/marker/` may still contain absolute paths captured when sources were processed
  under `/Users/maxholden/GitHub/larrak-audio/`. Treat those paths as **historical
  snapshots**, not guaranteed current locations. If tooling must resolve files,
  prefer `LARRICK_AUDIO_ROOT` (or clone `larrak-audio` and search-replace the prefix
  locally) rather than editing dozens of manifests blindly.
