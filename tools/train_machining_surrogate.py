import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from larrak_engines.gear.manufacturability_limits import (
    DEFAULT_DURATION_GRID_DEG,
    PROFILE_NAMES,
    ManufacturingProcessParams,
    _build_all_candidates,
    _surrogate_check,
)


# Re-implement cache key logic locally to ensure match
def make_cache_key(theta, r_planet, wire_d, overcut, cm, ml, voxel):
    h = hashlib.sha256()
    h.update(np.asarray(theta, dtype=np.float64).tobytes())
    h.update(np.asarray(r_planet, dtype=np.float64).tobytes())
    h.update(f"{wire_d},{overcut},{cm},{ml},{voxel}".encode())
    return h.hexdigest()


class MachiningSurrogate(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_surrogate():
    print("Loading cache...")
    cache_path = Path("src/larrak2/gear/picogk_cache.pkl")
    if not cache_path.exists():
        print("Cache not found!")
        return

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    print(f"Cache size: {len(cache)}")

    # 1. Dataset Generation
    X = []
    Y = []

    process = ManufacturingProcessParams(
        kerf_mm=0.2,
        overcut_mm=0.05,
        min_ligament_mm=0.25,  # Matches DOE
    )

    durations = DEFAULT_DURATION_GRID_DEG.astype(float)
    amps = np.linspace(-1.5, 4.0, 61)
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)

    shape_map = {name: i for i, name in enumerate(PROFILE_NAMES)}

    print("Reconstructing dataset from grid...")
    hit_count = 0

    for dur in durations:
        for amp in amps:
            candidates = _build_all_candidates(theta, dur, amp)

            for shape_name, ratio_profile in candidates:
                # We need to replicate logic to check cache
                if not _surrogate_check(theta, ratio_profile, process, strict=False):
                    continue

                ratio_safe = np.maximum(ratio_profile, 1e-6)
                r_planet = 80.0 / ratio_safe

                key = make_cache_key(
                    theta,
                    r_planet,
                    process.kerf_mm,
                    process.overcut_mm,
                    0.0,
                    process.min_ligament_mm,
                    0.1,
                )

                if key in cache:
                    res = cache[key]
                    hit_count += 1

                    # Inputs: Duration, Amplitude, ShapeOneHot
                    # Normalize:
                    # Dur: 0-360 -> 0-1
                    # Amp: -1.5..4.0 -> 0-1

                    x_vec = [
                        dur / 360.0,
                        (amp + 1.5) / 5.5,
                        # Shape OneHot
                        *[
                            1.0 if i == shape_map[shape_name] else 0.0
                            for i in range(len(PROFILE_NAMES))
                        ],
                    ]

                    # Outputs:
                    # TMinProxy, BMaxSurvivable, MinHoleDiam, MinHoleCurv
                    # Handling None/0.0
                    y_vec = [
                        res.get("t_min_proxy_mm", 0.0),
                        res.get("b_max_survivable_mm", 0.0),
                        res.get("min_hole_diameter_mm", 0.0),
                        res.get("min_hole_curvature_radius_mm", 0.0),
                    ]

                    X.append(x_vec)
                    Y.append(y_vec)

    print(f"Matched {len(X)} samples.")

    if not X:
        print("No matches found. Check matching logic.")
        return

    X_tn = torch.tensor(X, dtype=torch.float32)
    Y_tn = torch.tensor(Y, dtype=torch.float32)

    # 2. Training
    model = MachiningSurrogate(input_dim=len(X[0]))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Training...")
    for epoch in range(500):
        optimizer.zero_grad()
        pred = model(X_tn)
        loss = criterion(pred, Y_tn)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    print("Training Complete.")
    torch.save(model.state_dict(), "machining_surrogate.pth")
    print("Model saved to machining_surrogate.pth")

    # Validation
    with torch.no_grad():
        pred = model(X_tn)
        mse = criterion(pred, Y_tn).item()
        print(f"Final MSE: {mse:.4f}")
        print(f"Sample Pred: {pred[0].numpy()}")
        print(f"Sample True: {Y_tn[0].numpy()}")


if __name__ == "__main__":
    train_surrogate()
