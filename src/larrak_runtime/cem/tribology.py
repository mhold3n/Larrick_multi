"""ISO-grounded tribology calculations backed by registry datasets.

Implements:
- Specific film thickness lambda (EHL film / composite roughness proxy)
- Scuffing temperature margin for flash and integral methods
- Micropitting safety factor S_lambda

Data contracts (required in strict mode):
- data/cem/tribology_ehl_coefficients.csv
- data/cem/scuffing_critical_temperatures.csv
- data/cem/micropitting_lambda_perm.csv
- data/cem/fzg_step_load_map.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from larrak_runtime.cem.registry import get_registry

# ---------------------------------------------------------------------------
# Defaults used only for warn/off degraded execution paths.
# Strict mode requires dataset-backed rows.
# ---------------------------------------------------------------------------

_DEFAULT_EHL_CONSTANT = 0.045
_DEFAULT_EHL_SPEED_EXP = 0.70
_DEFAULT_EHL_PRESSURE_EXP = 0.13
_DEFAULT_EHL_TEMP_REF_C = 90.0
_DEFAULT_EHL_TEMP_EXP = 0.18

_DEFAULT_T_SCUFF_CRIT_C = 400.0
_DEFAULT_LAMBDA_PERM = 0.30

_VALIDATION_MODES = {"strict", "warn", "off"}
_SCUFF_METHODS = {"auto", "flash", "integral"}


class LubeRegime(Enum):
    """Lubrication regime classified by specific film thickness λ.

    Thresholds per AGMA 925-A03 / NASA correlation work:
        BOUNDARY:  λ ≤ 0.4
        MIXED:     0.4 < λ ≤ 1.0
        FULL_EHL:  λ > 1.0
    """

    BOUNDARY = "boundary"
    MIXED = "mixed"
    FULL_EHL = "full_ehl"


@dataclass(frozen=True)
class TribologyParams:
    """Operating-point tribology inputs.

    All values are for a single phase bin; the caller is responsible
    for evaluating across the phase grid and finding worst-case.

    Attributes:
        hertz_stress_MPa: Maximum Hertzian contact stress.
        sliding_velocity_m_s: Local sliding velocity at the contact.
        entrainment_velocity_m_s: Mean rolling (entrainment) velocity.
        oil_viscosity_cSt: Dynamic viscosity at contact inlet temperature.
        composite_roughness_um: Combined RMS roughness σ* (µm).
        bulk_temp_C: Bulk gear body temperature.
        oil_inlet_temp_C: Oil temperature at contact inlet.
        oil_type: Oil family identifier used for dataset lookup.
        additive_package: Additive-package key used for scuff/FZG lookup.
        finish_tier: Surface finish key used for EHL/micropitting lookup.
        friction_coeff: Effective friction coefficient for temperature-rise proxy.
    """

    hertz_stress_MPa: float = 1200.0
    sliding_velocity_m_s: float = 5.0
    entrainment_velocity_m_s: float = 15.0
    oil_viscosity_cSt: float = 12.0
    composite_roughness_um: float = 0.4
    bulk_temp_C: float = 150.0
    oil_inlet_temp_C: float = 90.0
    oil_type: str = "generic_ep"
    additive_package: str = "standard_ep"
    finish_tier: str = "fine_ground"
    friction_coeff: float = 0.06


@dataclass(frozen=True)
class TribologyEvaluation:
    """Single operating-point tribology evaluation bundle."""

    lambda_min: float
    lambda_perm: float
    micropitting_safety: float
    lube_regime: str
    scuff_margin_flash_C: float
    scuff_margin_integral_C: float
    scuff_margin_C: float
    tribology_method_used: str
    tribology_data_status: str
    tribology_data_messages: tuple[str, ...]
    tribology_provenance: dict[str, dict[str, str]]


def _normalize_validation_mode(validation_mode: str) -> str:
    mode = str(validation_mode).strip().lower()
    if mode not in _VALIDATION_MODES:
        raise ValueError(
            f"validation_mode must be one of {{'strict', 'warn', 'off'}}, got {validation_mode!r}"
        )
    return mode


def _normalize_scuff_method(scuff_method: str) -> str:
    method = str(scuff_method).strip().lower()
    if method not in _SCUFF_METHODS:
        raise ValueError(
            f"scuff_method must be one of {{'auto', 'flash', 'integral'}}, got {scuff_method!r}"
        )
    return method


def _is_blank(value: Any) -> bool:
    text = str(value).strip().lower()
    return text == "" or text in {"none", "nan", "null"}


def _as_float(
    raw: Any,
    *,
    dataset: str,
    column: str,
    mode: str,
    messages: list[str],
    default: float,
) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        msg = (
            f"Dataset '{dataset}' has non-numeric value for '{column}': {raw!r}. "
            f"Using fallback {default}."
        )
        if mode == "strict":
            raise ValueError(msg)
        messages.append(msg)
        return float(default)


def _table_rows(table: dict[str, list]) -> list[dict[str, Any]]:
    if not table:
        return []
    cols = list(table.keys())
    n_rows = max((len(table.get(c, [])) for c in cols), default=0)
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        row: dict[str, Any] = {}
        for c in cols:
            vals = table.get(c, [])
            row[c] = vals[i] if i < len(vals) else ""
        rows.append(row)
    return rows


def _load_rows(
    dataset: str,
    *,
    mode: str,
    key_columns: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    reg = get_registry()
    table, table_messages = reg.load_required_table(
        dataset,
        validation_mode=mode,
        key_columns=key_columns,
    )
    rows = _table_rows(table)
    return rows, list(table_messages)


def _extract_meta(row: dict[str, Any]) -> dict[str, str]:
    return {
        "provenance": str(row.get("provenance", "")).strip(),
        "version": str(row.get("version", "")).strip(),
    }


def _eq_text(a: Any, b: Any) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def _pick_temperature_row(
    rows: list[dict[str, Any]],
    *,
    temperature_C: float,
    mode: str,
    messages: list[str],
) -> dict[str, Any]:
    candidates: list[tuple[float, float, dict[str, Any], float, float]] = []
    for row in rows:
        try:
            t_min = float(row.get("temp_C_min", ""))
            t_max = float(row.get("temp_C_max", ""))
        except (TypeError, ValueError):
            if mode == "strict":
                raise ValueError(
                    "tribology_ehl_coefficients has non-numeric temp_C_min/temp_C_max values."
                )
            continue

        if t_max < t_min:
            t_min, t_max = t_max, t_min

        if t_min <= temperature_C <= t_max:
            distance = 0.0
        else:
            distance = min(abs(temperature_C - t_min), abs(temperature_C - t_max))
        span = max(t_max - t_min, 1e-9)
        candidates.append((distance, span, row, t_min, t_max))

    if not candidates:
        msg = (
            "No valid tribology_ehl_coefficients row with numeric temp_C_min/temp_C_max was found."
        )
        if mode == "strict":
            raise ValueError(msg)
        messages.append(msg)
        return rows[0]

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_distance, _, best_row, t_min, t_max = candidates[0]
    if best_distance > 0.0:
        msg = (
            "EHL coefficient temperature outside calibrated range "
            f"[{t_min:.1f}, {t_max:.1f}] C for requested {temperature_C:.1f} C; "
            "using nearest row."
        )
        if mode == "strict":
            raise ValueError(msg)
        messages.append(msg)
    return best_row


def _resolve_ehl_coefficients(
    params: TribologyParams,
    *,
    mode: str,
) -> tuple[dict[str, float], list[str], dict[str, str]]:
    rows, messages = _load_rows(
        "tribology_ehl_coefficients",
        mode=mode,
        key_columns=("oil_type", "finish_tier", "ehl_constant"),
    )

    matched = [
        row
        for row in rows
        if _eq_text(row.get("oil_type"), params.oil_type)
        and _eq_text(row.get("finish_tier"), params.finish_tier)
    ]

    if not matched:
        msg = (
            "No tribology_ehl_coefficients row for "
            f"oil_type={params.oil_type!r}, finish_tier={params.finish_tier!r}."
        )
        if mode == "strict":
            raise ValueError(msg)
        messages.append(msg)
        matched = rows

    row = _pick_temperature_row(
        matched,
        temperature_C=float(params.oil_inlet_temp_C),
        mode=mode,
        messages=messages,
    )

    coeffs = {
        "ehl_constant": _as_float(
            row.get("ehl_constant"),
            dataset="tribology_ehl_coefficients",
            column="ehl_constant",
            mode=mode,
            messages=messages,
            default=_DEFAULT_EHL_CONSTANT,
        ),
        "viscosity_speed_exp": _as_float(
            row.get("viscosity_speed_exp"),
            dataset="tribology_ehl_coefficients",
            column="viscosity_speed_exp",
            mode=mode,
            messages=messages,
            default=_DEFAULT_EHL_SPEED_EXP,
        ),
        "pressure_exp": _as_float(
            row.get("pressure_exp"),
            dataset="tribology_ehl_coefficients",
            column="pressure_exp",
            mode=mode,
            messages=messages,
            default=_DEFAULT_EHL_PRESSURE_EXP,
        ),
        "temp_ref_C": _as_float(
            row.get("temp_ref_C"),
            dataset="tribology_ehl_coefficients",
            column="temp_ref_C",
            mode=mode,
            messages=messages,
            default=_DEFAULT_EHL_TEMP_REF_C,
        ),
        "temp_exp": _as_float(
            row.get("temp_exp"),
            dataset="tribology_ehl_coefficients",
            column="temp_exp",
            mode=mode,
            messages=messages,
            default=_DEFAULT_EHL_TEMP_EXP,
        ),
    }
    return coeffs, messages, _extract_meta(row)


def _pick_scuff_reference_row(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    messages: list[str],
    method: str,
    params: TribologyParams,
) -> dict[str, Any]:
    exact = [
        row
        for row in rows
        if _eq_text(row.get("oil_type"), params.oil_type)
        and _eq_text(row.get("additive_package"), params.additive_package)
        and _eq_text(row.get("method"), method)
    ]
    if exact:
        candidates = exact
    else:
        msg = (
            "No scuffing_critical_temperatures row for "
            f"oil_type={params.oil_type!r}, additive_package={params.additive_package!r}, "
            f"method={method!r}."
        )
        if mode == "strict":
            raise ValueError(msg)
        messages.append(msg)
        by_method = [row for row in rows if _eq_text(row.get("method"), method)]
        candidates = by_method or rows

    # Conservative choice: lowest critical temperature across candidate rows.
    best_row = candidates[0]
    best_t = None
    for row in candidates:
        t_crit = _as_float(
            row.get("T_crit_C"),
            dataset="scuffing_critical_temperatures",
            column="T_crit_C",
            mode=mode,
            messages=messages,
            default=_DEFAULT_T_SCUFF_CRIT_C,
        )
        if best_t is None or t_crit < best_t:
            best_t = t_crit
            best_row = row
    return best_row


def _resolve_scuff_critical_temperature(
    *,
    params: TribologyParams,
    method: str,
    mode: str,
) -> tuple[float, list[str], dict[str, dict[str, str]]]:
    rows_scuff, messages = _load_rows(
        "scuffing_critical_temperatures",
        mode=mode,
        key_columns=("oil_type", "additive_package", "method", "T_crit_C"),
    )
    scuff_row = _pick_scuff_reference_row(
        rows_scuff,
        mode=mode,
        messages=messages,
        method=method,
        params=params,
    )

    t_crit_scuff = _as_float(
        scuff_row.get("T_crit_C"),
        dataset="scuffing_critical_temperatures",
        column="T_crit_C",
        mode=mode,
        messages=messages,
        default=_DEFAULT_T_SCUFF_CRIT_C,
    )
    load_stage = _as_float(
        scuff_row.get("load_stage"),
        dataset="scuffing_critical_temperatures",
        column="load_stage",
        mode=mode,
        messages=messages,
        default=0.0,
    )
    test_method = str(scuff_row.get("test_method", "")).strip()

    rows_fzg, fzg_messages = _load_rows(
        "fzg_step_load_map",
        mode=mode,
        key_columns=("test_standard", "test_method", "load_stage", "T_crit_C"),
    )
    messages.extend(fzg_messages)

    fzg_candidates = [
        row
        for row in rows_fzg
        if _eq_text(row.get("oil_type"), params.oil_type)
        and _eq_text(row.get("additive_package"), params.additive_package)
    ]
    if test_method:
        method_rows = [
            row for row in fzg_candidates if _eq_text(row.get("test_method"), test_method)
        ]
        if method_rows:
            fzg_candidates = method_rows

    if not fzg_candidates:
        msg = (
            "No fzg_step_load_map row for "
            f"oil_type={params.oil_type!r}, additive_package={params.additive_package!r}."
        )
        if mode == "strict":
            raise ValueError(msg)
        messages.append(msg)
        fzg_row = None
    else:
        # Pick nearest load stage to the scuff-row stage.
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in fzg_candidates:
            stage = _as_float(
                row.get("load_stage"),
                dataset="fzg_step_load_map",
                column="load_stage",
                mode=mode,
                messages=messages,
                default=load_stage,
            )
            scored.append((abs(stage - load_stage), row))
        scored.sort(key=lambda item: item[0])
        fzg_row = scored[0][1]

    t_crit_final = t_crit_scuff
    provenance = {
        "scuffing_critical_temperatures": _extract_meta(scuff_row),
    }
    if fzg_row is not None:
        t_crit_final = _as_float(
            fzg_row.get("T_crit_C"),
            dataset="fzg_step_load_map",
            column="T_crit_C",
            mode=mode,
            messages=messages,
            default=t_crit_scuff,
        )
        provenance["fzg_step_load_map"] = _extract_meta(fzg_row)

    return t_crit_final, messages, provenance


def _resolve_lambda_perm(
    *,
    params: TribologyParams,
    mode: str,
) -> tuple[float, list[str], dict[str, str]]:
    rows, messages = _load_rows(
        "micropitting_lambda_perm",
        mode=mode,
        key_columns=("oil_type", "finish_tier", "lambda_perm"),
    )

    matched = [
        row
        for row in rows
        if _eq_text(row.get("oil_type"), params.oil_type)
        and _eq_text(row.get("finish_tier"), params.finish_tier)
    ]
    if not matched:
        msg = (
            "No micropitting_lambda_perm row for "
            f"oil_type={params.oil_type!r}, finish_tier={params.finish_tier!r}."
        )
        if mode == "strict":
            raise ValueError(msg)
        messages.append(msg)
        matched = rows

    row = matched[0]
    lambda_perm = _as_float(
        row.get("lambda_perm"),
        dataset="micropitting_lambda_perm",
        column="lambda_perm",
        mode=mode,
        messages=messages,
        default=_DEFAULT_LAMBDA_PERM,
    )
    return lambda_perm, messages, _extract_meta(row)


def _data_status(mode: str, messages: list[str]) -> str:
    if not messages:
        return "ok"
    if mode == "warn":
        return "degraded_warn"
    if mode == "off":
        return "degraded_off"
    return "ok"


def _compute_lambda_from_coeffs(params: TribologyParams, coeffs: dict[str, float]) -> float:
    if params.composite_roughness_um <= 0:
        return 10.0

    viscosity_speed = max(
        float(params.oil_viscosity_cSt) * abs(float(params.entrainment_velocity_m_s)),
        0.0,
    )
    if viscosity_speed <= 0.0:
        return 0.0

    h_min = float(coeffs["ehl_constant"]) * (
        viscosity_speed ** float(coeffs["viscosity_speed_exp"])
    )

    stress = max(float(params.hertz_stress_MPa), 100.0)
    pressure_factor = (1500.0 / stress) ** float(coeffs["pressure_exp"])
    h_min *= pressure_factor

    # Proxy thermal-viscosity correction around calibrated reference temperature.
    temp_ref = max(float(coeffs["temp_ref_C"]), 1.0)
    temp_eval = max(0.5 * (float(params.bulk_temp_C) + float(params.oil_inlet_temp_C)), 1.0)
    temp_factor = (temp_ref / temp_eval) ** float(coeffs["temp_exp"])
    h_min *= temp_factor

    lambda_val = h_min / float(params.composite_roughness_um)
    return float(np.clip(lambda_val, 0.0, 10.0))


def _flash_contact_temp(params: TribologyParams) -> float:
    sliding = abs(float(params.sliding_velocity_m_s))
    entrainment = max(abs(float(params.entrainment_velocity_m_s)), 0.1)
    load_proxy = (max(float(params.hertz_stress_MPa), 100.0) / 1000.0) ** 2
    flash_rise = float(params.friction_coeff) * load_proxy * sliding / (0.01 * np.sqrt(entrainment))
    flash_rise = float(np.clip(flash_rise, 0.0, 650.0))
    return float(params.bulk_temp_C) + flash_rise


def _integral_contact_temp(params: TribologyParams) -> float:
    sliding = abs(float(params.sliding_velocity_m_s))
    entrainment = max(abs(float(params.entrainment_velocity_m_s)), 0.1)
    load_proxy = (max(float(params.hertz_stress_MPa), 100.0) / 1000.0) ** 2
    thermal_path = 0.6 * sliding + 0.4 * entrainment
    integral_rise = (
        float(params.friction_coeff)
        * load_proxy
        * thermal_path
        / (0.018 * np.power(entrainment, 0.25))
    )
    integral_rise = float(np.clip(integral_rise, 0.0, 650.0))
    base_temp = 0.6 * float(params.bulk_temp_C) + 0.4 * float(params.oil_inlet_temp_C)
    return base_temp + integral_rise


def evaluate_tribology(
    params: TribologyParams,
    *,
    scuff_method: str = "auto",
    validation_mode: str = "strict",
) -> TribologyEvaluation:
    """Evaluate lambda, scuff margins, and micropitting safety in one pass."""

    mode = _normalize_validation_mode(validation_mode)
    method = _normalize_scuff_method(scuff_method)

    messages: list[str] = []
    provenance: dict[str, dict[str, str]] = {}

    coeffs, msg_ehl, meta_ehl = _resolve_ehl_coefficients(params, mode=mode)
    messages.extend(msg_ehl)
    provenance["tribology_ehl_coefficients"] = meta_ehl
    lambda_val = _compute_lambda_from_coeffs(params, coeffs)

    t_crit_flash, msg_flash, meta_flash = _resolve_scuff_critical_temperature(
        params=params, method="flash", mode=mode
    )
    messages.extend(msg_flash)
    provenance.update(meta_flash)

    t_crit_integral, msg_integral, meta_integral = _resolve_scuff_critical_temperature(
        params=params, method="integral", mode=mode
    )
    messages.extend(msg_integral)
    provenance.update(meta_integral)

    margin_flash = float(t_crit_flash - _flash_contact_temp(params))
    margin_integral = float(t_crit_integral - _integral_contact_temp(params))

    if method == "flash":
        selected_margin = margin_flash
        method_used = "flash"
    elif method == "integral":
        selected_margin = margin_integral
        method_used = "integral"
    else:
        selected_margin = min(margin_flash, margin_integral)
        method_used = "flash" if margin_flash <= margin_integral else "integral"

    lambda_perm, msg_lambda, meta_lambda = _resolve_lambda_perm(params=params, mode=mode)
    messages.extend(msg_lambda)
    provenance["micropitting_lambda_perm"] = meta_lambda

    micropitting_sf = compute_micropitting_safety(lambda_val, lambda_perm=lambda_perm)

    dedup_messages = tuple(dict.fromkeys(msg for msg in messages if not _is_blank(msg)))
    status = _data_status(mode, list(dedup_messages))

    return TribologyEvaluation(
        lambda_min=lambda_val,
        lambda_perm=float(lambda_perm),
        micropitting_safety=float(micropitting_sf),
        lube_regime=classify_regime(lambda_val).value,
        scuff_margin_flash_C=float(margin_flash),
        scuff_margin_integral_C=float(margin_integral),
        scuff_margin_C=float(selected_margin),
        tribology_method_used=str(method_used),
        tribology_data_status=str(status),
        tribology_data_messages=dedup_messages,
        tribology_provenance=provenance,
    )


def compute_lambda(
    params: TribologyParams,
    *,
    validation_mode: str = "strict",
) -> float:
    """Compute specific film thickness lambda from data-backed EHL coefficients."""
    mode = _normalize_validation_mode(validation_mode)
    coeffs, _, _ = _resolve_ehl_coefficients(params, mode=mode)
    return _compute_lambda_from_coeffs(params, coeffs)


def classify_regime(lambda_val: float) -> LubeRegime:
    """Classify lubrication regime from specific film thickness."""
    if lambda_val <= 0.4:
        return LubeRegime.BOUNDARY
    elif lambda_val <= 1.0:
        return LubeRegime.MIXED
    else:
        return LubeRegime.FULL_EHL


def compute_scuff_margins(
    params: TribologyParams,
    *,
    scuff_method: str = "auto",
    validation_mode: str = "strict",
) -> dict[str, Any]:
    """Compute both scuff margins and selected margin for a method policy."""
    ev = evaluate_tribology(params, scuff_method=scuff_method, validation_mode=validation_mode)
    return {
        "scuff_margin_flash_C": float(ev.scuff_margin_flash_C),
        "scuff_margin_integral_C": float(ev.scuff_margin_integral_C),
        "scuff_margin_C": float(ev.scuff_margin_C),
        "tribology_method_used": str(ev.tribology_method_used),
        "tribology_data_status": str(ev.tribology_data_status),
        "tribology_data_messages": list(ev.tribology_data_messages),
        "tribology_provenance": dict(ev.tribology_provenance),
    }


def compute_scuff_margin(
    params: TribologyParams,
    *,
    scuff_method: str = "auto",
    validation_mode: str = "strict",
) -> float:
    """Compute selected scuffing temperature margin (deg C)."""
    ev = evaluate_tribology(params, scuff_method=scuff_method, validation_mode=validation_mode)
    return float(ev.scuff_margin_C)


def compute_micropitting_safety(
    lambda_val: float,
    lambda_perm: float | None = None,
    *,
    params: TribologyParams | None = None,
    validation_mode: str = "strict",
) -> float:
    """Compute micropitting safety factor S_lambda = lambda_min / lambda_perm."""
    perm = lambda_perm
    if perm is None:
        mode = _normalize_validation_mode(validation_mode)
        base_params = params or TribologyParams()
        perm, _, _ = _resolve_lambda_perm(params=base_params, mode=mode)
    if float(perm) <= 0:
        return 10.0
    return float(np.clip(float(lambda_val) / float(perm), 0.0, 10.0))
