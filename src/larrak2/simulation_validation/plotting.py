"""Comparison plot generation using plotly for validation metrics."""

from __future__ import annotations

from pathlib import Path

from .models import ValidationMetricResult


def _safe_import_plotly():
    """Import plotly with graceful fallback."""
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp

        return go, sp
    except ImportError:
        return None, None


def generate_metric_comparison_plot(
    results: list[ValidationMetricResult],
    title: str,
    output_path: str | Path,
) -> str | None:
    """Generate a plotly comparison bar chart for metric results.

    Returns the output path on success, None if plotly unavailable.
    """
    go, _ = _safe_import_plotly()
    if go is None:
        return None

    metric_ids = [r.metric_id for r in results]
    measured = [r.measured_value for r in results]
    simulated = [r.simulated_value for r in results]
    errors = [r.error for r in results]
    colors = ["#2ecc71" if r.passed else "#e74c3c" for r in results]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Measured",
            x=metric_ids,
            y=measured,
            marker_color="#3498db",
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Simulated",
            x=metric_ids,
            y=simulated,
            marker_color=colors,
            opacity=0.8,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    return str(out)


def generate_error_summary_plot(
    results: list[ValidationMetricResult],
    title: str,
    output_path: str | Path,
) -> str | None:
    """Generate error vs tolerance band plot."""
    go, _ = _safe_import_plotly()
    if go is None:
        return None

    metric_ids = [r.metric_id for r in results]
    errors = [r.error for r in results]
    tolerances = [r.tolerance_used for r in results]
    colors = ["#2ecc71" if r.passed else "#e74c3c" for r in results]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Error",
            x=metric_ids,
            y=errors,
            marker_color=colors,
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Tolerance",
            x=metric_ids,
            y=tolerances,
            mode="markers+lines",
            marker=dict(color="#f1c40f", size=10, symbol="diamond"),
            line=dict(dash="dash", color="#f1c40f"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Error / Tolerance",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    return str(out)


def generate_parity_plot(
    results: list[ValidationMetricResult],
    title: str,
    output_path: str | Path,
) -> str | None:
    """Generate measured-vs-simulated parity plot."""
    go, _ = _safe_import_plotly()
    if go is None:
        return None

    measured = [r.measured_value for r in results]
    simulated = [r.simulated_value for r in results]
    labels = [r.metric_id for r in results]
    colors = ["#2ecc71" if r.passed else "#e74c3c" for r in results]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=measured,
            y=simulated,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(color=colors, size=12),
        )
    )

    # Perfect agreement line
    all_vals = measured + simulated
    if all_vals:
        lo = min(all_vals) * 0.9
        hi = max(all_vals) * 1.1
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(dash="dash", color="#95a5a6"),
                name="Perfect Agreement",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Measured",
        yaxis_title="Simulated",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    return str(out)
