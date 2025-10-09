#!/usr/bin/env python

import click
import pathlib
import numpy as np
import pandas as pd
import json
import datetime
import sys
import logging
from rich.logging import RichHandler
from importlib.metadata import distributions

from ._io import validate_input_directory, load_io
from ._anomaly import anomaly_scores, select_idx
from ._explain import generate_cf
from ._plots import plot_timeline, plot_cf_overlay, plot_opt_trace
from ._report import generate_report
from ._schemas import Meta, Environment, Data, DataShapes, Model, Selection, Counterfactual, Report

log = logging.getLogger("anodex")

@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose (DEBUG) logging.")
@click.pass_context
def cli(ctx, verbose):
    """CLI Orchestrator for Time-Series Anomaly Explanation & Reporting"""
    level = logging.DEBUG if verbose else logging.INFO
    log.setLevel(level)
    handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        log_time_format="[%Y-%m-%d %H:%M:%S.%f]"
    )
    log.addHandler(handler)
    ctx.ensure_object(dict)

@cli.command()
@click.option('--in', 'in_dir', required=True, type=click.Path(exists=True, file_okay=False), help='Input directory')
@click.option('--out', 'out_dir', required=True, type=click.Path(file_okay=False), help='Output directory')
@click.option('--select', 'selection_policy', required=True, help='Anomaly selection policy (e.g., "topk:1", "idx:137", "threshold:0.95")')
@click.pass_context
def detect(ctx, in_dir, out_dir, selection_policy):
    """Detect anomalies and select one."""
    log.info(f"Detecting anomalies in '{in_dir}'...")
    
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, _, X_test, _, features, model_name = load_io(in_dir)
        scores = anomaly_scores(model, X_test)
        log.debug(f"Generated {len(scores)} anomaly scores.")
        selected_idx = select_idx(scores, selection_policy)
        log.info(f"Selected index [bold]{selected_idx}[/] based on policy '{selection_policy}'.")

        np.savetxt(out_dir / "scores.csv", scores, delimiter=",")
        (out_dir / "selected_idx.txt").write_text(str(selected_idx))
        np.save(out_dir / "x.npy", X_test[selected_idx])

        log.debug(f"Saved scores to '{out_dir / 'scores.csv'}'")
        log.debug(f"Saved selected index to '{out_dir / 'selected_idx.txt'}'")
        log.debug(f"Saved original instance to '{out_dir / 'x.npy'}'")
        log.info("[bold green]Detection complete![/bold green]")
        ctx.obj['selected_idx'] = selected_idx

    except (FileNotFoundError, AttributeError, ValueError) as e:
        log.exception("Error during detection phase.", extra={"rich_traceback": True})
        raise click.Abort()

@cli.command()
@click.option('--in', 'in_dir', required=True, type=click.Path(exists=True, file_okay=False), help='Input directory')
@click.option('--out', 'out_dir', required=True, type=click.Path(exists=True, file_okay=False), help='Output directory')
@click.option('--idx', type=int, help='Index of the anomaly to explain. Defaults to the one in selected_idx.txt')
@click.option('--cf-engine', default='tsinterpret', show_default=True, help='Counterfactual engine')
@click.option('--lambda', 'lambda_', type=float, default=0.3, show_default=True, help='Lambda for counterfactual generation')
@click.option('--smooth', type=float, default=0.05, show_default=True, help='Smoothness for counterfactual generation')
@click.option('--delta', type=float, default=1.5, show_default=True, help='Delta for counterfactual generation')
@click.option('--max-iters', type=int, default=400, show_default=True, help='Max iterations for counterfactual generation')
@click.pass_context
def explain(ctx, in_dir, out_dir, idx, cf_engine, lambda_, smooth, delta, max_iters):
    """Generate a counterfactual explanation for an anomaly."""
    out_dir = pathlib.Path(out_dir)
    if idx is None:
        try:
            idx = int((out_dir / "selected_idx.txt").read_text())
        except FileNotFoundError:
            log.error("--idx not provided and 'selected_idx.txt' not found.")
            log.error("Run the 'detect' command first or provide an index with --idx.")
            raise click.Abort()

    log.info(f"Explaining anomaly index [bold]{idx}[/] from '{in_dir}' into '{out_dir}'...")
    log.info(f"CF Engine: [bold cyan]{cf_engine}[/bold cyan], Max Iterations: {max_iters}")

    try:
        model, X_train, _, _, _, _ = load_io(in_dir)
        x = np.load(out_dir / "x.npy")
        log.debug(f"Loaded instance 'x' with shape {x.shape} from '{out_dir / 'x.npy'}'")
        x_cf, history = generate_cf(x, model, X_train, lambda_, smooth, delta, max_iters)

        np.save(out_dir / "x_cf.npy", x_cf)
        pd.DataFrame(history).to_csv(out_dir / "opt_hist.csv", index=False)

        log.debug(f"Saved counterfactual to '{out_dir / 'x_cf.npy'}'")
        log.debug(f"Saved optimization history to '{out_dir / 'opt_hist.csv'}'")
        log.info("[bold green]Explanation complete![/bold green]")

    except (FileNotFoundError, AttributeError, ValueError) as e:
        log.exception("Error during explanation phase.", extra={"rich_traceback": True})
        raise click.Abort()

@cli.command()
@click.option('--in', 'in_dir', required=True, type=click.Path(exists=True, file_okay=False), help='Input directory')
@click.option('--out', 'out_dir', required=True, type=click.Path(exists=True, file_okay=False), help='Output directory')
@click.option('--pdf', 'pdf_path', required=True, type=click.Path(dir_okay=False), help='Path to save the PDF report')
@click.pass_context
def report(ctx, in_dir, out_dir, pdf_path):
    """Generate a PDF report."""
    log.info(f"Generating report from '{out_dir}' to '{pdf_path}'...")
    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)
    pdf_path = pathlib.Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    figs_dir = out_dir / 'figs'
    figs_dir.mkdir(exist_ok=True)

    log.info("Loading data for report...")
    # Load data
    model, X_train, X_test, y_test, features, model_name = load_io(in_dir)
    scores = np.loadtxt(out_dir / 'scores.csv', delimiter=',')
    selected_idx = int((out_dir / 'selected_idx.txt').read_text())
    x = np.load(out_dir / 'x.npy')
    x_cf = np.load(out_dir / 'x_cf.npy')
    history_df = pd.read_csv(out_dir / 'opt_hist.csv')

    log.info("Generating plots...")
    # Generate plots
    plot_timeline(scores, selected_idx, figs_dir / 'timeline.png')
    plot_cf_overlay(x, x_cf, features['names'] if features else None, figs_dir)
    plot_opt_trace(history_df, figs_dir / 'opt_trace.png')

    log.info("Gathering metadata...")
    # Create meta.json
    p_anom_before = model.predict_proba(x.reshape(1, -1))[0, 1] if x.ndim == 1 else model.predict_proba(x.reshape(1, x.shape[0], x.shape[1]))[0,1]
    p_anom_after = model.predict_proba(x_cf.reshape(1, -1))[0, 1] if x.ndim == 1 else model.predict_proba(x_cf.reshape(1, x.shape[0], x.shape[1]))[0,1]
    
    packages = {dist.metadata["name"]: dist.version for dist in distributions()}
    env = Environment(python=f"{sys.version_info.major}.{sys.version_info.minor}", packages={
        'ts-interpret': packages.get('ts-interpret', 'not-found'),
        'aeon': packages.get('aeon', 'not-found'),
        'sklearn': packages.get('scikit-learn', 'not-found'),
        'pyod': packages.get('pyod', 'not-found'),
    })

    data_meta = Data(
        mode="timeseries" if X_train.ndim == 3 else "tabular",
        shapes=DataShapes(X_train=list(X_train.shape), X_test=list(X_test.shape)),
        features="features.json" if features else "absent",
        labels_present=y_test is not None
    )

    model_meta = Model(
        path=model_name,
        requires="predict_proba",
        supports=[m for m in ["decision_function", "predict"] if hasattr(model, m)]
    )

    selection_meta = Selection(
        policy=ctx.obj.get('selection_policy', 'unknown'),
        selected_idx=selected_idx,
        score_at_idx=scores[selected_idx],
        p_anom_at_idx=p_anom_before
    )

    cf_meta = Counterfactual(
        engine='tsinterpret',
        lambda_=ctx.obj.get('lambda_', 0.3),
        smooth=ctx.obj.get('smooth', 0.05),
        delta=ctx.obj.get('delta', 1.5),
        max_iters=ctx.obj.get('max_iters', 400),
        p_anom_before=p_anom_before,
        p_anom_after=p_anom_after,
        distance_l2=np.linalg.norm(x - x_cf)
    )

    report_meta = Report(
        pdf=str(pdf_path),
        figures=[str(p) for p in figs_dir.glob('*.png')]
    )

    meta = Meta(
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        environment=env,
        data=data_meta,
        model=model_meta,
        selection=selection_meta,
        counterfactual=cf_meta,
        report=report_meta
    )
    meta_path = out_dir / 'meta.json'
    meta.to_json(meta_path)

    log.info("Generating PDF document...")
    # Generate PDF
    generate_report(meta_path, out_dir, pdf_path)
    log.info(f"[bold green]Report generated successfully at '{pdf_path}'[/bold green]")

@cli.command()
@click.option('--in', 'in_dir', required=True, type=click.Path(exists=True, file_okay=False), help='Input directory')
@click.option('--out', 'out_dir', required=True, type=click.Path(file_okay=False), help='Output directory')
@click.option('--select', 'selection_policy', required=True, help='Anomaly selection policy')
@click.option('--pdf', 'pdf_path', required=True, type=click.Path(dir_okay=False), help='Path to save the PDF report')
@click.option('--cf-engine', default='tsinterpret', help='Counterfactual engine')
@click.option('--lambda', 'lambda_', type=float, default=0.3, help='Lambda for counterfactual generation')
@click.option('--smooth', type=float, default=0.05, help='Smoothness for counterfactual generation')
@click.option('--delta', type=float, default=1.5, help='Delta for counterfactual generation')
@click.option('--max-iters', type=int, default=400, help='Max iterations for counterfactual generation')
@click.pass_context
def run(ctx, in_dir, out_dir, selection_policy, pdf_path, cf_engine, lambda_, smooth, delta, max_iters):
    """Run the full pipeline: detect, explain, and report."""
    log.info("[bold blue]Running full pipeline: Detect -> Explain -> Report[/bold blue]")
    ctx.obj['selection_policy'] = selection_policy
    ctx.obj['lambda_'] = lambda_
    ctx.obj['smooth'] = smooth
    ctx.obj['delta'] = delta
    ctx.obj['max_iters'] = max_iters

    ctx.invoke(detect, in_dir=in_dir, out_dir=out_dir, selection_policy=selection_policy)
    selected_idx = ctx.obj.get('selected_idx')
    if selected_idx is not None:
        ctx.invoke(explain, in_dir=in_dir, out_dir=out_dir, idx=selected_idx, cf_engine=cf_engine, lambda_=lambda_, smooth=smooth, delta=delta, max_iters=max_iters)
        ctx.invoke(report, in_dir=in_dir, out_dir=out_dir, pdf_path=pdf_path)

@cli.command()
@click.option('--in', 'in_dir', required=True, type=click.Path(exists=True, file_okay=False), help='Input directory')
def validate(in_dir):
    """Validate the input directory structure and files."""
    log.info(f"Validating input directory '{in_dir}'...")
    files, errors = validate_input_directory(in_dir)
    if errors:
        for error in errors:
            log.error(f"Validation failed: {error}")
        raise click.Abort()
    
    log.info("[bold green]Validation successful![/bold green]")
    for name, path in files.items():
        if path:
            log.debug(f"  - Found {name}: {path}")
        else:
            log.debug(f"  - Optional {name} not found.")

if __name__ == '__main__':
    cli()
