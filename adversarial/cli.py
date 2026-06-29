from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

app = typer.Typer(
    help="Adversarial ML — fool image classifiers with imperceptible perturbations",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def attack(
    image: Path = typer.Argument(..., help="Path to input image (JPEG/PNG)"),
    epsilon: float = typer.Option(0.03, "--epsilon", "-e", help="Max L-inf perturbation in [0,1] (default 0.03)"),
    method: str = typer.Option("fgsm", "--method", "-m", help="Attack method: fgsm (default) or pgd"),
    fmt: str = typer.Option("both", "--format", "-f", help="Report format: md, both (JSON always written)"),
    output_dir: Path = typer.Option(Path("reports"), "--output", "-o", help="Output directory"),
) -> None:
    """Attack a single image and show how the classifier is fooled."""
    sys.stdout.reconfigure(encoding="utf-8")

    if not image.exists():
        console.print(f"[red]Image not found:[/red] {image}")
        raise typer.Exit(1)

    if method not in ("fgsm", "pgd"):
        console.print(f"[red]Unknown method:[/red] {method}. Choose fgsm or pgd.")
        raise typer.Exit(1)

    # Lazy imports so --help is instant even without PyTorch installed
    import torch
    from .classifier import get_device, load_model, load_image, classify, tensor_to_pil, perturbation_to_pil
    from .attacks.fgsm import fgsm_attack
    from .models import AttackConfig, AttackResult
    from .reporter import save_reports

    device = get_device()

    console.print(Panel(
        f"[bold cyan]Adversarial ML[/bold cyan]  attacking [yellow]{image.name}[/yellow]  "
        f"method=[green]{method.upper()}[/green]  ε=[green]{epsilon}[/green]  "
        f"device=[dim]{device}[/dim]",
        box=box.ROUNDED,
    ))

    with console.status("[cyan]Loading ResNet-50...[/cyan]"):
        model, categories = load_model(device)

    with console.status("[cyan]Classifying original image...[/cyan]"):
        image_01 = load_image(str(image), device)
        orig_preds = classify(model, image_01, categories)

    console.print(f"\n[bold]Original:[/bold] [green]{orig_preds[0].class_name}[/green] "
                  f"({orig_preds[0].confidence:.1%})")

    with console.status(f"[cyan]Running {method.upper()} attack (ε={epsilon})...[/cyan]"):
        true_class = orig_preds[0].class_id
        if method == "fgsm":
            adversarial_01 = fgsm_attack(model, image_01, true_class, epsilon)
        else:
            from .attacks.pgd import pgd_attack
            adversarial_01 = pgd_attack(model, image_01, true_class, epsilon)

    with console.status("[cyan]Classifying adversarial image...[/cyan]"):
        adv_preds = classify(model, adversarial_01, categories)

    diff = (adversarial_01 - image_01).abs()
    l_inf = diff.max().item()
    l2 = diff.pow(2).mean().sqrt().item()
    successful = adv_preds[0].class_id != orig_preds[0].class_id

    # ── Display results ───────────────────────────────────────────────────────
    status_color = "green" if successful else "yellow"
    status_label = "FOOLED" if successful else "RESISTED"
    console.print(f"[bold]Adversarial:[/bold] [{status_color}]{adv_preds[0].class_name}[/{status_color}] "
                  f"({adv_preds[0].confidence:.1%})  [{status_color}]{status_label}[/{status_color}]")
    console.print(f"[dim]L∞={l_inf:.5f}  L2={l2:.5f}[/dim]")

    table = Table(title="Top-5 Before vs After", box=box.SIMPLE_HEAVY)
    table.add_column("Rank", width=5)
    table.add_column("Original class", style="cyan")
    table.add_column("Conf", width=8)
    table.add_column("Adversarial class", style="yellow")
    table.add_column("Conf", width=8)
    for i, (o, a) in enumerate(zip(orig_preds, adv_preds), 1):
        table.add_row(
            str(i),
            o.class_name, f"{o.confidence:.1%}",
            a.class_name, f"{a.confidence:.1%}",
        )
    console.print(table)

    # ── Save images ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image.stem
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    adv_path  = output_dir / f"{stem}_{method}_adv_{ts}.png"
    pert_path = output_dir / f"{stem}_{method}_perturbation_{ts}.png"
    tensor_to_pil(adversarial_01).save(adv_path)
    perturbation_to_pil(image_01, adversarial_01).save(pert_path)

    result = AttackResult(
        image_path=str(image),
        original_top1=orig_preds[0],
        original_top5=orig_preds,
        adversarial_top1=adv_preds[0],
        adversarial_top5=adv_preds,
        config=AttackConfig(method=method, epsilon=epsilon),
        attack_successful=successful,
        l_inf=l_inf,
        l2=l2,
        adversarial_image_path=str(adv_path),
        perturbation_image_path=str(pert_path),
        timestamp=datetime.now(timezone.utc),
    )

    paths = save_reports(result, output_dir, fmt=fmt)
    console.print("\n[green]Reports saved:[/green]")
    for key, path in paths.items():
        console.print(f"  [dim]{key}:[/dim] {path}")
    console.print(f"  [dim]adversarial image:[/dim] {adv_path}")
    console.print(f"  [dim]perturbation (10× amplified):[/dim] {pert_path}")


def main() -> None:
    app()
