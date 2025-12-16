"""
CQE Command-Line Interface
"""

import click
from cqe import CQEClient, __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """CQE: Cartan-Quadratic Equivalence Framework"""
    pass


@main.command()
@click.argument('text')
@click.option('--optimize/--no-optimize', default=True, help='Apply MORSR optimization')
def embed(text, optimize):
    """Embed text into E8 space"""
    client = CQEClient()
    overlay = client.embed(text, optimize=optimize)

    click.echo(f"Overlay ID: {overlay.hash_id}")
    click.echo(f"Active slots: {len(overlay.active_slots)}/248")
    click.echo(f"Cartan active: {overlay.cartan_active}/8")

    metrics = client.get_phi_metrics(overlay)
    click.echo(f"\nΦ Metrics:")
    for key, value in metrics.items():
        click.echo(f"  {key}: {value:.3f}")


@main.command()
def info():
    """Display CQE system information"""
    client = CQEClient()

    click.echo("CQE System Information")
    click.echo("=" * 40)
    click.echo(f"Version: {__version__}")

    lattice_info = client.lattice.info()
    click.echo(f"\nE8 Lattice:")
    for key, value in lattice_info.items():
        click.echo(f"  {key}: {value}")

    cache_stats = client.get_cache_stats()
    click.echo(f"\nCache:")
    click.echo(f"  Size: {cache_stats['size']} overlays")


if __name__ == '__main__':
    main()
