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
#!/usr/bin/env python3
"""
Populate golden test data

Creates reference data and directory structure on cold start.
"""

import os
import json
from pathlib import Path
import sys

