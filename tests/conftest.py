import os
from pathlib import Path

from typer.testing import CliRunner

from makeragents.cli import app

runner = CliRunner()


# TODO: remove os.chdir for parallel safety
def _invoke_in(tmp_path: Path, *args: str):
    """Invoke the CLI app with *args* after chdir-ing to *tmp_path*."""
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        return runner.invoke(app, list(args))
    finally:
        os.chdir(cwd)
