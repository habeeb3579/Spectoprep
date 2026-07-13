"""Tests for the Typer command-line interface."""

from typer.testing import CliRunner

from spectoprep import __version__
from spectoprep.cli import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_command_lists_steps():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "preprocessing transforms available" in result.stdout


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
