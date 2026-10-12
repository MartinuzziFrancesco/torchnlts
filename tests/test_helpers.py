"""Tests for the shared helpers in tests/conftest.py."""

import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tests.conftest import REPOSITORY_ROOT, LargestOutput, run_julia


@pytest.fixture
def fake_julia(monkeypatch):
    """Puts a fake ``julia`` on PATH whose runs are recorded in ``calls`` and finish with ``outcome``."""
    calls = []
    outcome = SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return outcome

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return SimpleNamespace(calls=calls, outcome=outcome)


def test_run_julia_without_julia_skips(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("TORCHNLTS_REQUIRE_JULIA", raising=False)
    with pytest.raises(pytest.skip.Exception, match="julia not available"):
        run_julia("1")


def test_run_julia_without_julia_fails_when_required(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("TORCHNLTS_REQUIRE_JULIA", "1")
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception), match="TORCHNLTS_REQUIRE_JULIA") as raised:
        run_julia("1")
    assert raised.type is pytest.fail.Exception


def test_run_julia_returns_stdout_and_resolves_the_project_against_the_repository_root(fake_julia):
    fake_julia.outcome.stdout = "42\n"
    assert run_julia("println(42)", "juliaenv_causality", timeout=5) == "42\n"
    ((command, options),) = fake_julia.calls
    assert command == ["julia", f"--project={REPOSITORY_ROOT / 'juliaenv_causality'}", "-e", "println(42)"]
    assert options["timeout"] == 5


def test_run_julia_defaults_to_juliaenv_and_keeps_absolute_projects(fake_julia, tmp_path):
    run_julia("1")
    run_julia("1", str(tmp_path))
    assert [command[1] for command, _ in fake_julia.calls] == [
        f"--project={REPOSITORY_ROOT / 'juliaenv'}",
        f"--project={tmp_path}",
    ]


def test_run_julia_nonzero_exit_fails_with_stderr(fake_julia):
    fake_julia.outcome.returncode = 1
    fake_julia.outcome.stderr = "ERROR: boom"
    with pytest.raises(pytest.fail.Exception, match="ERROR: boom"):
        run_julia("error()")


def test_largest_output_records_the_largest_operation_overall_and_per_dtype():
    with LargestOutput() as recorder:
        torch.zeros(7, 5)
        torch.ones(3)
        torch.ones(2, dtype=torch.float64)
    assert recorder.numel == 35
    assert recorder.numel_by_dtype[torch.float32] == 35
    assert recorder.numel_by_dtype[torch.float64] == 2
