"""All four GPU-kind detectors agree on GB300 (oracle dispatch, perf.json labels, --device-kind)."""
import subprocess

import pytest
import torch

import bench
import gpu_lock
import oracle_harness
import repro_harness


@pytest.fixture
def uncached_gpu_kind():
    # get_gpu_kind caches its answer on the function; keep other tests' value intact.
    saved = oracle_harness.get_gpu_kind.__dict__.pop("_cached", None)
    yield
    oracle_harness.get_gpu_kind.__dict__.pop("_cached", None)
    if saved is not None:
        oracle_harness.get_gpu_kind._cached = saved


@pytest.mark.parametrize(
    "name,kind",
    [("NVIDIA GB300", "GB300"), ("NVIDIA B200", "B200"), ("NVIDIA GB200", "B200"), ("NVIDIA H100 80GB HBM3", "H100")],
)
def test_gpu_kind_detectors_agree(monkeypatch, uncached_gpu_kind, name, kind):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: name + "\n")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)

    assert oracle_harness.get_gpu_kind() == kind
    assert repro_harness._detect_hardware() == kind
    assert bench.current_hardware_label() == kind
    assert gpu_lock.device_kind_from_name(name) == kind
