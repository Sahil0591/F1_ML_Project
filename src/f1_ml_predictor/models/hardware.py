"""Explicit, bounded hardware probes, isolated from normal tests and imports."""

import ctypes
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from f1_ml_predictor.models.boosting import BACKENDS, estimator


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "scikit-learn", "xgboost", "lightgbm", "catboost"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cuda_api_versions() -> dict[str, Any]:
    """Read actual driver/runtime API versions on Windows without changing PATH."""
    result: dict[str, Any] = {}
    if os.name != "nt":
        return {"status": "not_probed", "reason": "Windows API probe only"}
    try:
        driver = ctypes.WinDLL("nvcuda.dll")
        value = ctypes.c_int()
        result["driver_init_returncode"] = driver.cuInit(0)
        result["driver_version_returncode"] = driver.cuDriverGetVersion(ctypes.byref(value))
        result["driver_api_version"] = value.value
    except OSError as exc:
        result["driver_error"] = str(exc)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        root = Path(cuda_path) / "bin"
        candidates = list(root.glob("cudart64_*.dll")) + list(root.glob("x64/cudart64_*.dll"))
        for path in sorted(candidates):
            try:
                runtime = ctypes.WinDLL(str(path.resolve()))
                value = ctypes.c_int()
                code = runtime.cudaRuntimeGetVersion(ctypes.byref(value))
                result["toolkit_runtime"] = {
                    "path": str(path.resolve()),
                    "returncode": code,
                    "api_version": value.value,
                }
                break
            except OSError as exc:
                result["runtime_error"] = str(exc)
    if "toolkit_runtime" not in result:
        result["runtime_status"] = "no standalone toolkit runtime resolved"
    result["note"] = "Toolkit runtime is separate from statically linked estimator runtimes."
    return result


def _command(command: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()[-3000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def probe_worker(backend: str, device: str, workload_rows: int) -> dict[str, Any]:
    """Exercise both estimator tasks and verify XGBoost did not silently use CPU."""
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(workload_rows, 32)).astype(np.float32)
    position = matrix[:, 0] * 2 + matrix[:, 1] + rng.normal(size=workload_rows)
    dnf = (matrix[:, 2] + rng.normal(size=workload_rows) > 1).astype(int)
    elapsed = []
    warning_messages = []
    build: dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        if backend == "xgboost":
            build = importlib.import_module("xgboost").build_info()
            if device == "cuda" and not build.get("USE_CUDA"):
                raise RuntimeError("installed XGBoost build has no CUDA")
        for _ in range(2):
            start = time.perf_counter()
            for task, target in (("position", position), ("dnf", dnf)):
                model = estimator(backend, task, 42, device)
                model.fit(matrix, target)
                model.predict(matrix[:20])
                if backend == "xgboost" and device == "cuda":
                    config = json.loads(model.get_booster().save_config())
                    if not config["learner"]["generic_param"]["device"].startswith("cuda"):
                        raise RuntimeError("XGBoost silently fell back to CPU")
            elapsed.append(time.perf_counter() - start)
        warning_messages = sorted({str(item.message) for item in captured})
    return {
        "status": "usable",
        "device": device,
        "seconds": elapsed,
        "median_seconds": float(np.median(elapsed)),
        "build_info": build,
        "warnings": warning_messages,
        "reproducibility": "GPU floating-point results may vary"
        if device != "cpu"
        else "fixed seed, one CPU thread",
    }


def inspect_hardware(output: Path, *, workload_rows: int = 8000) -> dict[str, Any]:
    if isinstance(workload_rows, bool) or not 256 <= workload_rows <= 50000:
        raise ValueError("hardware workload must have 256 to 50000 rows")
    versions = library_versions()
    report: dict[str, Any] = {
        "version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": versions,
        "workload": {
            "rows": workload_rows,
            "features": 32,
            "trees": 80,
            "repeats": 2,
            "synthetic": True,
        },
        "nvidia_smi": _command(["nvidia-smi"]),
        "gpu_inventory": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "cuda_toolkit": _command(["nvcc", "--version"]),
        "cuda_api": cuda_api_versions(),
        "note": "nvidia-smi reports driver capability; nvcc reports toolkit, not loaded runtime. "
        "CUDA library build information and actual training probes establish usability.",
        "backends": {},
    }
    for backend in BACKENDS:
        if backend != "hist" and versions[backend] is None:
            report["backends"][backend] = {"status": "not_installed"}
            continue
        devices = (
            ("cpu",)
            if backend == "hist"
            else (("cpu", "cuda", "gpu") if backend == "lightgbm" else ("cpu", "cuda"))
        )
        probes = {}
        for device in devices:
            response = _command(
                [
                    sys.executable,
                    "-m",
                    "f1_ml_predictor.models.hardware",
                    backend,
                    device,
                    str(workload_rows),
                ],
                timeout=60,
            )
            # Libraries may print diagnostics before the final JSON line.
            try:
                last_line = response.get("stdout", "").splitlines()[-1]
                probe = json.loads(last_line)
                if response.get("returncode") != 0:
                    raise ValueError("worker failed")
            except (IndexError, ValueError):
                probe = {"status": "unavailable", "details": response}
            probes[device] = probe
        report["backends"][backend] = {"status": "probed", "devices": probes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def choose_device(
    backend: str,
    rows: int,
    requested: str,
    hardware: dict[str, Any] | None,
) -> tuple[str, str]:
    if requested not in {"cpu", "auto", "cuda"}:
        raise ValueError("device must be cpu, auto, or cuda")
    if backend == "hist" or requested == "cpu":
        return "cpu", "CPU requested or CPU-only estimator"
    if hardware is None or hardware.get("libraries") != library_versions():
        return "cpu", "no matching verified hardware report"
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(hardware["captured_at"])).total_seconds()
    except (KeyError, ValueError, TypeError):
        age = -1
    if not 0 <= age <= 86400:
        return "cpu", "hardware report is missing a recent capture time"
    devices = hardware.get("backends", {}).get(backend, {}).get("devices", {})
    for device in ("cuda", "gpu") if backend == "lightgbm" else ("cuda",):
        probe = devices.get(device, {})
        if probe.get("status") != "usable":
            continue
        if requested == "cuda":
            return device, "explicit GPU request with successful training probe"
        cpu = devices.get("cpu", {})
        workload_rows = hardware["workload"]["rows"]
        if (
            rows >= workload_rows
            and cpu.get("status") == "usable"
            and probe["median_seconds"] < 0.85 * cpu["median_seconds"]
        ):
            return device, "measured at least 15% faster at a supported workload size"
    return "cpu", "GPU unavailable, workload smaller than benchmark, or no measured benefit"


if __name__ == "__main__":
    try:
        result = probe_worker(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    except Exception as exc:
        result = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
