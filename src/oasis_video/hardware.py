"""Best-effort local compute inventory; unknown values remain explicitly unknown."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _system_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
    return None


def _cpu_model() -> str | None:
    model = platform.processor().strip()
    if model:
        return model
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith(("model name", "hardware")) and ":" in line:
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return None


def _nvidia_gpus() -> list[dict[str, Any]] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    devices = []
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",", 2)]
        if len(columns) == 3:
            try:
                memory_mib = int(columns[1])
            except ValueError:
                memory_mib = None
            devices.append({"name": columns[0], "memory_mib": memory_mib, "driver_version": columns[2]})
    return devices


def inspect_hardware(storage_path: str | Path = ".") -> dict[str, Any]:
    """Return machine/software inventory without guessing at absent hardware."""
    storage = None
    try:
        storage_target = Path(storage_path).resolve()
        while not storage_target.exists() and storage_target != storage_target.parent:
            storage_target = storage_target.parent
        usage = shutil.disk_usage(storage_target)
        storage = {"total_bytes": usage.total, "free_bytes": usage.free, "measured_path": str(storage_target)}
    except OSError:
        pass

    result: dict[str, Any] = {
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "python_version": platform.python_version(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "system_memory_bytes": _system_memory_bytes(),
        "storage": storage,
        "nvidia_gpus": _nvidia_gpus(),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "accelerator_backend": "unknown",
        "framework_version": None,
        "cuda_available": None,
        "mps_available": None,
        "notes": [],
    }
    if result["nvidia_gpus"] is None:
        result["notes"].append("nvidia-smi unavailable or failed; NVIDIA GPU inventory is unknown")
    if result["torch_installed"]:
        try:
            import torch

            result["framework_version"] = torch.__version__
            result["cuda_available"] = bool(torch.cuda.is_available())
            result["mps_available"] = bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            )
            if result["cuda_available"]:
                result["accelerator_backend"] = "cuda"
            elif result["mps_available"]:
                result["accelerator_backend"] = "mps"
            else:
                result["accelerator_backend"] = "cpu"
        except Exception as exc:
            result["notes"].append(f"PyTorch import/probe failed: {type(exc).__name__}: {exc}")
    else:
        result["notes"].append("PyTorch is not installed; accelerator compatibility cannot be established")
    return result
