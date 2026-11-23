# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
"""WorldFoundry runtime environment resolution and redaction utilities.

Provides path resolution helpers for cache, data, model, checkpoint, and
artifact directories, plus environment-variable redaction and preflight
capture for manifests and diagnostics.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping as TypingMapping

from scenemembench._native.core.io.paths import (
    artifact_root_path,
    cache_root_path,
    checkpoint_root_path,
    hfd_root_path,
    local_data_root_path,
    local_model_root_path,
)


EnvMapping = TypingMapping[str, str]

# ── Public environment variable groups ───────────────────────────────────────

CORE_ENV_KEYS = (
    "SCENEMEMBENCH_HOME",
    "SCENEMEMBENCH_CACHE_DIR",
    "SCENEMEMBENCH_CKPT_DIR",
    "SCENEMEMBENCH_DATA_DIR",
    "SCENEMEMBENCH_MODEL_DIR",
    "SCENEMEMBENCH_MODEL_SOURCE_DIR",
    "SCENEMEMBENCH_ARTIFACT_DIR",
    "SCENEMEMBENCH_HFD_ROOT",
)
HF_ENV_KEYS = (
    "HF_HOME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
)
RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CONDA_DEFAULT_ENV",
    "VIRTUAL_ENV",
)
_SENSITIVE_MARKERS = ("TOKEN", "API_KEY", "SECRET", "PASSWORD", "CREDENTIAL")

# ── Sourceable environment templates ─────────────────────────────────────────

SOURCEABLE_ENV_BASE_LINES = (
    'export SCENEMEMBENCH_HOME="${SCENEMEMBENCH_HOME:-${HOME}/.cache/worldfoundry}"',
    'export SCENEMEMBENCH_CACHE_DIR="${SCENEMEMBENCH_CACHE_DIR:-${SCENEMEMBENCH_HOME}/cache}"',
    'export SCENEMEMBENCH_DATA_DIR="${SCENEMEMBENCH_DATA_DIR:-${SCENEMEMBENCH_HOME}/data}"',
    'export SCENEMEMBENCH_ARTIFACT_DIR="${SCENEMEMBENCH_ARTIFACT_DIR:-${SCENEMEMBENCH_HOME}/artifacts}"',
    'export SCENEMEMBENCH_MODEL_DIR="${SCENEMEMBENCH_MODEL_DIR:-${SCENEMEMBENCH_HOME}/models}"',
    'export SCENEMEMBENCH_CKPT_DIR="${SCENEMEMBENCH_CKPT_DIR:-${SCENEMEMBENCH_HOME}/checkpoints}"',
    'export SCENEMEMBENCH_HFD_ROOT="${SCENEMEMBENCH_HFD_ROOT:-${SCENEMEMBENCH_CKPT_DIR}/hfd}"',
    'export SCENEMEMBENCH_MODEL_SOURCE_DIR="${SCENEMEMBENCH_MODEL_SOURCE_DIR:-${SCENEMEMBENCH_CACHE_DIR}/official_runtime_repos}"',
    'export SCENEMEMBENCH_LOCAL_ASSET_MANIFEST="${SCENEMEMBENCH_LOCAL_ASSET_MANIFEST:-${SCENEMEMBENCH_CACHE_DIR}/manifests/local_assets_manifest.yaml}"',
)
_SOURCEABLE_SENSITIVE_MARKERS = (*_SENSITIVE_MARKERS, "ACCESS_KEY")
_MANUAL_ENV_NAMES = {"SCENEMEMBENCH_WORLDMODELBENCH_JUDGE"}


@dataclass(frozen=True)
class RequiredEnvReport:
    """Describe missing and present variables for a required environment check.

    Args:
        missing: Required environment variable names that were not set.
        present: Required environment variable names that were set.
    """

    missing: tuple[str, ...]
    present: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether all required variables were present.

        Args:
            None.
        """

        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report for manifests and JSON preflight output.

        Args:
            None.
        """

        return {
            "ok": self.ok,
            "missing": list(self.missing),
            "present": list(self.present),
        }


@dataclass(frozen=True)
class WorldFoundryEnv:
    """Expose WorldFoundry runtime paths and public environment metadata.

    Args:
        values: Environment mapping used for path and metadata resolution.
    """

    values: EnvMapping

    @classmethod
    def from_os(cls) -> "WorldFoundryEnv":
        """Build an environment view from the current process environment.

        Args:
            None.
        """

        return cls(os.environ)

    def resolve_cache_dir(self) -> Path:
        """Resolve the root cache directory for downloads and transient assets.

        Args:
            None.
        """

        return resolve_cache_dir(self.values)

    def resolve_data_dir(self) -> Path:
        """Resolve the shared local dataset directory.

        Args:
            None.
        """

        return resolve_data_dir(self.values)

    def resolve_model_dir(self) -> Path:
        """Resolve the local checkpoint and model asset directory.

        Args:
            None.
        """

        return resolve_model_dir(self.values)

    def resolve_ckpt_dir(self) -> Path:
        """Resolve the local checkpoint directory.

        Args:
            None.
        """

        return resolve_ckpt_dir(self.values)

    def resolve_artifact_dir(self) -> Path:
        """Resolve the generated artifact output directory.

        Args:
            None.
        """

        return resolve_artifact_dir(self.values)

    def resolve_hf_cache_dir(self) -> Path:
        """Resolve the Hugging Face dataset/model cache directory.

        Args:
            None.
        """

        return resolve_hf_cache_dir(self.values)

    def redact_for_manifest(self, keys: Sequence[str] | None = None) -> dict[str, Any]:
        """Return manifest-safe environment values with secrets reduced to presence.

        Args:
            keys: Optional variable names to include instead of the standard public set.
        """

        return redact_env_for_manifest(self.values, keys)

    def check_required(self, names: Sequence[str]) -> RequiredEnvReport:
        """Check that required variables are set without exposing their values.

        Args:
            names: Environment variable names required by a runtime.
        """

        return check_required_env(names, self.values)

    def capture_runtime(
        self, *, include_torch: bool = True, include_nvidia_smi: bool = True
    ) -> dict[str, Any]:
        """Capture local Python, CUDA, GPU, and Torch metadata for preflight logs.

        Args:
            include_torch: Whether to import Torch and record CUDA availability.
            include_nvidia_smi: Whether to call ``nvidia-smi`` when available.
        """

        return capture_runtime_environment(
            self.values,
            include_torch=include_torch,
            include_nvidia_smi=include_nvidia_smi,
        )


def _path(value: str | Path) -> Path:
    """Expand a user-provided filesystem path.

    Args:
        value: Path-like value to normalize.
    """

    return Path(value).expanduser()


def _is_sensitive_key(name: str) -> bool:
    """Return whether an environment key should never expose its value.

    Args:
        name: Environment variable name.
    """

    upper = name.upper()
    return any(marker in upper for marker in _SENSITIVE_MARKERS)


def first_env_value(names: Sequence[str], env: EnvMapping | None = None) -> str | None:
    """Return the first non-empty environment value from a priority list.

    Args:
        names: Environment variable names in highest-priority-first order.
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    environ = os.environ if env is None else env
    for name in names:
        value = environ.get(name)
        if value:
            return value
    return None


def first_env_path(
    names: Sequence[str],
    env: EnvMapping | None = None,
    default: str | Path | None = None,
) -> Path | None:
    """Return the first non-empty environment path from a priority list.

    Args:
        names: Environment variable names in highest-priority-first order.
        env: Optional environment mapping; defaults to ``os.environ``.
        default: Optional fallback path when no variable is set.
    """

    value = first_env_value(names, env)
    if value:
        return _path(value)
    return _path(default) if default is not None else None


def resolve_cache_dir(env: EnvMapping | None = None) -> Path:
    """Resolve the root cache directory from WorldFoundry variables.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    return cache_root_path(os.environ if env is None else env)


def benchmark_repo_cache_root(env: EnvMapping | None = None) -> Path:
    """Return the canonical cache directory for cloned benchmark upstream repos."""

    return resolve_cache_dir(env) / "repos"


def resolve_data_dir(env: EnvMapping | None = None) -> Path:
    """Resolve the shared local data directory from WorldFoundry variables.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    return local_data_root_path(os.environ if env is None else env)


def resolve_model_dir(env: EnvMapping | None = None) -> Path:
    """Resolve the local model directory from WorldFoundry variables.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    return local_model_root_path(os.environ if env is None else env)


def resolve_ckpt_dir(env: EnvMapping | None = None) -> Path:
    """Resolve the local checkpoint directory from WorldFoundry variables.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    return checkpoint_root_path(env=os.environ if env is None else env)


def resolve_artifact_dir(env: EnvMapping | None = None) -> Path:
    """Resolve the generated artifact directory from WorldFoundry variables.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    return artifact_root_path(os.environ if env is None else env)


def resolve_hf_cache_dir(env: EnvMapping | None = None) -> Path:
    """Resolve the Hugging Face cache root using WorldFoundry and HF conventions.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    environ = os.environ if env is None else env
    explicit = first_env_path(
        (
            "SCENEMEMBENCH_HF_CACHE_DIR",
            "HF_DATASETS_CACHE",
            "TRANSFORMERS_CACHE",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
        ),
        environ,
    )
    if explicit is not None:
        return explicit
    hf_home = first_env_path(("HF_HOME",), environ)
    if hf_home is not None:
        return hf_home / "hub"
    return resolve_cache_dir(environ) / "huggingface" / "hub"


def resolve_hfd_root(env: EnvMapping | None = None) -> Path:
    """Resolve the local HFD-style model checkpoint root.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    return hfd_root_path(env=os.environ if env is None else env)


def redact_env_for_manifest(
    env: EnvMapping | None = None, keys: Sequence[str] | None = None
) -> dict[str, Any]:
    """Return manifest-safe environment values without leaking secrets.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
        keys: Optional variable names to include instead of the standard public set.
    """

    environ = os.environ if env is None else env
    selected_keys = keys or (*CORE_ENV_KEYS, *HF_ENV_KEYS, *RUNTIME_ENV_KEYS)
    redacted: dict[str, Any] = {}
    for key in selected_keys:
        value = environ.get(key)
        if _is_sensitive_key(key):
            redacted[key] = {"present": bool(value)}
        elif value:
            redacted[key] = value
    return redacted


def check_required_env(
    names: Sequence[str], env: EnvMapping | None = None
) -> RequiredEnvReport:
    """Check required environment variables without exposing secret values.

    Args:
        names: Environment variable names required by a runtime.
        env: Optional environment mapping; defaults to ``os.environ``.
    """

    environ = os.environ if env is None else env
    present = tuple(name for name in names if environ.get(name))
    missing = tuple(name for name in names if not environ.get(name))
    return RequiredEnvReport(missing=missing, present=present)


def primary_benchmark_slug(benchmark_ids: Any) -> str:
    """Return a filesystem-friendly primary benchmark slug.

    Args:
        benchmark_ids: Benchmark id or sequence of benchmark ids requiring an env value.
    """

    if isinstance(benchmark_ids, (list, tuple)) and benchmark_ids:
        return str(benchmark_ids[0]).strip().replace("_", "-").lower() or "benchmark"
    text = str(benchmark_ids).strip()
    return text.replace("_", "-").lower() if text else "benchmark"


def suggested_sourceable_env_value(name: str, benchmark_ids: Any) -> str:
    """Return a shell-template default for non-secret benchmark environment variables.

    Args:
        name: Environment variable name.
        benchmark_ids: Benchmarks that require this variable.
    """

    if (
        any(marker in name for marker in _SOURCEABLE_SENSITIVE_MARKERS)
        or name in _MANUAL_ENV_NAMES
    ):
        return ""

    slug = primary_benchmark_slug(benchmark_ids)
    if name == "DATA_PATH":
        return "${SCENEMEMBENCH_DATA_DIR}"
    if name == "COPPELIASIM_ROOT":
        return "${SCENEMEMBENCH_MODEL_SOURCE_DIR}/coppeliasim"
    if name == "SCENEMEMBENCH_HF_CACHE_DIR":
        return "${SCENEMEMBENCH_CACHE_DIR}/huggingface/hub"
    if name == "SCENEMEMBENCH_CHRONOMAGIC_CHSCORE_CKPT":
        return "${SCENEMEMBENCH_CKPT_DIR}/hfd/BestWishYsh--ChronoMagic-Bench/CHScore/cotracker2.pth"
    if name == "SCENEMEMBENCH_CHRONOMAGIC_MTSCORE_CKPT":
        return "${SCENEMEMBENCH_CKPT_DIR}/hfd/BestWishYsh--ChronoMagic-Bench/MTScore/InternVideo2-stage2_1b-224p-f4.pt"
    if name == "SCENEMEMBENCH_GENERATED_ARTIFACT_DIR":
        return "${SCENEMEMBENCH_ARTIFACT_DIR}/generated"
    if name == "SCENEMEMBENCH_MODEL_SOURCE_DIR":
        return "${SCENEMEMBENCH_CACHE_DIR}/official_runtime_repos"
    if name == "SCENEMEMBENCH_IWORLD_BENCH_ROOT":
        return "${SCENEMEMBENCH_MODEL_SOURCE_DIR}/EmbodiedCity--iWorld-Bench"
    if name == "SCENEMEMBENCH_IWORLD_BENCH_DATASET_ROOT":
        return "${SCENEMEMBENCH_HFD_DATASET_ROOT}/EmbodiedCity__iWorld-Bench-Dataset"
    if name == "SCENEMEMBENCH_IWORLD_BENCH_GENERATED_VIDEO_DIR":
        return "${SCENEMEMBENCH_ARTIFACT_DIR}/generated/iworld-bench/videos"
    if name == "SCENEMEMBENCH_IPV_BENCH_ROOT":
        return "${SCENEMEMBENCH_CACHE_DIR}/repos/github.com_showlab_Impossible-Videos"
    if name == "SCENEMEMBENCH_IPV_BENCH_PROMPT_MANIFEST":
        return "${SCENEMEMBENCH_IPV_BENCH_ROOT}/ipv_txt_prompt_suite.json"
    if name == "SCENEMEMBENCH_IWORLD_BENCH_PROMPT_MANIFEST":
        return (
            "${SCENEMEMBENCH_IWORLD_BENCH_DATASET_ROOT}/dataset/all_pack/metadata.csv"
        )
    if name.endswith("_RESULTS_PATH"):
        return f"${{SCENEMEMBENCH_ARTIFACT_DIR}}/results/{slug}.jsonl"
    if name.endswith("_RESULTS"):
        return f"${{SCENEMEMBENCH_ARTIFACT_DIR}}/results/{slug}.jsonl"
    if name.endswith("_ARTIFACTS"):
        return f"${{SCENEMEMBENCH_ARTIFACT_DIR}}/generated/{slug}"
    if name.endswith("_MODEL_OUTPUT_DIR"):
        return f"${{SCENEMEMBENCH_ARTIFACT_DIR}}/model_outputs/{slug}"
    if name.endswith("_GENERATED_VIDEO_DIR"):
        return f"${{SCENEMEMBENCH_ARTIFACT_DIR}}/generated/{slug}/videos"
    if name.endswith("_INPUT_FOLDER"):
        return f"${{SCENEMEMBENCH_DATA_DIR}}/inputs/{slug}"
    if name.endswith("_GT_PATH"):
        return f"${{SCENEMEMBENCH_DATA_DIR}}/annotations/{slug}/ground_truth.json"
    if name.endswith("_CSV_DIR"):
        return f"${{SCENEMEMBENCH_DATA_DIR}}/annotations/{slug}"
    if name.endswith("_FRAMES_DIR"):
        return f"${{SCENEMEMBENCH_DATA_DIR}}/frames/{slug}"
    if name.endswith("_OFFICIAL_VIDEOS_ROOT"):
        return f"${{SCENEMEMBENCH_DATA_DIR}}/videos/{slug}/official"
    if name.endswith("_VIDEO_ROOT"):
        return f"${{SCENEMEMBENCH_DATA_DIR}}/videos/{slug}"
    if name.endswith("_ANNOTATION_ROOT"):
        return f"${{SCENEMEMBENCH_DATA_DIR}}/annotations/{slug}"
    if name.endswith("_DATASET_ROOT"):
        return f"${{SCENEMEMBENCH_DATA_DIR}}/datasets/{slug}"
    if name.endswith("_ROOT"):
        return f"${{SCENEMEMBENCH_MODEL_SOURCE_DIR}}/{slug}"
    return ""


def _run_command(argv: Sequence[str]) -> str | None:
    """Run a short diagnostic command when the binary exists.

    Args:
        argv: Command and arguments to execute.
    """

    if shutil.which(argv[0]) is None:
        return None
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    output = completed.stdout.strip() or completed.stderr.strip()
    return output or None


def _capture_torch() -> dict[str, Any]:
    """Capture Torch version and CUDA availability when Torch is importable.

    Args:
        None.
    """

    if "torch" in sys.modules:
        torch = sys.modules["torch"]
    else:
        spec = importlib.util.find_spec("torch")
        if spec is None:
            return {"installed": False}
        torch = __import__("torch")
    cuda = torch.cuda
    return {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(cuda.is_available()),
        "cuda_device_count": int(cuda.device_count()) if cuda.is_available() else 0,
    }


def capture_runtime_environment(
    env: EnvMapping | None = None,
    *,
    include_torch: bool = True,
    include_nvidia_smi: bool = True,
) -> dict[str, Any]:
    """Capture local runtime facts for preflight output without secret values.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.
        include_torch: Whether to import Torch and record CUDA availability.
        include_nvidia_smi: Whether to call ``nvidia-smi`` when available.
    """

    environ = os.environ if env is None else env
    return {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "environment": redact_env_for_manifest(environ),
        "cuda": {
            "visible_devices": environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": (
                _run_command(
                    (
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    )
                )
                if include_nvidia_smi
                else None
            ),
        },
        "torch": _capture_torch() if include_torch else {"checked": False},
    }
