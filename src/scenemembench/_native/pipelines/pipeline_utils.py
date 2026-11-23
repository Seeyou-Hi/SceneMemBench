# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
from typing import Any, ClassVar


class PipelineABC:
    """Shared, non-strict base for WorldFoundry pipelines.

    The class intentionally avoids abstract methods because many existing
    pipelines predate this contract. Subclasses can override any method while
    still sharing a stable framework surface.
    """

    MODEL_ID: ClassVar[str | None] = None
    OPERATOR_CLS: ClassVar[type[Any] | None] = None
    MEMORY_CLS: ClassVar[type[Any] | None] = None
    SYNTHESIS_CLS: ClassVar[type[Any] | None] = None
    MODEL_PATH_OPTION: ClassVar[str] = "repo_root"
    MEMORY_RECORD_TYPE: ClassVar[str] = "runtime_result"
    FRAMEWORK_LOADING_OPTION_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "adapter",
            "adapter_target",
            "acquisition_root",
            "hf_models_root",
            "manifest_path",
            "model_adapter",
            "model_id",
            "pipeline_target",
            "profile_id",
            "profile_path",
        }
    )
    RESERVED_SYNTHESIS_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"prompt", "images", "video", "actions", "ref_image_path", "extra_inputs"}
    )

    def __init__(
        self,
        model_id: str | None = None,
        operators: Any = None,
        operator: Any = None,
        synthesis_model: Any = None,
        memory_module: Any = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.model_id = model_id or self.MODEL_ID
        self.synthesis_model = synthesis_model
        self.operator = operators if operators is not None else operator
        self.operators = self.operator
        self.memory_module = (
            memory_module
            if memory_module is not None
            else self._create_memory_module_if_available()
        )
        self.device = device
        self.options = dict(kwargs)

    @classmethod
    def _create_memory_module_if_available(cls) -> Any:
        """Instantiate and return the memory module if specified."""
        if getattr(cls, "MEMORY_CLS", None) is None:
            return None
        return cls._component("MEMORY_CLS")(model_id=cls.MODEL_ID)

    @classmethod
    def _runtime_options(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct the combined runtime options dictionary."""
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options[cls.MODEL_PATH_OPTION] = str(model_path)
        options.update(required_components or {})
        options.update(kwargs or {})
        return options

    @classmethod
    def _resolve_model_id(
        cls, options: dict[str, Any], model_id: str | None = None
    ) -> str:
        """Resolve the target model or profile identifier."""
        resolved_model_id = str(
            options.get("model_id")
            or options.get("profile_id")
            or model_id
            or cls.MODEL_ID
            or ""
        )
        if not resolved_model_id:
            raise ValueError(
                f"{cls.__name__}.from_pretrained requires model_id/profile_id."
            )
        return resolved_model_id

    @classmethod
    def _strip_framework_loading_options(
        cls, options: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove WorldFoundry loader metadata before calling model internals."""
        for key in cls.FRAMEWORK_LOADING_OPTION_KEYS:
            options.pop(key, None)
        return options
