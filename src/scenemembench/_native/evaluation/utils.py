# Modified for SceneMemBench: standalone imports, configuration and runtime extraction.
# Original provenance and licenses: docs/source_manifest.json and THIRD_PARTY_NOTICES.md.
from pathlib import Path
import yaml


def worldfoundry_data_path(*parts):
    return Path(__file__).resolve().parents[1].joinpath("data", *parts)


def load_manifest(path):
    return yaml.safe_load(Path(path).read_text())
