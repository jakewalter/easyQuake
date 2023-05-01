from easyQuake.seisbench_gpd import *
from .annotations import Detection, Pick
from .decorators import log_lifecycle
from .files import (
    callback_if_uncached,
    download_ftp,
    download_http,
    ls_webdav,
    precheck_url,
    #safe_extract_tar,
)
from .torch_helpers import worker_seeding
from .trace_ops import (
    rotate_stream_to_zne,
    stream_to_array,
    trace_has_spikes,
    waveform_id_to_network_station_location,
)

from .gpd import GPD
from .base import SeisBenchModel, WaveformModel, WaveformPipeline
from .eqtransformer import EQTransformer
from .phasenet import PhaseNet
from .aepicker import BasicPhaseAE

import json as _json
import logging as _logging
import os as _os
from pathlib import Path as _Path
from urllib.parse import urljoin as _urljoin

import pkg_resources

__all__ = [
    "cache_root",
    "cache_data_root",
    "cache_model_root",
    "remote_root",
    "remote_data_root",
    "remote_model_root",
    "__version__",
    "config",
]

# global variable: cache_root
cache_root = _Path(
    _os.getenv("EASYQUAKE_CACHE_ROOT", _Path(_Path.home(), ".easyQuake"))
)

cache_data_root = cache_root / "datasets"
cache_model_root = cache_root / "models" / "v3"

remote_root = "https://dcache-demo.desy.de:2443/Helmholtz/HelmholtzAI/SeisBench/"

remote_data_root = _urljoin(remote_root, "datasets/")
remote_model_root = _urljoin(remote_root, "models/v3/")

if not cache_root.is_dir():
    cache_root.mkdir(parents=True, exist_ok=True)

if not cache_data_root.is_dir():
    cache_data_root.mkdir(parents=True, exist_ok=True)

if not cache_model_root.is_dir():
    cache_model_root.mkdir(parents=True, exist_ok=True)

_config_path = cache_root / "config.json"
if not _config_path.is_file():
    config = {"dimension_order": "NCW", "component_order": "ZNE"}
    with open(_config_path, "w") as _fconfig:
        _json.dump(config, _fconfig, indent=4, sort_keys=True)
else:
    with open(_config_path, "r") as _fconfig:
        config = _json.load(_fconfig)

# Version number
__version__ = pkg_resources.get_distribution("easyQuake").version

logger = _logging.getLogger("easyQuake.seisbench_gpd")
_ch = _logging.StreamHandler()
_ch.setFormatter(
    _logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
logger.addHandler(_ch)
