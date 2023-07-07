from pathlib import Path

from .common import ScanKey
from ...schemata.config import PathKind


def get_generic_session_path(scan_key: ScanKey):
    scan_key = ScanKey.model_validate(scan_key).model_dump()

    import datajoint as dj

    experiment = dj.create_virtual_module("experiment", "pipeline_experiment")
    acq = dj.create_virtual_module("acq", "acq")

    ephys_start_time_rel = dj.U("ephys_start_time") & (
        experiment.ScanEphysLink & scan_key
    )
    acq_ephys_rel = acq.Ephys - acq.EphysIgnore
    ephys_path = (acq_ephys_rel & ephys_start_time_rel).fetch1("ephys_path")
    return Path(ephys_path)


def get_session_path(scan_key: ScanKey) -> Path:
    return PathKind.SESSION.normalize(get_generic_session_path(scan_key))
