from pathlib import Path

from .common import ScanKey
from ...schemata.config import pipeline_config


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


def get_session_path(scan_key: ScanKey, include_generic=False) -> Path:
    generic_path = get_generic_session_path(scan_key).parent
    session_path = pipeline_config().specify(generic_path)
    if include_generic:
        return session_path, generic_path
    else:
        return session_path
