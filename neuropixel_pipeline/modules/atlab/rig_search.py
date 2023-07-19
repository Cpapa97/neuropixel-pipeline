from .common import ScanKey

def get_rig(scan_key: ScanKey):
    import datajoint as dj

    experiment = dj.create_virtual_module("experiment", "pipeline_experiment")

    session_rel = experiment.Session() & scan_key.model_dump(by_alias=True)
    experiment_rig = session_rel.fetch1("rig")

    return experiment_rig
