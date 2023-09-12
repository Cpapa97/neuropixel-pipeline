# flake8: noqa

from __future__ import annotations

from pathlib import Path
from pydantic import validate_call
import datajoint as dj
import numpy as np

from . import probe
from .. import utils
from . import SCHEMA_PREFIX
from .config import pipeline_config
from ..api.metadata import InsertionData
from ..readers import labview, kilosort


schema = dj.schema(SCHEMA_PREFIX + "ephys")


### ----------------------------- Table declarations ----------------------

# ------------ Pre-Clustering --------------


@schema
class Session(dj.Manual):
    """Session key"""

    definition = """
    # Session: table connection
    session_id : int # Session primary key
    ---
    animal_id=null: int unsigned # animal id
    session=null: smallint unsigned # original session id
    scan_idx=null: smallint unsigned # scan idx
    rig='': varchar(60) # recording rig
    timestamp=CURRENT_TIMESTAMP: timestamp # timestamp when this session was inserted
    """

    @classmethod
    def add_session(cls, session_meta, error_on_duplicate=True):
        if not cls & session_meta:
            # Synthesize session id, auto_increment cannot be used here if it's used later
            # Additionally this does make it more difficult to accidentally add two of the same session
            session_id = dj.U().aggr(cls, n="ifnull(max(session_id)+1,1)").fetch1("n")
            session_meta["session_id"] = session_id
            cls.insert1(
                session_meta
            )  # should just hash as the primary key and put the rest as a longblob?
        elif error_on_duplicate:
            raise ValueError("Duplicate secondary keys")
        else:
            pass

    @classmethod
    def get_session_id(cls, scan_key: dict, rel=False):
        if rel:
            return (cls & scan_key).proj()
        else:
            return (cls & scan_key).fetch1("session_id")


@schema
class SkullReference(dj.Lookup):
    """Reference area from the skull"""

    definition = """
    skull_reference   : varchar(60)
    """
    contents = zip(["Bregma", "Lambda"])


@schema
class AcquisitionSoftware(dj.Lookup):
    """Name of software used for recording electrophysiological data."""

    definition = """
    # Software used for recording of neuropixels probes
    acq_software: varchar(24)
    """
    contents = zip(["LabviewV1", "SpikeGLX", "OpenEphys"])


@schema
class ProbeInsertion(dj.Manual):
    """Information about probe insertion across subjects and sessions."""

    definition = """
    # Probe insertion implanted into an animal for a given session.
    -> Session
    insertion_number: tinyint unsigned
    ---
    -> probe.Probe
    """


@schema
class InsertionLocation(dj.Manual):
    """Stereotaxic location information for each probe insertion."""

    definition = """
    # Brain Location of a given probe insertion.
    -> ProbeInsertion
    ---
    -> SkullReference
    ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
    ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
    depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
    theta=null:  decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    phi=null:    decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    beta=null:   decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
    """

    @classmethod
    @validate_call
    def fill(cls, scan_key: dict, insertion_number: int, data: InsertionData):
        cls.insert1(
            dict(
                session_id=Session.get_session_id(scan_key),
                insertion_number=insertion_number,
                **data.model_dump(),
            )
        )


@schema
class EphysFile(dj.Manual):
    """Paths for ephys sessions"""

    definition = """
    # Paths for ephys sessions
    -> ProbeInsertion
    ---
    session_path: varchar(255) # file path or directory for an ephys session
    -> AcquisitionSoftware
    """


@schema
class EphysRecording(dj.Imported):
    """Automated table with electrophysiology recording information for each probe inserted during an experimental session."""

    definition = """
    # Ephys recording from a probe insertion for a given session.
    -> ProbeInsertion
    ---
    -> probe.ElectrodeConfig
    sampling_rate: float # (Hz)
    recording_datetime=null: datetime # datetime of the recording from this probe
    recording_duration=null: float # (seconds) duration of the recording from this probe
    """

    def make(self, key):
        """Populates table with electrophysiology recording information."""
        ephys_file_data = (EphysFile & key).fetch1()
        acq_software = ephys_file_data["acq_software"]
        session_path = pipeline_config().specify(ephys_file_data["session_path"])

        inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1(
            "probe"
        )

        if acq_software == "LabviewV1":
            labview_meta = labview.LabviewNeuropixelMeta.from_h5(session_path)
            if not str(labview_meta.serial_number) == inserted_probe_serial_number:
                raise FileNotFoundError(
                    "No Labview data found for probe insertion: {}".format(key)
                )

            probe_type = (probe.Probe & dict(probe=labview_meta.serial_number)).fetch1(
                "probe_type"
            )

            electrode_config_hash_key = probe.ElectrodeConfig.add_new_config(
                labview_meta, probe_type=probe_type
            )

            self.insert1(
                {
                    **key,
                    **electrode_config_hash_key,
                    "acq_software": acq_software,
                    "sampling_rate": labview_meta.sampling_rate,
                    "recording_datetime": None,
                    "recording_duration": None,
                },
                ignore_extra_fields=True,
            )
        else:
            raise NotImplementedError(
                f"Processing ephys files from"
                f" acquisition software of type {acq_software} is"
                f" not yet implemented"
            )


# ------------ Clustering --------------


@schema
class ClusteringMethod(dj.Lookup):
    """Kilosort clustering method."""

    definition = """
    # Method for clustering
    clustering_method: varchar(16)
    ---
    clustering_method_desc: varchar(1000)
    """

    contents = [
        ("kilosort2", "kilosort2 clustering method"),
        ("kilosort2.5", "kilosort2.5 clustering method"),
        ("kilosort3", "kilosort3 clustering method"),
        ("kilosort4", "kilosort4 clustering method"),
    ]


@schema
class ClusteringParamSet(dj.Lookup):
    """Parameters to be used in clustering procedure for spike sorting."""

    definition = """
    # Parameter set to be used in a clustering procedure
    paramset_idx:  smallint
    ---
    -> ClusteringMethod
    paramset_desc: varchar(128)
    paramset_hash: uuid
    unique index (clustering_method, paramset_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def fill(
        cls,
        params: dict,
        clustering_method: str,
        description: str = "",
        skip_duplicates=False,
    ):
        paramset_idx = dj.U().aggr(cls, n="ifnull(max(paramset_idx)+1,1)").fetch1("n")
        params_uuid = utils.dict_to_uuid(params)
        cls.insert1(
            dict(
                paramset_idx=paramset_idx,
                clustering_method=clustering_method,
                paramset_desc=description,
                paramset_hash=params_uuid,
                params=params,
            ),
            skip_duplicates=skip_duplicates,
        )


@schema
class ClusterQualityLabel(dj.Lookup):
    """Quality label for each spike sorted cluster."""

    definition = """
    # Quality
    cluster_quality_label:  varchar(100)  # cluster quality type - e.g. 'good', 'MUA', 'noise', etc.
    ---
    cluster_quality_description:  varchar(4000)
    """
    contents = [
        ("good", "single unit"),
        ("ok", "probably a single unit, but could be contaminated"),
        ("mua", "multi-unit activity"),
        ("noise", "bad unit"),
    ]


@schema
class ClusteringTask(dj.Manual):
    """A clustering task to spike sort electrophysiology datasets."""

    definition = """
    # Manual table for defining a clustering task ready to be run
    -> EphysRecording
    -> ClusteringParamSet
    ---
    clustering_output_dir='': varchar(255)  #  clustering output directory relative to the clustering root data directory
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """

    @classmethod
    def build_key_from_scan(
        cls, scan_key: dict, insertion_number: int, clustering_method: str, fetch=False
    ) -> dict:
        task_key = dict(
            session_id=(Session & scan_key).fetch1("session_id"),
            insertion_number=insertion_number,
            paramset_idx=(
                ClusteringParamSet & {"clustering_method": clustering_method}
            ).fetch1("paramset_idx"),
        )
        if fetch:
            return (cls & task_key).fetch()
        else:
            return task_key


@schema
class Clustering(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    # Clustering Procedure
    -> ClusteringTask
    ---
    clustering_time: datetime  # time of generation of this set of clustering results
    """

    def make(self, key):
        source_key = (ClusteringTask & key).fetch1()

        creation_time, _, _ = kilosort.Kilosort.extract_clustering_info(
            pipeline_config().specify(source_key["clustering_output_dir"])
        )
        self.insert1(
            dict(
                **source_key,
                clustering_time=creation_time,
            ),
            ignore_extra_fields=True,
        )


# Probably more layers above this are useful (for multiple users per curation, auto-curation maybe, etc.)
@schema
class CurationType(dj.Lookup):  # Table definition subject to change
    definition = """
    # Type of curation performed on the clustering
    curation: varchar(16)
    ---
    """

    contents = zip(["no curation", "phy"])


@schema
class Curation(dj.Manual):
    """Curation procedure table."""

    definition = """
    # Manual curation procedure
    -> Clustering
    curation_id: int
    ---
    curation_time: datetime             # time of generation of this set of curated clustering results
    curation_output_dir: varchar(255)   # output directory of the curated results, relative to root data directory
    -> CurationType                     # what type of curation has been performed on this clustering result?
    curation_note='': varchar(2000)
    """

    @classmethod
    def create1_from_clustering_task(
        cls, key, curation_output_dir=None, curation_note="", skip_duplicates=True
    ):
        """
        A function to create a new corresponding "Curation" for a particular
        "ClusteringTask"
        """
        if key not in Clustering():
            raise ValueError(
                f"No corresponding entry in Clustering available"
                f" for: {key}; do `Clustering.populate(key)`"
            )

        if curation_output_dir is None:
            curation_output_dir = (ClusteringTask & key).fetch1("clustering_output_dir")

        creation_time, _, _ = kilosort.Kilosort.extract_clustering_info(
            pipeline_config().specify(curation_output_dir)
        )

        with cls.connection.transaction:
            # Synthesize curation_id, no auto_increment for the same reason as Session
            curation_id = (
                dj.U().aggr(cls & key, n="ifnull(max(curation_id)+1,1)").fetch1("n")
            )

            cls.insert1(
                {
                    **key,
                    "curation_id": curation_id,
                    "curation_time": creation_time,
                    "curation_output_dir": curation_output_dir,
                    "curation_note": curation_note,
                },
                skip_duplicates=skip_duplicates,
            )
        return curation_id


# TODO: Remove longblob types, replace with external-blob (or some form of this)
@schema
class CuratedClustering(dj.Imported):
    """Clustering results after curation."""

    definition = """
    # Clustering results of a curation.
    -> Curation
    """

    class Unit(dj.Part):
        """Single unit properties after clustering and curation."""

        definition = """
        # Properties of a given unit from a round of clustering (and curation)
        -> master
        unit_id     : int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode with highest waveform amplitude for this unit
        -> ClusterQualityLabel
        depth       : float       # depth of the electrode
        spike_count : int         # how many spikes in this recording for this unit
        spike_times : longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        """

    def make(self, key):
        """Automated population of Unit information."""

        kilosort_dataset = kilosort.Kilosort(
            pipeline_config().specify((Curation & key).fetch1("curation_output_dir"))
        )
        sample_rate = (EphysRecording & key).fetch1("sampling_rate")

        sample_rate = kilosort_dataset.data["params"].get("sample_rate", sample_rate)

        # ---------- Unit ----------
        # -- Remove 0-spike units
        withspike_idx = [
            i
            for i, u in enumerate(kilosort_dataset.data["cluster_ids"])
            if (kilosort_dataset.data["spike_clusters"] == u).any()
        ]
        valid_units = kilosort_dataset.data["cluster_ids"][withspike_idx]
        valid_unit_labels = kilosort_dataset.data["cluster_groups"][withspike_idx]

        # -- Spike-times --
        # spike_times_sec_adj > spike_times_sec > spike_times
        spike_time_key = (
            "spike_times_sec_adj"
            if "spike_times_sec_adj" in kilosort_dataset.data
            else "spike_times_sec"
            if "spike_times_sec" in kilosort_dataset.data
            else "spike_times"
        )
        spike_times = kilosort_dataset.data[spike_time_key]

        electrode_config_hash = (EphysRecording * probe.ElectrodeConfig & key).fetch1(
            "electrode_config_hash"
        )

        serial_number = dj.U("probe") & (ProbeInsertion & key)
        probe_type = (probe.Probe & serial_number).fetch1("probe_type")

        # -- Insert unit, label, peak-chn
        units = []
        for unit, unit_lbl in zip(valid_units, valid_unit_labels):
            if (kilosort_dataset.data["spike_clusters"] == unit).any():
                unit_channel, depth, _ = kilosort_dataset.get_best_channel(unit)
                unit_spike_times = (
                    spike_times[kilosort_dataset.data["spike_clusters"] == unit]
                    / sample_rate
                )
                spike_count = len(unit_spike_times)

                units.append(
                    {
                        "unit_id": unit,
                        "cluster_quality_label": unit_lbl,
                        "electrode_config_hash": electrode_config_hash,
                        "probe_type": probe_type,
                        "electrode": unit_channel,
                        "depth": depth,
                        "spike_times": unit_spike_times,
                        "spike_count": spike_count,
                    }
                )

        self.insert1(key)
        insert_units = [{**key, **u} for u in units]
        self.Unit.insert(insert_units)


# important to note the original source of these quality metrics:
#   https://allensdk.readthedocs.io/en/latest/
#   https://github.com/AllenInstitute/ecephys_spike_sorting
#
@schema
class WaveformSet(dj.Imported):
    """A set of spike waveforms for units out of a given CuratedClustering."""

    definition = """
    # A set of spike waveforms for units out of a given CuratedClustering
    -> CuratedClustering
    """

    class PeakWaveform(dj.Part):
        """Mean waveform across spikes for a given unit."""

        definition = """
        # Mean waveform across spikes for a given unit at its representative electrode
        -> master
        -> CuratedClustering.Unit
        ---
        peak_electrode_waveform: longblob  # (uV) mean waveform for a given unit at its representative electrode
        """

    class Waveform(dj.Part):
        """Spike waveforms for a given unit."""

        definition = """
        # Spike waveforms and their mean across spikes for the given unit
        -> master
        -> CuratedClustering.Unit
        -> probe.ElectrodeConfig.Electrode
        ---
        waveform_mean: longblob   # (uV) mean waveform across spikes of the given unit
        """

    def make(self, key):
        """Populates waveform tables."""
        from neuropixel_pipeline.api.postclustering import (
            WaveformMetricsRunner,
            MEAN_WAVEFORMS_FILE,
        )

        curation_output_dir = pipeline_config().specify(
            (Curation & key).fetch1("curation_output_dir")
        )

        kilosort_dataset = kilosort.Kilosort(curation_output_dir)

        # -- Get channel and electrode-site mapping
        recording_key = (EphysRecording & key).fetch1("KEY")

        def channel2electrodes(ephys_recording_key):
            electrode_query = (
                probe.ProbeType.Electrode
                * probe.ElectrodeConfig.Electrode
                * EphysRecording
                & ephys_recording_key
            )

            probe_electrodes = {
                key["electrode"]: key for key in electrode_query.fetch("KEY")
            }

            return probe_electrodes

        probe_electrodes = channel2electrodes(recording_key)

        # Get all units
        units = {
            u["unit_id"]: u
            for u in (CuratedClustering.Unit & key).fetch(
                as_dict=True, order_by="unit_id"
            )
        }

        mean_waveform_fp = curation_output_dir / MEAN_WAVEFORMS_FILE
        if not mean_waveform_fp.exists():
            print("Constructing mean waveforms files")
            session_path = (EphysFile & key).fetch1("session_path")
            labview_meta = labview.LabviewNeuropixelMeta.from_h5(
                pipeline_config().specify()
            )
            bin_file = labview_meta.find_bin_from_prefix(session_path)
            results = WaveformMetricsRunner(
                generic_params=WaveformMetricsRunner.GenericParams(
                    bit_volts=labview_meta.scale[1]
                )
            ).calculate(mean_waveform_fp, bin_file=bin_file, has_sync_channel=True)
            print(f"WaveformMetricsRunner results: {results}")

        unit_waveforms = np.load(mean_waveform_fp)  # unit x channel x sample

        def yield_unit_waveforms():
            for unit_no, unit_waveform in zip(
                kilosort_dataset.data["cluster_ids"], unit_waveforms
            ):
                unit_peak_waveform = {}
                unit_electrode_waveforms = []
                if unit_no in units:
                    for channel, channel_waveform in zip(
                        kilosort_dataset.data["channel_map"], unit_waveform
                    ):
                        unit_electrode_waveforms.append(
                            {
                                **units[unit_no],
                                **probe_electrodes[channel],
                                "waveform_mean": channel_waveform,
                            }
                        )
                        if (
                            probe_electrodes[channel]["electrode"]
                            == units[unit_no]["electrode"]
                        ):
                            unit_peak_waveform = {
                                **units[unit_no],
                                "peak_electrode_waveform": channel_waveform,
                            }
                yield unit_peak_waveform, unit_electrode_waveforms

        # insert waveform on a per-unit basis to mitigate potential memory issue (might not be an issue when not sampling waveforms)
        self.insert1(key)
        for unit_peak_waveform, unit_electrode_waveforms in yield_unit_waveforms():
            if unit_peak_waveform:
                self.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
            if unit_electrode_waveforms:
                self.Waveform.insert(unit_electrode_waveforms, ignore_extra_fields=True)


# note the original source of these quality metric calculations:
#   https://allensdk.readthedocs.io/en/latest/
#   https://github.com/AllenInstitute/ecephys_spike_sorting
#
@schema
class QualityMetrics(dj.Imported):
    """Clustering and waveform quality metrics."""

    definition = """
    # Clusters and waveforms metrics
    -> CuratedClustering
    """

    class Cluster(dj.Part):
        """Cluster metrics for a unit."""

        definition = """
        # Cluster metrics for a particular unit
        -> master
        -> CuratedClustering.Unit
        ---
        firing_rate=null: float # (Hz) firing rate for a unit
        presence_ratio=null: float  # fraction of time in which spikes are present
        isi_violation=null: float   # rate of ISI violation as a fraction of overall rate
        number_violation=null: int  # total number of ISI violations
        amplitude_cutoff=null: float  # estimate of miss rate based on amplitude histogram
        """

    def make(self, key):
        """Populates tables with quality metrics data."""
        import pandas as pd
        from neuropixel_pipeline.api.postclustering import (
            QualityMetricsRunner,
            QUALITY_METRICS_FILE,
        )

        curation_output_dir = pipeline_config().specify(
            (Curation & key).fetch1("curation_output_dir")
        )

        metric_fp = curation_output_dir / QUALITY_METRICS_FILE
        rename_dict = {
            "isi_viol": "isi_violation",
            "num_viol": "number_violation",  # TODO: not calculated directly by AllenInstitute ecephys_spike_sorting
            "contam_rate": "contamination_rate",
        }

        if not metric_fp.exists():
            print("Constructing Quality Control metrics.csv file")
            results = QualityMetricsRunner().calculate(curation_output_dir)
            print(f"QualityMetricsRunner results: {results}")

        metrics_df = pd.read_csv(metric_fp)
        metrics_df.set_index("cluster_id", inplace=True)
        metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        metrics_df.columns = metrics_df.columns.str.lower()
        metrics_df.rename(columns=rename_dict, inplace=True)
        metrics_list = [
            dict(metrics_df.loc[unit_key["unit_id"]], **unit_key)
            for unit_key in (CuratedClustering.Unit & key).fetch("KEY")
        ]

        self.insert1(key)
        self.Cluster.insert(metrics_list, ignore_extra_fields=True)
