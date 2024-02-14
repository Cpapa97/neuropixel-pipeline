from __future__ import annotations
import abc
import time
import logging

from pydantic import BaseModel, Field
from enum import Enum
from typing import Union, Literal, Optional
from pathlib import Path

from .common import ScanKey
from . import (
    ACQ_SOFTWARE,
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_OUTPUT_RELATIVE,
)
from .probe_setup import probe_setup
from .session_search import get_session_path
from .rig_search import get_rig
from .kilosort_params import default_kilosort_parameters
from ...api import metadata, clustering, clustering_task
from ...api.clustering_task import ClusteringTaskMode, ClusteringTaskRunner
from ...readers.recording.labview import LabviewNeuropixelMeta
from ...utils import check_for_first_bin_with_prefix
from ...schemata import probe, ephys
from ...schemata.config import PipelineConfigStore, pipeline_config


class Runnable(abc.ABC):
    @abc.abstractmethod
    def run(self):
        pass


class PipelineMode(str, Enum):
    SETUP = "setup"
    MINION = "minion"
    NO_CURATION = "no curation"
    CURATED = "curated"
    INSERTION_META = "insertion meta"


class Setup(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.SETUP] = PipelineMode.SETUP

    def run(self):
        """Setup for neuropixel_probe"""
        ### Setup
        logging.info("starting setup section")
        if not PipelineConfigStore.Default():
            PipelineConfigStore.set_default("atlab")
        probe.ProbeType.fill_neuropixel_probes()
        probe_setup()

        # Clustering param set
        ephys.ClusteringParamSet.fill(
            params=default_kilosort_parameters(),
            clustering_method="kilosort4",
            description="default kilosort4 params",
            skip_duplicates=True,
        )
        ephys.ClusteringParamSet.fill(
            {},
            clustering_method="kilosort3",
            description="kilosort3 params (for ingesting)",
            skip_duplicates=True,
        )
        logging.info("done with setup section")


class Minion(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.MINION] = PipelineMode.MINION
    # base_dir: Optional[Path] = None

    def run(self, **populate_kwargs):
        def check_and_populate(self, key):
            """Drop-in replacement for minion.MinionOutput.make"""
            params = (minion.MinionInput & key).fetch1("params")
            pipeline = PipelineInput.model_validate(params)
            if pipeline.pipeline_mode is PipelineMode.MINION:
                raise ValueError(
                    "PipelineMode.MINION should NOT be an input to the MinionInput table, "
                    "it is only used by the pipeline minion internally"
                )
            pipeline.run()

        minion.MinionOutput.make = check_and_populate
        minion.MinionOutput.populate(**populate_kwargs)


class NoCuration(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.NO_CURATION] = PipelineMode.NO_CURATION
    scan_key: ScanKey
    insertion_id: int
    insertion_data: Optional[metadata.InsertionData] = None
    base_dir: Optional[Path] = None
    acq_software: str = ACQ_SOFTWARE
    config_attrs: Optional[dict] = None
    overwrite_config_attrs: bool = True
    clustering_method: str = DEFAULT_CLUSTERING_METHOD
    clustering_task_mode: clustering_task.ClusteringTaskMode = (
        clustering_task.ClusteringTaskMode.TRIGGER
    )
    clustering_output_dir: Optional[Path] = None
    clustering_output_suffix: Optional[Path] = None
    curation_input: clustering.CurationInput = clustering.CurationInput()
    check_for_existing_kilosort_results: bool = True
    run_insertion_meta: bool = True

    def run(self, **populate_kwargs):
        """Preclustering and Clustering"""
        if (
            self.clustering_output_dir is not None
            and self.clustering_output_suffix is not None
        ):
            raise ValueError(
                "clustering_output_dir and clustering_output_suffix can't both have values"
            )
        if self.base_dir is not None:
            pipeline_config().set_replacement_base(self.base_dir)

        ### ProbeInsertion
        if self.run_insertion_meta:
            InsertionMeta(
                scan_key=self.scan_key,
                base_dir=self.base_dir,
                insertion_id=self.insertion_id,
                insertion_data=self.insertion_data,
                config_attrs=self.config_attrs,
                overwrite_config_attrs=self.overwrite_config_attrs,
            ).run()

        ### PreClustering
        logging.info("starting preclustering section")
        session_meta = self.scan_key.model_dump()
        session_meta["insertion_id"] = self.insertion_id
        session_meta["rig"] = get_rig(self.scan_key.model_dump())
        ephys.Session.add_session(session_meta, error_on_duplicate=False)

        session_path, generic_path = get_session_path(
            self.scan_key, include_generic=True
        )

        inc_id = (ephys.Session & session_meta).fetch1("inc_id")
        session_key = dict(inc_id=inc_id)

        ephys.EphysFile.insert1(
            dict(
                **session_key,
                session_path=generic_path,
                acq_software=ACQ_SOFTWARE,
            ),
            skip_duplicates=True,
        )
        if self.config_attrs is not None:
            ephys.EphysFile.Metadata.fill(
                session_key=session_key,
                print_errors=True,
                config_attrs=self.config_attrs,
                overwrite_config_attrs=self.overwrite_config_attrs,
            )
        session_restriction = dict(**session_key)
        ephys.EphysRecording.populate(session_restriction, **populate_kwargs)

        # ephys.LFP.populate(session_restriction, **populate_kwargs)  # This isn't implemented yet

        logging.info("done with preclustering section")

        ### Clustering
        logging.info("starting clustering section")
        if self.clustering_output_dir is None:
            if self.clustering_output_suffix is None:
                self.clustering_output_dir = (
                    generic_path / DEFAULT_CLUSTERING_OUTPUT_RELATIVE
                )
            else:
                suffix_parts = self.clustering_output_suffix.parts
                clustering_output_dir = Path(generic_path)
                for part in suffix_parts:
                    if part == "..":
                        clustering_output_dir = clustering_output_dir.parent
                    else:
                        clustering_output_dir /= part
                self.clustering_output_dir = clustering_output_dir

        paramset_id = (
            ephys.ClusteringParamSet & {"clustering_method": self.clustering_method}
        ).fetch1("paramset_id")
        ephys.ClusteringTask.insert1(
            dict(
                **session_key,
                paramset_id=paramset_id,
                clustering_output_dir=self.clustering_output_dir,
                task_mode=self.clustering_task_mode.value,
            ),
            skip_duplicates=True,
        )

        if self.clustering_task_mode is ClusteringTaskMode.TRIGGER:
            from ...readers.recording.labview import NEUROPIXEL_PREFIX

            clustering_params = (
                (ephys.ClusteringParamSet & {"paramset_id": paramset_id})
                .fetch("params")
                .item()
            )
            task_runner = ClusteringTaskRunner(
                data_dir=session_path,
                results_dir=pipeline_config().specify(self.clustering_output_dir),
                filename=check_for_first_bin_with_prefix(
                    session_path, prefix=NEUROPIXEL_PREFIX
                ),
                clustering_params=clustering_params,
            )
            logging.info("attempting to trigger kilosort clustering")
            task_runner.trigger_clustering(self.check_for_existing_kilosort_results)
            logging.info("done with kilosort clustering")
        session_restriction["paramset_id"] = paramset_id
        ephys.Clustering.populate(session_restriction, **populate_kwargs)

        ### Curation Ingestion
        clustering_source_key = ephys.ClusteringTask.build_key_from_scan(
            self.scan_key.model_dump(), self.clustering_method
        )
        if self.curation_input.curation_output_dir is None:
            self.curation_input.curation_output_dir = (
                ephys.ClusteringTask() & clustering_source_key
            ).fetch1("clustering_output_dir")

        curation_id = ephys.Curation.create1_from_clustering_task(
            dict(
                **clustering_source_key,
                **self.curation_input.model_dump(),
            ),
        )
        session_restriction["curation_id"] = curation_id
        ephys.CuratedClustering.populate(session_restriction, **populate_kwargs)

        logging.info("done with clustering section")

        logging.info("starting post-clustering section")
        ephys.QualityMetrics.populate(session_restriction, **populate_kwargs)
        ephys.WaveformSet.populate(session_restriction, **populate_kwargs)
        logging.info("done with post-clustering section")


class Curated(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.CURATED] = PipelineMode.CURATED
    scan_key: ScanKey
    base_dir: Optional[Path] = None
    clustering_method: str = DEFAULT_CLUSTERING_METHOD
    curation_input: clustering.CurationInput

    def run(self, **populate_kwargs):
        """Ingesting curated results"""
        if self.base_dir is not None:
            pipeline_config().set_replacement_base(self.base_dir)

        ### Curation Ingestion
        clustering_source_key = ephys.ClusteringTask.build_key_from_scan(
            self.scan_key.model_dump(), self.clustering_method
        )
        if self.curation_input.curation_output_dir is None:
            self.curation_input.curation_output_dir = (
                ephys.ClusteringTask() & clustering_source_key
            ).fetch1("clustering_output_dir")
        curation_id = ephys.Curation.create1_from_clustering_task(
            dict(
                **clustering_source_key,
                **self.curation_input.model_dump(),
            ),
        )
        session_restriction = {
            "inc_id": clustering_source_key["inc_id"],
            "paramset_id": clustering_source_key["paramset_id"],
            "curation_id": curation_id,
        }
        ephys.CuratedClustering.populate(session_restriction, **populate_kwargs)

        logging.info("done with clustering section")

        logging.info("starting post-clustering section")
        ephys.QualityMetrics.populate(session_restriction, **populate_kwargs)
        ephys.WaveformSet.populate(session_restriction, **populate_kwargs)
        logging.info("done with post-clustering section")


class InsertionMeta(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.INSERTION_META] = PipelineMode.INSERTION_META
    scan_key: ScanKey
    insertion_id: int
    insertion_data: Optional[metadata.InsertionData] = None
    base_dir: Union[Optional[Path], Literal[False]] = Field(
        None,
        description="If set to False, will disable finding the probe serial number entirely",
    )
    config_attrs: Optional[dict] = None
    overwrite_config_attrs: bool = True
    

    def run(self):
        """Insertion data"""
        logging.info("starting probe insertion meta")
        find_probe = True
        if self.base_dir is False:
            find_probe = False

        insertion_key = dict(
            animal_id=self.scan_key.animal_id, insertion_id=self.insertion_id
        )
        ephys.ProbeInsertion.insert1(
            insertion_key,
            skip_duplicates=True,
        )

        if find_probe:
            if self.base_dir is not None:
                pipeline_config().set_replacement_base(self.base_dir)
            session_path = get_session_path(self.scan_key)

            # This is kind of finicky because currently I'm storing the metadata under EphysFile.Metadata
            # which is later in the pipeline than this.
            if self.config_attrs is None:
                labview_metadata = LabviewNeuropixelMeta.from_h5(directory=session_path)
            else:
                labview_metadata = LabviewNeuropixelMeta.from_h5(
                    directory=session_path,
                    config_attrs=self.config_attrs,
                    overwrite_config_attrs=self.overwrite_config_attrs,
                )
            ephys.ProbeInsertion.Probe.insert1(
                dict(**insertion_key, probe=labview_metadata.serial_number),
                skip_duplicates=True,
            )

        if self.insertion_data is not None:
            ephys.ProbeInsertion.Location.insert1(
                dict(**insertion_key, **self.insertion_data.model_dump()),
                skip_duplicates=True,
            )
        logging.info("done with probe insertion meta")


class PipelineInput(BaseModel, Runnable):
    params: Union[Setup, Minion, NoCuration, Curated, InsertionMeta] = Field(
        discriminator="pipeline_mode"
    )
    populate_kwargs: dict = {}  # {"reserve_jobs": True}

    def run(self):
        logging.info("starting neuropixel pipeline")
        logging.info(f"running in {self.params.pipeline_mode} mode")
        start_time = time.time()

        results = self.params.run(
            **self.populate_kwargs,
        )

        elapsed_time = round(time.time() - start_time, 2)
        logging.info(
            f"done with neuropixel pipeline, in mode {self.params.pipeline_mode}, elapsed_time: {elapsed_time}"
        )
        return results
