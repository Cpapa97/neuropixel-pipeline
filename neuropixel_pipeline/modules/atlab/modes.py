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
from ...readers.labview import LabviewNeuropixelMeta
from ...utils import check_for_first_bin_with_prefix
from ...schemata import probe, ephys


class Runnable(abc.ABC):
    @abc.abstractmethod
    def run(self):
        pass


class PipelineMode(str, Enum):
    SETUP = "setup"
    MINION = "minion"
    NO_CURATION = "no curation"
    CURATED = "curated"


class Setup(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.SETUP] = PipelineMode.SETUP
    setup: bool = True

    def run(self):
        """Setup for neuropixel_probe"""
        ### Setup
        logging.info("starting setup section")
        probe.ProbeType.fill_neuropixel_probes()
        probe_setup()
        ephys.ClusteringParamSet.fill(
            {},
            clustering_method="kilosort3",
            description="kilosort3 params (for ingesting)",
            skip_duplicates=True,
        )
        logging.info("done with setup section")


class Minion(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.MINION] = PipelineMode.MINION
    base_dir: Path

    def run(self, **populate_kwargs):
        # essentially should just run NoCuration or Curated based on keys from an IngestionTask table
        # might even be able to just call PipelineInput directly with the pipeline_mode
        pass


class NoCuration(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.NO_CURATION] = PipelineMode.NO_CURATION
    scan_key: ScanKey
    base_dir: Path
    acq_software: str = ACQ_SOFTWARE
    insertion_number: int
    # Will ephys.InsertionLocation just be inserted into directly from 2pmaster?
    insertion_location: Optional[metadata.InsertionData] = None
    clustering_method: str = DEFAULT_CLUSTERING_METHOD
    clustering_task_mode: clustering_task.ClusteringTaskMode = (
        clustering_task.ClusteringTaskMode.TRIGGER
    )
    clustering_output_dir: Optional[Path] = None
    clustering_output_suffix: Optional[Path] = None
    curation_input: clustering.CurationInput = clustering.CurationInput()
    check_for_existing_kilosort_results: bool = True

    def run(self, **populate_kwargs):
        """Preclustering and Clustering"""
        if (
            self.clustering_output_dir is not None
            and self.clustering_output_suffix is not None
        ):
            raise ValueError(
                "clustering_output_dir and clustering_output_suffix can't both have values"
            )

        ### PreClustering
        logging.info("starting preclustering section")
        session_meta = self.scan_key.model_dump()
        session_meta["rig"] = get_rig(self.scan_key.model_dump())
        ephys.Session.add_session(session_meta, error_on_duplicate=False)

        session_path = get_session_path(self.scan_key, base_dir=self.base_dir)

        labview_metadata = LabviewNeuropixelMeta.from_h5(session_path)

        session_id = (ephys.Session & session_meta).fetch1("session_id")
        insertion_key = dict(
            session_id=session_id, insertion_number=self.insertion_number
        )

        ephys.ProbeInsertion.insert1(
            dict(
                **insertion_key,
                probe=labview_metadata.serial_number,
            ),
            skip_duplicates=True,
        )

        if self.insertion_location is not None:
            ephys.InsertionLocation.insert(self.insertion_location.model_dict())

        ephys.EphysFile.insert1(
            dict(
                **insertion_key,
                session_path=session_path,
                acq_software=ACQ_SOFTWARE,
            ),
            skip_duplicates=True,
        )
        session_restriction = dict(**insertion_key)
        ephys.EphysRecording.populate(session_restriction, **populate_kwargs)

        # ephys.LFP.populate(session_restriction, **populate_kwargs)  # This isn't implemented yet

        logging.info("done with preclustering section")

        ### Clustering
        logging.info("starting clustering section")
        # This currently only supports the default kilosort parameters, which might be alright for atlab
        if self.clustering_method == DEFAULT_CLUSTERING_METHOD:
            ephys.ClusteringParamSet.fill(
                params=default_kilosort_parameters(),
                clustering_method="kilosort4",
                description="default kilosort4 params",
                skip_duplicates=True,
            )

        if self.clustering_output_dir is None:
            if self.clustering_output_suffix is None:
                self.clustering_output_dir = (
                    session_path / DEFAULT_CLUSTERING_OUTPUT_RELATIVE
                )
            else:
                suffix_parts = self.clustering_output_suffix.parts
                clustering_output_dir = Path(session_path)
                for part in suffix_parts:
                    if part == "..":
                        clustering_output_dir = clustering_output_dir.parent
                    else:
                        clustering_output_dir /= part

        paramset_idx = (
            ephys.ClusteringParamSet & {"clustering_method": self.clustering_method}
        ).fetch1("paramset_idx")
        task_source_key = dict(
            **insertion_key,
            paramset_idx=paramset_idx,
            clustering_output_dir=self.clustering_output_dir,
            task_mode=self.clustering_task_mode.value,
        )
        ephys.ClusteringTask.insert1(task_source_key, skip_duplicates=True)

        if self.clustering_task_mode is ClusteringTaskMode.TRIGGER:
            from ...readers.labview import NEUROPIXEL_PREFIX

            clustering_params = (
                (ephys.ClusteringParamSet & {"paramset_idx": paramset_idx})
                .fetch("params")
                .item()
            )
            task_runner = ClusteringTaskRunner(
                data_dir=session_path,
                results_dir=task_source_key["clustering_output_dir"],
                filename=check_for_first_bin_with_prefix(
                    session_path, prefix=NEUROPIXEL_PREFIX
                ),
                clustering_params=clustering_params,
            )
            logging.info("attempting to trigger kilosort clustering")
            task_runner.trigger_clustering(self.check_for_existing_kilosort_results)
            logging.info("done with kilosort clustering")
        session_restriction["paramset_idx"] = paramset_idx
        ephys.Clustering.populate(session_restriction, **populate_kwargs)

        ### Curation Ingestion
        clustering_source_key = ephys.ClusteringTask.build_key_from_scan(
            self.scan_key.model_dump(), self.insertion_number, self.clustering_method
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
        logging.info("done with post-clustering section")


class Curated(BaseModel, Runnable):
    pipeline_mode: Literal[PipelineMode.CURATED] = PipelineMode.CURATED
    scan_key: ScanKey
    base_dir: Path
    curation_input: clustering.CurationInput

    def run(self, **populate_kwargs):
        """Ingesting curated results"""
        ### Curation Ingestion
        clustering_source_key = ephys.ClusteringTask.build_key_from_scan(
            self.scan_key.model_dump(), self.insertion_number, self.clustering_method
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
            "session_id": clustering_source_key["session_id"],
            "insertion_number": clustering_source_key["insertion_number"],
            "paramset_idx": clustering_source_key["paramset_idx"],
            "curation_id": curation_id,
        }
        ephys.CuratedClustering.populate(session_restriction, **populate_kwargs)

        logging.info("done with clustering section")

        logging.info("starting post-clustering section")
        ephys.QualityMetrics.populate(session_restriction, **populate_kwargs)
        logging.info("done with post-clustering section")


class PipelineInput(BaseModel, Runnable):
    params: Union[Setup, Minion, NoCuration, Curated] = Field(
        discriminator="pipeline_mode"
    )
    populate_kwargs: dict = {"reserve_jobs": True}

    def run(self):
        logging.info("starting neuropixel pipeline")
        start_time = time.time()

        results = self.params.run(
            **self.populate_kwargs,
        )

        elapsed_time = round(time.time() - start_time, 2)
        logging.info(f"done with neuropixel pipeline, elapsed_time: {elapsed_time}")
        return results
