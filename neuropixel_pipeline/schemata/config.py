# flake8: noqa

from __future__ import annotations

import datajoint as dj
from pydantic import validate_call
from pathlib import Path
from enum import Enum

from ..config import PipelineConfig, atlab


schema = dj.schema("neuropixel_config")


def pipeline_config(name=None):
    if name is None:
        config = (PipelineConfigTable & PipelineConfigTable.Default).fetch1("config")
    else:
        config = (PipelineConfigTable & {"name": name}).fetch1("config")
    return PipelineConfig.model_validate_json(config)


@schema  # Rename PipelineConfig table to not shadow config.PipelineConfig?
class PipelineConfigTable(dj.Lookup):
    definition = """
    # Config that determines certain runtime behavior
    name: varchar(255)  # name of the config
    ---
    config: longblob    # PipelineConfig, validated by pydantic
    """

    contents = [
        ["atlab", atlab.atlab_pipeline_config.model_dump_json()]
    ]  # need to also consider that part of probe_setup is currently manual, except for at-lab where is isn't...

    class Default(dj.Part):
        definition = """
        # Default configuration (only one entry enforced at a time)
        default: enum('default')
        ---
        -> master
        """

    @classmethod
    @validate_call
    def add_config(
        cls, name: str, config: PipelineConfig, replace=False, make_default=False
    ):
        cls.insert1(dict(name=name, config=config.model_dump_json()), replace=replace)
        if make_default:
            cls.set_default(name)

    @classmethod
    @validate_call
    def set_default(cls, name: str, replace=True):
        cls.Default.insert1(
            dict(default="default", name=name),
            replace=replace,
        )

    @classmethod
    def get_default(cls) -> PipelineConfig:
        return PipelineConfig.model_validate_json((cls & cls.Default).fetch1("config"))


class PathKind(str, Enum):
    """
    Filepath Kind
    """

    SESSION = "session"
    CLUSTERING = "clustering"
    CURATION = "curation"

    def normalize(self, generic_path: Path):
        """
        Handles path kind specific nuances

        This is the function that should generally be used
        """
        specific_path = pipeline_config().specify(generic_path)
        if self is PathKind.SESSION:
            return specific_path.parent
        elif self is PathKind.CLUSTERING:
            return specific_path
        elif self is PathKind.CURATION:
            return specific_path
        else:
            raise NotImplementedError(
                f"this is not implemented for this PathKind: {self}"
            )
