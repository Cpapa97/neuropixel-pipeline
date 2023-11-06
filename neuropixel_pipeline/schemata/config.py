# flake8: noqa

from __future__ import annotations

import datajoint as dj
from pydantic import validate_call

from . import SCHEMA_PREFIX
from ..config import PipelineConfig, atlab


schema = dj.schema(SCHEMA_PREFIX + "config")


def pipeline_config(name=None, use_global=None, override_global=False):
    if use_global is None:
        use_global = pipeline_config(name=name, use_global=True).use_global_config

    if use_global:
        global instance_config

    if not override_global and use_global and "instance_config" in globals():
        return instance_config
    else:
        if name is None:
            config = (PipelineConfigStore & PipelineConfigStore.Default).fetch1(
                "config"
            )
        else:
            config = (PipelineConfigStore & {"name": name}).fetch1("config")
        instance_config = PipelineConfig.model_validate_json(config)
    return instance_config


@schema  # Rename PipelineConfig table to not shadow config.PipelineConfig?
class PipelineConfigStore(dj.Lookup):
    definition = """
    # Config that determines certain runtime behavior
    name: varchar(255)  # name of the config
    ---
    config: longblob    # PipelineConfig, validated by pydantic
    """

    # TODO: Probably don't need these here, the pipeline should be setup with their own config from the start
    contents = [
        ["atlab_mouse", atlab.atlab_mouse_pipeline_config.model_dump_json()],
        ["atlab_monkey", atlab.atlab_monkey_pipeline_config.model_dump_json()],
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
