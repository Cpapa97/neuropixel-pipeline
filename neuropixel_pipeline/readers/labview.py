"""
Custom labview neuropixel aquisition format reader
"""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, field_validator, constr
from typing import List, Tuple, Any, Optional, Dict
import numpy as np


from ..api import metadata, lfp
from neuropixel_pipeline.api.lfp import LfpMetrics
from .. import utils

NEUROPIXEL_PREFIX = "NPElectrophysiology"


class LabviewNeuropixelMeta(BaseModel, arbitrary_types_allowed=True):
    # probe serial number
    serial_number: constr(max_length=32) = Field(alias="SerialNum")

    # probe version
    version: float = Field(alias="Version")

    # sampling_rate as Fs
    sampling_rate: float = Field(alias="Fs")

    #
    channel_names: List[str] = Field(alias="channelNames")

    #
    class_name: str = Field(alias="class")

    #
    scale: Tuple[float, float]

    #
    t0: float = Field(alias="t0")

    #
    config_params: List[str] = Field(alias="ConfigParams")

    #
    config_data: Optional[Any] = Field(None, alias="Config")

    @field_validator("config_params", "channel_names", mode="before")
    def convert_to_list(cls, v):
        if isinstance(v, bytes):
            return v.decode().strip().split(",")
        elif isinstance(v, str):
            return v.strip().split(",")
        else:
            return v

    @field_validator("scale", mode="before")
    def check_scale_shape(cls, v):
        a, b = v
        return (a, b)

    @staticmethod
    def _validate_probe_naming_convention(
        meta: dict, original_key_name: str, normalized_key_name: str
    ):
        if normalized_key_name in original_key_name:
            meta[normalized_key_name] = meta.pop(original_key_name)

    # eventually might want to have a function wrapper above this that selects based on
    # an existing attribute file next to the bin or still uses h5 if not (if possible).
    @classmethod
    def from_h5(
        cls,
        directory: Path,
        family: str = "NPElectrophysiology%d.h5",
        load_config_data=True,
    ) -> LabviewNeuropixelMeta:
        """
        Uses an h5 family driver
        """
        import h5py

        directory = Path(directory)
        with h5py.File(directory / family, driver="family", memb_size=0) as f:
            meta = dict(f.attrs)

            # need eager keys evaluation here, therefore list is used
            for key in list(meta.keys()):
                cls._validate_probe_naming_convention(meta, key, "SerialNum")
                cls._validate_probe_naming_convention(meta, key, "t0")

            if load_config_data:
                for key in f.keys():
                    if "Config" in key:
                        meta["Config"] = np.array(f[key])

        return cls.model_validate(meta)

    @classmethod
    def from_metafile(cls) -> LabviewNeuropixelMeta:
        """
        This will be implemented when the metadata from labview is separated from the h5
        """
        raise NotImplementedError(
            "This will be implemented when the labview metadata is separate from the h5"
        )

    def channels(self) -> List[int]:
        # can use self.config_data instead, with config_params's channel and port
        return list(int(channel_name[-4:]) for channel_name in self.channel_names)

    def electrode_config(self) -> Dict[str, Any]:
        return dict(zip(self.config_params, self.config_data.T))

    def electrode_config_hash(self) -> str:
        return utils.dict_to_uuid(self.model_dump())

    def to_metadata(self) -> metadata.NeuropixelConfig:
        raise NotImplementedError(
            "This isn't implemented but is needed for neuropixel config generation"
        )


class LabviewBin(BaseModel):
    bin_path: Path

    @staticmethod
    def find_from_prefix(
        session_dir: Path,
        prefix: str = NEUROPIXEL_PREFIX,
    ):
        return LabviewBin(
            bin_path=utils.check_for_first_bin_with_prefix(
                session_dir=session_dir,
                prefix=prefix,
            )
        )

    @staticmethod
    def extract_lfp_metrics(
        self,
        microvolt_conversion_factor: float,
        num_channels=384,
        has_sync_channel=True,
    ) -> lfp.LfpMetrics:
        data = utils.extract_data_from_bin(
            bin_file=self.bin_path,
            num_channels=num_channels,
            has_sync_channel=has_sync_channel,
        )
        # TODO: calculate lfp metrics
        raise NotImplementedError(
            f"lfp not implemented yet for LabviewV1: here's the data though: {data}"
        )
        return lfp.LfpMetrics()
