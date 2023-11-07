"""
Custom labview neuropixel aquisition format reader
"""

from __future__ import annotations

import os
from enum import Enum

from pathlib import Path
from pydantic import BaseModel, Field, field_validator, constr
from typing import List, Tuple, Any, Optional, Dict
import numpy as np


from ..api import metadata, lfp
from neuropixel_pipeline.api.lfp import LfpMetrics
from .. import utils

NEUROPIXEL_PREFIX = "NPElectrophysiology"


class LabviewMetaType(Enum):
    HDF5 = ".h5"
    METAFILE = ".metafile"  # will probably replace this will something closer to the ext that'll be used
    MISSING = "_missing"


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

    @classmethod
    def _check_for_config(cls, directory: Path, family: str = None):
        """
        This function prefers the metafile over the h5 family of files
        """
        if family is None:

            def validate_h5_name(path: Path) -> bool:
                return True

        else:

            def validate_h5_name(path: Path) -> bool:
                return path.match(family.replace("%", "*"))

        metafile_exists = False
        metafile_paths = []
        h5_exists = False
        for path in os.listdir(directory):
            if path.is_file():
                ext = path.suffix
                if ext == LabviewMetaType.METAFILE.value():
                    metafile_exists = True
                    metafile_paths.append(path)
                elif ext == LabviewMetaType.HDF5.value() and validate_h5_name(path):
                    h5_exists = True
        if metafile_exists:
            assert (
                len(metafile_paths) == 1
            ), f"There should only be one metafile config, instead found these files: {metafile_paths}"
            metafile_path = metafile_paths[0]
            return LabviewMetaType.METAFILE, metafile_path
        elif h5_exists:
            return LabviewMetaType.HDF5, None
        else:
            return LabviewMetaType.MISSING, None

    @classmethod
    def from_any(
        cls,
        directory: Path,
        family: str = "NPElectrophysiology%d.h5",
        load_config_data=True,
    ) -> LabviewNeuropixelMeta:
        """
        Will check for the availability of config files,
        preferring the custom metafile over the h5 family of files
        """
        directory = Path(directory)
        file_type, filepath = cls._check_for_config(directory, family=family)
        if file_type is LabviewMetaType.METAFILE:
            cls.from_metafile(filepath=filepath, load_config_data=load_config_data)
        elif file_type is LabviewMetaType.HDF5:
            cls.from_h5(
                directory=directory, family=family, load_config_data=load_config_data
            )
        else:
            assert (
                file_type is LabviewMetaType.MISSING
            ), "Somehow got a variant other than MISSING"
            raise ValueError("No labview config found in this directory:", directory)

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
    def from_metafile(
        cls, filepath: Path, load_config_data=True
    ) -> LabviewNeuropixelMeta:
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

    @staticmethod
    def find_bin_from_prefix(
        session_dir: Path,
        prefix: str = NEUROPIXEL_PREFIX,
    ):
        return utils.check_for_first_bin_with_prefix(
            session_dir=session_dir,
            prefix=prefix,
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
