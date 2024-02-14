from pydantic import BaseModel
from enum import Enum

from . import labview


class RecordingSoftware(Enum):
    LABVIEW = "LabviewV1"
    SPIKE_GLX = "SpikeGLX"
    OPEN_EPHYS = "OpenEphys"


# TODO: Use this and the RecordingSoftware enum instead for selecting behavior in the DataJoint pipeline
class RecordingReader(BaseModel):
    software: RecordingSoftware

    def load(self, **kwargs):
        if RecordingSoftware.LABVIEW:
            return self.load_labview(**kwargs)
        else:
            raise ValueError("Only Labview aquisition software is currently supported")

    @staticmethod
    def load_labview(**kwargs):
        return labview.LabviewNeuropixelMeta.from_h5(**kwargs)
