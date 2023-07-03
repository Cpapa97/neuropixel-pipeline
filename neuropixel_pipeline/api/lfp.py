from pydantic import BaseModel, Field
from typing import Iterable
from datetime import datetime


class LfpMetrics(BaseModel):
    lfp_sampling_rate: float = Field(description="sampling rate in Hz")
    lfp_time_stamps: Iterable[datetime] = Field(
        description="(s) timestamps with respect to the start of the recording"
    )
    lfp_mean: Iterable[float] = Field(
        description="mean of LFP across electrodes - shape (time,) in microvolts (uV)"
    )
