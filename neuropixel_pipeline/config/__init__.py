from pydantic import BaseModel, Field
from typing import Optional, Tuple
from pathlib import Path

from .generic_directory import GenericDirectory


class PipelineConfig(BaseModel):
    generic_directory_suffix: Optional[GenericDirectory] = None

    def specify(self, path: Path):
        if self.generic_directory_suffix is not None:
            self.generic_directory_suffix.specify(path)
