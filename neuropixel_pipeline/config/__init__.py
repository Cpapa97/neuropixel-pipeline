from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from .generic_directory import GenericDirectory


class PipelineConfig(BaseModel):
    use_global_config: bool = False
    generic_directory_suffix: Optional[GenericDirectory] = None

    # @validate_call
    def specify(self, path: Path):
        path = Path(path)
        if self.generic_directory_suffix is not None:
            return self.generic_directory_suffix.specify(path)
        else:
            return path
