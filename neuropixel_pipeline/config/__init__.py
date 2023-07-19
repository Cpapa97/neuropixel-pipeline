from pydantic import BaseModel, validate_call
from typing import Optional
from pathlib import Path

from .generic_directory import GenericDirectory


class PipelineConfig(BaseModel):
    use_global_config: bool = False
    generic_directory_suffix: Optional[GenericDirectory] = None

    @validate_call
    def set_replacement_base(self, base_dir: Path):
        if self.generic_directory_suffix is not None:
            self.generic_directory_suffix.replacement = base_dir
        else:
            raise ValueError(
                "no self.generic_directory_suffix is set and missing self.generic_directory_suffix.`generic`"
            )

    @validate_call
    def specify(self, path: Path) -> Path:
        path = Path(path)
        if self.generic_directory_suffix is not None:
            return self.generic_directory_suffix.specify(path)
        else:
            return path
