from pydantic import BaseModel, Field
from typing import Optional, Tuple
from pathlib import Path

# I can possibly use json serialization to save this to a file and check for this file automatically,
# so a pipeline minion or docker image could have this config be made once
class PipelineConfig(BaseModel):
    generic_directory_suffix: Optional[Tuple[Path, Path]] = Field(
        description="Allows a generic path to be used, with a common suffix replaced by the passed in path"
    )


def choose_config(config_name: str):
    pass
