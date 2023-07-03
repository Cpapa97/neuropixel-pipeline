from pydantic import BaseModel, Field
from typing import Optional, Tuple
from pathlib import Path

# I can possibly use json serialization to save this to a file and check for this file automatically,
# so a pipeline minion or docker image could have this config be made once
# TODO: Even more ideally, it could be stored and/or fetched from a PipelineConfig table at the root of the schema,
#       that no other table should depend on directly.
# TODO: How would a default be chosen from the datajoint table though?
#       It would make sense to have the default be None for most fields, and have default behavior based on that,
#       but when a default is chosen, how do they keep it as the default? Is it just a secondary key that only
#       one table entry is allowed to have at one time?? Wouldn't that require Table._update to change?
class PipelineConfig(BaseModel):
    generic_directory_suffix: Optional[Tuple[Path, Path]] = Field(
        description="Allows a generic path to be used, with a common suffix replaced by the passed in path"
    )


def choose_config(config_name: str):
    pass
