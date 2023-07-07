from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


class GenericDirectory(BaseModel):
    """
    Generic directory with a common start that would be replaced.
    """

    generic: str = Field(description="generic part of the path to be replaced")
    replacement: Optional[Path] = Field(
        description="leave as None if the replacement path must be set by each process"
    )

    def specify(self, generic_path: Path) -> Path:
        parts = generic_path.parts
        index = parts.index(self.generic)
        return Path(self.replacement).joinpath(*parts[index + 1 :])
