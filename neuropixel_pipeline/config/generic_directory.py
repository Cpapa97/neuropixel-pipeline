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

    # @validate_call
    def specify(self, generic_path: Path) -> Path:
        if self.replacement is None:
            raise ValueError("self.replacement cannot be None for this method")
        generic_path = Path(generic_path)
        parts = generic_path.parts
        try:
            index = parts.index(self.generic)
        except ValueError:
            raise ValueError(
                f"Path does not contain the expected generic component: '{self.generic}', in path: '{generic_path}'"
            )
        return Path(self.replacement).joinpath(*parts[index + 1 :])
    
    # @validate_call
    def unspecify(self, specific_path: Path) -> Path:
        raise NotImplementedError("Unspecifying a specific path is not supported yet, but is planned")
