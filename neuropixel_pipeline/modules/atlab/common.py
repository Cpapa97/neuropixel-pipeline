from pydantic import BaseModel, Field, conint


class ScanKey(BaseModel):
    animal_id: conint(ge=0, le=2_147_483_647)
    session_id: conint(ge=0, le=32_767) = Field(alias="session", alias_priority=1)
    scan_id: conint(ge=0, le=32_767) = Field(alias="scan_idx", alias_priority=1)
