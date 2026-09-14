"""Pydantic schemas for agent tool inputs. The runtime rejects any call that fails them."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from linegate.model.leak_tests import RefitResult, ScopeResult, ShuffleResult

FeatureId = Annotated[str, StringConstraints(pattern=r"^f_[0-9a-f]{10}$")]

__all__ = ["FeatureRef", "QuarantineInput", "ApproveInput", "ShuffleResult", "RefitResult", "ScopeResult",
           "SchemaInput", "EmptyInput", "RunSQLInput", "ProposeInput", "EvaluateInput",
           "PartInput", "NeighborsInput", "CitationsInput", "DispositionInput"]


class FeatureRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_id: FeatureId


class QuarantineInput(FeatureRef):
    reason: str = Field(min_length=20, max_length=2000, description="Finding naming the test result that drove it")


class ApproveInput(FeatureRef):
    finding: str = Field(min_length=20, max_length=2000, description="Finding naming the test results that support it")


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SchemaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    station: Annotated[str, StringConstraints(pattern=r"^L[0-9]+_S[0-9]+$")] | None = None


class RunSQLInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis: str = Field(min_length=15, max_length=1000, description="What you expect to see and why, before the query")
    query: str = Field(min_length=10, max_length=20000)
    row_limit: int = Field(default=200, ge=1, le=1000)


class ProposeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,40}$")]
    hypothesis: str = Field(min_length=15, max_length=1000, description="Physical situation on the line this captures")
    sql: str = Field(min_length=10, max_length=60000, description="SELECT returning Id plus feature columns, one row per part")


class EvaluateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_ids: list[FeatureId] = Field(min_length=1, max_length=8)


class PartInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    part_id: int = Field(ge=1)


class NeighborsInput(PartInput):
    k: int = Field(default=20, ge=1, le=50)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["column", "part"]
    value: str = Field(min_length=1, max_length=40)


class CitationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citations: list[Citation] = Field(max_length=60)


class OutOfRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    value: float
    low: float
    high: float


class DispositionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: str = Field(max_length=500)
    out_of_range: list[OutOfRange] = Field(max_length=40)
    neighbor_summary: str = Field(max_length=600)
    recommendation: Literal["scrap", "ship", "senior_review"]
    reason: str = Field(min_length=20, max_length=600, description="Recommendation and reason in two sentences")
    citations: list[Citation] = Field(max_length=60)
