"""Pydantic schemas for agent tool inputs. The runtime rejects any call that fails them."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from linegate.model.leak_tests import RefitResult, ScopeResult, ShuffleResult

FeatureId = Annotated[str, StringConstraints(pattern=r"^f_[0-9a-f]{10}$")]

__all__ = ["FeatureRef", "QuarantineInput", "ApproveInput", "ShuffleResult", "RefitResult", "ScopeResult",
           "SchemaInput", "EmptyInput", "RunSQLInput", "ProposeInput", "EvaluateInput"]


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
