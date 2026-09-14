"""Pydantic schemas for agent tool inputs. The runtime rejects any call that fails them."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from linegate.model.leak_tests import RefitResult, ScopeResult, ShuffleResult

FeatureId = Annotated[str, StringConstraints(pattern=r"^f_[0-9a-f]{10}$")]

__all__ = ["FeatureRef", "QuarantineInput", "ApproveInput", "ShuffleResult", "RefitResult", "ScopeResult"]


class FeatureRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_id: FeatureId


class QuarantineInput(FeatureRef):
    reason: str = Field(min_length=20, max_length=2000, description="Finding naming the test result that drove it")


class ApproveInput(FeatureRef):
    finding: str = Field(min_length=20, max_length=2000, description="Finding naming the test results that support it")
