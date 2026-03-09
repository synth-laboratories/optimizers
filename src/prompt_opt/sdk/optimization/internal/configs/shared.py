"""Shared config base models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExtraModel(BaseModel):
    """Base model that preserves unknown keys for forward compatibility."""

    model_config = ConfigDict(extra="allow")

