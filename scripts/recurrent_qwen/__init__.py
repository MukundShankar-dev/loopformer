"""Shared-depth Qwen model mechanics, without import-time loading."""

from .model import RecurrentQwen
from .outputs import RecurrentOutput

__all__ = ["RecurrentQwen", "RecurrentOutput"]
