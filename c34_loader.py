"""Backward-compatible import path for C34.
Authoritative implementation: :mod:`sebrain.c34`.
"""
from .c34 import FabricRecord, KnowledgeFabricLoader, now_iso
__all__ = ["FabricRecord", "KnowledgeFabricLoader", "now_iso"]
