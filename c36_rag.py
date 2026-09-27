"""Backward-compatible import path for C36.
Authoritative implementation: :mod:`sebrain.c36`.
"""
from .c36 import RAGContext, RAGItem, RAGPipeline
__all__ = ["RAGContext", "RAGItem", "RAGPipeline"]
