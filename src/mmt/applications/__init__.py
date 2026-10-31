"""Application-specific dataset profiles."""

from .mosi import MosiAdapter, MosiSegment
from .pamap2 import Pamap2Adapter, Pamap2Recording

__all__ = ["MosiAdapter", "MosiSegment", "Pamap2Adapter", "Pamap2Recording"]
