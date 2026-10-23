"""Tracking and person attributes."""

from .attributes import assign_attributes
from .bytetrack import Tracker

__all__ = ["Tracker", "assign_attributes"]
