"""Compass: Intelligent follow-up question generation for LangGraph agents."""

from compass.helpers import extract_previous_followups, get_compass_instruction
from compass.node import CompassNode, ExampleRetriever
from compass.triggers import AlwaysTrigger, DefaultTriggerPolicy, NeverTrigger, TriggerPolicy

__all__ = [
    "CompassNode",
    "TriggerPolicy",
    "DefaultTriggerPolicy",
    "AlwaysTrigger",
    "NeverTrigger",
    "ExampleRetriever",
    "get_compass_instruction",
    "extract_previous_followups",
]

__version__ = "1.0.1"
