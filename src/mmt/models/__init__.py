"""Model components for multimodal temporal learning."""

from .source_encoder import EncoderOutput, ModelOutput, SourceBatch, SourceClassifier, SourceEncoder
from .fusion import V1GatedFusion, V2AttentionFusion
from .v0_late_fusion import V0LateFusion
from .baselines import BGridClassifier, GridBatch, MajorityReference
from .registry import MODEL_NAMES, build_majority_reference, build_model

__all__ = [
    "EncoderOutput",
    "BGridClassifier",
    "GridBatch",
    "MajorityReference",
    "MODEL_NAMES",
    "ModelOutput",
    "SourceBatch",
    "SourceClassifier",
    "SourceEncoder",
    "V0LateFusion",
    "V1GatedFusion",
    "V2AttentionFusion",
    "build_majority_reference",
    "build_model",
]