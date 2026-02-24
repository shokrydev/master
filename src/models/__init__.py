# Model definitions
# Currently using Unsloth's FastVisionModel directly
# Add custom model wrappers here as needed

from src.models.location_modality_projection import LocationModalityProjection
from src.models.satclip import get_satclip

__all__ = [
    "LocationModalityProjection",
    "get_satclip",
]
