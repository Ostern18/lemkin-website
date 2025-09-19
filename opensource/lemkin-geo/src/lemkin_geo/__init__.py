"""
Lemkin Geospatial Analysis Suite

Geographic analysis of evidence without requiring GIS expertise for legal investigations.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    CoordinateConverter,
    SatelliteAnalyzer,
    GeofenceProcessor,
    MappingGenerator,
    StandardCoordinate,
    BoundingBox,
    DateRange,
    SatelliteAnalysis,
    LocationCorrelation,
    InteractiveMap,
    Event,
    Evidence,
)

__all__ = [
    "CoordinateConverter",
    "SatelliteAnalyzer",
    "GeofenceProcessor",
    "MappingGenerator",
    "StandardCoordinate",
    "BoundingBox",
    "DateRange",
    "SatelliteAnalysis",
    "LocationCorrelation",
    "InteractiveMap",
    "Event",
    "Evidence",
]