"""
Sensors subpackage: Radar sensor simulation and telemetry structures.
"""

from src.sensors.radar_sensor import (
    RadarDetection,
    RadarSensor,
    SensorObservation,
    ShipTelemetry,
)

__all__ = [
    "RadarDetection",
    "RadarSensor",
    "SensorObservation",
    "ShipTelemetry",
]
