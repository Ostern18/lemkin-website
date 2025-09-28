"""
Tests for lemkin-geo core geospatial functionality.
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lemkin_geo.core import (
    CoordinateFormat,
    EventType,
    StandardCoordinate,
    BoundingBox,
    GeoEvent,
    LocationCorrelation,
    SatelliteAnalysis,
    InteractiveMap,
    GeospatialAnalyzer,
    CoordinateConverter,
    SatelliteAnalyzer,
    GeofenceProcessor,
    MappingGenerator
)


class TestStandardCoordinate:
    """Test StandardCoordinate model"""

    def test_valid_coordinate(self):
        """Test creation of valid coordinate"""
        coord = StandardCoordinate(
            latitude=40.7128,
            longitude=-74.0060,
            elevation=10.0,
            accuracy_meters=5.0
        )
        assert coord.latitude == 40.7128
        assert coord.longitude == -74.0060
        assert coord.elevation == 10.0
        assert coord.accuracy_meters == 5.0
        assert coord.coordinate_system == "WGS84"

    def test_invalid_latitude(self):
        """Test invalid latitude raises error"""
        with pytest.raises(ValueError):
            StandardCoordinate(latitude=91, longitude=0)

        with pytest.raises(ValueError):
            StandardCoordinate(latitude=-91, longitude=0)

    def test_invalid_longitude(self):
        """Test invalid longitude raises error"""
        with pytest.raises(ValueError):
            StandardCoordinate(latitude=0, longitude=181)

        with pytest.raises(ValueError):
            StandardCoordinate(latitude=0, longitude=-181)


class TestGeoEvent:
    """Test GeoEvent model"""

    def test_geo_event_creation(self):
        """Test creation of geo event"""
        event = GeoEvent(
            event_id="test-001",
            event_type=EventType.INCIDENT,
            coordinate=StandardCoordinate(latitude=40.7128, longitude=-74.0060),
            timestamp=datetime.now(timezone.utc),
            description="Test incident"
        )
        assert event.event_id == "test-001"
        assert event.event_type == EventType.INCIDENT
        assert event.description == "Test incident"
        assert event.coordinate.latitude == 40.7128

    def test_geo_event_with_metadata(self):
        """Test geo event with metadata"""
        metadata = {"severity": "high", "witnesses": 3}
        event = GeoEvent(
            event_id="test-002",
            event_type=EventType.WITNESS,
            coordinate=StandardCoordinate(latitude=40.7128, longitude=-74.0060),
            timestamp=datetime.now(timezone.utc),
            metadata=metadata
        )
        assert event.metadata == metadata
        assert event.metadata["severity"] == "high"
        assert event.metadata["witnesses"] == 3


class TestCoordinateConverter:
    """Test CoordinateConverter functionality"""

    def test_dms_to_decimal(self):
        """Test DMS to decimal conversion"""
        converter = CoordinateConverter()

        # Test positive coordinates
        dms = "40°42'46.0\"N 74°00'21.6\"W"
        result = converter.standardize_coordinates(dms, CoordinateFormat.DMS)
        assert abs(result.latitude - 40.7128) < 0.001
        assert abs(result.longitude - (-74.006)) < 0.001

        # Test negative coordinates
        dms = "33°52'12.0\"S 151°12'36.0\"E"
        result = converter.standardize_coordinates(dms, CoordinateFormat.DMS)
        assert abs(result.latitude - (-33.87)) < 0.01
        assert abs(result.longitude - 151.21) < 0.01

    def test_ddm_to_decimal(self):
        """Test DDM to decimal conversion"""
        converter = CoordinateConverter()

        ddm = "40°42.768'N 74°00.360'W"
        result = converter.standardize_coordinates(ddm, CoordinateFormat.DDM)
        assert abs(result.latitude - 40.7128) < 0.001
        assert abs(result.longitude - (-74.006)) < 0.001

    def test_decimal_passthrough(self):
        """Test decimal coordinates pass through correctly"""
        converter = CoordinateConverter()

        decimal = "40.7128, -74.0060"
        result = converter.standardize_coordinates(decimal, CoordinateFormat.DECIMAL)
        assert result.latitude == 40.7128
        assert result.longitude == -74.0060

    def test_invalid_format(self):
        """Test invalid coordinate format"""
        converter = CoordinateConverter()

        with pytest.raises(ValueError):
            converter.standardize_coordinates("invalid", CoordinateFormat.DECIMAL)


class TestGeofenceProcessor:
    """Test GeofenceProcessor functionality"""

    def test_correlate_events_within_radius(self):
        """Test correlation of events within radius"""
        processor = GeofenceProcessor()

        events = [
            GeoEvent(
                event_id="event-1",
                event_type=EventType.INCIDENT,
                coordinate=StandardCoordinate(latitude=40.7128, longitude=-74.0060),
                timestamp=datetime.now(timezone.utc),
                description="Event 1"
            ),
            GeoEvent(
                event_id="event-2",
                event_type=EventType.WITNESS,
                coordinate=StandardCoordinate(latitude=40.7130, longitude=-74.0062),
                timestamp=datetime.now(timezone.utc),
                description="Event 2"
            ),
            GeoEvent(
                event_id="event-3",
                event_type=EventType.EVIDENCE,
                coordinate=StandardCoordinate(latitude=41.7128, longitude=-74.0060),
                timestamp=datetime.now(timezone.utc),
                description="Event 3"
            )
        ]

        # Test with small radius (should correlate events 1 and 2)
        correlation = processor.correlate_events_by_location(events, radius_meters=500)
        assert len(correlation.correlated_clusters) >= 1

        # Check that events 1 and 2 are in same cluster
        cluster_with_both = None
        for cluster in correlation.correlated_clusters:
            event_ids = [e.event_id for e in cluster.events]
            if "event-1" in event_ids and "event-2" in event_ids:
                cluster_with_both = cluster
                break

        assert cluster_with_both is not None
        assert len(cluster_with_both.events) == 2

    def test_create_geofence(self):
        """Test geofence creation"""
        processor = GeofenceProcessor()

        center = StandardCoordinate(latitude=40.7128, longitude=-74.0060)
        geofence = processor.create_geofence(center, radius_meters=1000)

        assert geofence is not None
        # Geofence should be a polygon with multiple points
        assert isinstance(geofence, list)
        assert len(geofence) > 0

    def test_point_in_geofence(self):
        """Test checking if point is in geofence"""
        processor = GeofenceProcessor()

        center = StandardCoordinate(latitude=40.7128, longitude=-74.0060)
        geofence = processor.create_geofence(center, radius_meters=1000)

        # Point inside geofence
        inside_point = StandardCoordinate(latitude=40.7130, longitude=-74.0062)
        assert processor.point_in_geofence(inside_point, geofence) is True

        # Point outside geofence
        outside_point = StandardCoordinate(latitude=41.7128, longitude=-74.0060)
        assert processor.point_in_geofence(outside_point, geofence) is False


class TestSatelliteAnalyzer:
    """Test SatelliteAnalyzer functionality"""

    @patch('lemkin_geo.core.requests')
    def test_analyze_satellite_imagery(self, mock_requests):
        """Test satellite imagery analysis"""
        analyzer = SatelliteAnalyzer()

        # Mock satellite data response
        mock_response = Mock()
        mock_response.json.return_value = {
            "images": [
                {
                    "id": "sat-001",
                    "date": "2024-01-01",
                    "resolution": 10,
                    "cloud_coverage": 5
                }
            ]
        }
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response

        bbox = BoundingBox(
            north=40.8,
            south=40.6,
            east=-73.9,
            west=-74.1
        )

        date_range = {
            "start": datetime(2024, 1, 1),
            "end": datetime(2024, 1, 31)
        }

        result = analyzer.analyze_satellite_imagery(bbox, date_range)

        assert result is not None
        assert len(result.imagery_sources) > 0

    def test_detect_changes(self):
        """Test change detection in imagery"""
        analyzer = SatelliteAnalyzer()

        # Mock before and after images
        before_image = Mock()
        after_image = Mock()

        changes = analyzer.detect_changes(before_image, after_image)

        assert changes is not None


class TestMappingGenerator:
    """Test MappingGenerator functionality"""

    def test_generate_evidence_map(self):
        """Test evidence map generation"""
        generator = MappingGenerator()

        evidence = [
            GeoEvent(
                event_id="evidence-1",
                event_type=EventType.EVIDENCE,
                coordinate=StandardCoordinate(latitude=40.7128, longitude=-74.0060),
                timestamp=datetime.now(timezone.utc),
                description="Evidence 1"
            ),
            GeoEvent(
                event_id="evidence-2",
                event_type=EventType.WITNESS,
                coordinate=StandardCoordinate(latitude=40.7580, longitude=-73.9855),
                timestamp=datetime.now(timezone.utc),
                description="Evidence 2"
            )
        ]

        evidence_map = generator.generate_evidence_map(evidence)

        assert evidence_map is not None
        assert evidence_map.center_lat is not None
        assert evidence_map.center_lon is not None
        assert len(evidence_map.markers) == 2
        assert evidence_map.map_html is not None

    def test_add_heatmap_layer(self):
        """Test adding heatmap layer to map"""
        generator = MappingGenerator()

        events = [
            GeoEvent(
                event_id=f"event-{i}",
                event_type=EventType.INCIDENT,
                coordinate=StandardCoordinate(
                    latitude=40.7128 + i * 0.01,
                    longitude=-74.0060 + i * 0.01
                ),
                timestamp=datetime.now(timezone.utc)
            )
            for i in range(5)
        ]

        evidence_map = generator.generate_evidence_map(events)
        heatmap = generator.add_heatmap_layer(evidence_map, events)

        assert heatmap is not None
        assert "heatmap" in evidence_map.layers

    def test_export_map(self, tmp_path):
        """Test map export functionality"""
        generator = MappingGenerator()

        evidence = [
            GeoEvent(
                event_id="test-1",
                event_type=EventType.EVIDENCE,
                coordinate=StandardCoordinate(latitude=40.7128, longitude=-74.0060),
                timestamp=datetime.now(timezone.utc)
            )
        ]

        evidence_map = generator.generate_evidence_map(evidence)

        # Test HTML export
        html_path = tmp_path / "test_map.html"
        generator.export_map(evidence_map, str(html_path), format="html")
        assert html_path.exists()

        # Test GeoJSON export
        geojson_path = tmp_path / "test_map.geojson"
        generator.export_map(evidence_map, str(geojson_path), format="geojson")
        assert geojson_path.exists()


class TestGeospatialAnalyzer:
    """Test main GeospatialAnalyzer class"""

    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = GeospatialAnalyzer()

        assert analyzer.coordinate_converter is not None
        assert analyzer.satellite_analyzer is not None
        assert analyzer.geofence_processor is not None
        assert analyzer.mapping_generator is not None

    def test_full_analysis_workflow(self, tmp_path):
        """Test complete geospatial analysis workflow"""
        analyzer = GeospatialAnalyzer()

        # Create test events
        events = [
            GeoEvent(
                event_id=f"event-{i}",
                event_type=EventType.INCIDENT,
                coordinate=StandardCoordinate(
                    latitude=40.7128 + i * 0.001,
                    longitude=-74.0060 + i * 0.001
                ),
                timestamp=datetime.now(timezone.utc),
                description=f"Test event {i}"
            )
            for i in range(3)
        ]

        # Test correlation
        correlation = analyzer.correlate_events(events, radius_meters=500)
        assert correlation is not None
        assert len(correlation.correlated_clusters) > 0

        # Test map generation
        evidence_map = analyzer.create_map(events)
        assert evidence_map is not None

        # Test map export
        output_path = tmp_path / "analysis_map.html"
        analyzer.export_map(evidence_map, str(output_path))
        assert output_path.exists()