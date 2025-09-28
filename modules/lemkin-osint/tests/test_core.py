"""
Tests for OSINT core functionality.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from lemkin_osint.core import (
    OSINTCollector,
    WebArchiver,
    MetadataExtractor,
    SourceVerifier,
    Source,
    SourceType,
    CredibilityLevel,
    OSINTCollection,
    MediaMetadata,
)


class TestOSINTCollector:
    """Test OSINT collector functionality"""

    def test_initialization(self):
        """Test collector initialization"""
        collector = OSINTCollector()
        assert collector.user_agent.startswith("Lemkin-OSINT")
        assert collector.client is not None

    def test_custom_user_agent(self):
        """Test custom user agent"""
        custom_ua = "Custom-Agent/1.0"
        collector = OSINTCollector(user_agent=custom_ua)
        assert collector.user_agent == custom_ua

    def test_collect_social_media_evidence(self):
        """Test social media evidence collection"""
        collector = OSINTCollector()

        collection = collector.collect_social_media_evidence(
            query="test query",
            platforms=["wayback"],
            limit=10
        )

        assert isinstance(collection, OSINTCollection)
        assert collection.query == "test query"
        assert "wayback" in collection.platforms
        assert collection.total_items >= 0

    @patch('lemkin_osint.core.OSINTCollector._search_wayback')
    def test_collect_with_mock_wayback(self, mock_search):
        """Test collection with mocked Wayback search"""
        mock_search.return_value = [
            Source(
                url="https://example.com/test",
                title="Test Page",
                source_type=SourceType.WEBSITE
            )
        ]

        collector = OSINTCollector()
        collection = collector.collect_social_media_evidence(
            query="test",
            platforms=["wayback"],
            limit=10
        )

        assert collection.total_items == 1
        assert collection.sources[0].url == "https://example.com/test"


class TestWebArchiver:
    """Test web archiving functionality"""

    def test_initialization(self):
        """Test archiver initialization"""
        archiver = WebArchiver()
        assert archiver.user_agent.startswith("Lemkin-WebArchiver")

    @patch('waybackpy.Url')
    def test_archive_web_content(self, mock_wayback):
        """Test web content archiving"""
        # Mock Wayback Machine responses
        mock_archive = Mock()
        mock_archive.archive_url = "https://web.archive.org/web/20240101000000/https://example.com"
        mock_archive.timestamp = "20240101000000"

        mock_wayback_instance = Mock()
        mock_wayback_instance.save.return_value = mock_archive
        mock_wayback_instance.cdx_api.return_value = []

        mock_wayback.return_value = mock_wayback_instance

        archiver = WebArchiver()
        collection = archiver.archive_web_content(["https://example.com"])

        assert len(collection.archived_items) == 1
        assert collection.archived_items[0]["status"] == "success"
        assert "archive_url" in collection.archived_items[0]

    @patch('waybackpy.Url')
    def test_archive_failure_handling(self, mock_wayback):
        """Test handling of archive failures"""
        mock_wayback.side_effect = Exception("Archive failed")

        archiver = WebArchiver()
        collection = archiver.archive_web_content(["https://example.com"])

        assert len(collection.archived_items) == 1
        assert collection.archived_items[0]["status"] == "failed"
        assert "error" in collection.archived_items[0]


class TestMetadataExtractor:
    """Test metadata extraction functionality"""

    def test_initialization(self):
        """Test extractor initialization"""
        extractor = MetadataExtractor()
        assert extractor is not None

    def test_extract_metadata_nonexistent_file(self):
        """Test extraction with nonexistent file"""
        extractor = MetadataExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract_media_metadata(Path("/nonexistent/file.jpg"))

    def test_extract_metadata_text_file(self):
        """Test metadata extraction from text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)

        try:
            extractor = MetadataExtractor()
            metadata = extractor.extract_media_metadata(temp_path)

            assert isinstance(metadata, MediaMetadata)
            assert metadata.file_path == temp_path
            assert metadata.file_size > 0
            assert len(metadata.file_hash) == 64  # SHA-256 hash length
            assert metadata.mime_type == "text/plain"

        finally:
            temp_path.unlink(missing_ok=True)

    @patch('PIL.Image.open')
    def test_extract_image_metadata_with_exif(self, mock_image):
        """Test extraction from image with EXIF data"""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = Path(f.name)

        # Mock image with EXIF
        mock_img_instance = Mock()
        mock_img_instance._getexif.return_value = {
            271: "TestCamera",  # Make
            272: "Model X",     # Model
            305: "TestSoftware",  # Software
            36867: "2024:01:01 12:00:00",  # DateTimeOriginal
        }
        mock_img_instance.close = Mock()
        mock_image.return_value = mock_img_instance

        try:
            extractor = MetadataExtractor()
            metadata = extractor.extract_media_metadata(temp_path)

            assert metadata.camera_info == {"make": "TestCamera", "model": "Model X"}
            assert metadata.software_info == "TestSoftware"
            assert metadata.creation_date is not None
            assert len(metadata.exif_data) > 0

        finally:
            temp_path.unlink(missing_ok=True)

    def test_gps_coordinate_conversion(self):
        """Test GPS coordinate conversion"""
        extractor = MetadataExtractor()

        # Test conversion of GPS coordinates
        gps_coords = (40, 30, 0)  # 40 degrees, 30 minutes, 0 seconds
        result = extractor._convert_to_degrees(gps_coords)
        assert abs(result - 40.5) < 0.001  # 40.5 degrees


class TestSourceVerifier:
    """Test source verification functionality"""

    def test_initialization(self):
        """Test verifier initialization"""
        verifier = SourceVerifier()
        assert len(verifier.trusted_domains) > 0
        assert len(verifier.suspicious_indicators) > 0

    def test_verify_trusted_source(self):
        """Test verification of trusted source"""
        source = Source(
            url="https://reuters.com/article/test",
            title="News Article",
            source_type=SourceType.NEWS_ARTICLE
        )

        verifier = SourceVerifier()
        assessment = verifier.verify_source_credibility(source)

        assert assessment.credibility_level in [CredibilityLevel.HIGH, CredibilityLevel.MEDIUM]
        assert assessment.confidence_score > 0.5
        assert any(i.indicator == "trusted_domain" for i in assessment.indicators)

    def test_verify_suspicious_source(self):
        """Test verification of suspicious source"""
        source = Source(
            url="http://suspicious-site.com/fake-news",
            title="FAKE NEWS CLICKBAIT",
            source_type=SourceType.UNKNOWN
        )

        verifier = SourceVerifier()
        assessment = verifier.verify_source_credibility(source)

        assert assessment.credibility_level in [CredibilityLevel.LOW, CredibilityLevel.SUSPICIOUS]
        assert any(i.indicator == "suspicious_content" for i in assessment.indicators)
        assert len(assessment.concerns) > 0

    def test_verify_government_source(self):
        """Test verification of government source"""
        source = Source(
            url="https://un.org/document",
            title="UN Report",
            source_type=SourceType.GOVERNMENT
        )

        verifier = SourceVerifier()
        assessment = verifier.verify_source_credibility(source)

        assert assessment.credibility_level in [CredibilityLevel.HIGH, CredibilityLevel.MEDIUM]
        assert any(i.indicator == "authoritative_type" for i in assessment.indicators)
        assert any(i.indicator == "secure_connection" for i in assessment.indicators)

    def test_https_indicator(self):
        """Test HTTPS security indicator"""
        source_http = Source(url="http://example.com")
        source_https = Source(url="https://example.com")

        verifier = SourceVerifier()

        assessment_http = verifier.verify_source_credibility(source_http)
        assessment_https = verifier.verify_source_credibility(source_https)

        assert "No HTTPS encryption" in assessment_http.concerns
        assert any(i.indicator == "secure_connection" for i in assessment_https.indicators)