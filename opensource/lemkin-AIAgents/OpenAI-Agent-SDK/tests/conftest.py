"""
Shared pytest fixtures for LemkinAI Anthropic Claude SDK tests.

Provides common test fixtures including:
- Mock infrastructure (AuditLogger, EvidenceHandler)
- Mock API responses
- Test data generators
- Shared test utilities
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
import json
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

from shared import AuditLogger, EvidenceHandler, OutputFormatter
from shared.audit_logger import AuditEventType
from shared.evidence_handler import EvidenceType, EvidenceStatus


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def audit_logger(temp_directory):
    """Create an AuditLogger instance with temporary storage."""
    return AuditLogger(log_directory=temp_directory / "audit_logs")


@pytest.fixture
def evidence_handler(temp_directory):
    """Create an EvidenceHandler instance with temporary storage."""
    return EvidenceHandler(storage_directory=temp_directory / "evidence_store")


@pytest.fixture
def output_formatter():
    """Create an OutputFormatter instance."""
    return OutputFormatter()


@pytest.fixture
def shared_infrastructure(audit_logger, evidence_handler):
    """
    Create shared infrastructure for multi-agent testing.

    This fixture provides the standard infrastructure that all agents
    share for evidentiary compliance.
    """
    return {
        'audit_logger': audit_logger,
        'evidence_handler': evidence_handler
    }


@pytest.fixture
def mock_claude_response():
    """
    Create a mock Claude API response.

    Returns a factory function that generates mock responses.
    """
    def _create_response(
        content_text: str = "Test response",
        input_tokens: int = 100,
        output_tokens: int = 200,
        stop_reason: str = "end_turn"
    ):
        """Create a mock Claude response."""
        mock_response = MagicMock()

        # Create content block
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = content_text
        mock_response.content = [mock_content]

        # Create usage stats
        mock_usage = MagicMock()
        mock_usage.input_tokens = input_tokens
        mock_usage.output_tokens = output_tokens
        mock_response.usage = mock_usage

        # Other properties
        mock_response.stop_reason = stop_reason
        mock_response.model = "claude-sonnet-4-5-20250929"

        return mock_response

    return _create_response


@pytest.fixture
def mock_anthropic_client(mock_claude_response):
    """
    Create a mock Anthropic client for testing without API calls.

    This prevents actual API calls during testing and provides
    consistent responses.
    """
    with patch('anthropic.Anthropic') as mock_client_class:
        mock_client = MagicMock()
        mock_messages = MagicMock()

        # Default response
        mock_messages.create.return_value = mock_claude_response()

        mock_client.messages = mock_messages
        mock_client_class.return_value = mock_client

        yield mock_client


@pytest.fixture
def sample_pdf_data():
    """Generate sample PDF data for testing."""
    return b"%PDF-1.4\n%Mock PDF content for testing\n%%EOF"


@pytest.fixture
def sample_image_data():
    """Generate sample image data for testing."""
    # Minimal valid JPEG header
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


@pytest.fixture
def sample_evidence_data():
    """
    Generate sample evidence for testing.

    Returns a factory function that creates test evidence.
    """
    def _create_evidence(
        evidence_type: EvidenceType = EvidenceType.DOCUMENT_PDF,
        source: str = "Test Source",
        case_id: str = "TEST-001",
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Create sample evidence data."""
        file_data = b"Sample evidence file content"

        return {
            'file_data': file_data,
            'evidence_type': evidence_type,
            'source': source,
            'original_filename': 'test_evidence.pdf',
            'collected_by': 'Test Investigator',
            'collected_date': '2024-01-15T10:00:00Z',
            'case_id': case_id,
            'tags': tags or ['test', 'sample'],
            'mime_type': 'application/pdf'
        }

    return _create_evidence


@pytest.fixture
def sample_medical_record():
    """Sample medical record text for testing."""
    return """
    MEDICAL EXAMINATION REPORT

    Patient: John Doe
    Date: January 15, 2024
    Examiner: Dr. Jane Smith

    CHIEF COMPLAINT:
    Patient presents with multiple injuries following detention.

    PHYSICAL EXAMINATION:
    - Multiple contusions on upper back and shoulders
    - Linear marks consistent with impact from blunt object
    - Patient reports severe pain in lower back region
    - Evidence of stress positioning injuries to wrists and ankles

    ASSESSMENT:
    Injuries are consistent with physical abuse. Pattern suggests
    systematic application of blunt force trauma.

    RECOMMENDATIONS:
    Further documentation required. Psychiatric evaluation recommended.
    Istanbul Protocol documentation initiated.
    """


@pytest.fixture
def sample_osint_sources():
    """Sample OSINT sources for testing."""
    return [
        {
            'url': 'https://example.com/social-media-post-1',
            'platform': 'Twitter',
            'author': 'witness_account',
            'timestamp': '2024-01-15T14:30:00Z',
            'content': 'Witnessed military vehicles near detention center',
            'verified': False
        },
        {
            'url': 'https://example.com/news-article',
            'platform': 'News',
            'source': 'Local News Outlet',
            'timestamp': '2024-01-16T08:00:00Z',
            'content': 'Reports of civilian arrests in downtown area',
            'verified': True
        },
        {
            'url': 'https://example.com/video-evidence',
            'platform': 'YouTube',
            'author': 'citizen_journalist',
            'timestamp': '2024-01-15T16:45:00Z',
            'content': 'Video showing detention facility activity',
            'verified': False
        }
    ]


@pytest.fixture
def sample_legal_framework():
    """Sample legal framework data for testing."""
    return {
        'jurisdiction': 'International Criminal Court',
        'applicable_laws': [
            'Rome Statute Article 7 (Crimes Against Humanity)',
            'Rome Statute Article 8 (War Crimes)',
            'Geneva Conventions',
            'Convention Against Torture'
        ],
        'relevant_precedents': [
            'Prosecutor v. Tadić (ICTY)',
            'Prosecutor v. Akayesu (ICTR)'
        ]
    }


@pytest.fixture
def sample_satellite_imagery_metadata():
    """Sample satellite imagery metadata for testing."""
    return {
        'image_date': '2024-01-15',
        'coordinates': {'lat': 35.6892, 'lon': 51.3890},
        'resolution': '0.5m',
        'satellite': 'WorldView-3',
        'cloud_cover': '5%',
        'sun_elevation': '45°',
        'sensor_type': 'Multispectral'
    }


@pytest.fixture
def sample_chain_of_custody():
    """Sample chain of custody data for testing verification."""
    def _create_chain(evidence_id: str, num_events: int = 5) -> List[Dict[str, Any]]:
        """Create a sample chain of custody with N events."""
        chain = []
        previous_hash = None

        event_types = [
            AuditEventType.EVIDENCE_INGESTED,
            AuditEventType.EVIDENCE_PROCESSED,
            AuditEventType.ANALYSIS_PERFORMED,
            AuditEventType.OUTPUT_GENERATED,
            AuditEventType.CHAIN_VERIFIED
        ]

        for i in range(num_events):
            event = {
                'event_id': f'event_{i}',
                'timestamp': f'2024-01-15T{10+i}:00:00Z',
                'event_type': event_types[i % len(event_types)].value,
                'agent_id': f'agent_{i % 3}',
                'evidence_ids': [evidence_id],
                'details': {'step': i},
                'previous_event_hash': previous_hash
            }

            # Calculate hash
            event_json = json.dumps(event, sort_keys=True)
            event_hash = hashlib.sha256(event_json.encode()).hexdigest()
            event['event_hash'] = event_hash

            chain.append(event)
            previous_hash = event_hash

        return chain

    return _create_chain


@pytest.fixture
def mock_human_review_approved():
    """Mock human review that approves the request."""
    def _approve_review(review_request_id: str) -> Dict[str, Any]:
        return {
            'review_request_id': review_request_id,
            'decision': 'approved',
            'reviewer_id': 'test_reviewer',
            'notes': 'Analysis is accurate and well-supported',
            'reviewed_at': '2024-01-15T12:00:00Z'
        }
    return _approve_review


@pytest.fixture
def mock_human_review_rejected():
    """Mock human review that rejects the request."""
    def _reject_review(review_request_id: str) -> Dict[str, Any]:
        return {
            'review_request_id': review_request_id,
            'decision': 'rejected',
            'reviewer_id': 'test_reviewer',
            'notes': 'Insufficient evidence for conclusion',
            'reviewed_at': '2024-01-15T12:00:00Z'
        }
    return _reject_review


# Parametrized fixtures for testing multiple scenarios

@pytest.fixture(params=[
    EvidenceType.DOCUMENT_PDF,
    EvidenceType.PHOTO,
    EvidenceType.MEDICAL_RECORD,
    EvidenceType.WITNESS_STATEMENT
])
def various_evidence_types(request):
    """Parametrized fixture providing various evidence types."""
    return request.param


@pytest.fixture(params=[
    'torture',
    'genocide',
    'war_crimes',
    'crimes_against_humanity'
])
def various_crime_types(request):
    """Parametrized fixture providing various crime types."""
    return request.param


@pytest.fixture(params=[0.0, 0.2, 0.5, 1.0])
def various_temperatures(request):
    """Parametrized fixture providing various temperature settings."""
    return request.param


# Helper utilities

class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def create_mock_document(
        content: str,
        doc_type: str = 'pdf',
        metadata: Dict[str, Any] = None
    ) -> bytes:
        """Create a mock document with specified content."""
        if doc_type == 'pdf':
            header = b"%PDF-1.4\n"
            footer = b"\n%%EOF"
            return header + content.encode() + footer
        else:
            return content.encode()

    @staticmethod
    def create_mock_case(
        case_id: str = "TEST-001",
        num_evidence_items: int = 5
    ) -> Dict[str, Any]:
        """Create a complete mock case with multiple evidence items."""
        return {
            'case_id': case_id,
            'case_name': f'Investigation {case_id}',
            'charges': ['torture', 'unlawful_detention'],
            'evidence_count': num_evidence_items,
            'investigators': ['Investigator A', 'Investigator B'],
            'status': 'active'
        }


@pytest.fixture
def test_data_generator():
    """Provide TestDataGenerator utility."""
    return TestDataGenerator()


# Error simulation fixtures

@pytest.fixture
def mock_api_error():
    """Simulate API errors for error handling tests."""
    def _create_error(error_type: str = "rate_limit"):
        """Create mock API error."""
        if error_type == "rate_limit":
            from anthropic import RateLimitError
            return RateLimitError("Rate limit exceeded")
        elif error_type == "authentication":
            from anthropic import AuthenticationError
            return AuthenticationError("Invalid API key")
        elif error_type == "timeout":
            import requests
            return requests.exceptions.Timeout("Request timed out")
        else:
            return Exception("Generic API error")

    return _create_error


# Performance testing fixtures

@pytest.fixture
def performance_threshold():
    """Define performance thresholds for testing."""
    return {
        'max_response_time_seconds': 30,
        'max_tokens_per_request': 200000,
        'max_memory_mb': 1024,
        'min_accuracy_threshold': 0.85
    }
