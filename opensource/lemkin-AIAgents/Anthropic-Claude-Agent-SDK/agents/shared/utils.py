"""
Utility functions for LemkinAI Agents
Common helper functions used across agents.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text using common patterns.

    Args:
        text: Text to extract dates from

    Returns:
        List of date strings found
    """
    # Common date patterns
    patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # DD Month YYYY
    ]

    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)

    return dates


def extract_names(text: str) -> List[str]:
    """
    Extract potential names from text (simplified pattern matching).

    Args:
        text: Text to extract names from

    Returns:
        List of potential names
    """
    # Pattern: Capitalized words (2-3 words)
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b'
    names = re.findall(pattern, text)

    # Filter out common non-names
    common_words = {'The', 'This', 'That', 'These', 'Those', 'Where', 'When', 'What', 'Which', 'Who'}
    names = [name for name in names if not any(word in common_words for word in name.split())]

    return list(set(names))  # Remove duplicates


def extract_locations(text: str) -> List[str]:
    """
    Extract potential locations from text (simplified pattern matching).

    Args:
        text: Text to extract locations from

    Returns:
        List of potential locations
    """
    # Pattern: Capitalized words potentially followed by geographic terms
    geo_terms = ['city', 'town', 'village', 'province', 'district', 'region', 'country', 'street', 'road', 'avenue']
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:' + '|'.join(geo_terms) + r')\b'

    locations = re.findall(pattern, text, re.IGNORECASE)
    return list(set(locations))


def calculate_confidence_score(
    indicators: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate weighted confidence score from multiple indicators.

    Args:
        indicators: Dictionary of indicator names to scores (0-1)
        weights: Optional weights for each indicator (defaults to equal weighting)

    Returns:
        Overall confidence score (0-1)
    """
    if not indicators:
        return 0.0

    if weights is None:
        # Equal weighting
        weights = {k: 1.0 for k in indicators.keys()}

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted score
    score = sum(indicators.get(k, 0) * normalized_weights.get(k, 0) for k in indicators.keys())

    return max(0.0, min(1.0, score))  # Clamp to [0, 1]


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp in ISO 8601 format (UTC).

    Args:
        timestamp: Datetime to format (defaults to now)

    Returns:
        ISO 8601 formatted string
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return timestamp.isoformat()


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """
    Parse ISO 8601 timestamp string.

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        Datetime object
    """
    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem usage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(unsafe_chars, '_', filename)

    # Limit length
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        name = name[:max_length - len(ext) - 1]
        sanitized = f"{name}.{ext}" if ext else name

    return sanitized


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at word boundary
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:  # At least 80% of chunk size
                end = start + last_space

        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later dictionaries taking precedence.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def validate_evidence_id(evidence_id: str) -> bool:
    """
    Validate evidence ID format (UUID).

    Args:
        evidence_id: Evidence ID to validate

    Returns:
        True if valid UUID format, False otherwise
    """
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, evidence_id, re.IGNORECASE))


def classify_confidence_level(score: float) -> str:
    """
    Classify numeric confidence score into categorical level.

    Args:
        score: Confidence score (0-1)

    Returns:
        Confidence level (low/medium/high/very_high)
    """
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    elif score < 0.85:
        return "high"
    else:
        return "very_high"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
