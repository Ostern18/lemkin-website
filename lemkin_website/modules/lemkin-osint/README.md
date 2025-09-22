# Lemkin OSINT Collection Toolkit

## Purpose

The Lemkin OSINT Collection Toolkit provides systematic open-source intelligence gathering capabilities for legal investigations and human rights work. This toolkit enables ethical collection of publicly available information while respecting platform terms of service and privacy considerations.

## Safety & Ethics Notice

⚠️ **IMPORTANT**: This toolkit is designed for legitimate legal investigations and human rights documentation. Users must:
- Respect platform terms of service and rate limits
- Obtain proper legal authorization for investigations
- Protect witness and source confidentiality
- Follow applicable privacy laws and regulations
- Use collected data only for lawful purposes
- Verify information from multiple sources

## Key Features

- **Ethical Collection**: Respects platform ToS and rate limits
- **Web Archiving**: Integration with Wayback Machine for content preservation
- **Metadata Extraction**: EXIF and XMP extraction from media files
- **Source Verification**: Credibility assessment of OSINT sources
- **GPS Extraction**: Location data from image metadata
- **Batch Processing**: Handle multiple sources efficiently

## Quick Start

```bash
# Install the toolkit
pip install lemkin-osint

# Collect OSINT data
lemkin-osint collect "search query" --platforms wayback --limit 100

# Archive web content
lemkin-osint archive "https://example.com,https://example2.com"

# Extract media metadata
lemkin-osint extract-metadata image.jpg --show-gps

# Verify source credibility
lemkin-osint verify-source "https://news-site.com/article"
```

## Usage Examples

### 1. Collecting Evidence from Archives

```bash
# Search Wayback Machine for historical content
lemkin-osint collect "human rights violation site:example.org" \
    --platforms wayback \
    --limit 50 \
    --output evidence_collection.json
```

### 2. Preserving Web Evidence

```bash
# Archive current web pages for future reference
lemkin-osint archive \
    "https://witness-testimony.org/page1,https://evidence-site.com/report" \
    --output archive_results.json
```

### 3. Extracting Location from Images

```bash
# Extract GPS coordinates and camera information
lemkin-osint extract-metadata witness_photo.jpg \
    --show-gps \
    --output photo_metadata.json
```

### 4. Batch Source Verification

```bash
# Verify credibility of multiple sources
echo '["https://source1.com", "https://source2.org"]' > sources.json
lemkin-osint batch-verify sources.json --output verification_results.json
```

## Input/Output Specifications

### OSINT Collection Format
```python
{
    "collection_id": "uuid",
    "query": "search query",
    "platforms": ["wayback"],
    "sources": [
        {
            "url": "https://example.com",
            "title": "Page Title",
            "source_type": "website",
            "domain": "example.com",
            "collected_date": "2024-01-15T10:30:00Z"
        }
    ],
    "total_items": 10,
    "collected_date": "2024-01-15T10:30:00Z"
}
```

### Media Metadata Format
```python
{
    "file_path": "/path/to/image.jpg",
    "file_hash": "sha256_hash",
    "file_size": 2048576,
    "mime_type": "image/jpeg",
    "creation_date": "2024-01-10T14:30:00Z",
    "exif_data": {
        "Make": "Camera Brand",
        "Model": "Camera Model",
        "GPSLatitude": "40.7128",
        "GPSLongitude": "-74.0060"
    },
    "gps_data": {
        "decimal_coordinates": {
            "latitude": 40.7128,
            "longitude": -74.0060
        }
    }
}
```

## API Reference

### Core Classes

#### OSINTCollector
Manages ethical OSINT collection from various platforms.

```python
from lemkin_osint import OSINTCollector

collector = OSINTCollector()
collection = collector.collect_social_media_evidence(
    query="search terms",
    platforms=["wayback"],
    limit=100
)
```

#### MetadataExtractor
Extracts EXIF and XMP metadata from media files.

```python
from lemkin_osint import MetadataExtractor
from pathlib import Path

extractor = MetadataExtractor()
metadata = extractor.extract_media_metadata(Path("image.jpg"))

if metadata.gps_data:
    coords = metadata.gps_data.get('decimal_coordinates')
    print(f"Location: {coords['latitude']}, {coords['longitude']}")
```

#### SourceVerifier
Assesses credibility of OSINT sources.

```python
from lemkin_osint import SourceVerifier, Source

verifier = SourceVerifier()
source = Source(url="https://news-site.com/article")
assessment = verifier.verify_source_credibility(source)

print(f"Credibility: {assessment.credibility_level}")
print(f"Confidence: {assessment.confidence_score:.1%}")
```

## Evaluation & Limitations

### Performance Metrics
- Metadata extraction: ~100 images/minute
- Source verification: ~50 sources/minute
- Web archiving: Limited by Wayback Machine API rates

### Known Limitations
- Platform API rate limits apply
- Some platforms require API keys (not included)
- GPS extraction depends on EXIF availability
- Wayback Machine may not have all content
- Source verification is heuristic-based

### Failure Modes
- Network timeouts: Retry with exponential backoff
- API rate limits: Implement proper throttling
- Missing metadata: Not all files contain EXIF
- Archive failures: Some content cannot be archived

## Safety Guidelines

### Data Collection Ethics
1. **Respect Privacy**: Never collect private or protected information
2. **Platform Compliance**: Follow all terms of service
3. **Rate Limiting**: Implement appropriate delays
4. **Source Attribution**: Always maintain source records
5. **Verification**: Cross-reference multiple sources

### Legal Compliance
- Obtain proper authorization for investigations
- Follow data protection regulations (GDPR, etc.)
- Respect copyright and intellectual property
- Document chain of custody for evidence
- Preserve original data integrity

### Operational Security
1. **Use VPN/Tor**: Protect investigator identity when appropriate
2. **Secure Storage**: Encrypt sensitive collected data
3. **Access Control**: Limit access to investigation data
4. **Audit Trails**: Log all collection activities
5. **Data Disposal**: Securely delete data when required

## Contributing

We welcome contributions that enhance OSINT collection capabilities while maintaining ethical standards.

### Development Setup
```bash
# Clone repository
git clone https://github.com/lemkin-org/lemkin-osint.git
cd lemkin-osint

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
```

## License

Apache License 2.0 - see LICENSE file for details.

This toolkit is designed for legitimate legal investigations and human rights documentation. Users are responsible for ensuring compliance with all applicable laws and ethical guidelines.

---

*Part of the Lemkin AI open-source legal technology ecosystem.*