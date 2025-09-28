# Lemkin Image Verification Suite

## Purpose

The Lemkin Image Verification Suite provides comprehensive image authenticity verification and manipulation detection capabilities for legal investigations. This toolkit enables investigators to detect image manipulation, perform reverse image searches, analyze metadata, and attempt geolocation from visual content.

## Safety & Ethics Notice

⚠️ **IMPORTANT**: This toolkit is designed for legitimate legal investigations and human rights documentation. Users must:
- Obtain proper legal authorization before analyzing image evidence
- Respect privacy rights and intellectual property laws
- Maintain chain of custody for all image evidence
- Use detection results as investigative leads, not definitive proof
- Understand the limitations of automated detection systems
- Protect the privacy of individuals in images

## Key Features

- **Manipulation Detection**: Detect splicing, copy-move forgery, enhancement, upscaling, and AI generation
- **Reverse Image Search**: Search across multiple engines to find image sources and usage
- **Metadata Forensics**: Extract and analyze EXIF data for authenticity indicators
- **Visual Geolocation**: Attempt to determine location from visual content and landmarks
- **Comprehensive Analysis**: Combine all techniques for complete image verification
- **Batch Processing**: Analyze multiple images efficiently

## Quick Start

```bash
# Install the toolkit
pip install lemkin-images

# Detect image manipulation
lemkin-images detect-manipulation suspicious_photo.jpg --output analysis.json

# Perform reverse image search
lemkin-images reverse-search evidence_photo.jpg --engines "google,bing" --output search.json

# Analyze metadata for forensic indicators
lemkin-images analyze-metadata photo.jpg --show-exif --output metadata.json

# Attempt geolocation from visual content
lemkin-images geolocate location_photo.jpg --output geolocation.json

# Comprehensive analysis
lemkin-images comprehensive-analysis photo.jpg \
    --include-search --include-geolocation --output full_analysis.json
```

## Usage Examples

### 1. Manipulation Detection

```bash
# Detect various types of manipulation
lemkin-images detect-manipulation witness_photo.jpg \
    --output manipulation_report.json \
    --detailed
```

### 2. Reverse Image Search Investigation

```bash
# Search multiple engines for image sources
lemkin-images reverse-search evidence_image.jpg \
    --engines "google,bing,yandex" \
    --limit 50 \
    --output search_results.json
```

### 3. Metadata Forensics Analysis

```bash
# Extract all metadata and EXIF information
lemkin-images analyze-metadata digital_photo.jpg \
    --show-exif \
    --output metadata_analysis.json
```

### 4. Visual Geolocation

```bash
# Attempt to determine location from image content
lemkin-images geolocate scene_photo.jpg \
    --show-features \
    --output location_analysis.json
```

### 5. Complete Image Investigation Workflow

```bash
# Step 1: Basic manipulation check
lemkin-images detect-manipulation evidence.jpg --output step1_manipulation.json

# Step 2: Search for image sources
lemkin-images reverse-search evidence.jpg --engines "google,bing" --output step2_search.json

# Step 3: Analyze metadata
lemkin-images analyze-metadata evidence.jpg --show-exif --output step3_metadata.json

# Step 4: Comprehensive analysis
lemkin-images comprehensive-analysis evidence.jpg \
    --include-search --include-geolocation \
    --output final_report.json
```

## Input/Output Specifications

### Image Metadata Format
```python
{
    "file_path": "/path/to/image.jpg",
    "file_size": 1048576,
    "dimensions": [1920, 1080],
    "format": "JPEG",
    "mode": "RGB",
    "creation_date": "2024-01-15T10:30:00Z",
    "camera_make": "Canon",
    "camera_model": "EOS R5",
    "gps_coordinates": [40.7128, -74.0060],
    "exif_data": {
        "DateTime": "2024:01:15 10:30:00",
        "Make": "Canon",
        "Model": "EOS R5"
    },
    "file_hash": "abc123..."
}
```

### Manipulation Analysis Format
```python
{
    "analysis_id": "uuid",
    "image_path": "/path/to/image.jpg",
    "is_manipulated": true,
    "manipulation_types": ["splicing", "enhancement"],
    "confidence_score": 0.85,
    "manipulation_regions": [
        {
            "source_point": [100, 200],
            "target_point": [300, 400],
            "distance": 250.5
        }
    ],
    "analysis_details": {
        "splicing_detected": true,
        "enhancement_detected": true
    }
}
```

### Reverse Search Results Format
```python
{
    "search_id": "uuid",
    "query_image": "/path/to/image.jpg",
    "search_engine": "google",
    "results": [
        {
            "url": "https://example.com/article",
            "title": "News article with image",
            "source_domain": "example.com",
            "similarity_score": 0.95
        }
    ],
    "total_results": 15,
    "query_hash": "phash_string"
}
```

## API Reference

### Core Classes

#### ManipulationDetector
Detects various types of image manipulation and forgery.

```python
from lemkin_images import ManipulationDetector
from pathlib import Path

detector = ManipulationDetector()
analysis = detector.detect_image_manipulation(Path("image.jpg"))

if analysis.is_manipulated:
    print(f"Manipulation detected: {analysis.manipulation_types}")
    print(f"Confidence: {analysis.confidence_score:.1%}")
```

#### ReverseImageSearcher
Performs reverse image searches across multiple engines.

```python
from lemkin_images import ReverseImageSearcher, SearchEngine
from pathlib import Path

searcher = ReverseImageSearcher()
results = searcher.reverse_search_image(
    Path("image.jpg"),
    engines=[SearchEngine.GOOGLE, SearchEngine.BING],
    limit=20
)

print(f"Found {results.total_results} results")
for result in results.results[:5]:
    print(f"- {result.url} (similarity: {result.similarity_score:.1%})")
```

#### GeolocationHelper
Attempts to geolocate images from visual content.

```python
from lemkin_images import GeolocationHelper
from pathlib import Path

helper = GeolocationHelper()
result = helper.geolocate_image(Path("image.jpg"))

if result.estimated_location:
    lat, lon = result.estimated_location
    print(f"Estimated location: {lat:.6f}, {lon:.6f}")
    print(f"Confidence: {result.location_confidence:.1%}")
```

#### MetadataForensics
Analyzes image metadata for forensic indicators.

```python
from lemkin_images import MetadataForensics
from pathlib import Path

forensics = MetadataForensics()
metadata = forensics.analyze_image_metadata(Path("image.jpg"))

print(f"Camera: {metadata.camera_make} {metadata.camera_model}")
if metadata.gps_coordinates:
    lat, lon = metadata.gps_coordinates
    print(f"GPS: {lat:.6f}, {lon:.6f}")
```

## Detection Capabilities

### Manipulation Detection Techniques
1. **Splicing Detection**: Analyzes noise patterns to detect combined images
2. **Copy-Move Detection**: Uses feature matching to find duplicated regions
3. **Enhancement Detection**: Identifies over-sharpening and saturation artifacts
4. **Upscaling Detection**: Detects interpolation artifacts from resolution enhancement
5. **AI Generation Detection**: Identifies patterns typical of AI-generated images

### Metadata Analysis
1. **EXIF Extraction**: Comprehensive EXIF data parsing and analysis
2. **GPS Coordinates**: Extract and validate location information
3. **Camera Information**: Camera make, model, and settings analysis
4. **Timestamp Analysis**: Creation and modification date consistency
5. **Forensic Indicators**: Detect metadata inconsistencies and anomalies

### Visual Geolocation
1. **Feature Extraction**: Analyze architectural and environmental features
2. **Text Recognition**: Extract text clues from signs and labels
3. **Color Analysis**: Climate and environment indicators from color patterns
4. **Landmark Matching**: Match against known landmark databases
5. **GPS Metadata**: Extract and validate GPS coordinates

## Evaluation & Limitations

### Performance Metrics
- Manipulation detection accuracy: ~80% on standard datasets
- Reverse search: Dependent on search engine APIs and policies
- Metadata extraction: >95% success rate on standard formats
- Geolocation accuracy: Varies greatly based on image content

### Known Limitations
- **False Positives**: May flag legitimate processed images as manipulated
- **AI Detection**: New AI generation techniques may evade detection
- **Search Limitations**: Reverse search depends on public availability
- **Geolocation**: Visual geolocation is highly dependent on distinctive features
- **Metadata**: Can be easily stripped or falsified

### Failure Modes
- **Low Quality Images**: Detection accuracy decreases with image quality
- **Heavily Processed Images**: Multiple edits may mask manipulation signatures
- **API Limitations**: Search engines may limit or block automated queries
- **Format Support**: Some proprietary camera formats may not be fully supported

## Safety Guidelines

### Evidence Handling
1. **Original Preservation**: Never modify original image files
2. **Chain of Custody**: Maintain complete audit trail
3. **Hash Verification**: Verify file integrity before and after analysis
4. **Working Copies**: Always work with forensic copies
5. **Documentation**: Record all analysis parameters and results

### Legal Considerations
- **Expert Testimony**: Be prepared to explain detection methods in court
- **Limitations Disclosure**: Clearly communicate detection limitations
- **Multiple Methods**: Use several detection techniques for critical evidence
- **Human Verification**: Combine automated analysis with human expertise
- **Context Integration**: Consider image context and provenance

### Privacy and Ethics
1. **Individual Privacy**: Protect privacy of individuals in images
2. **Consent**: Ensure proper consent for image analysis
3. **Data Protection**: Secure storage and handling of image data
4. **Bias Awareness**: Understand potential biases in detection algorithms
5. **Responsible Disclosure**: Report findings appropriately

## Contributing

We welcome contributions that enhance image verification capabilities for legal investigations.

### Development Setup
```bash
# Clone repository
git clone https://github.com/lemkin-org/lemkin-images.git
cd lemkin-images

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
```

### Research Integration
- Integration with latest manipulation detection research
- Support for new image formats and standards
- Enhanced AI generation detection algorithms
- Improved geolocation techniques

## License

Apache License 2.0 - see LICENSE file for details.

This toolkit is designed for legitimate legal investigations and human rights documentation. Users are responsible for ensuring compliance with all applicable laws regarding image analysis and privacy rights.

---

*Part of the Lemkin AI open-source legal technology ecosystem.*