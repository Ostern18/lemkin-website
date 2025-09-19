# Contributing to Lemkin Image Verification Suite

Thank you for contributing to image verification capabilities for legal investigations!

## Development Setup

```bash
git clone https://github.com/lemkin-org/lemkin-images.git
cd lemkin-images
pip install -e ".[dev]"
pre-commit install
```

## Key Contribution Areas

- **Manipulation Detection**: Advanced image tampering and editing detection
- **Reverse Image Search**: Enhanced search capabilities across multiple engines
- **Metadata Forensics**: Comprehensive EXIF and metadata analysis
- **Geolocation**: Visual landmark identification and location matching

## Image-Specific Guidelines

### Privacy Protection
```python
def process_sensitive_image(
    image_path: Path,
    detect_faces: bool = True,
    strip_metadata: bool = True
) -> ImageAnalysis:
    \"\"\"Process image with privacy protections.

    Args:
        image_path: Path to image file
        detect_faces: Whether to detect and blur faces
        strip_metadata: Whether to remove sensitive metadata

    Returns:
        Image analysis with privacy protections applied
    \"\"\"
    with secure_temp_image(image_path) as temp_image:
        if detect_faces:
            temp_image = anonymize_detected_faces(temp_image)

        if strip_metadata:
            temp_image = remove_sensitive_exif(temp_image)

        return perform_image_analysis(temp_image)
```

### Security Considerations
- Always work with copies, never modify original evidence
- Implement PII detection for text in images
- Use secure protocols for reverse image searches
- Protect GPS coordinates and location metadata

### Image Processing Standards
- Support multiple image formats (JPEG, PNG, TIFF, RAW)
- Preserve image quality during processing
- Document all transformations for legal admissibility
- Implement robust error handling for corrupted images

## Testing Requirements

- Use synthetic or public domain images for testing
- Test with various image formats, sizes, and quality levels
- Verify manipulation detection accuracy with known test cases
- Include performance tests for batch processing

## Reverse Search Ethics

When implementing reverse search capabilities:
- Respect platform terms of service
- Implement appropriate rate limiting
- Protect investigation confidentiality
- Document search methodology for court

## Contact

- **Technical**: Open GitHub Issues
- **Computer Vision**: cv@lemkin.org
- **Ethics**: ethics@lemkin.org

---

*Help make image verification accessible for human rights investigations globally.*