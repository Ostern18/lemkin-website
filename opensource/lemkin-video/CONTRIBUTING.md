# Contributing to Lemkin Video Authentication Toolkit

Thank you for contributing to video authentication capabilities for legal investigations!

## Development Setup

```bash
git clone https://github.com/lemkin-org/lemkin-video.git
cd lemkin-video
pip install -e ".[dev]"
pre-commit install
```

## Key Contribution Areas

- **Deepfake Detection**: Advanced AI-generated video detection algorithms
- **Video Fingerprinting**: Robust video identification and matching
- **Compression Analysis**: Enhanced detection of video editing artifacts
- **Privacy Protection**: Face blurring and voice anonymization features

## Video-Specific Guidelines

### Security Considerations
```python
def process_sensitive_video(
    video_path: Path,
    anonymize_faces: bool = True,
    preserve_audio: bool = False
) -> VideoAnalysis:
    \"\"\"Process video with privacy protections.

    Args:
        video_path: Path to video file
        anonymize_faces: Whether to blur non-relevant faces
        preserve_audio: Whether to include audio in analysis

    Returns:
        Video analysis with appropriate privacy protections
    \"\"\"
    # Always work with secure temporary files
    with secure_temp_video(video_path) as temp_video:
        if anonymize_faces:
            temp_video = apply_face_anonymization(temp_video)

        return perform_video_analysis(temp_video)
```

### Privacy Protection
- Implement automatic face detection and blurring
- Secure voice data and speaker identification
- Use secure temporary storage with automatic cleanup
- Protect witness and victim identities in video content

### Performance Considerations
- Optimize for large video files and long processing times
- Implement efficient memory management for video analysis
- Use GPU acceleration where available and appropriate
- Provide progress indicators for long-running operations

## Testing Requirements

- Use synthetic or publicly available test videos only
- Test with various video formats and quality levels
- Verify privacy protection mechanisms work correctly
- Include performance benchmarks for different video sizes

## AI/ML Model Contributions

When contributing machine learning models:
- Document model architecture and training data
- Provide evaluation metrics and test results
- Ensure models are bias-tested and fair
- Include model versioning and update procedures

## Contact

- **Technical**: Open GitHub Issues
- **AI/ML Models**: ml@lemkin.org
- **Privacy**: privacy@lemkin.org

---

*Help make video authentication technology accessible for justice worldwide.*