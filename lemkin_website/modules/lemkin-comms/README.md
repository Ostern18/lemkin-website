# Lemkin Communication Analysis Analysis Toolkit

## Purpose

The Lemkin Communication Analysis Analysis Toolkit provides comprehensive audio analysis capabilities for legal investigations. This toolkit enables investigators to transcribe speech, identify speakers, verify audio authenticity, and enhance audio quality for better analysis.

## Safety & Ethics Notice

⚠️ **IMPORTANT**: This toolkit is designed for legitimate legal investigations and human rights documentation. Users must:
- Obtain proper legal authorization before analyzing audio evidence
- Respect privacy rights and wiretapping laws
- Maintain chain of custody for all audio evidence
- Use analysis results as investigative leads, not definitive proof
- Understand the limitations of automated analysis systems
- Protect the privacy of individuals in recordings

## Key Features

- **Speech Transcription**: Multi-language speech-to-text with timestamps and confidence scoring
- **Speaker Identification**: Voice profiling and speaker recognition using biometric features
- **Authenticity Verification**: Detect audio tampering, splicing, and manipulation
- **Communication Analysis Enhancement**: Noise reduction, gain normalization, and quality improvement
- **Comprehensive Analysis**: Combined analysis using all available techniques
- **Batch Processing**: Analyze multiple audio files efficiently

## Quick Start

```bash
# Install the toolkit
pip install lemkin-comms

# Transcribe speech from audio
lemkin-comms transcribe witness_interview.wav --language en-US --output transcription.json

# Verify audio authenticity
lemkin-comms verify-authenticity evidence_recording.wav --output authenticity_report.json

# Identify speakers using voice profiles
lemkin-comms identify-speaker phone_call.wav --profiles ./speaker_profiles/ --output speaker_analysis.json

# Enhance audio quality
lemkin-comms enhance noisy_recording.wav --output enhanced_audio.wav --noise-reduction --normalize

# Comprehensive analysis
lemkin-comms comprehensive-analysis evidence.wav \
    --transcription --authenticity --output full_analysis.json
```

## Usage Examples

### 1. Speech Transcription

```bash
# Basic transcription with automatic language detection
lemkin-comms transcribe interview.wav --output transcription.json

# Multi-language transcription with segments
lemkin-comms transcribe multilingual_call.wav \
    --language es-ES \
    --segments \
    --segment-length 15.0 \
    --output detailed_transcription.json
```

### 2. Speaker Identification

```bash
# Create a speaker profile from training samples
lemkin-comms identify-speaker john_sample1.wav \
    --create-profile "john_doe" \
    --profiles ./profiles/

# Identify speakers in unknown audio
lemkin-comms identify-speaker unknown_call.wav \
    --profiles ./profiles/ \
    --threshold 0.85 \
    --output speaker_results.json
```

### 3. Communication Analysis Authenticity Verification

```bash
# Comprehensive authenticity check
lemkin-comms verify-authenticity suspicious_recording.wav \
    --detailed \
    --output authenticity_analysis.json

# Quick authenticity verification
lemkin-comms verify-authenticity evidence.wav
```

### 4. Communication Analysis Enhancement

```bash
# Full enhancement pipeline
lemkin-comms enhance poor_quality.wav \
    --output enhanced.wav \
    --noise-reduction \
    --normalize \
    --frequency-filter \
    --low-cutoff 100 \
    --high-cutoff 7000 \
    --metrics

# Targeted noise reduction
lemkin-comms enhance noisy_call.wav \
    --output cleaned_call.wav \
    --spectral-subtraction \
    --echo-cancellation
```

### 5. Complete Communication Analysis Investigation Workflow

```bash
# Step 1: Enhance audio quality
lemkin-comms enhance evidence.wav --output evidence_enhanced.wav --noise-reduction

# Step 2: Verify authenticity
lemkin-comms verify-authenticity evidence_enhanced.wav --output step2_authenticity.json

# Step 3: Transcribe speech
lemkin-comms transcribe evidence_enhanced.wav --language en-US --segments --output step3_transcription.json

# Step 4: Identify speakers
lemkin-comms identify-speaker evidence_enhanced.wav --profiles ./known_speakers/ --output step4_speakers.json

# Step 5: Comprehensive analysis
lemkin-comms comprehensive-analysis evidence_enhanced.wav \
    --transcription --speaker-analysis --authenticity \
    --output final_analysis.json
```

## Input/Output Specifications

### Communication Analysis Metadata Format
```python
{
    "file_path": "/path/to/audio.wav",
    "file_size": 10485760,
    "duration_seconds": 120.5,
    "sample_rate": 44100,
    "channels": 2,
    "format": "wav",
    "bit_depth": 16,
    "creation_date": "2024-01-15T10:30:00Z",
    "file_hash": "abc123..."
}
```

### Transcription Results Format
```python
{
    "transcription_id": "uuid",
    "audio_path": "/path/to/audio.wav",
    "segments": [
        {
            "start_time": 0.0,
            "end_time": 5.2,
            "text": "This is the transcribed speech",
            "confidence": 0.95,
            "speaker_id": "speaker_1",
            "language": "en-US"
        }
    ],
    "full_text": "Complete transcribed text...",
    "total_duration": 120.5,
    "processing_time": 15.2
}
```

### Speaker Profile Format
```python
{
    "speaker_id": "john_doe",
    "voice_features": {
        "f0_mean": 125.5,
        "f0_std": 15.2,
        "spectral_centroid_mean": 2150.0,
        "mfcc_0_mean": -12.5
    },
    "fundamental_frequency": 125.5,
    "formant_frequencies": [800, 1200, 2500, 3500, 4500],
    "confidence_threshold": 0.8,
    "sample_count": 5,
    "created_at": "2024-01-15T10:30:00Z"
}
```

### Authenticity Report Format
```python
{
    "analysis_id": "uuid",
    "audio_path": "/path/to/audio.wav",
    "overall_authenticity": true,
    "authenticity_confidence": 0.89,
    "tampering_detected": false,
    "indicators": [
        {
            "indicator_name": "continuity_analysis",
            "is_authentic": true,
            "confidence": 0.92,
            "details": {
                "discontinuity_ratio": 0.02,
                "sudden_changes": 5
            }
        }
    ],
    "analysis_timestamp": "2024-01-15T10:30:00Z"
}
```

## API Reference

### Core Classes

#### SpeechTranscriber
Transcribes speech from audio files with multi-language support.

```python
from lemkin_comms import SpeechTranscriber, LanguageCode
from pathlib import Path

transcriber = SpeechTranscriber(model_size="base")
result = transcriber.transcribe_audio(
    Path("interview.wav"),
    language=LanguageCode.EN_US,
    enable_timestamps=True
)

print(f"Transcription: {result.full_text}")
print(f"Segments: {len(result.segments)}")
```

#### SpeakerIdentifier
Creates speaker profiles and identifies speakers in audio.

```python
from lemkin_comms import SpeakerIdentifier
from pathlib import Path

identifier = SpeakerIdentifier()

# Create speaker profile
profile = identifier.create_speaker_profile(
    speaker_id="witness_1",
    audio_samples=[Path("sample1.wav"), Path("sample2.wav")],
    confidence_threshold=0.8
)

# Identify speaker
result = identifier.identify_speaker(
    Path("unknown.wav"),
    known_profiles=["witness_1"]
)

if result.identified_speakers:
    speaker = result.identified_speakers[0]
    print(f"Speaker: {speaker['speaker_id']} (confidence: {speaker['confidence']:.1%})")
```

#### Communication AnalysisAuthenticator
Verifies audio authenticity and detects tampering.

```python
from lemkin_comms import Communication AnalysisAuthenticator
from pathlib import Path

authenticator = Communication AnalysisAuthenticator()
report = authenticator.verify_audio_authenticity(Path("evidence.wav"))

print(f"Authentic: {report.overall_authenticity}")
print(f"Confidence: {report.authenticity_confidence:.1%}")
print(f"Tampering detected: {report.tampering_detected}")

for indicator in report.indicators:
    print(f"- {indicator.indicator_name}: {'✓' if indicator.is_authentic else '✗'}")
```

#### Communication AnalysisEnhancer
Enhances audio quality using various processing techniques.

```python
from lemkin_comms import Communication AnalysisEnhancer, EnhancementSettings
from pathlib import Path

enhancer = Communication AnalysisEnhancer()
settings = EnhancementSettings(
    noise_reduction=True,
    gain_normalization=True,
    frequency_filtering=True,
    low_freq_cutoff=100.0,
    high_freq_cutoff=8000.0
)

result = enhancer.enhance_audio(
    Path("noisy.wav"),
    Path("enhanced.wav"),
    settings
)

print(f"Processing time: {result.processing_time:.2f}s")
for metric, improvement in result.quality_improvement.items():
    print(f"- {metric}: {improvement:+.3f}")
```

## Analysis Capabilities

### Speech Recognition Features
1. **Multi-language Support**: 11 languages including English, Spanish, French, German
2. **Timestamp Accuracy**: Precise start/end times for each speech segment
3. **Confidence Scoring**: Reliability metrics for transcription quality
4. **Segment Processing**: Handle long audio files with automatic segmentation
5. **Format Support**: WAV, MP3, FLAC, M4A, OGG, OPUS

### Speaker Analysis Techniques
1. **Voice Feature Extraction**: Fundamental frequency, MFCCs, spectral features
2. **Formant Analysis**: Extract vocal tract characteristics
3. **Biometric Profiling**: Create unique voice signatures
4. **Similarity Matching**: Compare unknown speakers to known profiles
5. **Voice Quality Assessment**: Analyze voice characteristics and consistency

### Authenticity Verification Methods
1. **Continuity Analysis**: Detect splicing and editing artifacts
2. **Compression Analysis**: Verify natural compression patterns
3. **Noise Floor Consistency**: Check for uniform background noise
4. **Frequency Spectrum Analysis**: Identify unnatural spectral characteristics
5. **Metadata Verification**: Analyze file structure and properties

### Enhancement Techniques
1. **Noise Reduction**: Spectral subtraction and advanced filtering
2. **Gain Normalization**: Optimize audio levels for analysis
3. **Frequency Filtering**: Remove unwanted frequency components
4. **Echo Cancellation**: Reduce reverb and echo artifacts
5. **Quality Metrics**: Quantify improvement in audio quality

## Evaluation & Limitations

### Performance Metrics
- Transcription accuracy: ~85% on clear audio (varies by language and quality)
- Speaker identification: ~80% accuracy with good quality training samples
- Authenticity detection: ~75% accuracy on common tampering techniques
- Enhancement effectiveness: Varies greatly based on original audio quality

### Known Limitations
- **Background Noise**: Performance degrades significantly with high noise levels
- **Multiple Speakers**: Speaker separation in overlapping speech is challenging
- **Accents and Dialects**: Recognition accuracy varies with speaker characteristics
- **Communication Analysis Quality**: Low-quality recordings may produce unreliable results
- **Language Detection**: Automatic language detection may fail with mixed languages

### Failure Modes
- **Codec Artifacts**: Some audio formats may introduce analysis artifacts
- **Very Short Samples**: Speaker identification requires sufficient audio duration
- **Synthetic Speech**: AI-generated speech may evade some detection methods
- **Heavy Processing**: Multiple layers of processing may mask authenticity indicators

## Safety Guidelines

### Evidence Handling
1. **Original Preservation**: Never modify original audio files
2. **Chain of Custody**: Maintain complete audit trail for all audio evidence
3. **Hash Verification**: Verify file integrity before and after analysis
4. **Working Copies**: Always work with forensic copies of evidence
5. **Documentation**: Record all analysis parameters and results

### Legal Considerations
- **Expert Testimony**: Be prepared to explain analysis methods in court
- **Limitations Disclosure**: Clearly communicate analysis limitations to stakeholders
- **Multiple Methods**: Use several analysis techniques for critical evidence
- **Human Verification**: Combine automated analysis with human expert review
- **Context Integration**: Consider audio context and provenance in analysis

### Privacy and Ethics
1. **Consent Requirements**: Ensure proper consent for audio analysis
2. **Data Protection**: Secure storage and handling of audio data
3. **Anonymization**: Protect privacy of non-relevant individuals in recordings
4. **Bias Awareness**: Understand potential biases in speech recognition and analysis
5. **Responsible Disclosure**: Report findings appropriately and ethically

## Batch Processing

Process multiple audio files efficiently:

```bash
# Transcribe all WAV files in a directory
lemkin-comms batch-process ./audio_files/ ./results/ \
    --type transcription \
    --pattern "*.wav" \
    --language en-US

# Comprehensive analysis of all audio evidence
lemkin-comms batch-process ./evidence/ ./analysis_results/ \
    --type comprehensive \
    --pattern "*.mp3"

# Authenticity verification for multiple files
lemkin-comms batch-process ./suspicious_audio/ ./authenticity_reports/ \
    --type authenticity \
    --pattern "*.*"
```

## Contributing

We welcome contributions that enhance audio analysis capabilities for legal investigations.

### Development Setup
```bash
# Clone repository
git clone https://github.com/lemkin-org/lemkin-comms.git
cd lemkin-comms

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
```

### Research Integration
- Integration with latest speech recognition models (Whisper, Wav2Vec2)
- Advanced speaker diarization techniques
- Improved audio authenticity detection methods
- Real-time audio analysis capabilities

## License

Apache License 2.0 - see LICENSE file for details.

This toolkit is designed for legitimate legal investigations and human rights documentation. Users are responsible for ensuring compliance with all applicable laws regarding audio recording and analysis.

---

*Part of the Lemkin AI open-source legal technology ecosystem.*