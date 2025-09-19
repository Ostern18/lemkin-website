"""
Lemkin Video Authentication Toolkit - CLI Interface

Command-line interface for the video authentication toolkit providing
comprehensive video analysis and manipulation detection capabilities.

Usage:
    lemkin-video detect-deepfake --video path/to/video.mp4 --output analysis.json
    lemkin-video fingerprint-video --video path/to/video.mp4 --output fingerprint.json
    lemkin-video analyze-compression --video path/to/video.mp4 --output compression.json
    lemkin-video extract-frames --video path/to/video.mp4 --output frames_analysis.json
    lemkin-video authenticate --video path/to/video.mp4 --case-number CASE-001 --output report.json
    lemkin-video compare-videos --video1 path/to/video1.mp4 --video2 path/to/video2.mp4
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
import logging
import click
from datetime import datetime

from .core import (
    VideoAuthConfig, VideoAuthenticator, AuthenticityLevel,
    TamperingType, AnalysisStatus
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def validate_video_file(ctx, param, value):
    """Validate that the video file exists and is readable"""
    if value is None:
        return value
    
    video_path = Path(value)
    if not video_path.exists():
        raise click.BadParameter(f"Video file not found: {video_path}")
    
    if not video_path.is_file():
        raise click.BadParameter(f"Path is not a file: {video_path}")
    
    # Check file size
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 1024:  # 1GB limit by default
        click.echo(f"Warning: Large video file ({file_size_mb:.1f} MB)", err=True)
    
    return video_path


def save_json_output(data: dict, output_path: str):
    """Save data as JSON with proper formatting"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        click.echo(f"Results saved to: {output_path}")
    except Exception as e:
        click.echo(f"Error saving to {output_path}: {e}", err=True)
        sys.exit(1)


def format_confidence_display(confidence: float) -> str:
    """Format confidence for display with appropriate styling"""
    if confidence >= 0.8:
        return f"{confidence:.1%}"
    elif confidence >= 0.6:
        return f"{confidence:.1%}"
    else:
        return f"{confidence:.1%}"


def format_authenticity_level(level: AuthenticityLevel) -> str:
    """Format authenticity level with color coding"""
    level_colors = {
        AuthenticityLevel.AUTHENTIC: "green",
        AuthenticityLevel.LIKELY_AUTHENTIC: "green",
        AuthenticityLevel.SUSPICIOUS: "yellow", 
        AuthenticityLevel.LIKELY_MANIPULATED: "red",
        AuthenticityLevel.MANIPULATED: "red",
        AuthenticityLevel.UNKNOWN: "white"
    }
    
    color = level_colors.get(level, "white")
    return click.style(level.value.upper().replace('_', ' '), fg=color, bold=True)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]):
    """
    Lemkin Video Authentication Toolkit
    
    Advanced video authenticity verification and manipulation detection
    for legal professionals and digital forensics investigators.
    """
    setup_logging(verbose)
    
    # Load configuration
    video_config = VideoAuthConfig()
    if config:
        try:
            with open(config, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                video_config = VideoAuthConfig(**config_data)
        except Exception as e:
            click.echo(f"Error loading config file: {e}", err=True)
            sys.exit(1)
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = video_config
    ctx.obj['verbose'] = verbose


@cli.command('detect-deepfake')
@click.option('--video', '-v', required=True, callback=validate_video_file,
              help='Path to video file to analyze')
@click.option('--model', '-m', help='Path to deepfake detection model')
@click.option('--threshold', '-t', type=float, default=0.7,
              help='Detection threshold (0.0-1.0)')
@click.option('--output', '-o', help='Output file path for results')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU acceleration if available')
@click.pass_context
def detect_deepfake(ctx, video: Path, model: Optional[str], threshold: float, 
                   output: Optional[str], gpu: bool):
    """Analyze video for deepfake manipulation"""
    
    try:
        config = ctx.obj['config']
        
        # Update configuration
        if model:
            config.deepfake_model_path = model
        config.deepfake_threshold = threshold
        config.use_gpu = gpu
        
        # Create authenticator
        authenticator = VideoAuthenticator(config)
        
        click.echo(f"Analyzing video for deepfake manipulation: {video}")
        click.echo(f"Detection threshold: {threshold:.1%}")
        click.echo(f"GPU acceleration: {'enabled' if gpu else 'disabled'}")
        
        # Perform deepfake detection
        start_time = datetime.now()
        deepfake_analysis = authenticator.detect_deepfake(video)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results
        click.echo(f"\nDeepfake Detection Results")
        click.echo("=" * 50)
        click.echo(f"Is Deepfake: {click.style('YES' if deepfake_analysis.is_deepfake else 'NO', 
                   fg='red' if deepfake_analysis.is_deepfake else 'green', bold=True)}")
        click.echo(f"Confidence: {format_confidence_display(deepfake_analysis.confidence)}")
        click.echo(f"Deepfake Probability: {deepfake_analysis.deepfake_probability:.1%}")
        click.echo(f"Model Used: {deepfake_analysis.model_used}")
        click.echo(f"Processing Time: {processing_time:.1f} seconds")
        
        # Frame analysis summary
        total_frames = deepfake_analysis.total_frames_analyzed
        if total_frames > 0:
            click.echo(f"\nFrame Analysis:")
            click.echo(f"  Total frames analyzed: {total_frames}")
            click.echo(f"  Positive detections: {deepfake_analysis.positive_frames} "
                      f"({deepfake_analysis.positive_frames / total_frames:.1%})")
            click.echo(f"  Negative detections: {deepfake_analysis.negative_frames} "
                      f"({deepfake_analysis.negative_frames / total_frames:.1%})")
            click.echo(f"  Uncertain frames: {deepfake_analysis.uncertain_frames} "
                      f"({deepfake_analysis.uncertain_frames / total_frames:.1%})")
        
        # Face analysis
        if deepfake_analysis.faces_detected > 0:
            click.echo(f"\nFace Analysis:")
            click.echo(f"  Faces detected: {deepfake_analysis.faces_detected}")
            if deepfake_analysis.face_consistency_score is not None:
                click.echo(f"  Face consistency: {deepfake_analysis.face_consistency_score:.1%}")
            if deepfake_analysis.identity_changes_detected > 0:
                click.echo(f"  Identity changes: {deepfake_analysis.identity_changes_detected}")
        
        # Technical indicators
        if deepfake_analysis.compression_artifacts:
            click.echo(f"\nTechnical Indicators:")
            click.echo(f"  Compression artifacts: {len(deepfake_analysis.compression_artifacts)}")
        
        if deepfake_analysis.temporal_inconsistencies:
            click.echo(f"  Temporal inconsistencies: {len(deepfake_analysis.temporal_inconsistencies)}")
        
        # Save output if specified
        if output:
            save_json_output(deepfake_analysis.dict(), output)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('fingerprint-video')
@click.option('--video', '-v', required=True, callback=validate_video_file,
              help='Path to video file to fingerprint')
@click.option('--algorithm', '-a', type=click.Choice(['dhash', 'phash', 'ahash', 'whash']),
              default='dhash', help='Hashing algorithm to use')
@click.option('--hash-size', type=int, default=8, help='Hash size (4-32)')
@click.option('--output', '-o', help='Output file path for fingerprint')
@click.option('--compare', help='Compare with existing fingerprint file')
@click.pass_context
def fingerprint_video(ctx, video: Path, algorithm: str, hash_size: int, 
                     output: Optional[str], compare: Optional[str]):
    """Generate content-based video fingerprint for duplicate detection"""
    
    try:
        config = ctx.obj['config']
        
        # Update configuration
        config.fingerprint_algorithm = algorithm
        config.hash_size = hash_size
        
        # Create authenticator
        authenticator = VideoAuthenticator(config)
        
        click.echo(f"Generating video fingerprint: {video}")
        click.echo(f"Algorithm: {algorithm}")
        click.echo(f"Hash size: {hash_size}")
        
        # Generate fingerprint
        start_time = datetime.now()
        fingerprint = authenticator.fingerprint_video(video)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results
        click.echo(f"\nVideo Fingerprint Results")
        click.echo("=" * 50)
        click.echo(f"Perceptual Hash: {fingerprint.perceptual_hash}")
        click.echo(f"Temporal Hash: {fingerprint.temporal_hash}")
        if fingerprint.audio_fingerprint:
            click.echo(f"Audio Fingerprint: {fingerprint.audio_fingerprint}")
        
        click.echo(f"\nVideo Properties:")
        click.echo(f"  Frame count: {fingerprint.frame_count}")
        click.echo(f"  Duration: {fingerprint.duration_seconds:.1f} seconds")
        click.echo(f"  Resolution: {fingerprint.resolution[0]}x{fingerprint.resolution[1]}")
        click.echo(f"  Average brightness: {fingerprint.average_brightness:.1%}")
        click.echo(f"  Quality score: {fingerprint.quality_score:.1%}")
        
        click.echo(f"\nProcessing:")
        click.echo(f"  Key frames: {len(fingerprint.key_frames_hashes)}")
        click.echo(f"  Processing time: {processing_time:.1f} seconds")
        
        # Compare with existing fingerprint if specified
        if compare:
            try:
                with open(compare, 'r') as f:
                    compare_data = json.load(f)
                    compare_fingerprint = VideoFingerprint(**compare_data)
                
                similarity = fingerprint.calculate_similarity(compare_fingerprint)
                
                click.echo(f"\nComparison Results:")
                click.echo(f"  Similarity score: {similarity:.1%}")
                click.echo(f"  Exact match: {'YES' if similarity >= fingerprint.exact_match_threshold else 'NO'}")
                click.echo(f"  Similar match: {'YES' if similarity >= fingerprint.similar_match_threshold else 'NO'}")
                
            except Exception as e:
                click.echo(f"Error comparing fingerprints: {e}", err=True)
        
        # Save output if specified
        if output:
            save_json_output(fingerprint.dict(), output)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('analyze-compression')
@click.option('--video', '-v', required=True, callback=validate_video_file,
              help='Path to video file to analyze')
@click.option('--output', '-o', help='Output file path for analysis')
@click.option('--detailed', is_flag=True, help='Include detailed compression metrics')
@click.pass_context
def analyze_compression(ctx, video: Path, output: Optional[str], detailed: bool):
    """Check compression artifacts for authenticity verification"""
    
    try:
        config = ctx.obj['config']
        
        # Create authenticator
        authenticator = VideoAuthenticator(config)
        
        click.echo(f"Analyzing compression artifacts: {video}")
        
        # Perform compression analysis
        start_time = datetime.now()
        compression_analysis = authenticator.analyze_compression_artifacts(video)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results
        click.echo(f"\nCompression Analysis Results")
        click.echo("=" * 50)
        click.echo(f"Compression Level: {compression_analysis.compression_level.value.upper().replace('_', ' ')}")
        click.echo(f"Is Recompressed: {click.style('YES' if compression_analysis.is_recompressed else 'NO',
                   fg='red' if compression_analysis.is_recompressed else 'green', bold=True)}")
        
        if compression_analysis.is_recompressed:
            click.echo(f"Recompression Count: {compression_analysis.recompression_count}")
        
        click.echo(f"\nQuality Metrics:")
        click.echo(f"  Overall quality score: {compression_analysis.overall_quality_score:.1%}")
        click.echo(f"  Bitrate consistency: {compression_analysis.bitrate_consistency:.1%}")
        click.echo(f"  Compression efficiency: {compression_analysis.compression_efficiency:.1%}")
        
        click.echo(f"\nArtifact Analysis:")
        click.echo(f"  Blocking artifacts: {compression_analysis.blocking_artifacts:.1%}")
        click.echo(f"  Ringing artifacts: {compression_analysis.ringing_artifacts:.1%}")
        click.echo(f"  Mosquito noise: {compression_analysis.mosquito_noise:.1%}")
        
        # Codec information
        if compression_analysis.codec_sequence:
            click.echo(f"\nCodec Information:")
            click.echo(f"  Codec sequence: {' -> '.join(compression_analysis.codec_sequence)}")
        
        if compression_analysis.quantization_parameters:
            qp_values = compression_analysis.quantization_parameters
            click.echo(f"  QP range: {min(qp_values):.1f} - {max(qp_values):.1f}")
            click.echo(f"  QP average: {sum(qp_values) / len(qp_values):.1f}")
        
        # Inconsistencies
        if compression_analysis.inconsistent_regions:
            click.echo(f"\nInconsistencies:")
            click.echo(f"  Inconsistent regions: {len(compression_analysis.inconsistent_regions)}")
        
        if compression_analysis.compression_boundaries:
            click.echo(f"  Quality boundaries: {len(compression_analysis.compression_boundaries)}")
        
        # Detailed information
        if detailed and compression_analysis.quality_variations:
            click.echo(f"\nDetailed Quality Analysis:")
            outliers = [v for v in compression_analysis.quality_variations if v.get('is_outlier', False)]
            if outliers:
                click.echo(f"  Quality outliers: {len(outliers)}")
        
        click.echo(f"\nProcessing time: {processing_time:.1f} seconds")
        
        # Save output if specified
        if output:
            save_json_output(compression_analysis.dict(), output)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('extract-frames')
@click.option('--video', '-v', required=True, callback=validate_video_file,
              help='Path to video file to analyze')
@click.option('--output', '-o', help='Output file path for frame analysis')
@click.option('--save-frames', help='Directory to save extracted key frames')
@click.option('--max-frames', type=int, default=100, help='Maximum number of frames to analyze')
@click.pass_context
def extract_frames(ctx, video: Path, output: Optional[str], save_frames: Optional[str], max_frames: int):
    """Extract and analyze key frames for tampering detection"""
    
    try:
        config = ctx.obj['config']
        
        # Create authenticator
        authenticator = VideoAuthenticator(config)
        
        click.echo(f"Extracting and analyzing key frames: {video}")
        click.echo(f"Maximum frames: {max_frames}")
        
        # Extract key frames
        start_time = datetime.now()
        key_frames = authenticator.extract_key_frames(video)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Limit frames if requested
        if len(key_frames) > max_frames:
            key_frames = key_frames[:max_frames]
            click.echo(f"Limited analysis to first {max_frames} key frames")
        
        # Display results
        click.echo(f"\nKey Frame Analysis Results")
        click.echo("=" * 50)
        click.echo(f"Key frames extracted: {len(key_frames)}")
        
        if key_frames:
            # Calculate summary statistics
            authenticity_scores = [kf.authenticity_score for kf in key_frames]
            quality_scores = [kf.quality_score for kf in key_frames if kf.quality_score is not None]
            
            avg_authenticity = sum(authenticity_scores) / len(authenticity_scores)
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            click.echo(f"Average authenticity score: {avg_authenticity:.1%}")
            if quality_scores:
                click.echo(f"Average quality score: {avg_quality:.1%}")
            
            # Count tampering indicators
            total_indicators = sum(len(kf.tampering_indicators) for kf in key_frames)
            critical_indicators = sum(1 for kf in key_frames 
                                    for indicator in kf.tampering_indicators 
                                    if indicator.is_critical)
            
            if total_indicators > 0:
                click.echo(f"\nTampering Indicators:")
                click.echo(f"  Total indicators: {total_indicators}")
                click.echo(f"  Critical indicators: {critical_indicators}")
                
                # Group by tampering type
                tampering_types = {}
                for kf in key_frames:
                    for indicator in kf.tampering_indicators:
                        tampering_type = indicator.tampering_type.value
                        tampering_types[tampering_type] = tampering_types.get(tampering_type, 0) + 1
                
                for tampering_type, count in tampering_types.items():
                    click.echo(f"    {tampering_type.replace('_', ' ').title()}: {count}")
            
            # Face analysis summary
            total_faces = sum(kf.faces_detected for kf in key_frames)
            frames_with_faces = sum(1 for kf in key_frames if kf.faces_detected > 0)
            
            if total_faces > 0:
                click.echo(f"\nFace Analysis:")
                click.echo(f"  Total faces detected: {total_faces}")
                click.echo(f"  Frames with faces: {frames_with_faces}")
                
                deepfake_probs = [kf.deepfake_probability for kf in key_frames 
                                if kf.deepfake_probability is not None]
                if deepfake_probs:
                    avg_deepfake_prob = sum(deepfake_probs) / len(deepfake_probs)
                    click.echo(f"  Average deepfake probability: {avg_deepfake_prob:.1%}")
            
            # Frame type distribution
            frame_types = {}
            for kf in key_frames:
                frame_type = kf.frame_type.value
                frame_types[frame_type] = frame_types.get(frame_type, 0) + 1
            
            click.echo(f"\nFrame Types:")
            for frame_type, count in frame_types.items():
                click.echo(f"  {frame_type.upper()}: {count}")
        
        click.echo(f"\nProcessing time: {processing_time:.1f} seconds")
        
        # Save frames if requested
        if save_frames and key_frames:
            import cv2
            frames_dir = Path(save_frames)
            frames_dir.mkdir(exist_ok=True)
            
            # This would require access to the original video frames
            # For now, just save the frame metadata
            frames_metadata = [kf.dict() for kf in key_frames]
            with open(frames_dir / "frames_metadata.json", 'w') as f:
                json.dump(frames_metadata, f, indent=2, default=str)
            
            click.echo(f"Frame metadata saved to: {frames_dir / 'frames_metadata.json'}")
        
        # Save output if specified
        if output:
            frames_data = [kf.dict() for kf in key_frames]
            save_json_output(frames_data, output)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('authenticate')
@click.option('--video', '-v', required=True, callback=validate_video_file,
              help='Path to video file to authenticate')
@click.option('--case-number', '-c', help='Case number for legal tracking')
@click.option('--investigator', '-i', help='Investigator name')
@click.option('--output', '-o', help='Output file path for comprehensive report')
@click.option('--format', type=click.Choice(['json', 'detailed']), default='detailed',
              help='Output format')
@click.pass_context
def authenticate(ctx, video: Path, case_number: Optional[str], investigator: Optional[str],
                output: Optional[str], format: str):
    """Perform comprehensive video authenticity analysis"""
    
    try:
        config = ctx.obj['config']
        
        # Create authenticator
        authenticator = VideoAuthenticator(config)
        
        click.echo(f"Performing comprehensive video authentication: {video}")
        if case_number:
            click.echo(f"Case number: {case_number}")
        if investigator:
            click.echo(f"Investigator: {investigator}")
        
        # Perform authentication
        start_time = datetime.now()
        report = authenticator.authenticate_video(video, case_number)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        
        # Update report with investigator info
        if investigator:
            report.investigator = investigator
        
        # Display results
        result = report.analysis_result
        
        click.echo(f"\nVideo Authentication Report")
        click.echo("=" * 60)
        click.echo(f"Case Number: {report.case_number or 'N/A'}")
        click.echo(f"Video File: {report.original_filename}")
        click.echo(f"File Hash: {report.file_hash}")
        click.echo(f"Analysis Date: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        click.echo(f"\nAuthenticity Assessment:")
        click.echo(f"  Level: {format_authenticity_level(result.authenticity_level)}")
        click.echo(f"  Overall Confidence: {format_confidence_display(result.overall_confidence)}")
        click.echo(f"  Authenticity Score: {result.authenticity_score:.1%}")
        
        if result.tampering_probability > 0:
            click.echo(f"  Tampering Probability: {result.tampering_probability:.1%}")
        
        # Component analysis summary
        click.echo(f"\nComponent Analysis:")
        
        if result.deepfake_analysis:
            df_analysis = result.deepfake_analysis
            status = "DETECTED" if df_analysis.is_deepfake else "NOT DETECTED"
            color = "red" if df_analysis.is_deepfake else "green"
            click.echo(f"  Deepfake: {click.style(status, fg=color)} "
                      f"(confidence: {df_analysis.confidence:.1%})")
        
        if result.compression_analysis:
            comp_analysis = result.compression_analysis
            status = "DETECTED" if comp_analysis.is_recompressed else "NOT DETECTED"
            color = "red" if comp_analysis.is_recompressed else "green"
            click.echo(f"  Recompression: {click.style(status, fg=color)} "
                      f"(quality: {comp_analysis.overall_quality_score:.1%})")
        
        if result.fingerprint_analysis:
            fp_analysis = result.fingerprint_analysis
            click.echo(f"  Fingerprint: Generated (quality: {fp_analysis.quality_score:.1%})")
        
        if result.key_frames:
            click.echo(f"  Key Frames: {len(result.key_frames)} analyzed")
        
        # Tampering indicators
        if result.tampering_indicators:
            click.echo(f"\nTampering Indicators ({len(result.tampering_indicators)}):")
            
            # Group by type and criticality
            critical_indicators = [t for t in result.tampering_indicators if t.is_critical]
            non_critical_indicators = [t for t in result.tampering_indicators if not t.is_critical]
            
            if critical_indicators:
                click.echo(f"  Critical Issues ({len(critical_indicators)}):")
                for indicator in critical_indicators[:5]:  # Show first 5
                    click.echo(f"    • {indicator.tampering_type.value.replace('_', ' ').title()}: "
                              f"{indicator.confidence:.1%} confidence")
                    click.echo(f"      {indicator.description}")
                
                if len(critical_indicators) > 5:
                    click.echo(f"    ... and {len(critical_indicators) - 5} more critical issues")
            
            if non_critical_indicators:
                click.echo(f"  Other Indicators ({len(non_critical_indicators)}):")
                indicator_types = {}
                for indicator in non_critical_indicators:
                    itype = indicator.tampering_type.value.replace('_', ' ').title()
                    indicator_types[itype] = indicator_types.get(itype, 0) + 1
                
                for itype, count in indicator_types.items():
                    click.echo(f"    • {itype}: {count}")
        
        # Executive summary
        click.echo(f"\nExecutive Summary:")
        click.echo(f"  {report.executive_summary}")
        
        # Key findings
        if report.key_findings:
            click.echo(f"\nKey Findings:")
            for finding in report.key_findings:
                click.echo(f"  • {finding}")
        
        # Warnings
        if result.warnings:
            click.echo(f"\nWarnings:")
            for warning in result.warnings:
                click.echo(f"  ⚠ {warning}")
        
        # Analysis metadata
        click.echo(f"\nAnalysis Metadata:")
        click.echo(f"  Status: {result.analysis_status.value.upper()}")
        click.echo(f"  Total processing time: {total_time:.1f} seconds")
        click.echo(f"  Chain of custody verified: {'YES' if report.chain_of_custody_verified else 'NO'}")
        
        # Save output if specified
        if output:
            if format == 'json':
                save_json_output(report.dict(), output)
            else:
                # Save detailed report (could be enhanced with HTML/PDF generation)
                save_json_output(report.dict(), output)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('compare-videos')
@click.option('--video1', required=True, callback=validate_video_file,
              help='Path to first video file')
@click.option('--video2', required=True, callback=validate_video_file,
              help='Path to second video file')
@click.option('--threshold', type=float, default=0.8,
              help='Similarity threshold for duplicate detection')
@click.option('--output', '-o', help='Output file path for comparison results')
@click.pass_context
def compare_videos(ctx, video1: Path, video2: Path, threshold: float, output: Optional[str]):
    """Compare two videos for similarity and potential duplication"""
    
    try:
        config = ctx.obj['config']
        config.perceptual_hash_threshold = threshold
        
        # Create authenticator
        authenticator = VideoAuthenticator(config)
        
        click.echo(f"Comparing videos for similarity:")
        click.echo(f"  Video 1: {video1}")
        click.echo(f"  Video 2: {video2}")
        click.echo(f"  Similarity threshold: {threshold:.1%}")
        
        # Compare videos
        start_time = datetime.now()
        comparison_result = authenticator.compare_videos(video1, video2)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results
        similarity = comparison_result['similarity_score']
        
        click.echo(f"\nVideo Comparison Results")
        click.echo("=" * 50)
        click.echo(f"Similarity Score: {similarity:.1%}")
        
        # Determine relationship
        if comparison_result['is_exact_match']:
            match_status = click.style("EXACT MATCH", fg="red", bold=True)
        elif comparison_result['is_similar']:
            match_status = click.style("SIMILAR", fg="yellow", bold=True)
        else:
            match_status = click.style("DIFFERENT", fg="green", bold=True)
        
        click.echo(f"Match Status: {match_status}")
        
        # Fingerprint details
        fp1 = comparison_result['fingerprint1']
        fp2 = comparison_result['fingerprint2']
        
        click.echo(f"\nVideo 1 Properties:")
        click.echo(f"  Duration: {fp1.duration_seconds:.1f} seconds")
        click.echo(f"  Resolution: {fp1.resolution[0]}x{fp1.resolution[1]}")
        click.echo(f"  Frame count: {fp1.frame_count}")
        click.echo(f"  Quality score: {fp1.quality_score:.1%}")
        
        click.echo(f"\nVideo 2 Properties:")
        click.echo(f"  Duration: {fp2.duration_seconds:.1f} seconds")
        click.echo(f"  Resolution: {fp2.resolution[0]}x{fp2.resolution[1]}")
        click.echo(f"  Frame count: {fp2.frame_count}")
        click.echo(f"  Quality score: {fp2.quality_score:.1%}")
        
        # Differences
        duration_diff = abs(fp1.duration_seconds - fp2.duration_seconds)
        resolution_diff = (fp1.resolution[0] != fp2.resolution[0] or 
                          fp1.resolution[1] != fp2.resolution[1])
        
        click.echo(f"\nDifferences:")
        click.echo(f"  Duration difference: {duration_diff:.1f} seconds")
        click.echo(f"  Resolution difference: {'YES' if resolution_diff else 'NO'}")
        click.echo(f"  Frame count difference: {abs(fp1.frame_count - fp2.frame_count)}")
        
        # Analysis interpretation
        click.echo(f"\nAnalysis Interpretation:")
        if comparison_result['is_exact_match']:
            click.echo("  These videos appear to be identical or near-identical copies.")
            click.echo("  This suggests potential video duplication or minimal processing.")
        elif comparison_result['is_similar']:
            click.echo("  These videos show significant similarity.")
            click.echo("  This could indicate: recompression, format conversion, or minor editing.")
        else:
            click.echo("  These videos appear to be substantially different.")
            click.echo("  They are likely unique content or heavily modified versions.")
        
        click.echo(f"\nProcessing time: {processing_time:.1f} seconds")
        
        # Save output if specified
        if output:
            save_json_output(comparison_result, output)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('config')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--create', help='Create configuration file at path')
@click.option('--set-param', nargs=2, multiple=True, help='Set configuration parameter (key value)')
@click.pass_context
def config_cmd(ctx, show: bool, create: Optional[str], set_param: List[tuple]):
    """Manage configuration settings"""
    
    try:
        config = ctx.obj['config']
        
        # Set parameters if specified
        for key, value in set_param:
            if hasattr(config, key):
                # Convert value to appropriate type
                current_value = getattr(config, key)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(config, key, value)
                click.echo(f"Set {key} = {value}")
            else:
                click.echo(f"Warning: Unknown parameter '{key}'", err=True)
        
        if show:
            click.echo("Current configuration:")
            click.echo("=" * 30)
            config_dict = config.dict()
            for key, value in config_dict.items():
                click.echo(f"  {key}: {value}")
        
        if create:
            config_dict = config.dict()
            save_json_output(config_dict, create)
            click.echo(f"Configuration saved to: {create}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('version')
def version():
    """Show version information"""
    click.echo("Lemkin Video Authentication Toolkit v0.1.0")
    click.echo("Advanced video authenticity verification and manipulation detection")
    click.echo("Berkeley Protocol compliant digital investigations")
    click.echo()
    click.echo("Features:")
    click.echo("  • Deepfake detection using state-of-the-art ML models")
    click.echo("  • Video fingerprinting for duplicate detection")
    click.echo("  • Compression analysis for authenticity verification")
    click.echo("  • Frame-level tampering detection")
    click.echo("  • Comprehensive authenticity reporting")


@cli.command('benchmark')
@click.option('--video', callback=validate_video_file, help='Test video file')
@click.option('--operations', multiple=True, 
              type=click.Choice(['deepfake', 'fingerprint', 'compression', 'frames', 'all']),
              default=['all'], help='Operations to benchmark')
@click.pass_context
def benchmark(ctx, video: Optional[Path], operations: List[str]):
    """Benchmark analysis performance"""
    
    if not video:
        click.echo("Please provide a test video with --video", err=True)
        sys.exit(1)
    
    config = ctx.obj['config']
    authenticator = VideoAuthenticator(config)
    
    # Determine operations to test
    if 'all' in operations:
        operations = ['deepfake', 'fingerprint', 'compression', 'frames']
    
    click.echo(f"Benchmarking video analysis performance")
    click.echo(f"Test video: {video}")
    click.echo(f"Operations: {', '.join(operations)}")
    click.echo("=" * 50)
    
    results = {}
    
    try:
        if 'deepfake' in operations:
            click.echo("Testing deepfake detection...")
            start_time = datetime.now()
            deepfake_result = authenticator.detect_deepfake(video)
            end_time = datetime.now()
            results['deepfake'] = (end_time - start_time).total_seconds()
            click.echo(f"  Deepfake detection: {results['deepfake']:.2f} seconds")
        
        if 'fingerprint' in operations:
            click.echo("Testing video fingerprinting...")
            start_time = datetime.now()
            fingerprint_result = authenticator.fingerprint_video(video)
            end_time = datetime.now()
            results['fingerprint'] = (end_time - start_time).total_seconds()
            click.echo(f"  Video fingerprinting: {results['fingerprint']:.2f} seconds")
        
        if 'compression' in operations:
            click.echo("Testing compression analysis...")
            start_time = datetime.now()
            compression_result = authenticator.analyze_compression_artifacts(video)
            end_time = datetime.now()
            results['compression'] = (end_time - start_time).total_seconds()
            click.echo(f"  Compression analysis: {results['compression']:.2f} seconds")
        
        if 'frames' in operations:
            click.echo("Testing frame analysis...")
            start_time = datetime.now()
            frames_result = authenticator.extract_key_frames(video)
            end_time = datetime.now()
            results['frames'] = (end_time - start_time).total_seconds()
            click.echo(f"  Frame analysis: {results['frames']:.2f} seconds")
        
        # Summary
        total_time = sum(results.values())
        click.echo(f"\nBenchmark Summary:")
        click.echo(f"  Total time: {total_time:.2f} seconds")
        click.echo(f"  Average per operation: {total_time / len(results):.2f} seconds")
        
        # Performance assessment
        file_size_mb = video.stat().st_size / (1024 * 1024)
        click.echo(f"  Processing rate: {file_size_mb / total_time:.2f} MB/second")
        
    except Exception as e:
        click.echo(f"Benchmark failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()