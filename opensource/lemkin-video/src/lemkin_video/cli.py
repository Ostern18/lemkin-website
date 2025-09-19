"""
Command-line interface for Lemkin Video Authentication Toolkit.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from .core import (
    DeepfakeDetector,
    VideoFingerprinter,
    CompressionAnalyzer,
    FrameAnalyzer,
    AuthenticityStatus,
    VideoMetadata,
)

app = typer.Typer(
    name="lemkin-video",
    help="Video authentication toolkit for legal investigations",
    no_args_is_help=True
)
console = Console()


@app.command()
def detect_deepfake(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    max_frames: int = typer.Option(300, help="Maximum frames to analyze")
):
    """Detect deepfake content in video"""

    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Analyzing video for deepfake content: {video_path}[/cyan]")

        with Progress() as progress:
            task = progress.add_task("[green]Detecting deepfakes...", total=100)

            detector = DeepfakeDetector()
            analysis = detector.detect_deepfake(video_path)

            progress.update(task, completed=100)

        # Display results
        status_color = "red" if analysis.is_deepfake else "green"
        panel_content = f"""
[bold]Video:[/bold] {video_path.name}
[bold]Deepfake Detected:[/bold] [{status_color}]{"YES" if analysis.is_deepfake else "NO"}[/{status_color}]
[bold]Confidence:[/bold] {analysis.confidence_score:.1%}
[bold]Frames Analyzed:[/bold] {analysis.frames_analyzed}
[bold]Suspicious Frames:[/bold] {len(analysis.suspicious_frames)}
[bold]Temporal Consistency:[/bold] {analysis.temporal_consistency:.1%}
        """

        console.print(Panel(panel_content, title="Deepfake Detection Results"))

        # Show detection metrics
        if analysis.detection_metrics:
            table = Table(title="Detection Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for metric, value in analysis.detection_metrics.items():
                if isinstance(value, float):
                    table.add_row(metric.replace('_', ' ').title(), f"{value:.3f}")
                else:
                    table.add_row(metric.replace('_', ' ').title(), str(value))

            console.print(table)

        # Show suspicious frame numbers if found
        if analysis.suspicious_frames:
            console.print("\n[yellow]Suspicious Frame Numbers:[/yellow]")
            frame_chunks = [analysis.suspicious_frames[i:i+10] for i in range(0, len(analysis.suspicious_frames), 10)]
            for chunk in frame_chunks[:5]:  # Show first 50 frames
                console.print(f"  {', '.join(map(str, chunk))}")

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(analysis.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Analysis saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error detecting deepfake: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def fingerprint_video(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    key_frames: int = typer.Option(10, help="Number of key frames to extract")
):
    """Generate video fingerprint for duplicate detection"""

    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Generating fingerprint for: {video_path}[/cyan]")

        with Progress() as progress:
            task = progress.add_task("[green]Creating fingerprint...", total=100)

            fingerprinter = VideoFingerprinter()
            fingerprint = fingerprinter.fingerprint_video(video_path)

            progress.update(task, completed=100)

        # Display results
        panel_content = f"""
[bold]Video:[/bold] {video_path.name}
[bold]Fingerprint ID:[/bold] {fingerprint.fingerprint_id}
[bold]Perceptual Hash:[/bold] {fingerprint.perceptual_hash}
[bold]Duration:[/bold] {fingerprint.duration:.2f} seconds
[bold]Key Frames:[/bold] {len(fingerprint.key_frames)}
[bold]Temporal Features:[/bold] {len(fingerprint.temporal_features)}
        """

        console.print(Panel(panel_content, title="Video Fingerprint Results"))

        # Show temporal features
        if fingerprint.temporal_features:
            console.print(f"\n[cyan]Temporal Features:[/cyan] {', '.join(fingerprint.temporal_features)}")

        # Show key frame information
        if fingerprint.key_frames:
            table = Table(title="Key Frames")
            table.add_column("Frame #", style="cyan")
            table.add_column("Timestamp", style="green")
            table.add_column("Hash", style="yellow")
            table.add_column("Brightness", style="white")

            for frame in fingerprint.key_frames[:10]:  # Show first 10
                brightness = frame.features.get('mean_brightness', 0)
                table.add_row(
                    str(frame.frame_number),
                    f"{frame.timestamp:.2f}s",
                    frame.frame_hash[:16] + "...",
                    f"{brightness:.1f}"
                )

            console.print(table)

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(fingerprint.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Fingerprint saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error generating fingerprint: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze_compression(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    detailed: bool = typer.Option(False, help="Show detailed analysis")
):
    """Analyze video compression artifacts"""

    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Analyzing compression artifacts: {video_path}[/cyan]")

        with Progress() as progress:
            task = progress.add_task("[green]Analyzing compression...", total=100)

            analyzer = CompressionAnalyzer()
            analysis = analyzer.analyze_compression_artifacts(video_path)

            progress.update(task, completed=100)

        # Display results
        panel_content = f"""
[bold]Video:[/bold] {video_path.name}
[bold]Compression Type:[/bold] {analysis.compression_type.value.upper()}
[bold]Recompression Count:[/bold] {analysis.recompression_count}
[bold]Artifacts Detected:[/bold] {len(analysis.artifacts_detected)}
[bold]Authenticity Indicators:[/bold] {len(analysis.authenticity_indicators)}
        """

        console.print(Panel(panel_content, title="Compression Analysis Results"))

        # Show quality metrics
        if analysis.quality_metrics:
            table = Table(title="Quality Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for metric, value in analysis.quality_metrics.items():
                if isinstance(value, float):
                    table.add_row(metric.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    table.add_row(metric.replace('_', ' ').title(), str(value))

            console.print(table)

        # Show artifacts detected
        if analysis.artifacts_detected:
            console.print(f"\n[yellow]Artifacts Detected:[/yellow]")
            for artifact in analysis.artifacts_detected:
                console.print(f"  • {artifact.replace('_', ' ').title()}")

        # Show authenticity indicators
        if analysis.authenticity_indicators:
            console.print(f"\n[cyan]Authenticity Indicators:[/cyan]")
            for indicator in analysis.authenticity_indicators:
                console.print(f"  • {indicator.replace('_', ' ').title()}")

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(analysis.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Analysis saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error analyzing compression: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def extract_frames(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    max_frames: int = typer.Option(50, help="Maximum key frames to extract")
):
    """Extract and analyze key frames from video"""

    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Extracting key frames from: {video_path}[/cyan]")

        with Progress() as progress:
            task = progress.add_task("[green]Extracting frames...", total=100)

            analyzer = FrameAnalyzer()
            key_frames = analyzer.extract_key_frames(video_path)

            progress.update(task, completed=100)

        # Display results
        panel_content = f"""
[bold]Video:[/bold] {video_path.name}
[bold]Key Frames Extracted:[/bold] {len(key_frames)}
[bold]Frames with Artifacts:[/bold] {sum(1 for f in key_frames if f.artifacts)}
        """

        console.print(Panel(panel_content, title="Key Frame Extraction Results"))

        # Show frame details
        table = Table(title="Key Frames")
        table.add_column("Frame #", style="cyan")
        table.add_column("Time", style="green")
        table.add_column("Hash", style="yellow")
        table.add_column("Artifacts", style="red")
        table.add_column("Brightness", style="white")

        for frame in key_frames[:20]:  # Show first 20
            artifacts_str = ", ".join(frame.artifacts) if frame.artifacts else "None"
            brightness = frame.features.get('mean_brightness', 0)

            table.add_row(
                str(frame.frame_number),
                f"{frame.timestamp:.1f}s",
                frame.frame_hash[:12] + "...",
                artifacts_str[:20] + ("..." if len(artifacts_str) > 20 else ""),
                f"{brightness:.1f}"
            )

        console.print(table)

        # Show artifact summary
        all_artifacts = []
        for frame in key_frames:
            all_artifacts.extend(frame.artifacts)

        if all_artifacts:
            from collections import Counter
            artifact_counts = Counter(all_artifacts)

            console.print(f"\n[yellow]Artifact Summary:[/yellow]")
            for artifact, count in artifact_counts.most_common():
                console.print(f"  • {artifact.replace('_', ' ').title()}: {count} frames")

        # Save results if requested
        if output:
            frame_data = [frame.model_dump(mode='json') for frame in key_frames]
            with open(output, 'w') as f:
                json.dump(frame_data, f, indent=2, default=str)
            console.print(f"[green]✓ Frame data saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error extracting frames: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare_videos(
    video1: Path = typer.Argument(..., help="Path to first video"),
    video2: Path = typer.Argument(..., help="Path to second video"),
    threshold: float = typer.Option(0.8, help="Similarity threshold (0.0-1.0)")
):
    """Compare two videos for similarity"""

    for video in [video1, video2]:
        if not video.exists():
            console.print(f"[red]Error: Video file not found: {video}[/red]")
            raise typer.Exit(1)

    try:
        console.print(f"[cyan]Comparing videos for similarity...[/cyan]")
        console.print(f"Video 1: {video1.name}")
        console.print(f"Video 2: {video2.name}")

        with Progress() as progress:
            task = progress.add_task("[green]Generating fingerprints...", total=100)

            fingerprinter = VideoFingerprinter()

            progress.update(task, completed=33)
            fingerprint1 = fingerprinter.fingerprint_video(video1)

            progress.update(task, completed=66)
            fingerprint2 = fingerprinter.fingerprint_video(video2)

            progress.update(task, completed=100)

        # Calculate similarity
        similarity_score = _calculate_similarity(fingerprint1, fingerprint2)
        is_similar = similarity_score >= threshold

        # Display results
        status_color = "green" if is_similar else "red"
        panel_content = f"""
[bold]Video 1:[/bold] {video1.name}
[bold]Video 2:[/bold] {video2.name}
[bold]Similarity Score:[/bold] {similarity_score:.3f}
[bold]Threshold:[/bold] {threshold}
[bold]Similar:[/bold] [{status_color}]{"YES" if is_similar else "NO"}[/{status_color}]
        """

        console.print(Panel(panel_content, title="Video Comparison Results"))

        # Show detailed comparison
        table = Table(title="Comparison Details")
        table.add_column("Property", style="cyan")
        table.add_column("Video 1", style="green")
        table.add_column("Video 2", style="yellow")
        table.add_column("Match", style="white")

        table.add_row(
            "Duration",
            f"{fingerprint1.duration:.2f}s",
            f"{fingerprint2.duration:.2f}s",
            "✓" if abs(fingerprint1.duration - fingerprint2.duration) < 1.0 else "✗"
        )

        table.add_row(
            "Key Frames",
            str(len(fingerprint1.key_frames)),
            str(len(fingerprint2.key_frames)),
            "✓" if abs(len(fingerprint1.key_frames) - len(fingerprint2.key_frames)) <= 2 else "✗"
        )

        table.add_row(
            "Perceptual Hash",
            fingerprint1.perceptual_hash[:16] + "...",
            fingerprint2.perceptual_hash[:16] + "...",
            "✓" if fingerprint1.perceptual_hash == fingerprint2.perceptual_hash else "✗"
        )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error comparing videos: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def get_metadata(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Extract video metadata"""

    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Extracting metadata from: {video_path}[/cyan]")

        # Extract metadata using OpenCV
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Get codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        # Get file info
        file_size = video_path.stat().st_size
        file_hash = ""

        # Create metadata object
        metadata = VideoMetadata(
            file_path=video_path,
            file_size=file_size,
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            codec=codec,
            file_hash=file_hash
        )

        # Display results
        panel_content = f"""
[bold]File:[/bold] {video_path.name}
[bold]Size:[/bold] {file_size:,} bytes
[bold]Duration:[/bold] {duration:.2f} seconds
[bold]Resolution:[/bold] {width}x{height}
[bold]FPS:[/bold] {fps:.2f}
[bold]Frame Count:[/bold] {frame_count:,}
[bold]Codec:[/bold] {codec}
[bold]File Hash:[/bold] {metadata.file_hash[:32]}...
        """

        console.print(Panel(panel_content, title="Video Metadata"))

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(metadata.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Metadata saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error extracting metadata: {e}[/red]")
        raise typer.Exit(1)


def _calculate_similarity(fp1: VideoFingerprint, fp2: VideoFingerprint) -> float:
    """Calculate similarity score between two video fingerprints"""
    similarity_factors = []

    # Compare perceptual hashes
    if fp1.perceptual_hash == fp2.perceptual_hash:
        similarity_factors.append(1.0)
    else:
        # Calculate Hamming distance for hash similarity
        hash1_bin = bin(int(fp1.perceptual_hash, 16))[2:].zfill(64)
        hash2_bin = bin(int(fp2.perceptual_hash, 16))[2:].zfill(64)
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1_bin, hash2_bin))
        hash_similarity = 1.0 - (hamming_distance / 64.0)
        similarity_factors.append(hash_similarity)

    # Compare duration
    duration_diff = abs(fp1.duration - fp2.duration)
    max_duration = max(fp1.duration, fp2.duration)
    duration_similarity = 1.0 - min(duration_diff / max_duration, 1.0) if max_duration > 0 else 1.0
    similarity_factors.append(duration_similarity)

    # Compare temporal features
    common_features = set(fp1.temporal_features) & set(fp2.temporal_features)
    total_features = set(fp1.temporal_features) | set(fp2.temporal_features)
    feature_similarity = len(common_features) / len(total_features) if total_features else 1.0
    similarity_factors.append(feature_similarity)

    # Return weighted average
    return sum(similarity_factors) / len(similarity_factors)


if __name__ == "__main__":
    app()