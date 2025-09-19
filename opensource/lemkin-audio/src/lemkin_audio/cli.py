"""
Command-line interface for the Lemkin Audio Analysis Toolkit.

Provides comprehensive CLI commands for audio transcription, speaker identification,
authenticity verification, and audio enhancement.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .core import (
    AudioAnalysis,
    AudioEnhancer,
    AudioAuthenticator,
    EnhancementSettings,
    LanguageCode,
    SpeakerIdentifier,
    SpeechTranscriber,
)

app = typer.Typer(
    name="lemkin-audio",
    help="Audio analysis toolkit for speech transcription, authentication and forensic analysis",
    no_args_is_help=True,
)

console = Console()


def validate_audio_file(file_path: str) -> Path:
    """Validate that the file exists and is a supported audio format."""
    path = Path(file_path)

    if not path.exists():
        raise typer.BadParameter(f"Audio file does not exist: {file_path}")

    supported_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}
    if path.suffix.lower() not in supported_extensions:
        raise typer.BadParameter(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported formats: {', '.join(supported_extensions)}"
        )

    return path


def save_results(results: dict, output_path: Optional[Path]) -> None:
    """Save analysis results to JSON file."""
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            console.print(f"✅ Results saved to: {output_path}")
        except Exception as e:
            console.print(f"❌ Failed to save results: {e}", style="red")


@app.command()
def transcribe(
    audio_file: str = typer.Argument(..., help="Path to audio file to transcribe"),
    language: Optional[LanguageCode] = typer.Option(
        None, "--language", "-l", help="Target language for transcription"
    ),
    segment_length: float = typer.Option(
        30.0, "--segment-length", "-s", help="Segment length in seconds", min=5.0, max=300.0
    ),
    model_size: str = typer.Option(
        "base", "--model", "-m", help="Transcription model size",
        click_type=typer.Choice(["tiny", "base", "small", "medium", "large"])
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for results (JSON)"
    ),
    show_segments: bool = typer.Option(
        False, "--segments", help="Show individual speech segments"
    ),
) -> None:
    """Transcribe speech from audio file to text with timestamps."""

    try:
        audio_path = validate_audio_file(audio_file)
        output_path = Path(output) if output else None

        console.print(f"🎙️ Transcribing audio: {audio_path.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Transcribing audio...", total=None)

            transcriber = SpeechTranscriber(model_size=model_size)
            result = transcriber.transcribe_audio(
                audio_path=audio_path,
                language=language,
                segment_length=segment_length
            )

            progress.update(task, description="Transcription completed!")

        # Display results
        console.print("\n" + "="*60)
        console.print(Panel(
            f"[bold green]Transcription Results[/bold green]\n\n"
            f"📁 File: {audio_path.name}\n"
            f"⏱️ Duration: {result.total_duration:.1f} seconds\n"
            f"🔤 Segments: {len(result.segments)}\n"
            f"🌐 Language: {result.detected_language or 'Auto-detected'}\n"
            f"⚡ Processing time: {result.processing_time:.2f}s",
            title="Audio Transcription"
        ))

        console.print(f"\n[bold blue]Full Transcription:[/bold blue]")
        console.print(Panel(result.full_text or "[No speech detected]"))

        if show_segments and result.segments:
            console.print(f"\n[bold blue]Speech Segments:[/bold blue]")

            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Start", width=8)
            table.add_column("End", width=8)
            table.add_column("Text", min_width=40)
            table.add_column("Confidence", width=10)

            for segment in result.segments:
                table.add_row(
                    f"{segment.start_time:.1f}s",
                    f"{segment.end_time:.1f}s",
                    segment.text[:80] + "..." if len(segment.text) > 80 else segment.text,
                    f"{segment.confidence:.1%}"
                )

            console.print(table)

        # Save results
        if output_path:
            results = {
                "transcription_result": result.dict(),
                "analysis_type": "speech_transcription"
            }
            save_results(results, output_path)

        console.print(f"\n✅ Transcription completed successfully!")

    except Exception as e:
        console.print(f"❌ Transcription failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def identify_speaker(
    audio_file: str = typer.Argument(..., help="Path to audio file to analyze"),
    profile_dir: Optional[str] = typer.Option(
        None, "--profiles", "-p", help="Directory containing speaker profiles"
    ),
    create_profile: Optional[str] = typer.Option(
        None, "--create-profile", help="Create new speaker profile with this ID"
    ),
    confidence_threshold: float = typer.Option(
        0.8, "--threshold", "-t", help="Confidence threshold for identification", min=0.0, max=1.0
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for results (JSON)"
    ),
) -> None:
    """Identify speakers in audio file using voice profiles."""

    try:
        audio_path = validate_audio_file(audio_file)
        output_path = Path(output) if output else None

        console.print(f"🔍 Analyzing speakers in: {audio_path.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing speakers...", total=None)

            identifier = SpeakerIdentifier()

            # Load existing profiles if directory provided
            known_profiles = []
            if profile_dir:
                profile_path = Path(profile_dir)
                if profile_path.exists():
                    # In a real implementation, load saved profiles
                    console.print(f"📂 Loading profiles from: {profile_path}")

            if create_profile:
                # Create new profile mode
                profile = identifier.create_speaker_profile(
                    speaker_id=create_profile,
                    audio_samples=[audio_path],
                    confidence_threshold=confidence_threshold
                )
                console.print(f"✅ Created speaker profile: {create_profile}")

                # Save profile if directory provided
                if profile_dir:
                    profile_path = Path(profile_dir)
                    profile_path.mkdir(exist_ok=True)
                    profile_file = profile_path / f"{create_profile}.json"

                    with open(profile_file, 'w') as f:
                        json.dump(profile.dict(), f, indent=2, default=str)

                    console.print(f"💾 Profile saved to: {profile_file}")

                return

            # Perform speaker identification
            result = identifier.identify_speaker(audio_path, known_profiles)

            progress.update(task, description="Speaker identification completed!")

        # Display results
        console.print("\n" + "="*60)
        console.print(Panel(
            f"[bold green]Speaker Identification Results[/bold green]\n\n"
            f"📁 File: {audio_path.name}\n"
            f"👥 Speakers detected: {result.total_speakers}\n"
            f"🎯 Analysis confidence: {result.analysis_confidence:.1%}",
            title="Speaker Analysis"
        ))

        if result.identified_speakers:
            console.print(f"\n[bold blue]Identified Speakers:[/bold blue]")

            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Speaker ID", width=15)
            table.add_column("Similarity", width=12)
            table.add_column("Confidence", width=12)

            for speaker in result.identified_speakers:
                table.add_row(
                    speaker["speaker_id"],
                    f"{speaker['similarity_score']:.1%}",
                    f"{speaker['confidence']:.1%}"
                )

            console.print(table)
        else:
            console.print("❓ No speakers identified with current profiles")

        if result.speaker_segments:
            console.print(f"\n[bold blue]Speaker Segments:[/bold blue]")

            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Start", width=8)
            table.add_column("End", width=8)
            table.add_column("Speaker", width=15)
            table.add_column("Confidence", width=12)

            for segment in result.speaker_segments:
                table.add_row(
                    f"{segment['start_time']:.1f}s",
                    f"{segment['end_time']:.1f}s",
                    segment["speaker_id"],
                    f"{segment['confidence']:.1%}"
                )

            console.print(table)

        # Save results
        if output_path:
            results = {
                "speaker_identification": result.dict(),
                "analysis_type": "speaker_identification"
            }
            save_results(results, output_path)

        console.print(f"\n✅ Speaker identification completed!")

    except Exception as e:
        console.print(f"❌ Speaker identification failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def verify_authenticity(
    audio_file: str = typer.Argument(..., help="Path to audio file to verify"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for results (JSON)"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed analysis for each indicator"
    ),
) -> None:
    """Verify audio authenticity and detect potential tampering."""

    try:
        audio_path = validate_audio_file(audio_file)
        output_path = Path(output) if output else None

        console.print(f"🔒 Verifying authenticity of: {audio_path.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing authenticity...", total=None)

            authenticator = AudioAuthenticator()
            report = authenticator.verify_audio_authenticity(audio_path)

            progress.update(task, description="Authenticity verification completed!")

        # Display results
        console.print("\n" + "="*60)

        # Overall result with color coding
        authenticity_color = "green" if report.overall_authenticity else "red"
        authenticity_text = "AUTHENTIC" if report.overall_authenticity else "SUSPICIOUS"
        tampering_text = "DETECTED" if report.tampering_detected else "NOT DETECTED"

        console.print(Panel(
            f"[bold {authenticity_color}]Overall Assessment: {authenticity_text}[/bold {authenticity_color}]\n\n"
            f"📁 File: {audio_path.name}\n"
            f"🎯 Confidence: {report.authenticity_confidence:.1%}\n"
            f"⚠️ Tampering: {tampering_text}\n"
            f"🔍 Indicators tested: {len(report.indicators)}",
            title="Audio Authenticity Report"
        ))

        # Authenticity indicators
        console.print(f"\n[bold blue]Authenticity Indicators:[/bold blue]")

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Indicator", width=20)
        table.add_column("Result", width=12)
        table.add_column("Confidence", width=12)
        table.add_column("Details", min_width=30)

        for indicator in report.indicators:
            result_text = "✅ Authentic" if indicator.is_authentic else "❌ Suspicious"
            result_color = "green" if indicator.is_authentic else "red"

            details_text = ""
            if detailed and indicator.details:
                key_details = []
                for key, value in indicator.details.items():
                    if key not in ['error'] and isinstance(value, (int, float)):
                        key_details.append(f"{key}: {value:.3f}")
                details_text = ", ".join(key_details[:2])  # Show first 2 details

            table.add_row(
                indicator.indicator_name.replace("_", " ").title(),
                Text(result_text, style=result_color),
                f"{indicator.confidence:.1%}",
                details_text
            )

        console.print(table)

        # Technical specifications
        if detailed and report.technical_analysis:
            console.print(f"\n[bold blue]Technical Analysis:[/bold blue]")
            tech_table = Table(show_header=True, header_style="bold blue")
            tech_table.add_column("Property", width=20)
            tech_table.add_column("Value", width=20)

            for key, value in report.technical_analysis.items():
                tech_table.add_row(key.replace("_", " ").title(), str(value))

            console.print(tech_table)

        # Save results
        if output_path:
            results = {
                "authenticity_report": report.dict(),
                "analysis_type": "audio_authenticity"
            }
            save_results(results, output_path)

        console.print(f"\n✅ Authenticity verification completed!")

        # Exit with appropriate code
        if not report.overall_authenticity:
            console.print("⚠️ [yellow]Warning: Audio shows signs of potential tampering[/yellow]")

    except Exception as e:
        console.print(f"❌ Authenticity verification failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def enhance(
    audio_file: str = typer.Argument(..., help="Path to audio file to enhance"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for enhanced audio"
    ),
    noise_reduction: bool = typer.Option(
        True, "--noise-reduction/--no-noise-reduction", help="Apply noise reduction"
    ),
    normalize: bool = typer.Option(
        True, "--normalize/--no-normalize", help="Normalize audio gain"
    ),
    frequency_filter: bool = typer.Option(
        False, "--frequency-filter", help="Apply frequency filtering"
    ),
    low_cutoff: float = typer.Option(
        80.0, "--low-cutoff", help="Low frequency cutoff (Hz)", min=0.0
    ),
    high_cutoff: float = typer.Option(
        8000.0, "--high-cutoff", help="High frequency cutoff (Hz)", min=0.0
    ),
    spectral_subtraction: bool = typer.Option(
        False, "--spectral-subtraction", help="Apply spectral subtraction"
    ),
    echo_cancellation: bool = typer.Option(
        False, "--echo-cancellation", help="Apply echo cancellation"
    ),
    show_metrics: bool = typer.Option(
        False, "--metrics", help="Show quality improvement metrics"
    ),
) -> None:
    """Enhance audio quality using various processing techniques."""

    try:
        audio_path = validate_audio_file(audio_file)
        output_path = Path(output) if output else None

        console.print(f"🎚️ Enhancing audio: {audio_path.name}")

        # Create enhancement settings
        settings = EnhancementSettings(
            noise_reduction=noise_reduction,
            gain_normalization=normalize,
            frequency_filtering=frequency_filter,
            low_freq_cutoff=low_cutoff,
            high_freq_cutoff=high_cutoff,
            spectral_subtraction=spectral_subtraction,
            echo_cancellation=echo_cancellation
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Enhancing audio...", total=None)

            enhancer = AudioEnhancer()
            result = enhancer.enhance_audio(
                audio_path=audio_path,
                output_path=output_path,
                settings=settings
            )

            progress.update(task, description="Audio enhancement completed!")

        # Display results
        console.print("\n" + "="*60)
        console.print(Panel(
            f"[bold green]Audio Enhancement Results[/bold green]\n\n"
            f"📁 Original: {Path(result.original_path).name}\n"
            f"✨ Enhanced: {Path(result.enhanced_path).name}\n"
            f"⚡ Processing time: {result.processing_time:.2f}s",
            title="Audio Enhancement"
        ))

        # Enhancement settings applied
        console.print(f"\n[bold blue]Applied Enhancements:[/bold blue]")
        settings_list = []
        if settings.noise_reduction:
            settings_list.append("✅ Noise Reduction")
        if settings.gain_normalization:
            settings_list.append("✅ Gain Normalization")
        if settings.frequency_filtering:
            settings_list.append(f"✅ Frequency Filtering ({settings.low_freq_cutoff}-{settings.high_freq_cutoff} Hz)")
        if settings.spectral_subtraction:
            settings_list.append("✅ Spectral Subtraction")
        if settings.echo_cancellation:
            settings_list.append("✅ Echo Cancellation")

        for setting in settings_list:
            console.print(f"  {setting}")

        # Quality metrics
        if show_metrics and result.quality_improvement:
            console.print(f"\n[bold blue]Quality Improvement Metrics:[/bold blue]")

            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Metric", width=20)
            table.add_column("Improvement", width=15)
            table.add_column("Status", width=10)

            for metric, improvement in result.quality_improvement.items():
                status = "✅ Better" if improvement > 0 else "➖ Neutral" if improvement == 0 else "❌ Worse"
                status_color = "green" if improvement > 0 else "yellow" if improvement == 0 else "red"

                table.add_row(
                    metric.replace("_", " ").title(),
                    f"{improvement:+.3f}",
                    Text(status, style=status_color)
                )

            console.print(table)

        console.print(f"\n✅ Audio enhancement completed!")
        console.print(f"📁 Enhanced audio saved to: {result.enhanced_path}")

    except Exception as e:
        console.print(f"❌ Audio enhancement failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def comprehensive_analysis(
    audio_file: str = typer.Argument(..., help="Path to audio file to analyze"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for complete analysis (JSON)"
    ),
    include_transcription: bool = typer.Option(
        True, "--transcription/--no-transcription", help="Include speech transcription"
    ),
    include_speaker_analysis: bool = typer.Option(
        False, "--speaker-analysis", help="Include speaker identification"
    ),
    include_authenticity: bool = typer.Option(
        True, "--authenticity/--no-authenticity", help="Include authenticity verification"
    ),
    include_enhancement: bool = typer.Option(
        False, "--enhancement", help="Include audio enhancement"
    ),
    language: Optional[LanguageCode] = typer.Option(
        None, "--language", "-l", help="Target language for transcription"
    ),
) -> None:
    """Perform comprehensive audio analysis including all available techniques."""

    try:
        audio_path = validate_audio_file(audio_file)
        output_path = Path(output) if output else None

        console.print(f"🔬 Comprehensive analysis of: {audio_path.name}")

        analysis = AudioAnalysis(
            audio_path=str(audio_path),
            file_metadata={},
            technical_specs={}
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Basic file analysis
            task = progress.add_task("Analyzing file metadata...", total=None)

            import librosa
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = len(y) / sr

            analysis.file_metadata = {
                "filename": audio_path.name,
                "file_size": audio_path.stat().st_size,
                "format": audio_path.suffix.lower()
            }

            analysis.technical_specs = {
                "sample_rate": sr,
                "duration_seconds": duration,
                "total_samples": len(y),
                "channels": 1  # librosa loads as mono
            }

            # Transcription
            if include_transcription:
                progress.update(task, description="Transcribing speech...")
                transcriber = SpeechTranscriber()
                analysis.transcription = transcriber.transcribe_audio(
                    audio_path=audio_path,
                    language=language
                )

            # Speaker analysis
            if include_speaker_analysis:
                progress.update(task, description="Analyzing speakers...")
                identifier = SpeakerIdentifier()
                analysis.speaker_analysis = identifier.identify_speaker(audio_path)

            # Authenticity verification
            if include_authenticity:
                progress.update(task, description="Verifying authenticity...")
                authenticator = AudioAuthenticator()
                analysis.authenticity_report = authenticator.verify_audio_authenticity(audio_path)

            # Audio enhancement
            if include_enhancement:
                progress.update(task, description="Enhancing audio quality...")
                enhancer = AudioEnhancer()
                analysis.enhancement_result = enhancer.enhance_audio(audio_path)

            progress.update(task, description="Comprehensive analysis completed!")

        # Display summary results
        console.print("\n" + "="*80)
        console.print(Panel(
            f"[bold green]Comprehensive Audio Analysis Results[/bold green]\n\n"
            f"📁 File: {audio_path.name}\n"
            f"💾 Size: {analysis.file_metadata['file_size']:,} bytes\n"
            f"⏱️ Duration: {analysis.technical_specs['duration_seconds']:.1f} seconds\n"
            f"📊 Sample Rate: {analysis.technical_specs['sample_rate']:,} Hz",
            title="Audio Analysis Summary"
        ))

        # Show results summary for each component
        if analysis.transcription:
            console.print(f"\n[bold blue]🎙️ Transcription:[/bold blue] {len(analysis.transcription.segments)} segments, {len(analysis.transcription.full_text)} characters")

        if analysis.speaker_analysis:
            console.print(f"\n[bold blue]👥 Speaker Analysis:[/bold blue] {analysis.speaker_analysis.total_speakers} speakers identified")

        if analysis.authenticity_report:
            auth_status = "AUTHENTIC" if analysis.authenticity_report.overall_authenticity else "SUSPICIOUS"
            auth_color = "green" if analysis.authenticity_report.overall_authenticity else "red"
            console.print(f"\n[bold blue]🔒 Authenticity:[/bold blue] [{auth_color}]{auth_status}[/{auth_color}] ({analysis.authenticity_report.authenticity_confidence:.1%} confidence)")

        if analysis.enhancement_result:
            console.print(f"\n[bold blue]✨ Enhancement:[/bold blue] Quality improved, saved to {Path(analysis.enhancement_result.enhanced_path).name}")

        # Save complete analysis
        if output_path:
            results = {
                "comprehensive_analysis": analysis.dict(),
                "analysis_type": "comprehensive_audio_analysis"
            }
            save_results(results, output_path)

        console.print(f"\n✅ Comprehensive analysis completed successfully!")

    except Exception as e:
        console.print(f"❌ Comprehensive analysis failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def batch_process(
    audio_dir: str = typer.Argument(..., help="Directory containing audio files to process"),
    output_dir: str = typer.Argument(..., help="Directory to save analysis results"),
    analysis_type: str = typer.Option(
        "transcription", "--type", "-t",
        help="Type of analysis to perform",
        click_type=typer.Choice(["transcription", "authenticity", "speaker", "enhancement", "comprehensive"])
    ),
    file_pattern: str = typer.Option(
        "*.wav", "--pattern", "-p", help="File pattern to match (e.g., '*.wav', '*.mp3')"
    ),
    language: Optional[LanguageCode] = typer.Option(
        None, "--language", "-l", help="Target language for transcription"
    ),
) -> None:
    """Process multiple audio files in batch mode."""

    try:
        input_dir = Path(audio_dir)
        output_path = Path(output_dir)

        if not input_dir.exists():
            raise typer.BadParameter(f"Input directory does not exist: {audio_dir}")

        output_path.mkdir(parents=True, exist_ok=True)

        # Find audio files
        audio_files = list(input_dir.glob(file_pattern))

        if not audio_files:
            console.print(f"❌ No audio files found matching pattern '{file_pattern}' in {input_dir}")
            raise typer.Exit(1)

        console.print(f"🔄 Processing {len(audio_files)} audio files...")
        console.print(f"📂 Input: {input_dir}")
        console.print(f"📁 Output: {output_path}")
        console.print(f"🎯 Analysis: {analysis_type}")

        # Process each file
        results_summary = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            main_task = progress.add_task(f"Processing files...", total=len(audio_files))

            for i, audio_file in enumerate(audio_files):
                try:
                    progress.update(main_task, description=f"Processing {audio_file.name}...")

                    # Determine output file
                    output_file = output_path / f"{audio_file.stem}_{analysis_type}.json"

                    # Perform analysis based on type
                    if analysis_type == "transcription":
                        transcriber = SpeechTranscriber()
                        result = transcriber.transcribe_audio(
                            audio_path=audio_file,
                            language=language
                        )
                        analysis_data = {"transcription_result": result.dict()}

                    elif analysis_type == "authenticity":
                        authenticator = AudioAuthenticator()
                        result = authenticator.verify_audio_authenticity(audio_file)
                        analysis_data = {"authenticity_report": result.dict()}

                    elif analysis_type == "speaker":
                        identifier = SpeakerIdentifier()
                        result = identifier.identify_speaker(audio_file)
                        analysis_data = {"speaker_identification": result.dict()}

                    elif analysis_type == "enhancement":
                        enhancer = AudioEnhancer()
                        enhanced_path = output_path / f"{audio_file.stem}_enhanced{audio_file.suffix}"
                        result = enhancer.enhance_audio(audio_file, enhanced_path)
                        analysis_data = {"enhancement_result": result.dict()}

                    elif analysis_type == "comprehensive":
                        # Comprehensive analysis
                        analysis = AudioAnalysis(
                            audio_path=str(audio_file),
                            file_metadata={"filename": audio_file.name},
                            technical_specs={}
                        )

                        # Add components
                        transcriber = SpeechTranscriber()
                        analysis.transcription = transcriber.transcribe_audio(audio_file, language)

                        authenticator = AudioAuthenticator()
                        analysis.authenticity_report = authenticator.verify_audio_authenticity(audio_file)

                        analysis_data = {"comprehensive_analysis": analysis.dict()}

                    # Save results
                    analysis_data["analysis_type"] = analysis_type
                    analysis_data["processed_file"] = str(audio_file)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, indent=2, default=str, ensure_ascii=False)

                    results_summary.append({
                        "file": audio_file.name,
                        "status": "success",
                        "output": output_file.name
                    })

                except Exception as e:
                    console.print(f"❌ Failed to process {audio_file.name}: {e}")
                    results_summary.append({
                        "file": audio_file.name,
                        "status": "error",
                        "error": str(e)
                    })

                progress.advance(main_task)

        # Show summary
        successful = sum(1 for r in results_summary if r["status"] == "success")
        failed = len(results_summary) - successful

        console.print(f"\n✅ Batch processing completed!")
        console.print(f"📊 Successfully processed: {successful}/{len(audio_files)} files")

        if failed > 0:
            console.print(f"❌ Failed: {failed} files")

            # Show failed files
            for result in results_summary:
                if result["status"] == "error":
                    console.print(f"  • {result['file']}: {result['error']}")

        # Save batch summary
        summary_file = output_path / f"batch_summary_{analysis_type}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "batch_analysis_type": analysis_type,
                "total_files": len(audio_files),
                "successful": successful,
                "failed": failed,
                "results": results_summary
            }, f, indent=2)

        console.print(f"📋 Batch summary saved to: {summary_file}")

    except Exception as e:
        console.print(f"❌ Batch processing failed: {str(e)}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()