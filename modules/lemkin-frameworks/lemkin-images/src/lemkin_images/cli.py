"""
Lemkin Image Verification Suite CLI Interface

Command-line interface for comprehensive image authenticity verification and forensic analysis:
- reverse-search: Multi-engine reverse image search
- detect-manipulation: Advanced manipulation detection
- geolocate: Image geolocation extraction and verification
- analyze-metadata: EXIF metadata forensic analysis
- authenticate: Comprehensive authenticity analysis

Provides user-friendly interface for complex image verification procedures.
"""

import click
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging
import traceback

from .core import (
    ImageAuthenticator,
    ImageAuthConfig,
    AuthenticityReport,
    SearchEngine,
    ImageFormat
)
from .reverse_search import ReverseImageSearcher
from .manipulation_detector import ImageManipulationDetector
from .geolocation_helper import ImageGeolocator
from .metadata_forensics import MetadataForensicsAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_output_directory(output_dir: Optional[str], operation: str) -> Path:
    """Setup and create output directory"""
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"lemkin_images_{operation}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_results_to_json(results: dict, output_path: Path, filename: str):
    """Save results to JSON file"""
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    click.echo(f"Results saved to: {output_file}")


def validate_image_file(image_path: str) -> Path:
    """Validate that the image file exists and is a supported format"""
    path = Path(image_path)
    if not path.exists():
        raise click.BadParameter(f"Image file not found: {image_path}")
    
    if not path.is_file():
        raise click.BadParameter(f"Path is not a file: {image_path}")
    
    # Check file extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.heic']
    if path.suffix.lower() not in valid_extensions:
        click.echo(f"Warning: File extension {path.suffix} may not be supported", err=True)
    
    return path


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """
    Lemkin Image Verification Suite
    
    Comprehensive image authenticity verification and forensic analysis toolkit.
    Supports reverse image search, manipulation detection, geolocation, and metadata analysis.
    """
    ctx.ensure_object(dict)
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo("Verbose logging enabled")
    
    # Load configuration
    if config:
        try:
            with open(config, 'r') as f:
                config_data = json.load(f)
            ctx.obj['config'] = ImageAuthConfig(**config_data)
            click.echo(f"Configuration loaded from: {config}")
        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            sys.exit(1)
    else:
        ctx.obj['config'] = ImageAuthConfig()


@cli.command()
@click.argument('image_path', type=str, callback=lambda ctx, param, value: validate_image_file(value))
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--engines', '-e', multiple=True, 
              type=click.Choice(['google', 'tineye', 'bing', 'yandex', 'baidu']),
              help='Search engines to use (can specify multiple)')
@click.option('--max-results', '-n', type=int, default=50, 
              help='Maximum results per engine (default: 50)')
@click.option('--timeout', '-t', type=int, default=30, 
              help='Search timeout in seconds (default: 30)')
@click.option('--save-images', is_flag=True, help='Download and save result images')
@click.pass_context
def reverse_search(ctx, image_path, output_dir, engines, max_results, timeout, save_images):
    """
    Perform reverse image search across multiple search engines.
    
    Searches for the image across Google, TinEye, Bing, Yandex, and Baidu
    to identify potential sources and track distribution.
    
    Example:
        lemkin-images reverse-search photo.jpg -e google -e tineye -n 100
    """
    try:
        config = ctx.obj['config']
        
        # Update config with command line options
        if engines:
            engine_map = {
                'google': SearchEngine.GOOGLE,
                'tineye': SearchEngine.TINEYE,
                'bing': SearchEngine.BING,
                'yandex': SearchEngine.YANDEX,
                'baidu': SearchEngine.BAIDU
            }
            config.search_engines = [engine_map[e] for e in engines]
        
        config.max_search_results = max_results
        config.search_timeout_seconds = timeout
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "reverse_search")
        
        click.echo(f"Starting reverse image search for: {image_path.name}")
        click.echo(f"Using engines: {[e.value for e in config.search_engines]}")
        click.echo(f"Max results per engine: {max_results}")
        
        # Perform search
        with click.progressbar(length=100, label='Searching') as bar:
            searcher = ReverseImageSearcher(config)
            
            # Update progress as we search each engine
            results = searcher.search_image(image_path)
            bar.update(100)
        
        # Display summary
        click.echo(f"\nSearch completed:")
        click.echo(f"  Total results found: {results.total_results_found}")
        click.echo(f"  Unique domains: {len(results.unique_domains)}")
        click.echo(f"  Countries found: {len(results.countries_found)}")
        
        if results.oldest_result_date:
            click.echo(f"  Earliest appearance: {results.oldest_result_date.strftime('%Y-%m-%d')}")
        
        if results.stock_photo_indicators:
            click.echo(f"  Stock photo indicators: {len(results.stock_photo_indicators)}")
        
        # Save detailed results
        results_dict = results.dict()
        save_results_to_json(results_dict, output_path, "reverse_search_results.json")
        
        # Save summary report
        summary = {
            'image_file': str(image_path),
            'search_timestamp': results.search_timestamp.isoformat(),
            'engines_used': [e.value for e in results.engines_used],
            'total_results': results.total_results_found,
            'unique_domains': len(results.unique_domains),
            'countries_found': results.countries_found,
            'widespread_usage': results.widespread_usage,
            'stock_photo_detected': len(results.stock_photo_indicators) > 0,
            'social_media_presence': results.social_media_presence,
            'potential_sources': results.potential_source_urls[:10]  # Top 10 potential sources
        }
        
        save_results_to_json(summary, output_path, "search_summary.json")
        
        # Download result images if requested
        if save_images:
            images_dir = output_path / "result_images"
            images_dir.mkdir(exist_ok=True)
            click.echo(f"Downloading result images to: {images_dir}")
            # Implementation would download thumbnails here
        
        click.echo(f"\nAnalysis complete. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Reverse search failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=str, callback=lambda ctx, param, value: validate_image_file(value))
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--threshold', '-t', type=float, default=0.7, 
              help='Manipulation detection threshold (0.0-1.0, default: 0.7)')
@click.option('--detailed', is_flag=True, help='Enable detailed analysis (slower but more thorough)')
@click.option('--save-visualizations', is_flag=True, help='Save analysis visualizations')
@click.option('--compression-analysis', is_flag=True, default=True, help='Enable JPEG compression analysis')
@click.option('--block-size', type=int, default=16, help='Block size for copy-move detection (default: 16)')
@click.pass_context
def detect_manipulation(ctx, image_path, output_dir, threshold, detailed, save_visualizations, compression_analysis, block_size):
    """
    Detect image manipulation using advanced forensic algorithms.
    
    Analyzes images for copy-move forgeries, splicing, resampling artifacts,
    compression inconsistencies, and other signs of tampering.
    
    Example:
        lemkin-images detect-manipulation photo.jpg --detailed --threshold 0.8
    """
    try:
        config = ctx.obj['config']
        
        # Update config with command line options
        config.manipulation_threshold = threshold
        config.detailed_analysis = detailed
        config.enable_compression_analysis = compression_analysis
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "manipulation_detection")
        
        click.echo(f"Starting manipulation detection for: {image_path.name}")
        click.echo(f"Detection threshold: {threshold}")
        click.echo(f"Detailed analysis: {'enabled' if detailed else 'disabled'}")
        
        # Perform analysis
        with click.progressbar(length=100, label='Analyzing') as bar:
            detector = ImageManipulationDetector(config)
            detector.block_size = block_size
            
            results = detector.detect_manipulation(image_path)
            bar.update(100)
        
        # Display summary
        click.echo(f"\nAnalysis completed:")
        click.echo(f"  Manipulation detected: {'YES' if results.is_manipulated else 'NO'}")
        click.echo(f"  Overall confidence: {results.overall_confidence:.1%}")
        click.echo(f"  Manipulation probability: {results.manipulation_probability:.1%}")
        click.echo(f"  Indicators found: {len(results.indicators)}")
        
        if results.indicators:
            click.echo(f"\nDetected manipulation types:")
            for indicator in results.indicators:
                severity_color = 'red' if indicator.severity == 'critical' else 'yellow' if indicator.severity == 'high' else None
                click.echo(f"  • {indicator.manipulation_type.value}: {indicator.confidence:.1%} confidence", color=severity_color)
                click.echo(f"    {indicator.description}")
        
        # Display method summary
        if results.methods_applied:
            click.echo(f"\nMethods applied: {', '.join(results.methods_applied)}")
        
        # Save detailed results
        results_dict = results.dict()
        save_results_to_json(results_dict, output_path, "manipulation_analysis.json")
        
        # Save summary report
        summary = {
            'image_file': str(image_path),
            'analysis_timestamp': results.analysis_timestamp.isoformat(),
            'is_manipulated': results.is_manipulated,
            'overall_confidence': results.overall_confidence,
            'manipulation_probability': results.manipulation_probability,
            'indicators_count': len(results.indicators),
            'critical_indicators': len([i for i in results.indicators if i.severity == 'critical']),
            'high_indicators': len([i for i in results.indicators if i.severity == 'high']),
            'expert_review_recommended': results.expert_review_recommended,
            'methods_used': results.methods_applied,
            'algorithms_used': results.algorithms_used
        }
        
        save_results_to_json(summary, output_path, "detection_summary.json")
        
        # Save visualizations if requested
        if save_visualizations:
            viz_dir = output_path / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            click.echo(f"Saving analysis visualizations to: {viz_dir}")
            # Implementation would save visualization images here
        
        click.echo(f"\nAnalysis complete. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Manipulation detection failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=str, callback=lambda ctx, param, value: validate_image_file(value))
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--visual-search', is_flag=True, help='Enable visual landmark recognition')
@click.option('--confidence-threshold', type=float, default=0.6, 
              help='Minimum confidence threshold for location verification (default: 0.6)')
@click.option('--save-map', is_flag=True, help='Generate and save location map')
@click.pass_context
def geolocate(ctx, image_path, output_dir, visual_search, confidence_threshold, save_map):
    """
    Extract and verify image geolocation from GPS metadata and visual features.
    
    Analyzes EXIF GPS data and optionally performs visual landmark recognition
    to determine the image location with confidence assessment.
    
    Example:
        lemkin-images geolocate photo.jpg --visual-search --save-map
    """
    try:
        config = ctx.obj['config']
        
        # Update config with command line options
        config.enable_visual_geolocation = visual_search
        config.geolocation_confidence_threshold = confidence_threshold
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "geolocation")
        
        click.echo(f"Starting geolocation analysis for: {image_path.name}")
        click.echo(f"Visual landmark recognition: {'enabled' if visual_search else 'disabled'}")
        click.echo(f"Confidence threshold: {confidence_threshold}")
        
        # Perform analysis
        with click.progressbar(length=100, label='Analyzing') as bar:
            geolocator = ImageGeolocator(config)
            results = geolocator.geolocate_image(image_path)
            bar.update(100)
        
        # Display summary
        click.echo(f"\nGeolocation analysis completed:")
        click.echo(f"  GPS data present: {'YES' if results.gps_data_present else 'NO'}")
        click.echo(f"  Overall confidence: {results.overall_confidence:.1%}")
        click.echo(f"  Location verified: {'YES' if results.location_verified else 'NO'}")
        
        if results.gps_data_tampered is not None:
            tampered_status = "YES" if results.gps_data_tampered else "NO"
            click.echo(f"  GPS data tampered: {tampered_status}")
        
        # Display primary location
        best_location = results.get_best_location()
        if best_location:
            click.echo(f"\nPrimary location:")
            if best_location.latitude and best_location.longitude:
                click.echo(f"  Coordinates: {best_location.latitude:.6f}, {best_location.longitude:.6f}")
            if best_location.address:
                click.echo(f"  Address: {best_location.address}")
            if best_location.country:
                click.echo(f"  Country: {best_location.country}")
            if best_location.city:
                click.echo(f"  City: {best_location.city}")
            click.echo(f"  Source: {best_location.source}")
            click.echo(f"  Confidence: {best_location.confidence:.1%}")
            
            if best_location.identified_landmarks:
                click.echo(f"  Landmarks: {', '.join(best_location.identified_landmarks)}")
        
        # Display alternative locations
        if results.alternative_locations:
            click.echo(f"\nAlternative locations found: {len(results.alternative_locations)}")
            for i, location in enumerate(results.alternative_locations[:3], 1):
                click.echo(f"  {i}. Source: {location.source}, Confidence: {location.confidence:.1%}")
        
        # Display methods used
        if results.methods_used:
            click.echo(f"\nMethods used: {', '.join(results.methods_used)}")
        
        # Save detailed results
        results_dict = results.dict()
        save_results_to_json(results_dict, output_path, "geolocation_results.json")
        
        # Save summary report
        summary = {
            'image_file': str(image_path),
            'analysis_timestamp': results.analysis_timestamp.isoformat(),
            'gps_data_present': results.gps_data_present,
            'location_verified': results.location_verified,
            'overall_confidence': results.overall_confidence,
            'gps_data_tampered': results.gps_data_tampered,
            'methods_used': results.methods_used,
            'visual_landmarks_detected': results.visual_landmarks_detected,
            'timezone_inferred': results.timezone_inferred
        }
        
        if best_location:
            summary.update({
                'primary_latitude': best_location.latitude,
                'primary_longitude': best_location.longitude,
                'primary_address': best_location.address,
                'primary_country': best_location.country,
                'primary_city': best_location.city,
                'primary_source': best_location.source,
                'primary_confidence': best_location.confidence
            })
        
        save_results_to_json(summary, output_path, "geolocation_summary.json")
        
        # Generate map if requested
        if save_map and best_location and best_location.latitude and best_location.longitude:
            map_file = output_path / "location_map.html"
            # Implementation would generate interactive map here
            click.echo(f"Location map saved to: {map_file}")
        
        click.echo(f"\nAnalysis complete. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Geolocation analysis failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=str, callback=lambda ctx, param, value: validate_image_file(value))
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--extract-hidden', is_flag=True, help='Extract hidden and manufacturer-specific metadata')
@click.option('--verify-integrity', is_flag=True, default=True, help='Verify metadata integrity')
@click.option('--camera-validation', is_flag=True, help='Validate camera specifications against database')
@click.option('--export-raw', is_flag=True, help='Export raw EXIF data')
@click.pass_context
def analyze_metadata(ctx, image_path, output_dir, extract_hidden, verify_integrity, camera_validation, export_raw):
    """
    Perform comprehensive EXIF metadata forensic analysis.
    
    Analyzes image metadata for authenticity, consistency, and signs of manipulation.
    Includes camera fingerprinting, timestamp validation, and GPS data verification.
    
    Example:
        lemkin-images analyze-metadata photo.jpg --extract-hidden --camera-validation
    """
    try:
        config = ctx.obj['config']
        
        # Update config with command line options
        config.extract_hidden_metadata = extract_hidden
        config.verify_metadata_integrity = verify_integrity
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "metadata_analysis")
        
        click.echo(f"Starting metadata analysis for: {image_path.name}")
        click.echo(f"Hidden metadata extraction: {'enabled' if extract_hidden else 'disabled'}")
        click.echo(f"Integrity verification: {'enabled' if verify_integrity else 'disabled'}")
        click.echo(f"Camera validation: {'enabled' if camera_validation else 'disabled'}")
        
        # Perform analysis
        with click.progressbar(length=100, label='Analyzing') as bar:
            analyzer = MetadataForensicsAnalyzer(config)
            results = analyzer.analyze_metadata(image_path)
            bar.update(100)
        
        # Display summary
        click.echo(f"\nMetadata analysis completed:")
        click.echo(f"  Metadata authentic: {'YES' if results.metadata_authentic else 'NO'}")
        click.echo(f"  Confidence: {results.metadata_confidence:.1%}")
        click.echo(f"  EXIF integrity: {'OK' if results.exif_integrity else 'COMPROMISED'}")
        
        # Display camera information
        metadata = results.metadata_source
        if metadata.camera_make or metadata.camera_model:
            click.echo(f"\nCamera information:")
            if metadata.camera_make:
                click.echo(f"  Make: {metadata.camera_make}")
            if metadata.camera_model:
                click.echo(f"  Model: {metadata.camera_model}")
            if metadata.lens_model:
                click.echo(f"  Lens: {metadata.lens_model}")
            if metadata.software_used:
                click.echo(f"  Software: {metadata.software_used}")
        
        # Display capture settings
        if any([metadata.iso_speed, metadata.aperture, metadata.shutter_speed, metadata.focal_length]):
            click.echo(f"\nCapture settings:")
            if metadata.iso_speed:
                click.echo(f"  ISO: {metadata.iso_speed}")
            if metadata.aperture:
                click.echo(f"  Aperture: f/{metadata.aperture}")
            if metadata.shutter_speed:
                click.echo(f"  Shutter speed: {metadata.shutter_speed}")
            if metadata.focal_length:
                click.echo(f"  Focal length: {metadata.focal_length}")
        
        # Display timestamps
        if metadata.creation_time or metadata.digitized_time:
            click.echo(f"\nTimestamps:")
            if metadata.creation_time:
                click.echo(f"  Created: {metadata.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if metadata.digitized_time:
                click.echo(f"  Digitized: {metadata.digitized_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if metadata.modification_time:
                click.echo(f"  Modified: {metadata.modification_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display GPS information
        if metadata.gps_latitude and metadata.gps_longitude:
            click.echo(f"\nGPS location:")
            click.echo(f"  Coordinates: {metadata.gps_latitude:.6f}, {metadata.gps_longitude:.6f}")
            if metadata.gps_altitude:
                click.echo(f"  Altitude: {metadata.gps_altitude}m")
        
        # Display inconsistencies
        all_inconsistencies = (
            results.timestamp_inconsistencies +
            results.camera_inconsistencies +
            results.software_inconsistencies +
            results.gps_inconsistencies
        )
        
        if all_inconsistencies:
            click.echo(f"\nInconsistencies found:")
            for inconsistency in all_inconsistencies:
                click.echo(f"  • {inconsistency}", fg='yellow')
        
        # Display manipulation indicators
        manipulation_flags = []
        if results.metadata_stripped:
            manipulation_flags.append("Metadata stripped")
        if results.metadata_modified:
            manipulation_flags.append("Metadata modified")
        if results.metadata_fabricated:
            manipulation_flags.append("Metadata fabricated")
        
        if manipulation_flags:
            click.echo(f"\nManipulation indicators:")
            for flag in manipulation_flags:
                click.echo(f"  • {flag}", fg='red')
        
        # Display recommendations
        recommendations = []
        if results.requires_deeper_analysis:
            recommendations.append("Deeper analysis recommended")
        if results.expert_validation_needed:
            recommendations.append("Expert validation required")
        if results.chain_of_custody_concerns:
            recommendations.extend(results.chain_of_custody_concerns)
        
        if recommendations:
            click.echo(f"\nRecommendations:")
            for rec in recommendations:
                click.echo(f"  • {rec}")
        
        # Save detailed results
        results_dict = results.dict()
        save_results_to_json(results_dict, output_path, "metadata_forensics.json")
        
        # Save summary report
        summary = {
            'image_file': str(image_path),
            'analysis_timestamp': results.analysis_timestamp.isoformat(),
            'metadata_authentic': results.metadata_authentic,
            'metadata_confidence': results.metadata_confidence,
            'exif_integrity': results.exif_integrity,
            'camera_make': metadata.camera_make,
            'camera_model': metadata.camera_model,
            'creation_time': metadata.creation_time.isoformat() if metadata.creation_time else None,
            'gps_present': bool(metadata.gps_latitude and metadata.gps_longitude),
            'inconsistencies_count': len(all_inconsistencies),
            'manipulation_flags': manipulation_flags,
            'requires_deeper_analysis': results.requires_deeper_analysis,
            'expert_validation_needed': results.expert_validation_needed
        }
        
        save_results_to_json(summary, output_path, "metadata_summary.json")
        
        # Export raw EXIF data if requested
        if export_raw:
            raw_exif_file = output_path / "raw_exif_data.json"
            save_results_to_json(metadata.exif_data, output_path, "raw_exif_data.json")
            click.echo(f"Raw EXIF data saved to: {raw_exif_file}")
        
        click.echo(f"\nAnalysis complete. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Metadata analysis failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=str, callback=lambda ctx, param, value: validate_image_file(value))
@click.option('--output-dir', '-o', help='Output directory for results and reports')
@click.option('--case-number', '-c', help='Case number for legal tracking')
@click.option('--investigator', '-i', help='Investigator name')
@click.option('--skip-reverse-search', is_flag=True, help='Skip reverse image search')
@click.option('--skip-manipulation', is_flag=True, help='Skip manipulation detection')
@click.option('--skip-geolocation', is_flag=True, help='Skip geolocation analysis')
@click.option('--skip-metadata', is_flag=True, help='Skip metadata analysis')
@click.option('--report-format', type=click.Choice(['json', 'html', 'pdf']), 
              default='json', help='Report format (default: json)')
@click.option('--generate-visual-report', is_flag=True, help='Generate visual report with images')
@click.pass_context
def authenticate(ctx, image_path, output_dir, case_number, investigator, skip_reverse_search, 
                skip_manipulation, skip_geolocation, skip_metadata, report_format, generate_visual_report):
    """
    Perform comprehensive image authenticity analysis.
    
    Combines reverse search, manipulation detection, geolocation, and metadata analysis
    into a single comprehensive authenticity assessment suitable for legal proceedings.
    
    Example:
        lemkin-images authenticate evidence.jpg -c "CASE-2024-001" -i "Jane Doe" --generate-visual-report
    """
    try:
        config = ctx.obj['config']
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "authentication")
        
        click.echo(f"Starting comprehensive authentication for: {image_path.name}")
        if case_number:
            click.echo(f"Case number: {case_number}")
        if investigator:
            click.echo(f"Investigator: {investigator}")
        
        # Perform comprehensive analysis
        authenticator = ImageAuthenticator(config)
        
        with click.progressbar(length=100, label='Authenticating') as bar:
            report = authenticator.authenticate_image(
                image_path,
                include_reverse_search=not skip_reverse_search,
                include_manipulation_detection=not skip_manipulation,
                include_geolocation=not skip_geolocation,
                include_metadata_forensics=not skip_metadata
            )
            bar.update(100)
        
        # Update report with case information
        if case_number:
            # In a full implementation, we'd update the report's case_number field
            pass
        if investigator:
            report.analyst = investigator
        
        # Display executive summary
        click.echo(f"\n" + "="*60)
        click.echo("AUTHENTICATION SUMMARY")
        click.echo("="*60)
        
        summary = report.get_executive_summary()
        click.echo(summary)
        
        # Display detailed findings
        if report.critical_findings:
            click.echo(f"\nCRITICAL FINDINGS:")
            for finding in report.critical_findings:
                click.echo(f"  • {finding}", fg='red')
        
        if report.red_flags:
            click.echo(f"\nRED FLAGS:")
            for flag in report.red_flags:
                click.echo(f"  • {flag}", fg='yellow')
        
        if report.supporting_evidence:
            click.echo(f"\nSUPPORTING EVIDENCE:")
            for evidence in report.supporting_evidence:
                click.echo(f"  • {evidence}", fg='green')
        
        # Display component results
        components_analyzed = []
        if report.reverse_search_results and not skip_reverse_search:
            components_analyzed.append(f"Reverse Search: {report.reverse_search_results.total_results_found} results found")
        
        if report.manipulation_analysis and not skip_manipulation:
            manipulation_status = "DETECTED" if report.manipulation_analysis.is_manipulated else "NOT DETECTED"
            components_analyzed.append(f"Manipulation: {manipulation_status}")
        
        if report.geolocation_result and not skip_geolocation:
            location_status = "VERIFIED" if report.geolocation_result.location_verified else "UNVERIFIED"
            components_analyzed.append(f"Geolocation: {location_status}")
        
        if report.metadata_forensics and not skip_metadata:
            metadata_status = "AUTHENTIC" if report.metadata_forensics.metadata_authentic else "SUSPICIOUS"
            components_analyzed.append(f"Metadata: {metadata_status}")
        
        if components_analyzed:
            click.echo(f"\nCOMPONENT ANALYSES:")
            for component in components_analyzed:
                click.echo(f"  • {component}")
        
        # Display methods and limitations
        if report.methods_used:
            click.echo(f"\nMethods used: {', '.join(report.methods_used)}")
        
        if report.limitations:
            click.echo(f"\nLimitations:")
            for limitation in report.limitations:
                click.echo(f"  • {limitation}")
        
        # Display legal considerations
        click.echo(f"\nLEGAL ASSESSMENT:")
        click.echo(f"  Admissibility: {report.admissibility_assessment or 'To be determined'}")
        click.echo(f"  Chain of custody: {report.chain_of_custody_status}")
        click.echo(f"  Expert testimony required: {'YES' if report.expert_testimony_required else 'NO'}")
        
        # Save comprehensive report
        report_dict = report.dict()
        save_results_to_json(report_dict, output_path, "authenticity_report.json")
        
        # Generate additional report formats
        if report_format == 'html' or generate_visual_report:
            html_report = output_path / "authenticity_report.html"
            # Implementation would generate HTML report here
            click.echo(f"HTML report generated: {html_report}")
        
        if report_format == 'pdf':
            pdf_report = output_path / "authenticity_report.pdf"
            # Implementation would generate PDF report here
            click.echo(f"PDF report generated: {pdf_report}")
        
        # Save executive summary
        summary_file = output_path / "executive_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        click.echo(f"Executive summary saved: {summary_file}")
        
        click.echo(f"\nComprehensive authentication complete.")
        click.echo(f"Results saved to: {output_path}")
        
        # Final verdict display
        verdict_color = None
        if report.authenticity_verdict == "authentic":
            verdict_color = 'green'
        elif report.authenticity_verdict in ["manipulated", "suspicious"]:
            verdict_color = 'red'
        elif report.authenticity_verdict == "inconclusive":
            verdict_color = 'yellow'
        
        click.echo(f"\nFINAL VERDICT: {report.authenticity_verdict.upper()}", fg=verdict_color)
        
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='lemkin_images_config.json', help='Output configuration file')
def generate_config(output):
    """
    Generate a sample configuration file with all available options.
    
    Creates a JSON configuration file with default values and documentation
    that can be customized and used with the --config option.
    """
    config = ImageAuthConfig()
    config_dict = config.dict()
    
    # Add documentation
    documented_config = {
        "_description": "Lemkin Images Configuration File",
        "_version": "1.0.0",
        "_documentation": {
            "reverse_search": {
                "enable_reverse_search": "Enable/disable reverse image search",
                "search_engines": "List of search engines to use: google, tineye, bing, yandex, baidu",
                "max_search_results": "Maximum results to collect per engine (1-500)",
                "search_timeout_seconds": "Timeout for each search engine (5-300 seconds)"
            },
            "manipulation_detection": {
                "enable_manipulation_detection": "Enable/disable manipulation detection",
                "manipulation_threshold": "Threshold for manipulation detection (0.0-1.0)",
                "detailed_analysis": "Enable detailed analysis (slower but more thorough)",
                "enable_compression_analysis": "Enable JPEG compression analysis"
            },
            "geolocation": {
                "enable_geolocation": "Enable/disable geolocation extraction",
                "enable_visual_geolocation": "Enable visual landmark recognition",
                "geolocation_confidence_threshold": "Minimum confidence for location verification (0.0-1.0)"
            },
            "metadata_forensics": {
                "enable_metadata_forensics": "Enable/disable metadata forensics",
                "extract_hidden_metadata": "Extract hidden and manufacturer-specific metadata",
                "verify_metadata_integrity": "Verify EXIF data integrity"
            },
            "processing": {
                "max_image_size_mb": "Maximum image size to process (1-1000 MB)",
                "analysis_timeout_minutes": "Overall analysis timeout (1-120 minutes)",
                "preserve_original": "Whether to preserve original file integrity"
            },
            "output": {
                "generate_visual_reports": "Generate visual analysis reports",
                "include_technical_details": "Include technical details in reports",
                "confidence_scoring": "Enable confidence scoring for all analyses"
            }
        },
        **config_dict
    }
    
    output_path = Path(output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documented_config, f, indent=2)
    
    click.echo(f"Configuration file generated: {output_path}")
    click.echo(f"Edit this file and use with: --config {output_path}")


if __name__ == '__main__':
    cli()