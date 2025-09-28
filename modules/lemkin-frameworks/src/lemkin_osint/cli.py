"""
Lemkin OSINT Collection Toolkit - CLI Interface

Command-line interface for the OSINT collection toolkit providing
comprehensive digital investigation capabilities.

Usage:
    lemkin-osint collect-social --query "search term" --platforms twitter,reddit
    lemkin-osint archive-web --urls urls.txt --output archive_results.json
    lemkin-osint extract-metadata --files /path/to/images --output metadata_report.json
    lemkin-osint verify-source --url "https://example.com" --output assessment.json
    lemkin-osint search-archives --query "search term" --domain example.com
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
    OSINTConfig, OSINTCollector, Source, PlatformType, ContentType,
    CollectionResult
)
from .social_scraper import SocialMediaScraper
from .web_archiver import WebArchiver
from .metadata_extractor import MetadataExtractor
from .source_verifier import SourceVerifier

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


def parse_platforms(platforms_str: str) -> List[PlatformType]:
    """Parse comma-separated platform string"""
    if not platforms_str:
        return []
    
    platform_map = {
        'twitter': PlatformType.TWITTER,
        'facebook': PlatformType.FACEBOOK,
        'youtube': PlatformType.YOUTUBE,
        'linkedin': PlatformType.LINKEDIN,
        'instagram': PlatformType.INSTAGRAM,
        'reddit': PlatformType.REDDIT,
        'telegram': PlatformType.TELEGRAM,
        'tiktok': PlatformType.TIKTOK
    }
    
    platforms = []
    for platform_str in platforms_str.split(','):
        platform_str = platform_str.strip().lower()
        if platform_str in platform_map:
            platforms.append(platform_map[platform_str])
        else:
            click.echo(f"Warning: Unknown platform '{platform_str}' ignored", err=True)
    
    return platforms


def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from text file (one per line)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return urls
    except Exception as e:
        click.echo(f"Error reading URLs from {file_path}: {e}", err=True)
        sys.exit(1)


def save_json_output(data: dict, output_path: str):
    """Save data as JSON with proper formatting"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        click.echo(f"Results saved to: {output_path}")
    except Exception as e:
        click.echo(f"Error saving to {output_path}: {e}", err=True)
        sys.exit(1)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]):
    """
    Lemkin OSINT Collection Toolkit
    
    Ethical open-source intelligence gathering and digital investigation tools
    compliant with the Berkeley Protocol for Digital Investigations.
    """
    setup_logging(verbose)
    
    # Load configuration
    osint_config = OSINTConfig()
    if config:
        try:
            with open(config, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                osint_config = OSINTConfig(**config_data)
        except Exception as e:
            click.echo(f"Error loading config file: {e}", err=True)
            sys.exit(1)
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = osint_config
    ctx.obj['verbose'] = verbose


@cli.command('collect-social')
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--platforms', '-p', required=True, help='Comma-separated platforms (twitter,reddit,youtube)')
@click.option('--max-results', '-m', type=int, help='Maximum results to collect')
@click.option('--output', '-o', help='Output file path')
@click.option('--collection-name', help='Name for the collection')
@click.pass_context
def collect_social(ctx, query: str, platforms: str, max_results: Optional[int], 
                   output: Optional[str], collection_name: Optional[str]):
    """Collect social media evidence ethically across platforms"""
    
    try:
        config = ctx.obj['config']
        
        # Parse platforms
        platform_list = parse_platforms(platforms)
        if not platform_list:
            click.echo("Error: No valid platforms specified", err=True)
            sys.exit(1)
        
        # Create scraper and collector
        scraper = SocialMediaScraper(config)
        collector = OSINTCollector(config)
        
        # Set max results if specified
        if max_results:
            config.max_results_per_query = max_results
        
        click.echo(f"🔍 Collecting social media evidence for: {query}")
        click.echo(f"📱 Platforms: {', '.join([p.value for p in platform_list])}")
        click.echo(f"📊 Max results: {config.max_results_per_query}")
        
        # Run collection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            collection = loop.run_until_complete(
                scraper.collect_social_media_evidence(
                    query, platform_list, max_results
                )
            )
            
            # Display results
            click.echo(f"\n✅ Collection completed!")
            click.echo(f"📈 Status: {collection.status}")
            click.echo(f"📊 Total items: {collection.total_items_collected}")
            click.echo(f"⚠️  ToS violations: {len(collection.tos_violations)}")
            click.echo(f"🚦 Rate limit hits: {collection.rate_limit_hits}")
            
            # Save output if specified
            if output:
                save_json_output(collection.dict(), output)
            else:
                # Print summary
                for post in collection.social_posts[:5]:  # Show first 5
                    click.echo(f"\n📝 {post.platform.value}: {post.text_content[:100]}...")
                
                if len(collection.social_posts) > 5:
                    click.echo(f"\n... and {len(collection.social_posts) - 5} more posts")
        
        finally:
            loop.close()
            scraper.close()
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command('archive-web')
@click.option('--urls', '-u', help='Comma-separated URLs or path to file with URLs')
@click.option('--url-file', type=click.Path(exists=True), help='File containing URLs (one per line)')
@click.option('--output', '-o', help='Output file path')
@click.option('--package', help='Create preservation package at this path')
@click.pass_context
def archive_web(ctx, urls: Optional[str], url_file: Optional[str], 
                output: Optional[str], package: Optional[str]):
    """Archive web content for preservation"""
    
    try:
        config = ctx.obj['config']
        
        # Get URL list
        url_list = []
        if urls:
            url_list = [url.strip() for url in urls.split(',')]
        elif url_file:
            url_list = load_urls_from_file(url_file)
        else:
            click.echo("Error: Must specify --urls or --url-file", err=True)
            sys.exit(1)
        
        if not url_list:
            click.echo("Error: No URLs to archive", err=True)
            sys.exit(1)
        
        # Create archiver
        archiver = WebArchiver(config)
        
        click.echo(f"🌐 Archiving {len(url_list)} URLs...")
        
        # Run archiving
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            collection = loop.run_until_complete(
                archiver.archive_web_content(url_list)
            )
            
            # Display results
            click.echo(f"\n✅ Archiving completed!")
            click.echo(f"📊 Total URLs: {collection.total_urls}")
            click.echo(f"✅ Successful: {collection.successful_archives}")
            click.echo(f"❌ Failed: {collection.failed_archives}")
            
            # Save output
            if output:
                save_json_output(collection.dict(), output)
            
            # Create preservation package if requested
            if package:
                if archiver.create_preservation_package(collection, package):
                    click.echo(f"📦 Preservation package created: {package}")
                else:
                    click.echo("❌ Failed to create preservation package", err=True)
            
            # Show sample results
            for i, archive in enumerate(collection.archives[:3]):
                click.echo(f"\n📄 {archive.original_url}")
                click.echo(f"   Archived: {archive.archived_url}")
                click.echo(f"   Time: {archive.archive_timestamp}")
        
        finally:
            loop.close()
            archiver.close()
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command('extract-metadata')
@click.option('--files', '-f', help='Comma-separated file paths or directory')
@click.option('--directory', '-d', type=click.Path(exists=True), help='Directory to process')
@click.option('--recursive', '-r', is_flag=True, help='Process directory recursively')
@click.option('--output', '-o', help='Output file path')
@click.option('--report', help='Generate comprehensive report at this path')
@click.pass_context
def extract_metadata(ctx, files: Optional[str], directory: Optional[str], 
                     recursive: bool, output: Optional[str], report: Optional[str]):
    """Extract EXIF/XMP metadata from media files"""
    
    try:
        config = ctx.obj['config']
        
        # Get file list
        file_list = []
        if files:
            file_list = [Path(f.strip()) for f in files.split(',')]
        elif directory:
            dir_path = Path(directory)
            if recursive:
                # Find all media files recursively
                extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.mp4', '.avi', '.mov'}
                file_list = [f for f in dir_path.rglob('*') if f.suffix.lower() in extensions]
            else:
                # Just files in directory
                extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.mp4', '.avi', '.mov'}
                file_list = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        else:
            click.echo("Error: Must specify --files or --directory", err=True)
            sys.exit(1)
        
        if not file_list:
            click.echo("Error: No files to process", err=True)
            sys.exit(1)
        
        # Create extractor
        extractor = MetadataExtractor(config)
        
        click.echo(f"📷 Extracting metadata from {len(file_list)} files...")
        
        # Process files
        metadata_list = []
        for i, file_path in enumerate(file_list):
            try:
                click.echo(f"Processing {i+1}/{len(file_list)}: {file_path.name}")
                metadata = extractor.extract_media_metadata(file_path)
                metadata_list.append(metadata)
            except Exception as e:
                click.echo(f"⚠️  Error processing {file_path}: {e}", err=True)
        
        # Display results
        click.echo(f"\n✅ Metadata extraction completed!")
        click.echo(f"📊 Total files processed: {len(metadata_list)}")
        
        # Count by type
        type_counts = {}
        gps_count = 0
        camera_count = 0
        
        for metadata in metadata_list:
            content_type = metadata.content_type.value
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
            
            if metadata.gps_coordinates:
                gps_count += 1
            if metadata.camera_make or metadata.camera_model:
                camera_count += 1
        
        click.echo(f"📸 Files with GPS data: {gps_count}")
        click.echo(f"📷 Files with camera info: {camera_count}")
        
        for content_type, count in type_counts.items():
            click.echo(f"📁 {content_type}: {count}")
        
        # Save output
        if output:
            output_data = [metadata.dict() for metadata in metadata_list]
            save_json_output(output_data, output)
        
        # Generate report
        if report:
            if extractor.create_metadata_report(metadata_list, Path(report)):
                click.echo(f"📋 Report generated: {report}")
            else:
                click.echo("❌ Failed to generate report", err=True)
        
        # Show sample metadata
        if metadata_list and not output and not report:
            sample = metadata_list[0]
            click.echo(f"\n📝 Sample metadata for {sample.file_path}:")
            click.echo(f"   Size: {sample.file_size:,} bytes")
            click.echo(f"   Type: {sample.content_type}")
            if sample.camera_make:
                click.echo(f"   Camera: {sample.camera_make} {sample.camera_model}")
            if sample.gps_coordinates:
                click.echo(f"   GPS: {sample.gps_coordinates}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command('verify-source')
@click.option('--url', '-u', help='URL to verify')
@click.option('--name', '-n', help='Source name')
@click.option('--platform', help='Platform type (twitter, facebook, etc.)')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def verify_source(ctx, url: Optional[str], name: Optional[str], 
                 platform: Optional[str], output: Optional[str]):
    """Assess source credibility and verification"""
    
    try:
        config = ctx.obj['config']
        
        if not url and not name:
            click.echo("Error: Must specify --url or --name", err=True)
            sys.exit(1)
        
        # Create source object
        source = Source(
            name=name or "Unknown Source",
            url=url,
            platform=PlatformType(platform) if platform else None
        )
        
        # Create verifier
        verifier = SourceVerifier(config)
        
        click.echo(f"🔍 Verifying source credibility...")
        click.echo(f"📝 Name: {source.name}")
        if source.url:
            click.echo(f"🌐 URL: {source.url}")
        if source.platform:
            click.echo(f"📱 Platform: {source.platform.value}")
        
        # Perform assessment
        assessment = verifier.verify_source_credibility(source)
        
        # Display results
        click.echo(f"\n✅ Assessment completed!")
        click.echo(f"📊 Credibility Score: {assessment.credibility_score:.1f}/10")
        click.echo(f"📈 Credibility Level: {assessment.credibility_level.value.upper()}")
        click.echo(f"🎯 Confidence: {assessment.confidence:.0%}")
        
        # Show key factors
        if assessment.domain_reputation:
            click.echo(f"🌐 Domain Reputation: {assessment.domain_reputation:.1f}/10")
        if assessment.ssl_valid is not None:
            status = "✅ Valid" if assessment.ssl_valid else "❌ Invalid"
            click.echo(f"🔒 SSL Certificate: {status}")
        
        # Show warnings
        if assessment.warning_flags:
            click.echo(f"\n⚠️  Warning flags:")
            for flag in assessment.warning_flags:
                click.echo(f"   • {flag}")
        
        # Show assessment notes
        if assessment.assessment_notes:
            click.echo(f"\n📋 Assessment Notes:")
            click.echo(assessment.assessment_notes)
        
        # Save output
        if output:
            save_json_output(assessment.dict(), output)
            
        verifier.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command('search-archives')
@click.option('--query', '-q', help='Search query')
@click.option('--domain', '-d', help='Limit search to specific domain')
@click.option('--from-date', help='Start date (YYYY-MM-DD)')
@click.option('--to-date', help='End date (YYYY-MM-DD)')
@click.option('--limit', '-l', type=int, default=50, help='Maximum results')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def search_archives(ctx, query: Optional[str], domain: Optional[str], 
                   from_date: Optional[str], to_date: Optional[str],
                   limit: int, output: Optional[str]):
    """Search archived web content"""
    
    try:
        config = ctx.obj['config']
        
        if not query and not domain:
            click.echo("Error: Must specify --query or --domain", err=True)
            sys.exit(1)
        
        # Parse dates
        date_range = None
        if from_date or to_date:
            try:
                start_date = datetime.strptime(from_date, '%Y-%m-%d') if from_date else None
                end_date = datetime.strptime(to_date, '%Y-%m-%d') if to_date else None
                if start_date and end_date:
                    date_range = (start_date, end_date)
            except ValueError as e:
                click.echo(f"Error parsing dates: {e}", err=True)
                sys.exit(1)
        
        # Create archiver for search
        archiver = WebArchiver(config)
        
        click.echo(f"🔍 Searching archived content...")
        if query:
            click.echo(f"📝 Query: {query}")
        if domain:
            click.echo(f"🌐 Domain: {domain}")
        if date_range:
            click.echo(f"📅 Date range: {from_date} to {to_date}")
        
        # Perform search
        results = archiver.search_archived_content(
            query or "", domain, date_range
        )
        
        # Limit results
        results = results[:limit]
        
        # Display results
        click.echo(f"\n✅ Search completed!")
        click.echo(f"📊 Results found: {len(results)}")
        
        for i, result in enumerate(results[:10], 1):  # Show first 10
            click.echo(f"\n{i}. {result['url']}")
            click.echo(f"   Archived: {result['timestamp']}")
            click.echo(f"   URL: {result['archived_url']}")
        
        if len(results) > 10:
            click.echo(f"\n... and {len(results) - 10} more results")
        
        # Save output
        if output:
            save_json_output(results, output)
            
        archiver.close()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command('config')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--create', help='Create configuration file at path')
@click.pass_context
def config_cmd(ctx, show: bool, create: Optional[str]):
    """Manage configuration settings"""
    
    try:
        config = ctx.obj['config']
        
        if show:
            click.echo("📋 Current configuration:")
            config_dict = config.dict()
            for key, value in config_dict.items():
                click.echo(f"  {key}: {value}")
        
        if create:
            config_dict = config.dict()
            save_json_output(config_dict, create)
            click.echo(f"⚙️  Configuration saved to: {create}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command('version')
def version():
    """Show version information"""
    click.echo("Lemkin OSINT Collection Toolkit v1.0")
    click.echo("Ethical open-source intelligence gathering")
    click.echo("Berkeley Protocol compliant digital investigations")


if __name__ == '__main__':
    cli()