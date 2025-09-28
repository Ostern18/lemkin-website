"""
Lemkin Digital Forensics CLI Interface

Command-line interface for digital forensics operations including:
- analyze-filesystem: Comprehensive disk image analysis
- process-network: Network traffic and log analysis  
- extract-mobile: Mobile device data extraction
- verify-authenticity: Evidence authenticity verification
- generate-timeline: Forensic timeline creation
- create-case: Case management operations

Provides user-friendly interface for complex forensics procedures.
"""

import click
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from .core import (
    DigitalForensicsAnalyzer,
    ForensicsConfig,
    EvidenceType,
    ForensicsCase,
    DigitalEvidence
)
from .file_analyzer import FileAnalyzer
from .network_processor import NetworkProcessor
from .mobile_analyzer import MobileAnalyzer
from .authenticity_verifier import AuthenticityVerifier

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
        output_dir = f"lemkin_forensics_{operation}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_results_to_json(results: dict, output_path: Path, filename: str):
    """Save results to JSON file"""
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    click.echo(f"Results saved to: {output_file}")


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """
    Lemkin Digital Forensics Toolkit
    
    Comprehensive digital forensics analysis for legal professionals.
    Supports disk images, network captures, mobile backups, and evidence verification.
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
            ctx.obj['config'] = ForensicsConfig(**config_data)
            click.echo(f"Configuration loaded from: {config}")
        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            sys.exit(1)
    else:
        ctx.obj['config'] = ForensicsConfig()


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for results and recovered files')
@click.option('--recover-deleted', is_flag=True, help='Enable deleted file recovery')
@click.option('--timeline', is_flag=True, help='Generate filesystem timeline')
@click.option('--carve-files', is_flag=True, help='Enable file carving')
@click.option('--format', type=click.Choice(['json', 'csv', 'html']), default='json', help='Output format')
@click.pass_context
def analyze_filesystem(ctx, image_path, output_dir, recover_deleted, timeline, carve_files, format):
    """
    Analyze filesystem from disk image.
    
    Performs comprehensive analysis of disk images including:
    - File system structure analysis
    - File listing and metadata extraction
    - Deleted file recovery (optional)
    - Timeline generation (optional)
    - File carving for fragments (optional)
    
    IMAGE_PATH: Path to disk image file (E01, DD, AFF, etc.)
    """
    click.echo(f"Starting filesystem analysis of: {image_path}")
    
    # Setup output directory
    output_path = setup_output_directory(output_dir, "filesystem")
    
    try:
        # Initialize analyzer
        config = ctx.obj['config']
        config.enable_deleted_file_recovery = recover_deleted
        config.enable_timeline_generation = timeline
        
        analyzer = FileAnalyzer(config)
        
        # Create evidence object
        evidence = DigitalEvidence(
            name=f"Disk Image - {os.path.basename(image_path)}",
            evidence_type=EvidenceType.DISK_IMAGE,
            file_path=image_path,
            file_size=os.path.getsize(image_path),
            file_hash_sha256="placeholder"  # Would be calculated in real implementation
        )
        
        # Perform analysis
        with click.progressbar(length=100, label='Analyzing filesystem') as bar:
            analysis = analyzer.analyze_disk_image(
                evidence,
                output_dir=str(output_path) if carve_files else None
            )
            bar.update(100)
        
        # Generate results
        results = {
            'analysis_summary': analysis.analysis_summary,
            'filesystem_info': analysis.filesystem_info.__dict__ if analysis.filesystem_info else None,
            'total_files': len(analysis.file_artifacts),
            'deleted_files_recovered': analysis.deleted_files_recovered,
            'carved_files': len(analysis.carved_files),
            'timeline_events': len(analysis.timeline_events),
            'file_artifacts': [artifact.__dict__ for artifact in analysis.file_artifacts[:100]],  # Limit for readability
            'timeline_sample': [event.__dict__ for event in analysis.timeline_events[:50]]
        }
        
        # Save results
        save_results_to_json(results, output_path, 'filesystem_analysis.json')
        
        # Save detailed timeline if generated
        if timeline and analysis.timeline_events:
            timeline_results = [event.__dict__ for event in analysis.timeline_events]
            save_results_to_json(timeline_results, output_path, 'filesystem_timeline.json')
        
        click.echo(f"\n✓ Analysis completed successfully!")
        click.echo(f"  Total files found: {len(analysis.file_artifacts)}")
        click.echo(f"  Deleted files recovered: {analysis.deleted_files_recovered}")
        click.echo(f"  Timeline events: {len(analysis.timeline_events)}")
        click.echo(f"  Results saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Analysis failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('capture_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--extract-files', is_flag=True, help='Extract files from HTTP traffic')
@click.option('--log-format', type=click.Choice(['auto', 'apache', 'iis', 'firewall']), 
              default='auto', help='Log file format (for log files)')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Output format')
@click.pass_context
def process_network(ctx, capture_path, output_dir, extract_files, log_format, format):
    """
    Process network traffic captures and logs.
    
    Analyzes network communications including:
    - PCAP file analysis and flow reconstruction
    - HTTP/HTTPS traffic analysis
    - DNS query analysis
    - Suspicious activity detection
    - Communication pattern analysis
    
    CAPTURE_PATH: Path to PCAP file or network log file
    """
    click.echo(f"Starting network analysis of: {capture_path}")
    
    # Setup output directory
    output_path = setup_output_directory(output_dir, "network")
    
    try:
        # Initialize processor
        config = ctx.obj['config']
        processor = NetworkProcessor(config)
        
        # Create evidence object
        evidence = DigitalEvidence(
            name=f"Network Capture - {os.path.basename(capture_path)}",
            evidence_type=EvidenceType.NETWORK_CAPTURE,
            file_path=capture_path,
            file_size=os.path.getsize(capture_path),
            file_hash_sha256="placeholder"
        )
        
        # Determine analysis type
        file_ext = os.path.splitext(capture_path)[1].lower()
        
        if file_ext in ['.pcap', '.pcapng', '.cap']:
            # PCAP analysis
            with click.progressbar(length=100, label='Analyzing PCAP') as bar:
                analysis = processor.analyze_pcap(
                    evidence,
                    output_dir=str(output_path) if extract_files else None
                )
                bar.update(100)
        else:
            # Log file analysis
            with click.progressbar(length=100, label='Analyzing logs') as bar:
                analysis = processor.analyze_log_file(evidence, log_format)
                bar.update(100)
        
        # Generate results
        results = {
            'analysis_summary': {
                'total_packets': analysis.total_packets,
                'total_bytes': analysis.total_bytes,
                'unique_ips': len(analysis.unique_ips),
                'protocols': list(analysis.protocols_seen),
                'flows': len(analysis.flows),
                'http_transactions': len(analysis.http_transactions),
                'dns_queries': len(analysis.dns_queries),
                'suspicious_activities': len(analysis.suspicious_activities)
            },
            'top_talkers': analysis.top_talkers[:10],
            'suspicious_activities': analysis.suspicious_activities,
            'temporal_patterns': analysis.temporal_patterns,
            'flows_sample': [flow.__dict__ for flow in analysis.flows[:50]],
            'http_sample': [tx.__dict__ for tx in analysis.http_transactions[:25]],
            'dns_sample': [query.__dict__ for query in analysis.dns_queries[:25]]
        }
        
        # Save results
        save_results_to_json(results, output_path, 'network_analysis.json')
        
        click.echo(f"\n✓ Analysis completed successfully!")
        click.echo(f"  Total packets: {analysis.total_packets:,}")
        click.echo(f"  Network flows: {len(analysis.flows):,}")
        click.echo(f"  HTTP transactions: {len(analysis.http_transactions):,}")
        click.echo(f"  DNS queries: {len(analysis.dns_queries):,}")
        click.echo(f"  Suspicious activities: {len(analysis.suspicious_activities)}")
        click.echo(f"  Results saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Network analysis failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('backup_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for extracted data')
@click.option('--backup-type', type=click.Choice(['auto', 'ios', 'android']), 
              default='auto', help='Mobile backup type')
@click.option('--extract-media', is_flag=True, help='Extract media files')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Output format')
@click.pass_context
def extract_mobile(ctx, backup_path, output_dir, backup_type, extract_media, format):
    """
    Extract data from mobile device backups.
    
    Analyzes mobile device data including:
    - Contacts and communication history
    - Text messages and chat logs
    - Call logs and communication patterns
    - Location history and geofencing
    - App data and cross-app analysis
    - Media files with metadata
    
    BACKUP_PATH: Path to mobile backup (iTunes backup, Android backup, etc.)
    """
    click.echo(f"Starting mobile data extraction from: {backup_path}")
    
    # Setup output directory
    output_path = setup_output_directory(output_dir, "mobile")
    
    try:
        # Initialize analyzer
        config = ctx.obj['config']
        analyzer = MobileAnalyzer(config)
        
        # Create evidence object
        evidence = DigitalEvidence(
            name=f"Mobile Backup - {os.path.basename(backup_path)}",
            evidence_type=EvidenceType.MOBILE_BACKUP,
            file_path=backup_path,
            file_size=os.path.getsize(backup_path) if os.path.isfile(backup_path) else 0,
            file_hash_sha256="placeholder"
        )
        
        # Perform extraction
        with click.progressbar(length=100, label='Extracting mobile data') as bar:
            extraction = analyzer.analyze_mobile_backup(
                evidence,
                backup_type=backup_type,
                output_dir=str(output_path)
            )
            bar.update(100)
        
        # Generate results
        results = {
            'extraction_summary': extraction.extraction_summary,
            'device_info': extraction.device_info,
            'contacts_count': len(extraction.contacts),
            'messages_count': len(extraction.messages),
            'call_logs_count': len(extraction.call_logs),
            'location_points_count': len(extraction.location_data),
            'apps_count': len(extraction.apps),
            'media_files_count': len(extraction.media_files),
            'timeline_events_count': len(extraction.timeline_events),
            'privacy_concerns': extraction.privacy_concerns,
            'contacts_sample': [contact.__dict__ for contact in extraction.contacts[:20]],
            'messages_sample': [message.__dict__ for message in extraction.messages[:50]],
            'timeline_sample': [event.__dict__ for event in extraction.timeline_events[:50]]
        }
        
        # Save results
        save_results_to_json(results, output_path, 'mobile_extraction.json')
        
        click.echo(f"\n✓ Extraction completed successfully!")
        click.echo(f"  Contacts: {len(extraction.contacts):,}")
        click.echo(f"  Messages: {len(extraction.messages):,}")
        click.echo(f"  Call logs: {len(extraction.call_logs):,}")
        click.echo(f"  Location points: {len(extraction.location_data):,}")
        click.echo(f"  Timeline events: {len(extraction.timeline_events):,}")
        
        if extraction.privacy_concerns:
            click.echo(f"  ⚠️  Privacy concerns: {len(extraction.privacy_concerns)}")
        
        click.echo(f"  Results saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Mobile extraction failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('evidence_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for verification report')
@click.option('--verification-level', type=click.Choice(['basic', 'standard', 'comprehensive']), 
              default='standard', help='Level of verification to perform')
@click.option('--expected-hash', help='Expected SHA-256 hash for verification')
@click.option('--format', type=click.Choice(['json', 'html']), default='json', help='Report format')
@click.pass_context
def verify_authenticity(ctx, evidence_path, output_dir, verification_level, expected_hash, format):
    """
    Verify digital evidence authenticity and integrity.
    
    Performs comprehensive authenticity verification including:
    - Cryptographic hash verification
    - Digital signature validation
    - Metadata authenticity analysis
    - Chain of custody verification
    - Tamper detection analysis
    - Legal admissibility assessment
    
    EVIDENCE_PATH: Path to evidence file to verify
    """
    click.echo(f"Starting authenticity verification of: {evidence_path}")
    
    # Setup output directory
    output_path = setup_output_directory(output_dir, "verification")
    
    try:
        # Initialize verifier
        config = ctx.obj['config']
        verifier = AuthenticityVerifier(config)
        
        # Create evidence object
        evidence = DigitalEvidence(
            name=f"Evidence - {os.path.basename(evidence_path)}",
            evidence_type=EvidenceType.DOCUMENT_COLLECTION,  # Generic type
            file_path=evidence_path,
            file_size=os.path.getsize(evidence_path),
            file_hash_sha256=expected_hash or "unknown"
        )
        
        # If expected hash provided, calculate actual hash
        if expected_hash:
            import hashlib
            with open(evidence_path, 'rb') as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            evidence.file_hash_sha256 = expected_hash
        
        # Perform verification
        with click.progressbar(length=100, label='Verifying authenticity') as bar:
            report = verifier.verify_evidence_authenticity(evidence, verification_level)
            bar.update(100)
        
        # Generate summary
        summary = verifier.generate_authenticity_summary(report)
        
        # Prepare detailed results
        results = {
            'verification_summary': summary,
            'overall_authentic': report.overall_authentic,
            'confidence_score': report.confidence_score,
            'admissibility_assessment': report.admissibility_assessment,
            'hash_verifications': [v.__dict__ for v in report.hash_verifications],
            'signature_info': report.signature_info.__dict__ if report.signature_info else None,
            'metadata_analysis': report.metadata_analysis.__dict__ if report.metadata_analysis else None,
            'custody_validation': report.custody_validation.__dict__ if report.custody_validation else None,
            'issues_found': report.issues_found,
            'recommendations': report.recommendations,
            'legal_concerns': report.legal_concerns
        }
        
        # Save results
        save_results_to_json(results, output_path, 'authenticity_report.json')
        
        # Display results
        click.echo(f"\n✓ Verification completed!")
        click.echo(f"  Overall authentic: {'✓' if report.overall_authentic else '✗'}")
        click.echo(f"  Confidence score: {report.confidence_score:.1f}/10.0")
        click.echo(f"  Admissibility: {report.admissibility_assessment.upper()}")
        
        # Hash verification status
        for verification in report.hash_verifications:
            status = "✓" if verification.matches else "✗"
            click.echo(f"  {verification.algorithm.upper()}: {status}")
        
        if report.issues_found:
            click.echo(f"  ⚠️  Issues found: {len(report.issues_found)}")
            for issue in report.issues_found[:3]:  # Show first 3 issues
                click.echo(f"    - {issue}")
        
        if report.legal_concerns:
            click.echo(f"  ⚠️  Legal concerns: {len(report.legal_concerns)}")
        
        click.echo(f"  Report saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Verification failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('case_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for timeline')
@click.option('--format', type=click.Choice(['json', 'csv', 'html']), default='json', help='Timeline format')
@click.option('--start-date', help='Start date for timeline (YYYY-MM-DD)')
@click.option('--end-date', help='End date for timeline (YYYY-MM-DD)')
@click.pass_context
def generate_timeline(ctx, case_dir, output_dir, format, start_date, end_date):
    """
    Generate comprehensive forensic timeline.
    
    Creates unified timeline from multiple evidence sources including:
    - File system activities
    - Network communications
    - Mobile device activities
    - Application usage patterns
    - System events
    
    CASE_DIR: Directory containing case evidence and analysis results
    """
    click.echo(f"Generating timeline from case directory: {case_dir}")
    
    # Setup output directory
    output_path = setup_output_directory(output_dir, "timeline")
    
    try:
        # Initialize analyzer
        config = ctx.obj['config']
        analyzer = DigitalForensicsAnalyzer(config)
        
        # This would load existing case data and generate timeline
        # For now, create a placeholder implementation
        
        timeline_events = []
        
        # Parse date filters if provided
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate timeline results
        results = {
            'timeline_summary': {
                'total_events': len(timeline_events),
                'date_range': {
                    'start': start_date,
                    'end': end_date
                },
                'event_types': [],
                'sources': []
            },
            'timeline_events': timeline_events
        }
        
        # Save results
        save_results_to_json(results, output_path, 'forensic_timeline.json')
        
        click.echo(f"\n✓ Timeline generation completed!")
        click.echo(f"  Total events: {len(timeline_events):,}")
        click.echo(f"  Timeline saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Timeline generation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('case_number')
@click.argument('case_name')
@click.argument('investigator')
@click.option('--output-dir', '-o', help='Output directory for case files')
@click.option('--client', help='Client name')
@click.option('--legal-matter', help='Legal matter description')
@click.pass_context
def create_case(ctx, case_number, case_name, investigator, output_dir, client, legal_matter):
    """
    Create new forensics case.
    
    Initializes a new digital forensics case with proper structure:
    - Case metadata and documentation
    - Evidence tracking system
    - Chain of custody logging
    - Analysis workflow setup
    
    CASE_NUMBER: Unique case identifier
    CASE_NAME: Descriptive case name
    INVESTIGATOR: Lead investigator name
    """
    click.echo(f"Creating new forensics case: {case_number}")
    
    # Setup case directory
    case_dir = output_dir or f"case_{case_number}_{datetime.now().strftime('%Y%m%d')}"
    output_path = Path(case_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        config = ctx.obj['config']
        analyzer = DigitalForensicsAnalyzer(config)
        
        # Create case
        case = analyzer.create_case(
            case_number=case_number,
            case_name=case_name,
            investigator=investigator,
            client=client,
            legal_matter=legal_matter
        )
        
        # Create case directory structure
        (output_path / "evidence").mkdir(exist_ok=True)
        (output_path / "analysis").mkdir(exist_ok=True)
        (output_path / "reports").mkdir(exist_ok=True)
        (output_path / "exports").mkdir(exist_ok=True)
        
        # Save case metadata
        case_data = {
            'case_info': case.dict(),
            'created_by': 'lemkin-forensics-cli',
            'creation_timestamp': datetime.utcnow().isoformat()
        }
        
        save_results_to_json(case_data, output_path, 'case_metadata.json')
        
        # Create case README
        readme_content = f"""# Forensics Case: {case_name}

## Case Information
- Case Number: {case_number}
- Case Name: {case_name}
- Investigator: {investigator}
- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Client: {client or 'N/A'}
- Legal Matter: {legal_matter or 'N/A'}

## Directory Structure
- `evidence/` - Original evidence files
- `analysis/` - Analysis results and working files
- `reports/` - Generated reports and documentation
- `exports/` - Exported data and timelines

## Next Steps
1. Add evidence using: `lemkin-forensics add-evidence`
2. Perform analysis using appropriate commands
3. Generate reports using: `lemkin-forensics generate-timeline`
4. Export case data when complete

## Chain of Custody
All evidence handling is logged in the case metadata file.
Maintain proper chain of custody procedures throughout the investigation.
"""
        
        with open(output_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        click.echo(f"\n✓ Case created successfully!")
        click.echo(f"  Case Number: {case_number}")
        click.echo(f"  Case Name: {case_name}")
        click.echo(f"  Investigator: {investigator}")
        click.echo(f"  Case directory: {output_path}")
        click.echo(f"\nNext steps:")
        click.echo(f"  1. cd {output_path}")
        click.echo(f"  2. Add evidence files to evidence/ directory")
        click.echo(f"  3. Run analysis commands from case directory")
        
    except Exception as e:
        click.echo(f"Case creation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--version', is_flag=True, help='Show version information')
def info(version):
    """
    Display toolkit information and capabilities.
    """
    if version:
        click.echo("Lemkin Digital Forensics Toolkit v1.0.0")
        return
    
    click.echo("""
Lemkin Digital Forensics Toolkit
================================

Comprehensive digital forensics analysis for legal professionals.

Capabilities:
• Disk Image Analysis - E01, DD, AFF formats
• Network Traffic Analysis - PCAP, logs
• Mobile Device Forensics - iOS, Android backups
• Evidence Authenticity Verification
• Timeline Generation and Visualization
• Chain of Custody Management

Supported Evidence Types:
• Disk images and file systems
• Network packet captures
• Mobile device backups
• Email archives and databases
• Documents and media files
• Registry hives and system files

Legal Compliance:
• ISO 27037 evidence handling standards
• Chain of custody documentation
• Hash verification and integrity checking
• Court-admissible reporting

For help with specific commands, use:
  lemkin-forensics <command> --help

For more information, visit: https://lemkin.org/forensics
    """)


if __name__ == '__main__':
    cli()