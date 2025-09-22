"""
Command Line Interface for Lemkin Report Generator Suite.

Provides comprehensive CLI commands for generating legal reports, managing
templates, and exporting documents in various formats.
"""

import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

import click
from loguru import logger

from .core import (
    ReportGenerator, ReportConfig, create_report_generator, create_default_config,
    create_case_data, FactSheet, EvidenceCatalog, LegalBrief, ExportedReport,
    PersonInfo, CaseInfo, Evidence, EvidenceType, EvidenceAuthenticity,
    ExportFormat, ExportSettings, ReportType, CitationStyle, DocumentStandard,
    ConfidentialityLevel, LegalCitation
)


# Global CLI state
class CLIState:
    def __init__(self):
        self.config = None
        self.generator = None
        self.verbose = False
        
    def initialize(self, config_file: Optional[str] = None, verbose: bool = False):
        """Initialize CLI state"""
        self.verbose = verbose
        
        if verbose:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.remove()
            logger.add(sys.stderr, level="INFO")
        
        # Load configuration
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                self.config = ReportConfig(**config_data)
        else:
            self.config = create_default_config()
        
        # Create report generator
        self.generator = create_report_generator(self.config)
        
        logger.info("Lemkin Report Generator CLI initialized")


# Global state instance
cli_state = CLIState()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(config: Optional[str], verbose: bool):
    """Lemkin Report Generator Suite - Professional Legal Document Generation"""
    cli_state.initialize(config, verbose)


@cli.group()
def generate():
    """Generate legal reports and documents"""
    pass


@generate.command('fact-sheet')
@click.option('--case-file', '-f', required=True, type=click.Path(exists=True), 
              help='JSON file containing case data')
@click.option('--template', '-t', default='standard', help='Template to use (default: standard)')
@click.option('--author', '-a', help='Author name for the fact sheet')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--export-format', '-e', type=click.Choice(['pdf', 'docx', 'html']), 
              default='pdf', help='Export format (default: pdf)')
def generate_fact_sheet(case_file: str, template: str, author: Optional[str], 
                       output: Optional[str], export_format: str):
    """Generate standardized fact sheet for a case"""
    try:
        # Load case data
        with open(case_file, 'r') as f:
            case_data_dict = json.load(f)
        
        case_data = _parse_case_data(case_data_dict)
        
        # Get author info
        author_info = None
        if author:
            author_info = PersonInfo(full_name=author, role="attorney")
        
        # Generate fact sheet
        click.echo(f"Generating fact sheet using {template} template...")
        fact_sheet = cli_state.generator.generate_fact_sheet(
            case_data=case_data,
            template=template,
            author=author_info
        )
        
        click.echo(f"✓ Fact sheet generated: {fact_sheet.title}")
        
        # Export if requested
        if output or export_format != 'json':
            _export_report(fact_sheet, export_format, output)
        
        # Save as JSON
        json_output = _get_json_output_path(output, "fact_sheet", case_data.case_info.case_number)
        with open(json_output, 'w') as f:
            f.write(fact_sheet.model_dump_json(indent=2))
        
        click.echo(f"✓ Fact sheet saved: {json_output}")
        
    except Exception as e:
        click.echo(f"✗ Error generating fact sheet: {str(e)}", err=True)
        sys.exit(1)


@generate.command('evidence-catalog')
@click.option('--case-file', '-f', required=True, type=click.Path(exists=True),
              help='JSON file containing case data with evidence')
@click.option('--custodian', '-c', required=True, help='Evidence custodian name')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--export-format', '-e', type=click.Choice(['pdf', 'docx', 'html']),
              default='pdf', help='Export format (default: pdf)')
def generate_evidence_catalog(case_file: str, custodian: str, output: Optional[str], 
                             export_format: str):
    """Generate comprehensive evidence inventory and catalog"""
    try:
        # Load case data
        with open(case_file, 'r') as f:
            case_data_dict = json.load(f)
        
        case_data = _parse_case_data(case_data_dict)
        
        if not case_data.evidence_list:
            click.echo("✗ No evidence items found in case data", err=True)
            sys.exit(1)
        
        # Create custodian info
        custodian_info = PersonInfo(full_name=custodian, role="custodian")
        
        # Generate evidence catalog
        click.echo(f"Cataloging {len(case_data.evidence_list)} evidence items...")
        catalog = cli_state.generator.catalog_evidence(
            evidence_list=case_data.evidence_list,
            case_data=case_data,
            custodian=custodian_info
        )
        
        click.echo(f"✓ Evidence catalog generated: {catalog.title}")
        
        # Export if requested
        if output or export_format != 'json':
            _export_report(catalog, export_format, output)
        
        # Save as JSON
        json_output = _get_json_output_path(output, "evidence_catalog", case_data.case_info.case_number)
        with open(json_output, 'w') as f:
            f.write(catalog.model_dump_json(indent=2))
        
        click.echo(f"✓ Evidence catalog saved: {json_output}")
        
    except Exception as e:
        click.echo(f"✗ Error generating evidence catalog: {str(e)}", err=True)
        sys.exit(1)


@generate.command('legal-brief')
@click.option('--case-file', '-f', required=True, type=click.Path(exists=True),
              help='JSON file containing case data')
@click.option('--template', '-t', default='motion', help='Template to use (motion, response, appellate)')
@click.option('--brief-type', '-b', required=True, help='Type of brief (motion, response, etc.)')
@click.option('--author', '-a', required=True, help='Author name for the brief')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--export-format', '-e', type=click.Choice(['pdf', 'docx', 'html', 'latex']),
              default='pdf', help='Export format (default: pdf)')
def generate_legal_brief(case_file: str, template: str, brief_type: str, author: str,
                        output: Optional[str], export_format: str):
    """Generate auto-populated legal brief from template"""
    try:
        # Load case data
        with open(case_file, 'r') as f:
            case_data_dict = json.load(f)
        
        case_data = _parse_case_data(case_data_dict)
        
        # Create author info
        author_info = PersonInfo(full_name=author, role="attorney")
        
        # Generate legal brief
        click.echo(f"Generating {brief_type} brief using {template} template...")
        brief = cli_state.generator.format_legal_brief(
            case_data=case_data,
            template=template,
            brief_type=brief_type,
            author=author_info
        )
        
        click.echo(f"✓ Legal brief generated: {brief.title}")
        click.echo(f"  Word count: {brief.word_count}")
        click.echo(f"  Argument sections: {len(brief.argument_sections)}")
        
        # Export if requested
        if output or export_format != 'json':
            _export_report(brief, export_format, output)
        
        # Save as JSON
        json_output = _get_json_output_path(output, "legal_brief", case_data.case_info.case_number)
        with open(json_output, 'w') as f:
            f.write(brief.model_dump_json(indent=2))
        
        click.echo(f"✓ Legal brief saved: {json_output}")
        
    except Exception as e:
        click.echo(f"✗ Error generating legal brief: {str(e)}", err=True)
        sys.exit(1)


@cli.command('batch-generate')
@click.option('--cases-dir', '-d', required=True, type=click.Path(exists=True),
              help='Directory containing case data JSON files')
@click.option('--report-types', '-r', multiple=True, required=True,
              type=click.Choice(['fact-sheet', 'evidence-catalog', 'legal-brief']),
              help='Types of reports to generate (can be specified multiple times)')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--export-format', '-e', type=click.Choice(['pdf', 'docx', 'html']),
              default='pdf', help='Export format (default: pdf)')
def batch_generate(cases_dir: str, report_types: tuple, output_dir: Optional[str], 
                  export_format: str):
    """Generate multiple reports for multiple cases"""
    try:
        cases_path = Path(cases_dir)
        case_files = list(cases_path.glob("*.json"))
        
        if not case_files:
            click.echo("✗ No JSON case files found in directory", err=True)
            sys.exit(1)
        
        click.echo(f"Found {len(case_files)} case files")
        click.echo(f"Report types: {', '.join(report_types)}")
        
        # Load all case data
        cases_data = []
        for case_file in case_files:
            try:
                with open(case_file, 'r') as f:
                    case_data_dict = json.load(f)
                case_data = _parse_case_data(case_data_dict)
                cases_data.append(case_data)
                click.echo(f"  ✓ Loaded: {case_data.case_info.case_name}")
            except Exception as e:
                click.echo(f"  ✗ Error loading {case_file}: {str(e)}", err=True)
                continue
        
        # Convert report types
        report_type_mapping = {
            'fact-sheet': ReportType.FACT_SHEET,
            'evidence-catalog': ReportType.EVIDENCE_CATALOG,
            'legal-brief': ReportType.LEGAL_BRIEF
        }
        
        report_type_enums = [report_type_mapping[rt] for rt in report_types]
        
        # Generate reports
        click.echo("Starting batch generation...")
        reports = cli_state.generator.batch_generate_reports(
            cases=cases_data,
            report_types=report_type_enums
        )
        
        click.echo(f"✓ Generated {len(reports)} reports")
        
        # Export all reports
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        for report in reports:
            try:
                case_number = report.case_data.case_info.case_number
                _export_report(report, export_format, 
                             str(output_path) if output_dir else None)
                click.echo(f"  ✓ Exported: {type(report).__name__} for {case_number}")
            except Exception as e:
                click.echo(f"  ✗ Export error: {str(e)}", err=True)
        
        click.echo("✓ Batch generation completed")
        
    except Exception as e:
        click.echo(f"✗ Error in batch generation: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def export():
    """Export reports to various formats"""
    pass


@export.command('report')
@click.option('--report-file', '-f', required=True, type=click.Path(exists=True),
              help='JSON file containing report data')
@click.option('--format', '-t', required=True, 
              type=click.Choice(['pdf', 'docx', 'html', 'latex']),
              help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--template', help='Custom export template')
def export_report(report_file: str, format: str, output: Optional[str], template: Optional[str]):
    """Export a saved report to specified format"""
    try:
        # Load report data
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        # Determine report type and recreate object
        report = _parse_report_data(report_data)
        
        # Create export settings
        export_format = ExportFormat(format)
        settings = ExportSettings(export_format=export_format)
        
        if output:
            settings.output_directory = Path(output).parent
            settings.filename_template = Path(output).stem
        
        # Export report
        click.echo(f"Exporting {type(report).__name__} to {format.upper()}...")
        exported = cli_state.generator.export_report(
            report=report,
            format_type=export_format,
            settings=settings
        )
        
        if exported.export_successful:
            click.echo(f"✓ Report exported: {exported.output_path}")
            click.echo(f"  File size: {exported.file_size_bytes:,} bytes")
        else:
            click.echo("✗ Export failed:", err=True)
            for error in exported.export_errors:
                click.echo(f"  - {error}", err=True)
        
    except Exception as e:
        click.echo(f"✗ Error exporting report: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def templates():
    """Manage report templates"""
    pass


@templates.command('list')
@click.option('--type', '-t', type=click.Choice(['standard', 'court_specific', 'custom']),
              help='Filter by template type')
def list_templates(type: Optional[str]):
    """List available report templates"""
    try:
        templates = cli_state.generator.get_available_templates()
        
        if type:
            templates = [t for t in templates if type in t.tags]
        
        if not templates:
            click.echo("No templates found")
            return
        
        click.echo("Available Templates:")
        click.echo("-" * 50)
        
        for template in templates:
            click.echo(f"ID: {template.template_id}")
            click.echo(f"Name: {template.name}")
            click.echo(f"Category: {template.category}")
            click.echo(f"Usage Count: {template.times_used}")
            if template.user_rating:
                click.echo(f"Rating: {template.user_rating:.1f}/5.0")
            click.echo(f"Tags: {', '.join(template.tags)}")
            click.echo("-" * 30)
        
    except Exception as e:
        click.echo(f"✗ Error listing templates: {str(e)}", err=True)


@templates.command('validate')
@click.option('--template-file', '-f', required=True, type=click.Path(exists=True),
              help='JSON file containing template definition')
def validate_template(template_file: str):
    """Validate a custom template definition"""
    try:
        with open(template_file, 'r') as f:
            template_data = json.load(f)
        
        # Create template object for validation
        from .core import LegalTemplate, TemplateType, DocumentStandard
        template = LegalTemplate(**template_data)
        
        click.echo("✓ Template structure is valid")
        click.echo(f"Template: {template.name}")
        click.echo(f"Type: {template.template_type}")
        click.echo(f"Sections: {len(template.sections)}")
        
        # Additional validation could be added here
        
    except Exception as e:
        click.echo(f"✗ Template validation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def validate():
    """Validate reports and documents"""
    pass


@validate.command('report')
@click.option('--report-file', '-f', required=True, type=click.Path(exists=True),
              help='JSON file containing report data')
def validate_report(report_file: str):
    """Validate a generated report"""
    try:
        # Load and parse report
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        report = _parse_report_data(report_data)
        
        # Validate report
        validation_result = cli_state.generator.validate_report(report)
        
        click.echo("Report Validation Results:")
        click.echo("-" * 30)
        
        if validation_result["valid"]:
            click.echo("✓ Report is valid")
        else:
            click.echo("✗ Report has validation issues")
        
        click.echo(f"Completeness Score: {validation_result['completeness_score']:.1%}")
        click.echo(f"Quality Score: {validation_result['quality_score']:.1%}")
        
        if validation_result["errors"]:
            click.echo("\nErrors:")
            for error in validation_result["errors"]:
                click.echo(f"  ✗ {error}")
        
        if validation_result["warnings"]:
            click.echo("\nWarnings:")
            for warning in validation_result["warnings"]:
                click.echo(f"  ⚠ {warning}")
        
        if validation_result["recommendations"]:
            click.echo("\nRecommendations:")
            for rec in validation_result["recommendations"]:
                click.echo(f"  → {rec}")
        
    except Exception as e:
        click.echo(f"✗ Error validating report: {str(e)}", err=True)
        sys.exit(1)


@cli.command('config')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--create', type=click.Path(), help='Create default config file')
@click.option('--set', 'set_values', multiple=True, help='Set configuration value (key=value)')
def config(show: bool, create: Optional[str], set_values: tuple):
    """Manage configuration settings"""
    try:
        if show:
            click.echo("Current Configuration:")
            click.echo("-" * 30)
            config_dict = cli_state.config.model_dump()
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    click.echo(f"{key}:")
                    for sub_key, sub_value in value.items():
                        click.echo(f"  {sub_key}: {sub_value}")
                else:
                    click.echo(f"{key}: {value}")
        
        if create:
            config_path = Path(create)
            default_config = create_default_config()
            
            with open(config_path, 'w') as f:
                f.write(default_config.model_dump_json(indent=2))
            
            click.echo(f"✓ Default configuration created: {config_path}")
        
        if set_values:
            click.echo("Configuration updates not yet implemented")
            # TODO: Implement configuration updates
        
    except Exception as e:
        click.echo(f"✗ Error managing configuration: {str(e)}", err=True)


@cli.command('version')
def version():
    """Show version information"""
    click.echo("Lemkin Report Generator Suite")
    click.echo("Version: 1.0.0")
    click.echo("Author: Lemkin Legal Technologies")


# Helper functions

def _parse_case_data(case_data_dict: Dict[str, Any]) -> 'CaseData':
    """Parse case data from dictionary"""
    # Parse case info
    case_info_dict = case_data_dict.get('case_info', {})
    case_info = CaseInfo(**case_info_dict)
    
    # Parse evidence list
    evidence_list = []
    for evidence_dict in case_data_dict.get('evidence_list', []):
        # Convert date strings to date objects
        if 'date_collected' in evidence_dict and isinstance(evidence_dict['date_collected'], str):
            evidence_dict['date_collected'] = datetime.fromisoformat(evidence_dict['date_collected']).date()
        if 'authentication_date' in evidence_dict and isinstance(evidence_dict['authentication_date'], str):
            evidence_dict['authentication_date'] = datetime.fromisoformat(evidence_dict['authentication_date']).date()
        
        evidence = Evidence(**evidence_dict)
        evidence_list.append(evidence)
    
    # Parse other fields
    witnesses = [PersonInfo(**w) for w in case_data_dict.get('witnesses', [])]
    attorneys = [PersonInfo(**a) for a in case_data_dict.get('attorneys', [])]
    precedent_cases = [LegalCitation(**p) for p in case_data_dict.get('precedent_cases', [])]
    
    # Create case data
    case_data = create_case_data(
        case_info.case_number,
        case_info.case_name,
        case_info.court,
        case_info.jurisdiction,
        case_info.practice_area
    )
    
    # Update with parsed data
    case_data.case_info = case_info
    case_data.evidence_list = evidence_list
    case_data.witnesses = witnesses
    case_data.attorneys = attorneys
    case_data.precedent_cases = precedent_cases
    
    # Parse other optional fields
    for field in ['legal_theories', 'causes_of_action', 'defenses', 'key_dates', 
                  'chronology', 'statement_of_facts', 'disputed_facts', 'undisputed_facts',
                  'strengths', 'weaknesses', 'risks', 'opportunities']:
        if field in case_data_dict:
            setattr(case_data, field, case_data_dict[field])
    
    return case_data


def _parse_report_data(report_data: Dict[str, Any]):
    """Parse report data from dictionary and return appropriate report object"""
    report_type = report_data.get('report_type')
    
    if report_type == 'fact_sheet':
        return FactSheet(**report_data)
    elif report_type == 'evidence_catalog':
        return EvidenceCatalog(**report_data)
    elif report_type == 'legal_brief':
        return LegalBrief(**report_data)
    else:
        raise ValueError(f"Unknown report type: {report_type}")


def _export_report(report, export_format: str, output_path: Optional[str]):
    """Export report to specified format"""
    format_enum = ExportFormat(export_format)
    settings = ExportSettings(export_format=format_enum)
    
    if output_path:
        settings.output_directory = Path(output_path).parent
        settings.filename_template = Path(output_path).stem
    
    exported = cli_state.generator.export_report(
        report=report,
        format_type=format_enum,
        settings=settings
    )
    
    if exported.export_successful:
        click.echo(f"✓ Exported to {export_format.upper()}: {exported.output_path}")
    else:
        click.echo(f"✗ Export to {export_format.upper()} failed:")
        for error in exported.export_errors:
            click.echo(f"  - {error}")


def _get_json_output_path(output: Optional[str], report_type: str, case_number: str) -> Path:
    """Generate JSON output path"""
    if output:
        return Path(output).with_suffix('.json')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{case_number.replace(' ', '_')}_{timestamp}.json"
        return Path(filename)


if __name__ == '__main__':
    cli()