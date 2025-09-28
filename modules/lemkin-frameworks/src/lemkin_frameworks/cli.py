"""
Command Line Interface for Legal Framework Mapper.

This module provides a comprehensive CLI for analyzing evidence against
various international legal frameworks including the Rome Statute,
Geneva Conventions, and human rights instruments.
"""

import json
import sys
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any

import click
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.text import Text
from rich.tree import Tree

from .core import (
    LegalFrameworkMapper, Evidence, LegalAssessment, FrameworkAnalysis,
    LegalFramework, EvidenceType, ConfidenceLevel
)
from .rome_statute import RomeStatuteAnalyzer, analyze_rome_statute_elements
from .geneva_conventions import GenevaAnalyzer, assess_geneva_violations
from .human_rights_frameworks import HumanRightsAnalyzer, analyze_human_rights_violations


console = Console()


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special types."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', help='Log file path')
def cli(verbose: bool, log_file: Optional[str]):
    """Legal Framework Mapper - Analyze evidence against international legal frameworks."""
    
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<green>{time}</green> | <level>{level}</level> | {message}")
    
    if log_file:
        logger.add(log_file, level=log_level, format="{time} | {level} | {message}")
    
    console.print(Panel.fit(
        "[bold blue]Legal Framework Mapper[/bold blue]\n"
        "Analyze evidence against international legal frameworks",
        title="🏛️ Lemkin Frameworks"
    ))


@cli.command()
@click.argument('evidence_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path (JSON format)')
@click.option('--confidence-threshold', '-t', default=0.6, type=float, 
              help='Minimum confidence threshold for reporting violations')
@click.option('--include-elements', '-e', multiple=True, 
              help='Specific elements to analyze (e.g., genocide_6a, cah_7_1_a)')
@click.option('--exclude-elements', multiple=True,
              help='Elements to exclude from analysis')
def analyze_rome_statute(evidence_file: str, output: Optional[str], 
                        confidence_threshold: float, include_elements: tuple,
                        exclude_elements: tuple):
    """Analyze evidence against Rome Statute crimes (genocide, crimes against humanity, war crimes)."""
    
    console.print("\n[bold yellow]🔍 Rome Statute Analysis[/bold yellow]")
    
    # Load evidence
    evidence = load_evidence_from_file(evidence_file)
    if not evidence:
        console.print("[red]❌ No evidence loaded. Exiting.[/red]")
        return
    
    console.print(f"📄 Loaded {len(evidence)} pieces of evidence")
    
    # Initialize analyzer
    with console.status("[bold green]Initializing Rome Statute analyzer..."):
        analyzer = RomeStatuteAnalyzer()
    
    # Filter elements if specified
    if include_elements or exclude_elements:
        filtered_elements = filter_legal_elements(
            analyzer.legal_elements, include_elements, exclude_elements
        )
        analyzer.legal_elements = filtered_elements
        console.print(f"🎯 Analyzing {len(filtered_elements)} selected elements")
    
    # Perform analysis
    with console.status("[bold green]Analyzing evidence against Rome Statute elements..."):
        analysis = analyzer.analyze(evidence)
    
    # Display results
    display_framework_analysis(analysis, confidence_threshold)
    
    # Rome Statute specific analysis
    rome_analysis = analyze_rome_statute_elements(evidence)
    display_rome_statute_specific_results(rome_analysis)
    
    # Save output if requested
    if output:
        save_analysis_results({
            'framework_analysis': analysis,
            'rome_statute_analysis': rome_analysis
        }, output)
        console.print(f"💾 Results saved to {output}")


@cli.command()
@click.argument('evidence_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path (JSON format)')
@click.option('--confidence-threshold', '-t', default=0.6, type=float)
@click.option('--conflict-type', type=click.Choice(['international', 'non-international', 'auto']),
              default='auto', help='Type of armed conflict for analysis context')
@click.option('--protected-persons', '-p', multiple=True,
              type=click.Choice(['civilians', 'prisoners_of_war', 'wounded_sick', 'medical_personnel']),
              help='Focus on specific categories of protected persons')
def analyze_geneva(evidence_file: str, output: Optional[str], confidence_threshold: float,
                  conflict_type: str, protected_persons: tuple):
    """Analyze evidence against Geneva Conventions and IHL violations."""
    
    console.print("\n[bold red]⚔️ Geneva Conventions Analysis[/bold red]")
    
    # Load evidence
    evidence = load_evidence_from_file(evidence_file)
    if not evidence:
        console.print("[red]❌ No evidence loaded. Exiting.[/red]")
        return
    
    console.print(f"📄 Loaded {len(evidence)} pieces of evidence")
    console.print(f"⚔️ Conflict classification: {conflict_type}")
    
    if protected_persons:
        console.print(f"👥 Focus on protected persons: {', '.join(protected_persons)}")
    
    # Initialize analyzer
    with console.status("[bold green]Initializing Geneva Conventions analyzer..."):
        analyzer = GenevaAnalyzer()
    
    # Perform analysis
    with console.status("[bold green]Analyzing evidence against IHL provisions..."):
        analysis = analyzer.analyze(evidence)
    
    # Display results
    display_framework_analysis(analysis, confidence_threshold)
    
    # Geneva-specific analysis
    geneva_analysis = assess_geneva_violations(evidence)
    display_geneva_specific_results(geneva_analysis, protected_persons)
    
    # Save output if requested
    if output:
        save_analysis_results({
            'framework_analysis': analysis,
            'geneva_analysis': geneva_analysis,
            'conflict_type': conflict_type,
            'protected_persons_focus': list(protected_persons)
        }, output)
        console.print(f"💾 Results saved to {output}")


@cli.command()
@click.argument('evidence_file', type=click.Path(exists=True))
@click.argument('framework', type=click.Choice(['iccpr', 'echr', 'achr', 'achpr', 'udhr']))
@click.option('--output', '-o', help='Output file path (JSON format)')
@click.option('--confidence-threshold', '-t', default=0.6, type=float)
@click.option('--rights-focus', '-r', multiple=True,
              type=click.Choice(['right_to_life', 'prohibition_of_torture', 'right_to_liberty', 
                               'right_to_fair_trial', 'freedom_of_expression', 'freedom_of_assembly']),
              help='Focus on specific human rights')
def analyze_human_rights(evidence_file: str, framework: str, output: Optional[str], 
                        confidence_threshold: float, rights_focus: tuple):
    """Analyze evidence against human rights frameworks (ICCPR, ECHR, ACHR, ACHPR, UDHR)."""
    
    framework_names = {
        'iccpr': 'International Covenant on Civil and Political Rights',
        'echr': 'European Convention on Human Rights',
        'achr': 'American Convention on Human Rights',
        'achpr': 'African Charter on Human and Peoples\' Rights',
        'udhr': 'Universal Declaration of Human Rights'
    }
    
    console.print(f"\n[bold cyan]🏛️ {framework_names[framework]} Analysis[/bold cyan]")
    
    # Load evidence
    evidence = load_evidence_from_file(evidence_file)
    if not evidence:
        console.print("[red]❌ No evidence loaded. Exiting.[/red]")
        return
    
    console.print(f"📄 Loaded {len(evidence)} pieces of evidence")
    
    if rights_focus:
        console.print(f"🎯 Focus on rights: {', '.join(rights_focus)}")
    
    # Initialize analyzer
    legal_framework = LegalFramework(framework)
    with console.status(f"[bold green]Initializing {framework.upper()} analyzer..."):
        analyzer = HumanRightsAnalyzer(legal_framework)
    
    # Perform analysis
    with console.status("[bold green]Analyzing evidence against human rights provisions..."):
        analysis = analyzer.analyze(evidence)
    
    # Display results
    display_framework_analysis(analysis, confidence_threshold)
    
    # Human rights specific analysis
    hr_analysis = analyze_human_rights_violations(evidence, legal_framework)
    display_human_rights_specific_results(hr_analysis, rights_focus)
    
    # Save output if requested
    if output:
        save_analysis_results({
            'framework_analysis': analysis,
            'human_rights_analysis': hr_analysis,
            'framework': framework,
            'rights_focus': list(rights_focus)
        }, output)
        console.print(f"💾 Results saved to {output}")


@cli.command()
@click.argument('evidence_file', type=click.Path(exists=True))
@click.option('--frameworks', '-f', multiple=True, required=True,
              type=click.Choice(['rome_statute', 'geneva_conventions', 'iccpr', 'echr', 'achr', 'achpr', 'udhr']),
              help='Legal frameworks to include in assessment')
@click.option('--output', '-o', help='Output file path (JSON format)')
@click.option('--title', default='Legal Assessment', help='Title for the assessment')
@click.option('--description', help='Description of the case or incident')
@click.option('--confidence-threshold', '-t', default=0.6, type=float)
@click.option('--include-cross-analysis', is_flag=True, 
              help='Include cross-framework pattern analysis')
def generate_assessment(evidence_file: str, frameworks: tuple, output: Optional[str],
                       title: str, description: Optional[str], confidence_threshold: float,
                       include_cross_analysis: bool):
    """Generate comprehensive legal assessment across multiple frameworks."""
    
    console.print(f"\n[bold magenta]📊 {title}[/bold magenta]")
    
    if description:
        console.print(f"📝 {description}")
    
    # Load evidence
    evidence = load_evidence_from_file(evidence_file)
    if not evidence:
        console.print("[red]❌ No evidence loaded. Exiting.[/red]")
        return
    
    console.print(f"📄 Loaded {len(evidence)} pieces of evidence")
    console.print(f"⚖️ Frameworks: {', '.join(frameworks)}")
    
    # Convert framework names to enum values
    framework_enums = []
    for fw in frameworks:
        try:
            framework_enums.append(LegalFramework(fw))
        except ValueError:
            console.print(f"[red]❌ Unknown framework: {fw}[/red]")
            return
    
    # Initialize mapper
    with console.status("[bold green]Initializing Legal Framework Mapper..."):
        mapper = LegalFrameworkMapper()
    
    # Generate assessment
    with console.status("[bold green]Generating comprehensive legal assessment..."):
        assessment = mapper.generate_legal_assessment(
            evidence=evidence,
            frameworks=framework_enums,
            title=title,
            description=description or ""
        )
    
    # Display results
    display_legal_assessment(assessment, confidence_threshold, include_cross_analysis)
    
    # Save output if requested
    if output:
        save_analysis_results({
            'legal_assessment': assessment,
            'parameters': {
                'frameworks': list(frameworks),
                'confidence_threshold': confidence_threshold,
                'include_cross_analysis': include_cross_analysis
            }
        }, output)
        console.print(f"💾 Assessment saved to {output}")


@cli.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed framework information')
def list_frameworks(detailed: bool):
    """List all available legal frameworks and their capabilities."""
    
    console.print("\n[bold green]📚 Available Legal Frameworks[/bold green]")
    
    frameworks_info = {
        'rome_statute': {
            'name': 'Rome Statute of the International Criminal Court',
            'description': 'Analyze genocide, crimes against humanity, war crimes, and crime of aggression',
            'elements': ['Genocide (Article 6)', 'Crimes Against Humanity (Article 7)', 
                        'War Crimes (Article 8)', 'General Elements'],
            'jurisdiction': 'International Criminal Court',
            'coverage': 'International crimes committed after July 1, 2002'
        },
        'geneva_conventions': {
            'name': 'Geneva Conventions and Additional Protocols',
            'description': 'Analyze violations of International Humanitarian Law',
            'elements': ['Geneva Convention I-IV', 'Additional Protocol I-II', 
                        'Grave Breaches', 'Protected Persons'],
            'jurisdiction': 'Universal jurisdiction for grave breaches',
            'coverage': 'International and non-international armed conflicts'
        },
        'iccpr': {
            'name': 'International Covenant on Civil and Political Rights',
            'description': 'Analyze violations of civil and political rights',
            'elements': ['Right to Life', 'Prohibition of Torture', 'Right to Liberty',
                        'Fair Trial Rights', 'Freedom of Expression'],
            'jurisdiction': 'UN Human Rights Committee',
            'coverage': 'States parties to the ICCPR'
        },
        'echr': {
            'name': 'European Convention on Human Rights',
            'description': 'Analyze human rights violations in European context',
            'elements': ['Right to Life', 'Prohibition of Torture', 'Right to Liberty',
                        'Private Life', 'Freedom of Expression'],
            'jurisdiction': 'European Court of Human Rights',
            'coverage': 'Council of Europe member states'
        },
        'achr': {
            'name': 'American Convention on Human Rights',
            'description': 'Analyze human rights violations in Americas',
            'elements': ['Right to Life', 'Right to Humane Treatment', 'Liberty Rights'],
            'jurisdiction': 'Inter-American Court of Human Rights',
            'coverage': 'OAS member states (ratifying parties)'
        },
        'achpr': {
            'name': 'African Charter on Human and Peoples\' Rights',
            'description': 'Analyze human rights violations in African context',
            'elements': ['Right to Life', 'Prohibition of Torture', 'Individual Rights'],
            'jurisdiction': 'African Court on Human and Peoples\' Rights',
            'coverage': 'African Union member states'
        },
        'udhr': {
            'name': 'Universal Declaration of Human Rights',
            'description': 'Foundational human rights analysis',
            'elements': ['Right to Life', 'Prohibition of Torture', 'Fundamental Rights'],
            'jurisdiction': 'Moral authority (non-binding)',
            'coverage': 'Universal application'
        }
    }
    
    if detailed:
        for fw_key, fw_info in frameworks_info.items():
            panel_content = []
            panel_content.append(f"[bold]{fw_info['description']}[/bold]\n")
            panel_content.append(f"📍 Jurisdiction: {fw_info['jurisdiction']}")
            panel_content.append(f"🌍 Coverage: {fw_info['coverage']}\n")
            panel_content.append("📋 Key Elements:")
            for element in fw_info['elements']:
                panel_content.append(f"  • {element}")
            
            console.print(Panel(
                "\n".join(panel_content),
                title=f"[cyan]{fw_info['name']}[/cyan]",
                title_align="left"
            ))
    else:
        table = Table(title="Legal Frameworks")
        table.add_column("Framework", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Jurisdiction", style="yellow")
        
        for fw_key, fw_info in frameworks_info.items():
            table.add_row(fw_key, fw_info['name'], fw_info['jurisdiction'])
        
        console.print(table)


@cli.command()
@click.argument('evidence_files', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--output-dir', '-d', help='Output directory for processed evidence')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']), 
              default='json', help='Output format')
@click.option('--merge', is_flag=True, help='Merge all evidence into single file')
def process_evidence(evidence_files: tuple, output_dir: Optional[str], 
                    output_format: str, merge: bool):
    """Process and standardize evidence files for analysis."""
    
    console.print(f"\n[bold blue]📁 Processing {len(evidence_files)} evidence file(s)[/bold blue]")
    
    all_evidence = []
    
    for file_path in track(evidence_files, description="Processing files..."):
        evidence = load_evidence_from_file(file_path)
        if evidence:
            all_evidence.extend(evidence)
            console.print(f"✅ Processed {len(evidence)} items from {file_path}")
        else:
            console.print(f"⚠️ No evidence found in {file_path}")
    
    console.print(f"📊 Total evidence items: {len(all_evidence)}")
    
    # Display evidence summary
    display_evidence_summary(all_evidence)
    
    # Save processed evidence
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if merge:
            # Save all evidence in one file
            filename = f"merged_evidence.{output_format}"
            file_path = output_path / filename
            save_evidence_to_file(all_evidence, str(file_path), output_format)
            console.print(f"💾 Merged evidence saved to {file_path}")
        else:
            # Save evidence by source file
            evidence_by_source = {}
            for ev in all_evidence:
                source = ev.source
                if source not in evidence_by_source:
                    evidence_by_source[source] = []
                evidence_by_source[source].append(ev)
            
            for source, evidence_list in evidence_by_source.items():
                safe_filename = "".join(c for c in source if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{safe_filename[:50]}_evidence.{output_format}"
                file_path = output_path / filename
                save_evidence_to_file(evidence_list, str(file_path), output_format)
                console.print(f"💾 {len(evidence_list)} items saved to {file_path}")


@cli.command()
@click.argument('template_name', type=click.Choice(['basic', 'rome_statute', 'geneva', 'human_rights']))
@click.option('--output', '-o', help='Output file path')
@click.option('--count', '-c', default=5, help='Number of sample evidence items to generate')
def create_evidence_template(template_name: str, output: Optional[str], count: int):
    """Create evidence template files for testing and examples."""
    
    console.print(f"\n[bold green]📋 Creating {template_name} evidence template[/bold green]")
    
    templates = {
        'basic': create_basic_evidence_template,
        'rome_statute': create_rome_statute_evidence_template,
        'geneva': create_geneva_evidence_template,
        'human_rights': create_human_rights_evidence_template
    }
    
    template_func = templates[template_name]
    evidence_items = template_func(count)
    
    output_file = output or f"{template_name}_evidence_template.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([ev.__dict__ for ev in evidence_items], f, 
                 indent=2, cls=CustomJSONEncoder, ensure_ascii=False)
    
    console.print(f"✅ Created {len(evidence_items)} evidence items")
    console.print(f"💾 Template saved to {output_file}")
    
    # Display template structure
    if evidence_items:
        console.print("\n📋 Template Structure:")
        sample_ev = evidence_items[0]
        for key, value in sample_ev.__dict__.items():
            console.print(f"  • {key}: {type(value).__name__}")


# Helper functions

def load_evidence_from_file(file_path: str) -> List[Evidence]:
    """Load evidence from a file (JSON, CSV, or TXT format)."""
    try:
        path = Path(file_path)
        
        if path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            evidence_list = []
            for item in data:
                if isinstance(item, dict):
                    # Handle date strings
                    for date_field in ['date_collected', 'incident_date']:
                        if date_field in item and isinstance(item[date_field], str):
                            try:
                                item[date_field] = datetime.fromisoformat(item[date_field].replace('Z', '+00:00'))
                            except ValueError:
                                item[date_field] = None
                    
                    # Handle UUID strings
                    if 'id' in item and isinstance(item['id'], str):
                        try:
                            item['id'] = uuid.UUID(item['id'])
                        except ValueError:
                            item['id'] = uuid.uuid4()
                    
                    # Create Evidence object
                    evidence = Evidence(**item)
                    evidence_list.append(evidence)
                    
            return evidence_list
            
        elif path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            evidence_list = []
            
            for _, row in df.iterrows():
                evidence_data = {
                    'title': row.get('title', 'Untitled Evidence'),
                    'content': row.get('content', ''),
                    'evidence_type': row.get('evidence_type', 'document'),
                    'source': row.get('source', 'Unknown'),
                    'tags': row.get('tags', '').split(',') if row.get('tags') else [],
                    'reliability_score': float(row.get('reliability_score', 0.5))
                }
                
                # Handle optional fields
                for field in ['location', 'date_collected', 'incident_date']:
                    if field in row and pd.notna(row[field]):
                        if 'date' in field:
                            try:
                                evidence_data[field] = pd.to_datetime(row[field])
                            except:
                                evidence_data[field] = None
                        else:
                            evidence_data[field] = row[field]
                
                evidence = Evidence(**evidence_data)
                evidence_list.append(evidence)
                
            return evidence_list
            
        else:
            # Treat as plain text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            evidence = Evidence(
                title=f"Evidence from {path.name}",
                content=content,
                evidence_type=EvidenceType.DOCUMENT,
                source=str(path),
                reliability_score=0.7
            )
            return [evidence]
            
    except Exception as e:
        logger.error(f"Error loading evidence from {file_path}: {e}")
        return []


def save_evidence_to_file(evidence: List[Evidence], file_path: str, format_type: str):
    """Save evidence to file in specified format."""
    try:
        if format_type == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([ev.__dict__ for ev in evidence], f, 
                         indent=2, cls=CustomJSONEncoder, ensure_ascii=False)
        elif format_type == 'csv':
            # Convert to DataFrame
            data = []
            for ev in evidence:
                row = ev.__dict__.copy()
                # Handle special types
                if 'tags' in row:
                    row['tags'] = ','.join(row['tags']) if row['tags'] else ''
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8')
        
    except Exception as e:
        logger.error(f"Error saving evidence to {file_path}: {e}")


def save_analysis_results(results: Dict[str, Any], file_path: str):
    """Save analysis results to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, cls=CustomJSONEncoder, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving results to {file_path}: {e}")


def filter_legal_elements(elements: Dict[str, Any], include: tuple, exclude: tuple) -> Dict[str, Any]:
    """Filter legal elements based on include/exclude lists."""
    filtered = elements.copy()
    
    if include:
        # Only keep included elements
        filtered = {k: v for k, v in elements.items() if k in include}
    
    if exclude:
        # Remove excluded elements
        filtered = {k: v for k, v in filtered.items() if k not in exclude}
    
    return filtered


def display_framework_analysis(analysis: FrameworkAnalysis, threshold: float):
    """Display framework analysis results."""
    console.print(f"\n[bold green]📊 {analysis.framework.value.replace('_', ' ').title()} Analysis Results[/bold green]")
    console.print(f"📅 Analysis Date: {analysis.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"📄 Evidence Analyzed: {analysis.evidence_count}")
    console.print(f"⚖️ Overall Confidence: {analysis.overall_confidence:.1%}")
    
    # Summary
    console.print(Panel(analysis.summary, title="📋 Executive Summary"))
    
    # Violations
    if analysis.violations_identified:
        violations_above_threshold = [
            v for s in analysis.element_satisfactions 
            for v in analysis.violations_identified
            if s.confidence >= threshold
        ]
        
        if violations_above_threshold:
            console.print(f"\n[bold red]⚠️ Violations Identified (≥{threshold:.1%} confidence)[/bold red]")
            for violation in violations_above_threshold[:10]:  # Show top 10
                console.print(f"  • {violation}")
        else:
            console.print(f"\n[yellow]⚠️ No violations meet the {threshold:.1%} confidence threshold[/yellow]")
    else:
        console.print("\n[green]✅ No violations identified[/green]")
    
    # Element satisfactions summary
    satisfied = sum(1 for s in analysis.element_satisfactions if s.status == 'satisfied')
    partial = sum(1 for s in analysis.element_satisfactions if s.status == 'partially_satisfied')
    insufficient = sum(1 for s in analysis.element_satisfactions if s.status == 'insufficient_evidence')
    not_satisfied = sum(1 for s in analysis.element_satisfactions if s.status == 'not_satisfied')
    
    table = Table(title="Element Satisfaction Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    
    total = len(analysis.element_satisfactions)
    table.add_row("Satisfied", str(satisfied), f"{satisfied/total:.1%}")
    table.add_row("Partially Satisfied", str(partial), f"{partial/total:.1%}")
    table.add_row("Insufficient Evidence", str(insufficient), f"{insufficient/total:.1%}")
    table.add_row("Not Satisfied", str(not_satisfied), f"{not_satisfied/total:.1%}")
    
    console.print(table)
    
    # Gap analysis
    if analysis.gap_analysis.missing_elements or analysis.gap_analysis.weak_elements:
        console.print(f"\n[bold yellow]📈 Gap Analysis[/bold yellow]")
        console.print(f"Priority Score: {analysis.gap_analysis.priority_score:.1%}")
        
        if analysis.gap_analysis.missing_elements:
            console.print(f"Missing Elements: {len(analysis.gap_analysis.missing_elements)}")
        
        if analysis.gap_analysis.weak_elements:
            console.print(f"Weak Elements: {len(analysis.gap_analysis.weak_elements)}")
        
        if analysis.gap_analysis.recommendations:
            console.print("\n📋 Recommendations:")
            for i, rec in enumerate(analysis.gap_analysis.recommendations[:5], 1):
                console.print(f"  {i}. {rec}")


def display_rome_statute_specific_results(rome_analysis):
    """Display Rome Statute specific analysis results."""
    console.print(f"\n[bold yellow]🏛️ ICC Jurisdictional Assessment[/bold yellow]")
    
    # Jurisdictional elements
    jurisdiction_table = Table(title="Jurisdictional Elements")
    jurisdiction_table.add_column("Element", style="cyan")
    jurisdiction_table.add_column("Status", style="white")
    
    for element, status in rome_analysis.jurisdictional_elements.items():
        status_display = "✅ Met" if status else "❌ Not Met"
        jurisdiction_table.add_row(element.replace('_', ' ').title(), status_display)
    
    console.print(jurisdiction_table)
    
    # Admissibility assessment
    console.print(f"\n📊 Admissibility Assessment:")
    for criterion, assessment in rome_analysis.admissibility_assessment.items():
        console.print(f"  • {criterion.replace('_', ' ').title()}: {assessment}")
    
    # Crime-specific findings
    if rome_analysis.genocide_findings:
        console.print(f"\n[bold red]🔺 Genocide Findings: {len(rome_analysis.genocide_findings)}[/bold red]")
    
    if rome_analysis.crimes_against_humanity_findings:
        console.print(f"[bold orange]🔶 Crimes Against Humanity: {len(rome_analysis.crimes_against_humanity_findings)}[/bold orange]")
    
    if rome_analysis.war_crimes_findings:
        console.print(f"[bold yellow]⚔️ War Crimes: {len(rome_analysis.war_crimes_findings)}[/bold yellow]")


def display_geneva_specific_results(geneva_analysis, protected_persons_focus):
    """Display Geneva Conventions specific results."""
    console.print(f"\n[bold red]⚔️ International Humanitarian Law Assessment[/bold red]")
    
    # Conflict classification
    console.print(f"📋 Conflict Classification:")
    for key, value in geneva_analysis.conflict_classification.items():
        console.print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Violations by severity
    if geneva_analysis.grave_breaches_found:
        console.print(f"\n[bold red]🚨 Grave Breaches Found: {len(geneva_analysis.grave_breaches_found)}[/bold red]")
    
    if geneva_analysis.serious_violations_found:
        console.print(f"[bold orange]⚠️ Serious Violations Found: {len(geneva_analysis.serious_violations_found)}[/bold orange]")
    
    # Protected persons analysis
    if geneva_analysis.protected_persons_analysis:
        console.print(f"\n👥 Protected Persons Analysis: {len(geneva_analysis.protected_persons_analysis)} findings")
    
    # Specialized analysis areas
    analysis_areas = [
        ("Medical Facilities", geneva_analysis.medical_facilities_analysis),
        ("Civilian Objects", geneva_analysis.civilian_objects_analysis),
        ("Proportionality", geneva_analysis.proportionality_analysis)
    ]
    
    for area_name, analysis_data in analysis_areas:
        if any(v != "requires_assessment" for v in analysis_data.values()):
            console.print(f"\n📊 {area_name} Analysis:")
            for key, value in analysis_data.items():
                if value != "requires_assessment":
                    console.print(f"  • {key.replace('_', ' ').title()}: {value}")


def display_human_rights_specific_results(hr_analysis, rights_focus):
    """Display human rights specific results."""
    console.print(f"\n[bold cyan]🏛️ Human Rights Assessment[/bold cyan]")
    
    # Universal vs Regional findings
    if hr_analysis.universal_rights_findings:
        console.print(f"🌍 Universal Rights Findings: {len(hr_analysis.universal_rights_findings)}")
    
    if hr_analysis.regional_rights_findings:
        console.print(f"🗺️ Regional Rights Findings: {len(hr_analysis.regional_rights_findings)}")
    
    # State responsibility
    console.print(f"\n⚖️ State Responsibility Assessment:")
    for key, value in hr_analysis.state_responsibility_assessment.items():
        console.print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Individual remedies
    if hr_analysis.individual_remedies:
        console.print(f"\n💊 Individual Remedies Available:")
        for remedy in hr_analysis.individual_remedies:
            console.print(f"  • {remedy}")
    
    # Systemic issues
    if hr_analysis.systemic_issues_identified:
        console.print(f"\n🔍 Systemic Issues Identified:")
        for issue in hr_analysis.systemic_issues_identified:
            console.print(f"  • {issue}")


def display_legal_assessment(assessment: LegalAssessment, threshold: float, include_cross_analysis: bool):
    """Display comprehensive legal assessment."""
    console.print(f"\n[bold magenta]📊 {assessment.title}[/bold magenta]")
    
    if assessment.description:
        console.print(f"📝 {assessment.description}")
    
    console.print(f"📅 Assessment Date: {assessment.assessment_date.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"⚖️ Frameworks Analyzed: {len(assessment.frameworks_analyzed)}")
    console.print(f"💪 Overall Strength: {assessment.strength_of_case.value.replace('_', ' ').title()}")
    
    # Overall assessment
    console.print(Panel(assessment.overall_assessment, title="📋 Overall Assessment"))
    
    # Framework-specific summaries
    console.print(f"\n[bold green]📚 Framework Analysis Summary[/bold green]")
    
    framework_table = Table(title="Framework Results")
    framework_table.add_column("Framework", style="cyan")
    framework_table.add_column("Violations", justify="right")
    framework_table.add_column("Confidence", justify="right")
    framework_table.add_column("Status")
    
    for analysis in assessment.framework_analyses:
        framework_name = analysis.framework.value.replace('_', ' ').title()
        violations_count = len(analysis.violations_identified)
        confidence = f"{analysis.overall_confidence:.1%}"
        
        if analysis.overall_confidence >= 0.8:
            status = "[green]Strong[/green]"
        elif analysis.overall_confidence >= 0.6:
            status = "[yellow]Moderate[/yellow]"
        else:
            status = "[red]Weak[/red]"
        
        framework_table.add_row(framework_name, str(violations_count), confidence, status)
    
    console.print(framework_table)
    
    # Cross-framework findings
    if include_cross_analysis and assessment.cross_framework_findings:
        console.print(f"\n[bold blue]🔗 Cross-Framework Analysis[/bold blue]")
        
        findings = assessment.cross_framework_findings
        
        if findings.get('common_violations'):
            console.print(f"📊 Common Violations: {', '.join(findings['common_violations'])}")
        
        if findings.get('overlapping_evidence'):
            console.print(f"🔄 Overlapping Evidence: {len(findings['overlapping_evidence'])} items")
    
    # Jurisdiction recommendations
    if assessment.jurisdiction_recommendations:
        console.print(f"\n[bold purple]🏛️ Jurisdiction Recommendations[/bold purple]")
        for i, rec in enumerate(assessment.jurisdiction_recommendations, 1):
            console.print(f"  {i}. {rec}")
    
    # Next steps
    if assessment.next_steps:
        console.print(f"\n[bold yellow]📋 Next Steps[/bold yellow]")
        for i, step in enumerate(assessment.next_steps[:8], 1):
            console.print(f"  {i}. {step}")
    
    # Generated citations
    if assessment.generated_citations:
        console.print(f"\n[bold green]📚 Legal Citations[/bold green]")
        for citation in assessment.generated_citations[:5]:
            console.print(f"  • {citation}")


def display_evidence_summary(evidence: List[Evidence]):
    """Display summary of evidence."""
    console.print(f"\n[bold blue]📊 Evidence Summary[/bold blue]")
    
    # Evidence by type
    type_counts = {}
    for ev in evidence:
        ev_type = ev.evidence_type.value if hasattr(ev.evidence_type, 'value') else str(ev.evidence_type)
        type_counts[ev_type] = type_counts.get(ev_type, 0) + 1
    
    type_table = Table(title="Evidence by Type")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", justify="right")
    type_table.add_column("Percentage", justify="right")
    
    total = len(evidence)
    for ev_type, count in sorted(type_counts.items()):
        type_table.add_row(ev_type.replace('_', ' ').title(), str(count), f"{count/total:.1%}")
    
    console.print(type_table)
    
    # Reliability distribution
    reliability_scores = [ev.reliability_score for ev in evidence]
    avg_reliability = sum(reliability_scores) / len(reliability_scores)
    console.print(f"📊 Average Reliability: {avg_reliability:.1%}")
    
    # Date range
    dates = [ev.incident_date for ev in evidence if ev.incident_date]
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        console.print(f"📅 Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")


# Template creation functions

def create_basic_evidence_template(count: int) -> List[Evidence]:
    """Create basic evidence template."""
    evidence_items = []
    
    for i in range(count):
        evidence = Evidence(
            title=f"Sample Evidence Item {i+1}",
            content=f"This is sample evidence content for item {i+1}. Replace with actual evidence content.",
            evidence_type=EvidenceType.DOCUMENT,
            source=f"Sample Source {i+1}",
            date_collected=datetime.now(),
            incident_date=datetime.now(),
            location="Sample Location",
            tags=["sample", f"item_{i+1}"],
            metadata={"sample_metadata": f"value_{i+1}"},
            reliability_score=0.8
        )
        evidence_items.append(evidence)
    
    return evidence_items


def create_rome_statute_evidence_template(count: int) -> List[Evidence]:
    """Create Rome Statute specific evidence template."""
    evidence_items = []
    
    templates = [
        {
            "title": "Witness Testimony - Mass Killing",
            "content": "Witness observed systematic killing of civilians belonging to specific ethnic group. Multiple victims identified. Perpetrators wore military uniforms.",
            "evidence_type": EvidenceType.TESTIMONY,
            "tags": ["genocide", "killing", "witness", "systematic"]
        },
        {
            "title": "Official Document - Deportation Order",
            "content": "Official order for forcible transfer of civilian population from territory. Document shows systematic policy implementation.",
            "evidence_type": EvidenceType.DOCUMENT,
            "tags": ["crimes_against_humanity", "deportation", "official_document"]
        },
        {
            "title": "Video Evidence - Attack on Hospital",
            "content": "Video showing deliberate attack on medical facility clearly marked with Red Cross symbol. No military presence visible.",
            "evidence_type": EvidenceType.VIDEO,
            "tags": ["war_crimes", "medical_facility", "attack", "protected_objects"]
        },
        {
            "title": "Expert Report - Forensic Analysis",
            "content": "Forensic analysis of mass grave site. Evidence of execution-style killings. DNA analysis confirms identity of victims from protected group.",
            "evidence_type": EvidenceType.EXPERT_REPORT,
            "tags": ["genocide", "forensic", "mass_grave", "expert_analysis"]
        },
        {
            "title": "Photograph - Detention Facility",
            "content": "Photographs showing inhumane conditions in detention facility. Overcrowding, lack of sanitation, signs of torture visible.",
            "evidence_type": EvidenceType.PHOTO,
            "tags": ["crimes_against_humanity", "torture", "detention", "conditions"]
        }
    ]
    
    for i in range(count):
        template = templates[i % len(templates)]
        evidence = Evidence(
            title=template["title"],
            content=template["content"],
            evidence_type=template["evidence_type"],
            source=f"ICC Investigation File {i+1}",
            date_collected=datetime.now(),
            incident_date=datetime.now(),
            location="Conflict Zone",
            tags=template["tags"],
            metadata={"case_related": True, "framework": "rome_statute"},
            reliability_score=0.85
        )
        evidence_items.append(evidence)
    
    return evidence_items


def create_geneva_evidence_template(count: int) -> List[Evidence]:
    """Create Geneva Conventions evidence template."""
    evidence_items = []
    
    templates = [
        {
            "title": "Medical Personnel Attack",
            "content": "Deliberate targeting of medical personnel wearing Red Cross insignia during medical evacuation mission.",
            "evidence_type": EvidenceType.TESTIMONY,
            "tags": ["medical_personnel", "red_cross", "protected_persons", "attack"]
        },
        {
            "title": "POW Interrogation Record",
            "content": "Record showing prisoner of war subjected to coercive interrogation methods beyond name, rank, serial number, date of birth.",
            "evidence_type": EvidenceType.DOCUMENT,
            "tags": ["prisoners_of_war", "interrogation", "treatment", "geneva_iii"]
        },
        {
            "title": "Civilian Displacement",
            "content": "Evidence of forcible transfer of civilian population from occupied territory without military necessity.",
            "evidence_type": EvidenceType.PHOTO,
            "tags": ["civilians", "forcible_transfer", "occupied_territory", "geneva_iv"]
        }
    ]
    
    for i in range(count):
        template = templates[i % len(templates)]
        evidence = Evidence(
            title=template["title"],
            content=template["content"],
            evidence_type=template["evidence_type"],
            source=f"ICRC Report {i+1}",
            date_collected=datetime.now(),
            incident_date=datetime.now(),
            location="Armed Conflict Zone",
            tags=template["tags"],
            metadata={"conflict_type": "international", "framework": "geneva_conventions"},
            reliability_score=0.9
        )
        evidence_items.append(evidence)
    
    return evidence_items


def create_human_rights_evidence_template(count: int) -> List[Evidence]:
    """Create human rights evidence template."""
    evidence_items = []
    
    templates = [
        {
            "title": "Arbitrary Detention Case",
            "content": "Individual detained without charges for extended period. No access to legal counsel or judicial review.",
            "evidence_type": EvidenceType.TESTIMONY,
            "tags": ["arbitrary_detention", "right_to_liberty", "due_process"]
        },
        {
            "title": "Torture Allegation",
            "content": "Medical evidence of torture during detention. Physical and psychological harm documented by independent medical examination.",
            "evidence_type": EvidenceType.EXPERT_REPORT,
            "tags": ["torture", "medical_evidence", "detention", "cruel_treatment"]
        },
        {
            "title": "Freedom of Expression Violation",
            "content": "Journalist arrested and charged for reporting on government activities. No legitimate justification for restriction.",
            "evidence_type": EvidenceType.DOCUMENT,
            "tags": ["freedom_of_expression", "journalist", "media_freedom", "restrictions"]
        }
    ]
    
    for i in range(count):
        template = templates[i % len(templates)]
        evidence = Evidence(
            title=template["title"],
            content=template["content"],
            evidence_type=template["evidence_type"],
            source=f"Human Rights Organization Report {i+1}",
            date_collected=datetime.now(),
            incident_date=datetime.now(),
            location="State Territory",
            tags=template["tags"],
            metadata={"human_rights_framework": True, "state_responsibility": True},
            reliability_score=0.8
        )
        evidence_items.append(evidence)
    
    return evidence_items


if __name__ == '__main__':
    cli()