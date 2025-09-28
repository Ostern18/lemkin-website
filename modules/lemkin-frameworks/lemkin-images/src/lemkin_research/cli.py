"""
Command-line interface for the Lemkin Legal Research Assistant.

This module provides a comprehensive CLI for all legal research operations
including case law search, precedent analysis, citation processing,
and research aggregation with memo generation.
"""

import asyncio
import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any
import webbrowser

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.tree import Tree
from loguru import logger

from .core import (
    LegalResearchAssistant, ResearchConfig, SearchQuery, 
    DatabaseType, CitationStyle, JurisdictionType, 
    create_research_assistant, create_default_config
)


# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="lemkin-research",
    help="Lemkin Legal Research Assistant - Comprehensive legal research and analysis tool",
    add_completion=False,
    rich_markup_mode="rich"
)

# Global research assistant instance
research_assistant: Optional[LegalResearchAssistant] = None


def get_research_assistant() -> LegalResearchAssistant:
    """Get or create the research assistant instance"""
    global research_assistant
    if research_assistant is None:
        config = create_default_config()
        research_assistant = create_research_assistant(config)
    return research_assistant


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()  # Remove default handler
    
    # Add console handler with appropriate level
    if verbose:
        logger.add(
            sys.stderr, 
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
    
    # Add file handler
    logger.add(
        "lemkin_research.log",
        level=log_level,
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def display_welcome():
    """Display welcome message"""
    welcome_text = """
# Lemkin Legal Research Assistant

Welcome to the comprehensive legal research and analysis tool.

## Available Commands:
- **search-cases**: Search legal databases for case law
- **find-precedents**: Find similar legal precedents 
- **parse-citations**: Parse and validate legal citations
- **aggregate-research**: Compile research from multiple sources
- **generate-memo**: Create legal research memorandum
- **config**: Manage configuration settings

Use `--help` with any command for detailed options.
"""
    console.print(Panel(Markdown(welcome_text), title="Lemkin Legal Research", border_style="blue"))


def format_case_results(results, limit: int = 10):
    """Format case law search results for display"""
    if not results.aggregated_cases:
        console.print("[yellow]No cases found.[/yellow]")
        return
    
    # Summary table
    summary_table = Table(title="Search Results Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Cases Found", str(results.total_results))
    summary_table.add_row("Unique Cases", str(len(results.aggregated_cases)))
    summary_table.add_row("Databases Searched", str(len(results.database_results)))
    summary_table.add_row("Search Duration", f"{results.search_duration:.2f}s")
    
    console.print(summary_table)
    console.print()
    
    # Cases table
    cases_table = Table(title=f"Top {min(limit, len(results.aggregated_cases))} Cases")
    cases_table.add_column("Case Name", style="bold cyan", max_width=40)
    cases_table.add_column("Citation", style="green")
    cases_table.add_column("Court", style="yellow")
    cases_table.add_column("Year", style="magenta")
    cases_table.add_column("Jurisdiction", style="blue")
    
    for case in results.aggregated_cases[:limit]:
        year = str(case.date_decided.year) if case.date_decided else "N/A"
        jurisdiction = case.jurisdiction.value if case.jurisdiction else "N/A"
        
        cases_table.add_row(
            case.case_name[:40] + "..." if len(case.case_name) > 40 else case.case_name,
            case.citation or "N/A",
            case.court or "N/A", 
            year,
            jurisdiction
        )
    
    console.print(cases_table)
    
    # Database breakdown
    if results.database_results:
        db_table = Table(title="Database Results")
        db_table.add_column("Database", style="cyan")
        db_table.add_column("Results", style="green")
        db_table.add_column("Search Time", style="yellow")
        
        for db_result in results.database_results:
            db_table.add_row(
                db_result.database.value,
                str(db_result.results_count),
                f"{db_result.search_time:.2f}s"
            )
        
        console.print(db_table)


def format_precedent_results(precedents: List, limit: int = 10):
    """Format precedent analysis results for display"""
    if not precedents:
        console.print("[yellow]No precedents found.[/yellow]")
        return
    
    precedents_table = Table(title=f"Top {min(limit, len(precedents))} Precedents")
    precedents_table.add_column("Case Name", style="bold cyan", max_width=35)
    precedents_table.add_column("Citation", style="green") 
    precedents_table.add_column("Similarity", style="yellow")
    precedents_table.add_column("Precedential Value", style="magenta")
    precedents_table.add_column("Binding Strength", style="blue")
    
    for match in precedents[:limit]:
        precedent = match.matched_precedent if hasattr(match, 'matched_precedent') else match
        case = precedent.case_opinion
        
        precedents_table.add_row(
            case.case_name[:35] + "..." if len(case.case_name) > 35 else case.case_name,
            case.citation or "N/A",
            f"{match.match_confidence:.3f}" if hasattr(match, 'match_confidence') else f"{precedent.precedential_value:.3f}",
            f"{precedent.precedential_value:.3f}",
            precedent.binding_strength
        )
    
    console.print(precedents_table)
    
    # Show analysis details for top match
    if precedents:
        top_match = precedents[0]
        if hasattr(top_match, 'supporting_evidence') and top_match.supporting_evidence:
            console.print(Panel(
                "\n".join([f"• {evidence}" for evidence in top_match.supporting_evidence[:5]]),
                title="Top Match - Supporting Evidence",
                border_style="green"
            ))


def format_citation_results(citation_match):
    """Format citation parsing results for display"""
    if not citation_match.citations_found:
        console.print("[yellow]No citations found.[/yellow]")
        return
    
    # Summary
    summary_table = Table(title="Citation Analysis Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Citations Found", str(len(citation_match.citations_found)))
    summary_table.add_row("Valid Citations", str(sum(1 for c in citation_match.citations_found if c.is_valid)))
    summary_table.add_row("Parsing Confidence", f"{citation_match.parsing_confidence:.2f}")
    summary_table.add_row("Validation Passed", "Yes" if citation_match.validation_passed else "No")
    
    console.print(summary_table)
    console.print()
    
    # Citations table
    citations_table = Table(title="Parsed Citations")
    citations_table.add_column("Original Citation", style="cyan", max_width=50)
    citations_table.add_column("Type", style="green")
    citations_table.add_column("Valid", style="yellow")
    citations_table.add_column("Standardized", style="blue", max_width=50)
    
    for citation in citation_match.citations_found:
        citations_table.add_row(
            citation.raw_citation[:50] + "..." if len(citation.raw_citation) > 50 else citation.raw_citation,
            citation.citation_type.value,
            "✓" if citation.is_valid else "✗",
            citation.parsed_citation[:50] + "..." if citation.parsed_citation and len(citation.parsed_citation) > 50 else citation.parsed_citation or "N/A"
        )
    
    console.print(citations_table)
    
    # Show validation errors if any
    errors = []
    for citation in citation_match.citations_found:
        if citation.validation_errors:
            errors.extend([f"{citation.raw_citation[:30]}...: {error}" for error in citation.validation_errors])
    
    if errors:
        console.print(Panel(
            "\n".join(errors[:10]),
            title="Validation Errors",
            border_style="red"
        ))


def format_research_summary(summary):
    """Format research summary for display"""
    # Key findings
    if summary.key_findings:
        findings_text = "\n".join([f"• {finding}" for finding in summary.key_findings[:10]])
        console.print(Panel(findings_text, title="Key Findings", border_style="green"))
        console.print()
    
    # Statistics table
    stats_table = Table(title="Research Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Primary Precedents", str(len(summary.primary_precedents)))
    stats_table.add_row("Supporting Cases", str(len(summary.supporting_cases)))
    stats_table.add_row("Contradictory Authority", str(len(summary.contradictory_authority)))
    stats_table.add_row("Citations Analyzed", str(summary.citations_analyzed))
    stats_table.add_row("Databases Searched", str(len(summary.databases_searched)))
    stats_table.add_row("Confidence Level", f"{summary.confidence_level:.2f}")
    
    console.print(stats_table)
    console.print()
    
    # Recommendations
    if summary.recommendations:
        recommendations_text = "\n".join([f"• {rec}" for rec in summary.recommendations[:10]])
        console.print(Panel(recommendations_text, title="Recommendations", border_style="blue"))
        console.print()
    
    # Research gaps
    if summary.research_gaps:
        gaps_text = "\n".join([f"• {gap}" for gap in summary.research_gaps[:5]])
        console.print(Panel(gaps_text, title="Research Gaps", border_style="yellow"))


# CLI Commands

@app.command()
def search_cases(
    query: str = typer.Argument(..., help="Search query for case law"),
    databases: Optional[List[str]] = typer.Option(None, "--database", "-d", help="Specific databases to search"),
    max_results: int = typer.Option(50, "--max-results", "-n", help="Maximum number of results"),
    jurisdiction: Optional[str] = typer.Option(None, "--jurisdiction", "-j", help="Filter by jurisdiction (federal/state/local)"),
    court: Optional[str] = typer.Option(None, "--court", "-c", help="Filter by specific court"),
    year_start: Optional[int] = typer.Option(None, "--year-start", help="Start year for date range"),
    year_end: Optional[int] = typer.Option(None, "--year-end", help="End year for date range"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("table", "--format", "-f", help="Output format (table/json/detailed)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Search legal databases for case law"""
    setup_logging(verbose)
    
    console.print(f"[bold blue]Searching case law for:[/bold blue] {query}")
    
    # Parse databases
    selected_databases = None
    if databases:
        selected_databases = []
        for db in databases:
            try:
                selected_databases.append(DatabaseType(db.lower()))
            except ValueError:
                console.print(f"[red]Warning: Unknown database '{db}'. Skipping.[/red]")
    
    # Build search query
    search_query = SearchQuery(
        query_text=query,
        max_results=max_results
    )
    
    if jurisdiction:
        try:
            search_query.jurisdiction = JurisdictionType(jurisdiction.lower())
        except ValueError:
            console.print(f"[red]Warning: Unknown jurisdiction '{jurisdiction}'[/red]")
    
    if court:
        search_query.court = court
    
    if year_start or year_end:
        date_range = {}
        if year_start:
            date_range['start'] = date(year_start, 1, 1)
        if year_end:
            date_range['end'] = date(year_end, 12, 31)
        search_query.date_range = date_range
    
    # Perform search
    assistant = get_research_assistant()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Searching databases...", total=None)
        
        try:
            # Run async search
            results = asyncio.run(assistant.case_searcher.search(search_query, selected_databases))
            progress.update(task, completed=True)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]Search failed: {e}[/red]")
            raise typer.Exit(1)
    
    # Display results
    if format_type == "table":
        format_case_results(results, max_results)
    elif format_type == "json":
        console.print_json(results.model_dump_json(indent=2))
    elif format_type == "detailed":
        format_case_results(results, max_results)
        console.print()
        
        # Show detailed view of top cases
        for i, case in enumerate(results.aggregated_cases[:5], 1):
            console.print(f"[bold cyan]Case {i}: {case.case_name}[/bold cyan]")
            
            if case.summary:
                console.print(f"[yellow]Summary:[/yellow] {case.summary[:200]}...")
                
            if case.holdings:
                console.print(f"[green]Holdings:[/green]")
                for holding in case.holdings[:3]:
                    console.print(f"  • {holding}")
            
            console.print()
    
    # Save output if requested
    if output:
        with open(output, 'w') as f:
            if format_type == "json":
                f.write(results.model_dump_json(indent=2))
            else:
                # Save as structured text
                f.write(f"Case Law Search Results\n")
                f.write(f"Query: {query}\n")
                f.write(f"Total Results: {results.total_results}\n")
                f.write(f"Search Duration: {results.search_duration:.2f}s\n\n")
                
                for case in results.aggregated_cases:
                    f.write(f"Case: {case.case_name}\n")
                    f.write(f"Citation: {case.citation or 'N/A'}\n")
                    f.write(f"Court: {case.court or 'N/A'}\n")
                    if case.date_decided:
                        f.write(f"Date: {case.date_decided}\n")
                    f.write(f"URL: {case.url or 'N/A'}\n")
                    f.write("-" * 80 + "\n")
        
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def find_precedents(
    reference_case: str = typer.Argument(..., help="Reference case name or citation"),
    max_results: int = typer.Option(10, "--max-results", "-n", help="Maximum number of precedents"),
    similarity_threshold: float = typer.Option(0.7, "--threshold", "-t", help="Similarity threshold (0.0-1.0)"),
    jurisdiction: Optional[str] = typer.Option(None, "--jurisdiction", "-j", help="Filter by jurisdiction"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("table", "--format", "-f", help="Output format (table/json/detailed)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Find similar legal precedents for a reference case"""
    setup_logging(verbose)
    
    console.print(f"[bold blue]Finding precedents for:[/bold blue] {reference_case}")
    
    assistant = get_research_assistant()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), 
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing precedents...", total=None)
        
        try:
            # First search for cases to build index
            search_query = SearchQuery(
                query_text=reference_case,
                max_results=200  # Get more cases for better precedent analysis
            )
            
            progress.update(task, description="Searching case databases...")
            case_results = asyncio.run(assistant.case_searcher.search(search_query))
            
            if not case_results.aggregated_cases:
                progress.stop()
                console.print("[yellow]No cases found to analyze for precedents.[/yellow]")
                return
            
            # Add cases to precedent analyzer
            progress.update(task, description="Building precedent index...")
            assistant.precedent_analyzer.add_cases_to_index(case_results.aggregated_cases)
            
            # Find precedents
            progress.update(task, description="Finding similar precedents...")
            precedents = assistant.precedent_analyzer.find_similar_precedents(
                reference_case,
                max_results,
                similarity_threshold
            )
            
            progress.update(task, completed=True)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]Precedent analysis failed: {e}[/red]")
            raise typer.Exit(1)
    
    # Display results
    if format_type == "table":
        format_precedent_results(precedents, max_results)
    elif format_type == "json":
        # Convert to JSON-serializable format
        precedents_data = []
        for match in precedents:
            precedent_data = {
                'case_name': match.matched_precedent.case_opinion.case_name,
                'citation': match.matched_precedent.case_opinion.citation,
                'match_confidence': match.match_confidence,
                'precedential_value': match.matched_precedent.precedential_value,
                'binding_strength': match.matched_precedent.binding_strength,
                'supporting_evidence': match.supporting_evidence,
                'distinguishing_factors': match.distinguishing_factors,
                'recommendation': match.recommendation
            }
            precedents_data.append(precedent_data)
        
        console.print_json(json.dumps(precedents_data, indent=2))
    elif format_type == "detailed":
        format_precedent_results(precedents, max_results)
        
        # Show detailed analysis for top precedents
        for i, match in enumerate(precedents[:3], 1):
            precedent = match.matched_precedent
            case = precedent.case_opinion
            
            console.print(f"\n[bold cyan]Precedent {i}: {case.case_name}[/bold cyan]")
            console.print(f"[green]Match Confidence:[/green] {match.match_confidence:.3f}")
            console.print(f"[green]Precedential Value:[/green] {precedent.precedential_value:.3f}")
            
            if match.supporting_evidence:
                console.print(f"[yellow]Supporting Evidence:[/yellow]")
                for evidence in match.supporting_evidence[:3]:
                    console.print(f"  • {evidence}")
            
            if match.distinguishing_factors:
                console.print(f"[red]Distinguishing Factors:[/red]")
                for factor in match.distinguishing_factors[:3]:
                    console.print(f"  • {factor}")
    
    # Save output if requested
    if output:
        precedents_data = []
        for match in precedents:
            precedent_data = {
                'case_name': match.matched_precedent.case_opinion.case_name,
                'citation': match.matched_precedent.case_opinion.citation,
                'court': match.matched_precedent.case_opinion.court,
                'date_decided': match.matched_precedent.case_opinion.date_decided.isoformat() if match.matched_precedent.case_opinion.date_decided else None,
                'match_confidence': match.match_confidence,
                'precedential_value': match.matched_precedent.precedential_value,
                'binding_strength': match.matched_precedent.binding_strength,
                'supporting_evidence': match.supporting_evidence,
                'distinguishing_factors': match.distinguishing_factors,
                'recommendation': match.recommendation
            }
            precedents_data.append(precedent_data)
        
        with open(output, 'w') as f:
            json.dump(precedents_data, f, indent=2)
        
        console.print(f"[green]Precedent analysis saved to {output}[/green]")


@app.command()
def parse_citations(
    text: str = typer.Argument(None, help="Text containing legal citations"),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Input file path"),
    citation_style: str = typer.Option("bluebook", "--style", "-s", help="Citation style (bluebook/alwd/apa)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("table", "--format", "-f", help="Output format (table/json/detailed)"),
    validate_only: bool = typer.Option(False, "--validate-only", help="Only validate, don't reformat"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Parse and validate legal citations from text"""
    setup_logging(verbose)
    
    # Get input text
    if input_file:
        if not input_file.exists():
            console.print(f"[red]Input file not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        console.print(f"[blue]Processing citations from file:[/blue] {input_file}")
    elif text:
        console.print(f"[blue]Processing citations from text...[/blue]")
    else:
        console.print("[red]Either provide text or use --input to specify a file[/red]")
        raise typer.Exit(1)
    
    # Parse citation style
    try:
        style = CitationStyle(citation_style.lower())
    except ValueError:
        console.print(f"[red]Unknown citation style: {citation_style}[/red]")
        console.print(f"[yellow]Available styles: {', '.join([s.value for s in CitationStyle])}[/yellow]")
        raise typer.Exit(1)
    
    assistant = get_research_assistant()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Parsing citations...", total=None)
        
        try:
            citation_match = assistant.citation_processor.parse(text, style)
            progress.update(task, completed=True)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]Citation parsing failed: {e}[/red]")
            raise typer.Exit(1)
    
    # Display results
    if format_type == "table":
        format_citation_results(citation_match)
    elif format_type == "json":
        # Convert to JSON-serializable format
        citations_data = {
            'original_text': citation_match.original_text,
            'parsing_confidence': citation_match.parsing_confidence,
            'validation_passed': citation_match.validation_passed,
            'citations': []
        }
        
        for citation in citation_match.citations_found:
            citation_data = {
                'raw_citation': citation.raw_citation,
                'parsed_citation': citation.parsed_citation,
                'citation_type': citation.citation_type.value,
                'is_valid': citation.is_valid,
                'validation_errors': citation.validation_errors,
                'case_name': citation.case_name,
                'reporter': citation.reporter,
                'volume': citation.volume,
                'page': citation.page,
                'year': citation.year,
                'court': citation.court,
                'jurisdiction': citation.jurisdiction.value if citation.jurisdiction else None
            }
            citations_data['citations'].append(citation_data)
        
        console.print_json(json.dumps(citations_data, indent=2))
    elif format_type == "detailed":
        format_citation_results(citation_match)
        
        # Show detailed analysis
        console.print(f"\n[bold cyan]Detailed Citation Analysis[/bold cyan]")
        for i, citation in enumerate(citation_match.citations_found, 1):
            console.print(f"\n[yellow]Citation {i}:[/yellow]")
            console.print(f"[green]Raw:[/green] {citation.raw_citation}")
            
            if citation.parsed_citation:
                console.print(f"[green]Formatted:[/green] {citation.parsed_citation}")
            
            console.print(f"[green]Type:[/green] {citation.citation_type.value}")
            console.print(f"[green]Valid:[/green] {'Yes' if citation.is_valid else 'No'}")
            
            if citation.validation_errors:
                console.print(f"[red]Errors:[/red] {'; '.join(citation.validation_errors)}")
            
            # Show extracted fields
            fields = []
            if citation.case_name:
                fields.append(f"Case: {citation.case_name}")
            if citation.volume:
                fields.append(f"Vol: {citation.volume}")
            if citation.reporter:
                fields.append(f"Reporter: {citation.reporter}")
            if citation.page:
                fields.append(f"Page: {citation.page}")
            if citation.year:
                fields.append(f"Year: {citation.year}")
            if citation.court:
                fields.append(f"Court: {citation.court}")
            
            if fields:
                console.print(f"[blue]Fields:[/blue] {'; '.join(fields)}")
    
    # Save output if requested
    if output:
        citations_data = {
            'parsing_confidence': citation_match.parsing_confidence,
            'validation_passed': citation_match.validation_passed,
            'citations_count': len(citation_match.citations_found),
            'citations': []
        }
        
        for citation in citation_match.citations_found:
            citation_data = {
                'raw_citation': citation.raw_citation,
                'parsed_citation': citation.parsed_citation,
                'citation_type': citation.citation_type.value,
                'is_valid': citation.is_valid,
                'validation_errors': citation.validation_errors
            }
            citations_data['citations'].append(citation_data)
        
        with open(output, 'w') as f:
            json.dump(citations_data, f, indent=2)
        
        console.print(f"[green]Citation analysis saved to {output}[/green]")


@app.command()
def aggregate_research(
    research_question: str = typer.Argument(..., help="Primary research question"),
    max_cases: int = typer.Option(50, "--max-cases", "-n", help="Maximum number of cases to analyze"),
    include_precedents: bool = typer.Option(True, "--precedents/--no-precedents", help="Include precedent analysis"),
    include_citations: bool = typer.Option(True, "--citations/--no-citations", help="Include citation analysis"),
    sources: Optional[List[str]] = typer.Option(None, "--source", "-s", help="Specific sources to include"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("summary", "--format", "-f", help="Output format (summary/json/detailed)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Compile research from multiple sources and synthesize findings"""
    setup_logging(verbose)
    
    console.print(f"[bold blue]Aggregating research for:[/bold blue] {research_question}")
    
    assistant = get_research_assistant()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Starting research aggregation...", total=100)
        
        try:
            # Set up components
            assistant.research_aggregator.set_components(
                assistant.case_searcher,
                assistant.precedent_analyzer, 
                assistant.citation_processor
            )
            
            progress.update(task, advance=10, description="Searching case databases...")
            
            # Perform aggregated research
            summary = asyncio.run(
                assistant.research_aggregator.aggregate_research(
                    research_question,
                    sources,
                    max_cases,
                    include_precedents,
                    include_citations
                )
            )
            
            progress.update(task, completed=100)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]Research aggregation failed: {e}[/red]")
            raise typer.Exit(1)
    
    # Display results
    if format_type == "summary":
        format_research_summary(summary)
    elif format_type == "json":
        console.print_json(summary.model_dump_json(indent=2))
    elif format_type == "detailed":
        format_research_summary(summary)
        
        # Additional detailed sections
        if summary.jurisdictional_analysis:
            console.print("\n[bold cyan]Jurisdictional Analysis[/bold cyan]")
            for jurisdiction, analysis in summary.jurisdictional_analysis.items():
                console.print(f"[yellow]{jurisdiction}:[/yellow] {analysis}")
        
        if summary.temporal_analysis:
            console.print("\n[bold cyan]Temporal Analysis[/bold cyan]")
            for aspect, analysis in summary.temporal_analysis.items():
                console.print(f"[yellow]{aspect}:[/yellow] {analysis}")
    
    # Save output if requested
    if output:
        with open(output, 'w') as f:
            f.write(summary.model_dump_json(indent=2))
        
        console.print(f"[green]Research summary saved to {output}[/green]")


@app.command()
def generate_memo(
    research_file: Path = typer.Argument(..., help="Research summary file (JSON)"),
    client: Optional[str] = typer.Option(None, "--client", "-c", help="Client name"),
    attorney: Optional[str] = typer.Option(None, "--attorney", "-a", help="Attorney name"),
    template: str = typer.Option("standard", "--template", "-t", help="Memo template (standard/brief/comprehensive)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("markdown", "--format", "-f", help="Output format (markdown/html/docx)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Generate a legal research memorandum from research summary"""
    setup_logging(verbose)
    
    if not research_file.exists():
        console.print(f"[red]Research file not found: {research_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold blue]Generating legal memo from:[/bold blue] {research_file}")
    
    # Load research summary
    try:
        from .core import ResearchSummary
        with open(research_file, 'r') as f:
            data = json.load(f)
        
        summary = ResearchSummary.model_validate(data)
        
    except Exception as e:
        console.print(f"[red]Failed to load research summary: {e}[/red]")
        raise typer.Exit(1)
    
    assistant = get_research_assistant()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating legal memo...", total=None)
        
        try:
            memo = assistant.research_aggregator.generate_memo(
                summary,
                template,
                client,
                attorney
            )
            
            progress.update(task, completed=True)
            
        except Exception as e:
            progress.stop()
            console.print(f"[red]Memo generation failed: {e}[/red]")
            raise typer.Exit(1)
    
    # Display memo preview
    console.print(Panel(
        f"[bold]{memo.title}[/bold]\n\n"
        f"Client: {memo.client or 'N/A'}\n"
        f"Attorney: {memo.attorney or 'N/A'}\n"
        f"Date: {memo.date_prepared}\n"
        f"Word Count: {memo.word_count}",
        title="Memo Summary",
        border_style="green"
    ))
    
    # Show brief answer
    if memo.brief_answer:
        console.print(Panel(
            memo.brief_answer,
            title="Brief Answer",
            border_style="blue"
        ))
    
    # Determine output file
    if not output:
        output = Path(f"memo_{memo.id}.md")
    
    # Generate memo content based on format
    if format_type == "markdown":
        memo_content = f"""# {memo.title}

**To:** {memo.client or '[CLIENT]'}
**From:** {memo.attorney or '[ATTORNEY]'}
**Date:** {memo.date_prepared}
**Re:** {memo.research_question}

---

## Brief Answer

{memo.brief_answer}

## Executive Summary

{memo.executive_summary}

## Factual Background

{memo.factual_background}

## Legal Analysis

{memo.legal_analysis}

## Conclusion

{memo.conclusion}

## Recommendations

{chr(10).join(['- ' + rec for rec in memo.recommendations])}

---

### Supporting Authority
{chr(10).join(['- ' + auth for auth in memo.supporting_authority])}

### Research Sources
{chr(10).join(['- ' + source for source in memo.research_sources])}

---
*Generated by Lemkin Legal Research Assistant*
*Word Count: {memo.word_count}*
"""
    elif format_type == "html":
        memo_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{memo.title}</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; border-bottom: 2px solid #000; margin-bottom: 20px; }}
        .section {{ margin-bottom: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .memo-info {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{memo.title}</h1>
    </div>
    
    <div class="memo-info">
        <p><strong>To:</strong> {memo.client or '[CLIENT]'}</p>
        <p><strong>From:</strong> {memo.attorney or '[ATTORNEY]'}</p>
        <p><strong>Date:</strong> {memo.date_prepared}</p>
        <p><strong>Re:</strong> {memo.research_question}</p>
    </div>
    
    <div class="section">
        <h2>Brief Answer</h2>
        <p>{memo.brief_answer}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>{memo.executive_summary}</p>
    </div>
    
    <div class="section">
        <h2>Legal Analysis</h2>
        <div>{memo.legal_analysis.replace(chr(10), '<br>')}</div>
    </div>
    
    <div class="section">
        <h2>Conclusion</h2>
        <p>{memo.conclusion}</p>
    </div>
    
    <hr>
    <p><em>Generated by Lemkin Legal Research Assistant - Word Count: {memo.word_count}</em></p>
</body>
</html>"""
    else:
        # Default to plain text
        memo_content = f"""{memo.title}

To: {memo.client or '[CLIENT]'}
From: {memo.attorney or '[ATTORNEY]'}
Date: {memo.date_prepared}
Re: {memo.research_question}

{'='*80}

BRIEF ANSWER

{memo.brief_answer}

EXECUTIVE SUMMARY

{memo.executive_summary}

LEGAL ANALYSIS

{memo.legal_analysis}

CONCLUSION

{memo.conclusion}

RECOMMENDATIONS

{chr(10).join(['- ' + rec for rec in memo.recommendations])}

{'='*80}
Generated by Lemkin Legal Research Assistant
Word Count: {memo.word_count}
"""
    
    # Save memo
    with open(output, 'w', encoding='utf-8') as f:
        f.write(memo_content)
    
    console.print(f"[green]Legal memo saved to {output}[/green]")
    
    # Offer to open the file
    if Confirm.ask(f"Open {output} in default application?"):
        webbrowser.open(f"file://{output.absolute()}")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    set_key: Optional[str] = typer.Option(None, "--set", help="Set configuration key"),
    value: Optional[str] = typer.Option(None, "--value", help="Configuration value"),
    database: Optional[str] = typer.Option(None, "--database", help="Configure database settings"),
    list_databases: bool = typer.Option(False, "--list-databases", help="List available databases"),
    test_connection: Optional[str] = typer.Option(None, "--test", help="Test database connection"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Manage configuration settings"""
    setup_logging(verbose)
    
    assistant = get_research_assistant()
    
    if list_databases:
        console.print("[bold cyan]Available Legal Databases:[/bold cyan]")
        
        databases_table = Table()
        databases_table.add_column("Database", style="cyan")
        databases_table.add_column("Type", style="green")
        databases_table.add_column("Status", style="yellow")
        databases_table.add_column("Description", style="blue")
        
        for db_type in DatabaseType:
            status = "✓ Enabled" if db_type in assistant.config.enabled_databases else "✗ Disabled"
            description = {
                DatabaseType.GOOGLE_SCHOLAR: "Free access to legal cases via Google Scholar",
                DatabaseType.COURTLISTENER: "Free legal database with comprehensive case law",
                DatabaseType.JUSTIA: "Free legal information and case law database",
                DatabaseType.WESTLAW: "Premium legal database (requires subscription)",
                DatabaseType.LEXIS: "Premium legal database (requires subscription)",
                DatabaseType.BLOOMBERG_LAW: "Premium legal database (requires subscription)",
                DatabaseType.FASTCASE: "Legal database (requires subscription)",
                DatabaseType.CASELAW_ACCESS: "Free case law access project",
                DatabaseType.FREE_LAW: "Free legal information databases",
                DatabaseType.CUSTOM: "Custom database configuration"
            }.get(db_type, "Legal database")
            
            databases_table.add_row(
                db_type.value,
                "Free" if db_type in [DatabaseType.GOOGLE_SCHOLAR, DatabaseType.COURTLISTENER, DatabaseType.JUSTIA] else "Premium",
                status,
                description
            )
        
        console.print(databases_table)
        return
    
    if test_connection:
        try:
            db_type = DatabaseType(test_connection.lower())
            
            console.print(f"[yellow]Testing connection to {db_type.value}...[/yellow]")
            
            result = asyncio.run(
                assistant.case_searcher.test_database_connection(db_type)
            )
            
            if result['available']:
                console.print(f"[green]✓ {db_type.value} connection successful[/green]")
                console.print(f"Response time: {result.get('response_time', 0):.2f}s")
                if 'test_results' in result:
                    console.print(f"Test results: {result['test_results']}")
            else:
                console.print(f"[red]✗ {db_type.value} connection failed[/red]")
                console.print(f"Error: {result.get('error', 'Unknown error')}")
            
        except ValueError:
            console.print(f"[red]Unknown database: {test_connection}[/red]")
            console.print("[yellow]Use --list-databases to see available options[/yellow]")
        except Exception as e:
            console.print(f"[red]Connection test failed: {e}[/red]")
        
        return
    
    if show:
        console.print("[bold cyan]Current Configuration:[/bold cyan]")
        
        config_table = Table()
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Enabled Databases", ", ".join([db.value for db in assistant.config.enabled_databases]))
        config_table.add_row("Default Max Results", str(assistant.config.default_max_results))
        config_table.add_row("Search Timeout", f"{assistant.config.search_timeout}s")
        config_table.add_row("Similarity Threshold", str(assistant.config.similarity_threshold))
        config_table.add_row("Precedent Threshold", str(assistant.config.precedent_threshold))
        config_table.add_row("Default Citation Style", assistant.config.default_citation_style.value)
        config_table.add_row("Embedding Model", assistant.config.embedding_model)
        config_table.add_row("Log Level", assistant.config.log_level)
        
        console.print(config_table)
        return
    
    if reset:
        if Confirm.ask("Reset all configuration to defaults?"):
            assistant.config = create_default_config()
            console.print("[green]Configuration reset to defaults[/green]")
        return
    
    if set_key and value:
        # Simple configuration setting (would be expanded in full implementation)
        console.print(f"[yellow]Setting {set_key} = {value}[/yellow]")
        console.print("[blue]Note: Configuration changes require restart to take effect[/blue]")
        return
    
    # Show help if no specific action
    console.print("[yellow]Use --show to view current configuration[/yellow]")
    console.print("[yellow]Use --list-databases to see available databases[/yellow]") 
    console.print("[yellow]Use --test DATABASE to test database connections[/yellow]")


@app.command()
def version():
    """Show version information"""
    from . import get_version, get_package_info
    
    info = get_package_info()
    
    version_table = Table(title="Lemkin Legal Research Assistant")
    version_table.add_column("Property", style="cyan")
    version_table.add_column("Value", style="green")
    
    version_table.add_row("Version", get_version())
    version_table.add_row("Description", info['description'])
    version_table.add_row("Author", info['author'])
    version_table.add_row("License", info['license'])
    version_table.add_row("URL", info['url'])
    
    console.print(version_table)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Lemkin Legal Research Assistant
    
    Comprehensive legal research and analysis tool for legal professionals.
    
    Features:
    • Multi-database case law search
    • Semantic precedent analysis  
    • Legal citation parsing and validation
    • Research synthesis and memo generation
    """
    if ctx.invoked_subcommand is None:
        display_welcome()


if __name__ == "__main__":
    app()