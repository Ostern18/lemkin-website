"""
Lemkin Reports CLI

Command-line interface for automated report generation, compliance validation,
and multi-format output for legal investigation workflows.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from loguru import logger

from .core import (
    ReportGenerator,
    ReportType,
    ReportFormat,
    ComplianceStandard,
    ReportData,
    ReportSection,
    ComplianceValidator,
    CitationManager
)

app = typer.Typer(help="Lemkin Automated Reporting System")
console = Console()


@app.command()
def create(
    case_id: str = typer.Argument(..., help="Case ID for the report"),
    report_type: ReportType = typer.Argument(..., help="Type of report to create"),
    title: str = typer.Option(..., help="Report title"),
    author: str = typer.Option(..., help="Report author"),
    template_id: Optional[str] = typer.Option(None, help="Template ID to use"),
    output_path: Optional[Path] = typer.Option(None, help="Output directory path")
):
    """Create a new investigation report."""
    console.print(f"[bold blue]Creating {report_type} report for case {case_id}[/bold blue]")

    try:
        generator = ReportGenerator()

        # Create report structure
        report = generator.create_report(
            report_type=report_type,
            case_id=case_id,
            title=title,
            author=author,
            template_id=template_id
        )

        console.print(f"[bold green]✓[/bold green] Report structure created!")
        console.print(f"Report ID: {report.report_id}")
        console.print(f"Sections: {len(report.sections)}")

        # Show report structure
        section_table = Table(title="Report Sections", show_header=True)
        section_table.add_column("Section ID", style="cyan")
        section_table.add_column("Title", style="magenta")
        section_table.add_column("Type", style="green")

        for section in report.sections:
            section_table.add_row(
                section.section_id,
                section.title,
                section.section_type
            )

        console.print(section_table)

        # Interactive content addition
        if Confirm.ask("Would you like to add content to sections interactively?"):
            add_content_interactive(report)

        # Save report structure
        report_file = Path(output_path or ".") / f"{report.report_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)

        console.print(f"[bold green]✓[/bold green] Report saved to: {report_file}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to create report: {e}")
        raise typer.Exit(1)


@app.command()
def generate(
    report_file: Path = typer.Argument(..., help="Report JSON file"),
    output_format: ReportFormat = typer.Option(ReportFormat.PDF, help="Output format"),
    data_file: Optional[Path] = typer.Option(None, help="Case data JSON file"),
    output_path: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Generate report in specified format from report definition."""
    console.print(f"[bold blue]Generating {output_format} report from {report_file}[/bold blue]")

    try:
        # Load report definition
        with open(report_file) as f:
            report_data = json.load(f)

        from .core import Report
        report = Report(**report_data)

        # Load case data if provided
        case_data = None
        if data_file and data_file.exists():
            with open(data_file) as f:
                data_dict = json.load(f)
            case_data = ReportData(**data_dict)
            console.print(f"[dim]Loaded case data with {len(case_data.entities)} entities and {len(case_data.timeline)} events[/dim]")

        generator = ReportGenerator()

        # Populate report with data
        if case_data:
            report = generator.populate_report_data(report, case_data)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Generating {output_format} report...", total=None)

            # Generate report
            output_file = generator.save_report(
                report=report,
                report_format=output_format,
                data=case_data,
                output_path=output_path
            )

            progress.update(task, completed=100)

        console.print(f"[bold green]✓[/bold green] Report generated successfully!")
        console.print(f"Output file: {output_file}")

        # Show report summary
        summary_content = f"""
[bold]Report Summary:[/bold]
• Case ID: {report.case_id}
• Report Type: {report.report_type}
• Author: {report.author}
• Sections: {len(report.sections)}
• Format: {output_format}
• File Size: {output_file.stat().st_size / 1024:.1f} KB
        """
        console.print(Panel(summary_content, title="Generation Complete", border_style="green"))

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Report generation failed: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    report_file: Path = typer.Argument(..., help="Report JSON file to validate"),
    compliance_standard: ComplianceStandard = typer.Option(
        ComplianceStandard.FEDERAL_RULES,
        help="Compliance standard to validate against"
    )
):
    """Validate report against compliance standards."""
    console.print(f"[bold blue]Validating report against {compliance_standard} standards[/bold blue]")

    try:
        # Load report
        with open(report_file) as f:
            report_data = json.load(f)

        from .core import Report
        report = Report(**report_data)

        validator = ComplianceValidator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Validating compliance...", total=None)

            validation_result = validator.validate_report(report, compliance_standard)

            progress.update(task, completed=100)

        # Display results
        is_valid = validation_result["valid"]
        compliance_score = validation_result.get("compliance_score", 0)

        status_color = "green" if is_valid else "red"
        status_text = "VALID" if is_valid else "INVALID"

        console.print(f"[bold {status_color}]{status_text}[/bold {status_color}] - Compliance Score: {compliance_score}/100")

        # Show warnings
        if validation_result["warnings"]:
            warning_table = Table(title="Warnings", show_header=True)
            warning_table.add_column("Warning", style="yellow")

            for warning in validation_result["warnings"]:
                warning_table.add_row(warning)

            console.print(warning_table)

        # Show errors
        if validation_result["errors"]:
            error_table = Table(title="Errors", show_header=True)
            error_table.add_column("Error", style="red")

            for error in validation_result["errors"]:
                error_table.add_row(error)

            console.print(error_table)

        if is_valid:
            console.print("[bold green]✓[/bold green] Report meets compliance requirements!")
        else:
            console.print("[bold red]✗[/bold red] Report has compliance issues that need to be addressed.")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def list_templates(
    report_type: Optional[ReportType] = typer.Option(None, help="Filter by report type")
):
    """List available report templates."""
    try:
        generator = ReportGenerator()

        # Get template files
        template_files = list(generator.templates_path.glob("*.html"))

        if not template_files:
            console.print("[yellow]No templates found.[/yellow]")
            return

        # Create table
        table = Table(title="Available Report Templates", show_header=True)
        table.add_column("Template", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Size", style="dim", justify="right")
        table.add_column("Modified", style="dim")

        for template_file in template_files:
            template_name = template_file.stem
            if report_type and report_type not in template_name:
                continue

            file_size = template_file.stat().st_size / 1024
            modified_time = datetime.fromtimestamp(template_file.stat().st_mtime)

            table.add_row(
                template_name,
                template_name.replace('_', ' ').title(),
                f"{file_size:.1f} KB",
                modified_time.strftime('%Y-%m-%d')
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to list templates: {e}")
        raise typer.Exit(1)


@app.command()
def edit_section(
    report_file: Path = typer.Argument(..., help="Report JSON file"),
    section_id: str = typer.Argument(..., help="Section ID to edit"),
    content_file: Optional[Path] = typer.Option(None, help="File containing section content")
):
    """Edit a specific section of a report."""
    console.print(f"[bold blue]Editing section '{section_id}' in {report_file}[/bold blue]")

    try:
        # Load report
        with open(report_file) as f:
            report_data = json.load(f)

        from .core import Report
        report = Report(**report_data)

        # Find section
        target_section = None
        for section in report.sections:
            if section.section_id == section_id:
                target_section = section
                break

        if not target_section:
            console.print(f"[bold red]✗[/bold red] Section '{section_id}' not found")
            raise typer.Exit(1)

        # Get new content
        if content_file and content_file.exists():
            new_content = content_file.read_text()
        else:
            console.print(f"\n[dim]Current content:[/dim]\n{target_section.content[:200]}{'...' if len(target_section.content) > 200 else ''}\n")
            new_content = typer.prompt("Enter new content (or press Enter to keep current)")

            if not new_content.strip():
                console.print("No changes made.")
                return

        # Update section
        target_section.content = new_content
        target_section.metadata["last_edited"] = datetime.now().isoformat()

        # Update report
        report.updated_at = datetime.now()

        # Save updated report
        with open(report_file, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)

        console.print(f"[bold green]✓[/bold green] Section '{section_id}' updated successfully!")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Section edit failed: {e}")
        raise typer.Exit(1)


@app.command()
def add_citation(
    report_file: Path = typer.Argument(..., help="Report JSON file"),
    section_id: str = typer.Argument(..., help="Section ID to add citation to"),
    citation_type: str = typer.Option("document", help="Type of citation (case_law, statute, document, expert_opinion)")
):
    """Add a citation to a report section."""
    console.print(f"[bold blue]Adding {citation_type} citation to section '{section_id}'[/bold blue]")

    try:
        # Load report
        with open(report_file) as f:
            report_data = json.load(f)

        from .core import Report
        report = Report(**report_data)

        # Find section
        target_section = None
        for section in report.sections:
            if section.section_id == section_id:
                target_section = section
                break

        if not target_section:
            console.print(f"[bold red]✗[/bold red] Section '{section_id}' not found")
            raise typer.Exit(1)

        citation_manager = CitationManager()

        # Collect citation information based on type
        citation_data = {}

        if citation_type == "case_law":
            citation_data["case_name"] = typer.prompt("Case name")
            citation_data["year"] = int(typer.prompt("Year"))
            citation_data["court"] = typer.prompt("Court")
            citation_data["citation"] = typer.prompt("Citation")
        elif citation_type == "statute":
            citation_data["statute_name"] = typer.prompt("Statute name")
            citation_data["section"] = typer.prompt("Section")
            citation_data["year"] = int(typer.prompt("Year"))
        elif citation_type == "document":
            citation_data["title"] = typer.prompt("Document title")
            citation_data["author"] = typer.prompt("Author")
            citation_data["date"] = typer.prompt("Date")
        elif citation_type == "expert_opinion":
            citation_data["expert_name"] = typer.prompt("Expert name")
            citation_data["title"] = typer.prompt("Opinion title")
            citation_data["date"] = typer.prompt("Date")

        # Generate citation
        formatted_citation = citation_manager.add_citation(citation_type, **citation_data)

        # Add to section
        target_section.citations.append({
            "type": citation_type,
            "formatted": formatted_citation,
            "data": citation_data,
            "added": datetime.now().isoformat()
        })

        # Update report
        report.updated_at = datetime.now()

        # Save updated report
        with open(report_file, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)

        console.print(f"[bold green]✓[/bold green] Citation added successfully!")
        console.print(f"Citation: {formatted_citation}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Citation addition failed: {e}")
        raise typer.Exit(1)


@app.command()
def batch_generate(
    config_file: Path = typer.Argument(..., help="Batch configuration JSON file"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory")
):
    """Generate multiple reports from batch configuration."""
    console.print(f"[bold blue]Running batch report generation from {config_file}[/bold blue]")

    try:
        # Load batch configuration
        with open(config_file) as f:
            batch_config = json.load(f)

        reports_to_generate = batch_config.get("reports", [])
        if not reports_to_generate:
            console.print("[yellow]No reports specified in configuration.[/yellow]")
            return

        console.print(f"Found {len(reports_to_generate)} reports to generate")

        generator = ReportGenerator()
        output_directory = output_dir or Path("batch_reports")
        output_directory.mkdir(exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            main_task = progress.add_task("Batch generation...", total=len(reports_to_generate))

            for i, report_config in enumerate(reports_to_generate):
                case_id = report_config["case_id"]
                report_type = ReportType(report_config["report_type"])
                title = report_config["title"]
                author = report_config["author"]
                output_format = ReportFormat(report_config.get("format", "pdf"))

                progress.update(main_task, description=f"Generating report for case {case_id}...")

                # Create report
                report = generator.create_report(
                    report_type=report_type,
                    case_id=case_id,
                    title=title,
                    author=author
                )

                # Load data if specified
                case_data = None
                if "data_file" in report_config:
                    data_file = Path(report_config["data_file"])
                    if data_file.exists():
                        with open(data_file) as f:
                            data_dict = json.load(f)
                        case_data = ReportData(**data_dict)

                # Populate and generate
                if case_data:
                    report = generator.populate_report_data(report, case_data)

                output_file = output_directory / f"{case_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
                generator.save_report(report, output_format, case_data, output_file)

                progress.advance(main_task)

        console.print(f"[bold green]✓[/bold green] Batch generation complete!")
        console.print(f"Generated {len(reports_to_generate)} reports in: {output_directory}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Batch generation failed: {e}")
        raise typer.Exit(1)


@app.command()
def preview(
    report_file: Path = typer.Argument(..., help="Report JSON file"),
    section_id: Optional[str] = typer.Option(None, help="Preview specific section only")
):
    """Preview report content in terminal."""
    try:
        # Load report
        with open(report_file) as f:
            report_data = json.load(f)

        from .core import Report
        report = Report(**report_data)

        if section_id:
            # Preview specific section
            target_section = None
            for section in report.sections:
                if section.section_id == section_id:
                    target_section = section
                    break

            if not target_section:
                console.print(f"[bold red]✗[/bold red] Section '{section_id}' not found")
                raise typer.Exit(1)

            console.print(Panel(
                target_section.content[:500] + ("..." if len(target_section.content) > 500 else ""),
                title=f"Section: {target_section.title}",
                border_style="blue"
            ))
        else:
            # Preview entire report
            console.print(Panel(
                f"[bold]Title:[/bold] {report.title}\n"
                f"[bold]Type:[/bold] {report.report_type}\n"
                f"[bold]Case ID:[/bold] {report.case_id}\n"
                f"[bold]Author:[/bold] {report.author}\n"
                f"[bold]Sections:[/bold] {len(report.sections)}\n"
                f"[bold]Created:[/bold] {report.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"[bold]Updated:[/bold] {report.updated_at.strftime('%Y-%m-%d %H:%M')}",
                title="Report Overview",
                border_style="green"
            ))

            # Show section summaries
            section_table = Table(title="Sections", show_header=True)
            section_table.add_column("ID", style="cyan")
            section_table.add_column("Title", style="magenta")
            section_table.add_column("Content Length", style="dim", justify="right")
            section_table.add_column("Citations", style="blue", justify="right")

            for section in report.sections:
                section_table.add_row(
                    section.section_id,
                    section.title,
                    f"{len(section.content)} chars",
                    str(len(section.citations))
                )

            console.print(section_table)

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Preview failed: {e}")
        raise typer.Exit(1)


def add_content_interactive(report):
    """Interactive content addition to report sections."""
    for section in report.sections:
        if Confirm.ask(f"Add content to '{section.title}' section?"):
            content = typer.prompt(f"Enter content for '{section.title}'", type=str, default="")
            if content.strip():
                section.content = content
                section.metadata["edited"] = datetime.now().isoformat()
                console.print(f"[green]✓[/green] Added content to '{section.title}'")


if __name__ == "__main__":
    app()