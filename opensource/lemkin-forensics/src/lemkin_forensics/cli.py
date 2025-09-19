"""
Command-line interface for Lemkin Digital Forensics Helpers.
"""

import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID

from .core import (
    FileAnalyzer,
    NetworkProcessor,
    MobileAnalyzer,
    AuthenticityVerifier,
    DigitalEvidence,
    FileType,
)

app = typer.Typer(
    name="lemkin-forensics",
    help="Digital forensics helpers for non-technical legal investigators",
    no_args_is_help=True
)
console = Console()


@app.command()
def analyze_files(
    source_path: Path = typer.Argument(..., help="Directory or disk image to analyze"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    include_deleted: bool = typer.Option(True, help="Search for deleted files"),
    show_hidden: bool = typer.Option(True, help="Include hidden files in analysis")
):
    """Analyze file system for forensic evidence"""

    if not source_path.exists():
        console.print(f"[red]Error: Path not found: {source_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Analyzing file system: {source_path}[/cyan]")

        with Progress() as progress:
            task = progress.add_task("[green]Scanning files...", total=100)

            analyzer = FileAnalyzer()
            analysis = analyzer.analyze_file_system(source_path)

            progress.update(task, completed=100)

        # Display summary
        panel_content = f"""
[bold]Analysis ID:[/bold] {analysis.analysis_id}
[bold]Source:[/bold] {analysis.source_path}
[bold]Total Files:[/bold] {analysis.total_files:,}
[bold]Total Size:[/bold] {analysis.total_size:,} bytes
[bold]Deleted Files:[/bold] {len(analysis.deleted_files)}
[bold]Suspicious Files:[/bold] {len(analysis.suspicious_files)}
[bold]Hidden Files:[/bold] {len(analysis.hidden_files)}
        """

        console.print(Panel(panel_content, title="File System Analysis Results"))

        # Show file type breakdown
        if analysis.file_types:
            table = Table(title="File Type Distribution")
            table.add_column("Type", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Percentage", style="yellow")

            total_files = sum(analysis.file_types.values())
            for file_type, count in sorted(analysis.file_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_files) * 100 if total_files > 0 else 0
                table.add_row(file_type, str(count), f"{percentage:.1f}%")

            console.print(table)

        # Show suspicious files if found
        if analysis.suspicious_files:
            console.print("\n[yellow]Suspicious Files Found:[/yellow]")
            sus_table = Table()
            sus_table.add_column("File", style="red")
            sus_table.add_column("Size", style="cyan")
            sus_table.add_column("Modified", style="yellow")

            for file_meta in analysis.suspicious_files[:10]:  # Show first 10
                sus_table.add_row(
                    file_meta.file_name,
                    f"{file_meta.file_size:,}",
                    file_meta.modified_date.strftime("%Y-%m-%d %H:%M") if file_meta.modified_date else "Unknown"
                )

            console.print(sus_table)

        # Show deleted files if found
        if analysis.deleted_files and include_deleted:
            console.print("\n[red]Deleted Files Found:[/red]")
            del_table = Table()
            del_table.add_column("File", style="red")
            del_table.add_column("Type", style="cyan")
            del_table.add_column("Size", style="yellow")

            for file_meta in analysis.deleted_files:
                del_table.add_row(
                    file_meta.file_name,
                    file_meta.file_type.value,
                    f"{file_meta.file_size:,}"
                )

            console.print(del_table)

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(analysis.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Analysis saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error analyzing files: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze_network(
    log_files: str = typer.Argument(..., help="Comma-separated network log files"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    show_suspicious: bool = typer.Option(True, help="Show suspicious activities")
):
    """Analyze network logs for forensic evidence"""

    log_paths = [Path(f.strip()) for f in log_files.split(",")]

    # Check if files exist
    for log_path in log_paths:
        if not log_path.exists():
            console.print(f"[red]Error: Log file not found: {log_path}[/red]")
            raise typer.Exit(1)

    try:
        console.print(f"[cyan]Analyzing {len(log_paths)} network log files...[/cyan]")

        processor = NetworkProcessor()
        analysis = processor.process_network_logs(log_paths)

        # Display summary
        date_range = f"{analysis.date_range['start'].strftime('%Y-%m-%d')} to {analysis.date_range['end'].strftime('%Y-%m-%d')}"
        panel_content = f"""
[bold]Analysis ID:[/bold] {analysis.analysis_id}
[bold]Total Entries:[/bold] {analysis.total_entries:,}
[bold]Date Range:[/bold] {date_range}
[bold]Top Sources:[/bold] {len(analysis.top_sources)}
[bold]Top Destinations:[/bold] {len(analysis.top_destinations)}
[bold]Suspicious Activities:[/bold] {len(analysis.suspicious_activities)}
        """

        console.print(Panel(panel_content, title="Network Analysis Results"))

        # Show top sources
        if analysis.top_sources:
            table = Table(title="Top Source IPs")
            table.add_column("IP Address", style="cyan")
            table.add_column("Connections", style="green")
            table.add_column("Total Bytes", style="yellow")

            for source in analysis.top_sources[:10]:
                table.add_row(
                    source["ip"],
                    str(source["connections"]),
                    f"{source['total_bytes']:,}"
                )

            console.print(table)

        # Show suspicious activities
        if analysis.suspicious_activities and show_suspicious:
            console.print("\n[red]Suspicious Activities:[/red]")
            sus_table = Table()
            sus_table.add_column("Type", style="red")
            sus_table.add_column("Source IP", style="cyan")
            sus_table.add_column("Description", style="yellow")

            for activity in analysis.suspicious_activities:
                sus_table.add_row(
                    activity["type"],
                    activity.get("source_ip", "Unknown"),
                    activity.get("description", "No description")
                )

            console.print(sus_table)

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(analysis.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Analysis saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error analyzing network logs: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def extract_mobile(
    backup_path: Path = typer.Argument(..., help="Path to mobile device backup"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    include_location: bool = typer.Option(False, help="Include location data"),
    include_messages: bool = typer.Option(True, help="Include message data")
):
    """Extract data from mobile device backup"""

    if not backup_path.exists():
        console.print(f"[red]Error: Backup path not found: {backup_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Extracting mobile data from: {backup_path}[/cyan]")

        analyzer = MobileAnalyzer()
        extraction = analyzer.extract_mobile_data(backup_path)

        # Display summary
        panel_content = f"""
[bold]Extraction ID:[/bold] {extraction.extraction_id}
[bold]Device Type:[/bold] {extraction.device_info.get('device_type', 'Unknown')}
[bold]Contacts:[/bold] {len(extraction.contacts)}
[bold]Messages:[/bold] {len(extraction.messages)}
[bold]Call Logs:[/bold] {len(extraction.call_logs)}
[bold]Apps:[/bold] {len(extraction.app_data)}
[bold]Location Points:[/bold] {len(extraction.location_data)}
        """

        console.print(Panel(panel_content, title="Mobile Data Extraction Results"))

        # Show contacts
        if extraction.contacts:
            console.print("\n[cyan]Contacts Found:[/cyan]")
            contacts_table = Table()
            contacts_table.add_column("Name", style="green")
            contacts_table.add_column("Phone", style="cyan")
            contacts_table.add_column("Email", style="yellow")

            for contact in extraction.contacts[:10]:  # Show first 10
                contacts_table.add_row(
                    contact.get("name", "Unknown"),
                    contact.get("phone", ""),
                    contact.get("email", "")
                )

            console.print(contacts_table)

        # Show messages if requested
        if extraction.messages and include_messages:
            console.print("\n[cyan]Recent Messages:[/cyan]")
            messages_table = Table()
            messages_table.add_column("From", style="green")
            messages_table.add_column("To", style="cyan")
            messages_table.add_column("Content", style="white", max_width=40)
            messages_table.add_column("Date", style="yellow")

            for message in extraction.messages[:5]:  # Show first 5
                messages_table.add_row(
                    message.get("sender", "Unknown"),
                    message.get("recipient", "Unknown"),
                    message.get("content", "")[:40] + "..." if len(message.get("content", "")) > 40 else message.get("content", ""),
                    message.get("timestamp", "")[:10]
                )

            console.print(messages_table)

        # Show location data if requested
        if extraction.location_data and include_location:
            console.print("\n[cyan]Location Data:[/cyan]")
            location_table = Table()
            location_table.add_column("Latitude", style="green")
            location_table.add_column("Longitude", style="cyan")
            location_table.add_column("Accuracy", style="yellow")
            location_table.add_column("Timestamp", style="white")

            for location in extraction.location_data[:10]:  # Show first 10
                location_table.add_row(
                    str(location.get("latitude", 0)),
                    str(location.get("longitude", 0)),
                    f"{location.get('accuracy', 0)}m",
                    location.get("timestamp", "")[:19]
                )

            console.print(location_table)

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(extraction.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Extraction saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error extracting mobile data: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify_authenticity(
    evidence_file: Path = typer.Argument(..., help="Path to evidence file"),
    evidence_id: str = typer.Option(..., help="Evidence identifier"),
    evidence_type: str = typer.Option("document", help="Evidence type"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Verify digital evidence authenticity"""

    if not evidence_file.exists():
        console.print(f"[red]Error: Evidence file not found: {evidence_file}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Verifying authenticity of: {evidence_file}[/cyan]")

        # Create evidence object
        evidence = DigitalEvidence(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            file_path=evidence_file
        )

        # Verify authenticity
        verifier = AuthenticityVerifier()
        report = verifier.verify_digital_authenticity(evidence)

        # Display results
        status_color = "green" if report.status.value == "authentic" else "red"
        panel_content = f"""
[bold]Evidence ID:[/bold] {report.evidence_id}
[bold]Status:[/bold] [{status_color}]{report.status.value.upper()}[/{status_color}]
[bold]Confidence:[/bold] {report.confidence_score:.1%}
[bold]Integrity Verified:[/bold] {"✓" if report.integrity_verified else "✗"}
[bold]Timestamp Verified:[/bold] {"✓" if report.timestamp_verified else "✗"}
[bold]Metadata Consistent:[/bold] {"✓" if report.metadata_consistent else "✗"}
        """

        console.print(Panel(panel_content, title="Authenticity Verification Results"))

        # Show findings
        if report.findings:
            console.print("\n[red]Findings:[/red]")
            for finding in report.findings:
                console.print(f"  • {finding}")

        # Show recommendations
        if report.recommendations:
            console.print("\n[yellow]Recommendations:[/yellow]")
            for recommendation in report.recommendations:
                console.print(f"  • {recommendation}")

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(report.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Report saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error verifying authenticity: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def hash_file(
    file_path: Path = typer.Argument(..., help="File to hash"),
    algorithms: str = typer.Option("md5,sha256", help="Hash algorithms to use")
):
    """Calculate cryptographic hashes of a file"""

    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Calculating hashes for: {file_path}[/cyan]")

        algorithms_list = [alg.strip().lower() for alg in algorithms.split(",")]
        file_size = file_path.stat().st_size

        # Calculate hashes
        hashes = {}
        with open(file_path, 'rb') as f:
            import hashlib

            hash_objects = {}
            for alg in algorithms_list:
                if alg == "md5":
                    hash_objects[alg] = hashlib.md5()
                elif alg == "sha1":
                    hash_objects[alg] = hashlib.sha1()
                elif alg == "sha256":
                    hash_objects[alg] = hashlib.sha256()
                elif alg == "sha512":
                    hash_objects[alg] = hashlib.sha512()
                else:
                    console.print(f"[yellow]Warning: Unsupported algorithm: {alg}[/yellow]")
                    continue

            while chunk := f.read(8192):
                for hash_obj in hash_objects.values():
                    hash_obj.update(chunk)

            for alg, hash_obj in hash_objects.items():
                hashes[alg] = hash_obj.hexdigest()

        # Display results
        table = Table(title="File Hash Results")
        table.add_column("Algorithm", style="cyan")
        table.add_column("Hash", style="green")

        for alg, hash_value in hashes.items():
            table.add_row(alg.upper(), hash_value)

        console.print(table)

        # Show file info
        console.print(f"\n[cyan]File Size:[/cyan] {file_size:,} bytes")
        console.print(f"[cyan]File Path:[/cyan] {file_path}")

    except Exception as e:
        console.print(f"[red]Error calculating hashes: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()