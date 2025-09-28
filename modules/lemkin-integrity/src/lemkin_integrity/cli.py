import typer
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from datetime import datetime, timezone

from .core import EvidenceIntegrityManager, EvidenceMetadata, ActionType

app = typer.Typer(
    name="lemkin-integrity",
    help="Evidence integrity verification and chain of custody management",
    no_args_is_help=True
)
console = Console()

@app.command()
def hash_evidence(
    file_path: Path = typer.Argument(..., help="Path to evidence file"),
    case_id: str = typer.Option(..., help="Case identifier"),
    collector: str = typer.Option(..., help="Evidence collector name"),
    source: str = typer.Option("Unknown", help="Evidence source"),
    location: Optional[str] = typer.Option(None, help="Collection location"),
    description: Optional[str] = typer.Option(None, help="Evidence description"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags"),
    db_path: str = typer.Option("evidence_integrity.db", help="Database path")
):
    """Generate cryptographic hash for evidence file"""
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    try:
        # Create metadata
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
        metadata = EvidenceMetadata(
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            mime_type="application/octet-stream",  # Would use python-magic in real implementation
            created_date=datetime.now(timezone.utc),
            source=source,
            case_id=case_id,
            collector=collector,
            location=location,
            description=description,
            tags=tag_list
        )
        
        # Initialize manager and generate hash
        manager = EvidenceIntegrityManager(db_path)
        evidence_hash = manager.generate_evidence_hash(file_path, metadata)
        
        # Display results
        table = Table(title="Evidence Hash Generated")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Evidence ID", evidence_hash.evidence_id)
        table.add_row("SHA-256 Hash", evidence_hash.sha256_hash)
        table.add_row("File Size", str(metadata.file_size))
        table.add_row("Case ID", case_id)
        table.add_row("Collector", collector)
        table.add_row("Timestamp", evidence_hash.timestamp.isoformat())
        
        console.print(table)
        console.print(f"[green]✓ Evidence hash stored in database: {db_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error generating hash: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def add_custody(
    evidence_id: str = typer.Argument(..., help="Evidence identifier"),
    action: str = typer.Argument(..., help="Action type (accessed, transferred, etc.)"),
    actor: str = typer.Argument(..., help="Person performing action"),
    location: Optional[str] = typer.Option(None, help="Location of action"),
    notes: Optional[str] = typer.Option(None, help="Additional notes"),
    db_path: str = typer.Option("evidence_integrity.db", help="Database path")
):
    """Add entry to chain of custody"""
    
    try:
        # Validate action type
        try:
            action_type = ActionType(action.lower())
        except ValueError:
            valid_actions = [a.value for a in ActionType]
            console.print(f"[red]Invalid action. Valid actions: {', '.join(valid_actions)}[/red]")
            raise typer.Exit(1)
        
        # Create custody entry
        manager = EvidenceIntegrityManager(db_path)
        custody_entry = manager.create_custody_entry(
            evidence_id=evidence_id,
            action=action_type,
            actor=actor,
            location=location,
            notes=notes
        )
        
        console.print(f"[green]✓ Custody entry created: {custody_entry.entry_id}[/green]")
        console.print(f"Action: {action_type.value}")
        console.print(f"Actor: {actor}")
        console.print(f"Timestamp: {custody_entry.timestamp.isoformat()}")
        
    except Exception as e:
        console.print(f"[red]Error adding custody entry: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def verify(
    evidence_id: str = typer.Argument(..., help="Evidence identifier"),
    file_path: Optional[Path] = typer.Option(None, help="Current file path for hash verification"),
    db_path: str = typer.Option("evidence_integrity.db", help="Database path")
):
    """Verify evidence integrity"""
    
    try:
        manager = EvidenceIntegrityManager(db_path)
        integrity_report = manager.verify_integrity(evidence_id, file_path)
        
        # Display results
        status_color = "green" if integrity_report.status.name == "VERIFIED" else "red"
        
        panel_content = f"""
[bold]Evidence ID:[/bold] {evidence_id}
[bold]Status:[/bold] [{status_color}]{integrity_report.status.value.upper()}[/{status_color}]
[bold]Hash Verified:[/bold] {"✓" if integrity_report.hash_verified else "✗"}
[bold]Custody Verified:[/bold] {"✓" if integrity_report.custody_verified else "✗"}
[bold]Admissible:[/bold] {"✓" if integrity_report.admissible else "✗"}
[bold]Timestamp:[/bold] {integrity_report.timestamp.isoformat()}
        """
        
        console.print(Panel(panel_content, title="Integrity Verification"))
        
        if integrity_report.issues:
            console.print("\n[red]Issues Found:[/red]")
            for issue in integrity_report.issues:
                console.print(f"  • {issue}")
        
        if integrity_report.recommendations:
            console.print("\n[yellow]Recommendations:[/yellow]")
            for rec in integrity_report.recommendations:
                console.print(f"  • {rec}")
        
    except Exception as e:
        console.print(f"[red]Error verifying integrity: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def custody_chain(
    evidence_id: str = typer.Argument(..., help="Evidence identifier"),
    db_path: str = typer.Option("evidence_integrity.db", help="Database path"),
    output_format: str = typer.Option("table", help="Output format: table or json")
):
    """Display chain of custody for evidence"""
    
    try:
        manager = EvidenceIntegrityManager(db_path)
        custody_entries = manager.get_custody_chain(evidence_id)
        
        if not custody_entries:
            console.print(f"[yellow]No custody entries found for evidence: {evidence_id}[/yellow]")
            return
        
        if output_format == "json":
            # JSON output
            custody_data = [entry.to_dict() for entry in custody_entries]
            console.print(JSON.from_data(custody_data))
        else:
            # Table output
            table = Table(title=f"Chain of Custody - {evidence_id}")
            table.add_column("Timestamp", style="cyan")
            table.add_column("Action", style="yellow")
            table.add_column("Actor", style="green")
            table.add_column("Location", style="blue")
            table.add_column("Notes", style="white")
            
            for entry in custody_entries:
                table.add_row(
                    entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    entry.action.value,
                    entry.actor,
                    entry.location or "-",
                    entry.notes or "-"
                )
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error retrieving custody chain: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def generate_manifest(
    case_id: str = typer.Argument(..., help="Case identifier"),
    output_file: Optional[Path] = typer.Option(None, help="Output file path"),
    db_path: str = typer.Option("evidence_integrity.db", help="Database path")
):
    """Generate court manifest for case"""
    
    try:
        manager = EvidenceIntegrityManager(db_path)
        manifest = manager.generate_court_manifest(case_id)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(manifest.to_dict(), f, indent=2, default=str)
            console.print(f"[green]✓ Manifest exported to: {output_file}[/green]")
        else:
            console.print(JSON.from_data(manifest.to_dict()))
        
        # Summary
        console.print(f"\n[bold]Manifest Summary:[/bold]")
        console.print(f"Case ID: {manifest.case_id}")
        console.print(f"Evidence Count: {manifest.evidence_count}")
        console.print(f"Total Size: {manifest.total_size:,} bytes")
        console.print(f"Verified: {manifest.integrity_summary['verified']}")
        console.print(f"Compromised: {manifest.integrity_summary['compromised']}")
        console.print(f"Unknown: {manifest.integrity_summary['unknown']}")
        
    except Exception as e:
        console.print(f"[red]Error generating manifest: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def export_package(
    case_id: str = typer.Argument(..., help="Case identifier"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    db_path: str = typer.Option("evidence_integrity.db", help="Database path")
):
    """Export complete evidence package for case"""
    
    try:
        manager = EvidenceIntegrityManager(db_path)
        export_summary = manager.export_evidence_package(case_id, output_dir)
        
        console.print(f"[green]✓ Evidence package exported successfully[/green]")
        console.print(f"Case ID: {export_summary['case_id']}")
        console.print(f"Evidence Count: {export_summary['evidence_count']}")
        console.print(f"Output Directory: {export_summary['output_directory']}")
        console.print(f"Files Created: {len(export_summary['files_created'])}")
        
    except Exception as e:
        console.print(f"[red]Error exporting package: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()