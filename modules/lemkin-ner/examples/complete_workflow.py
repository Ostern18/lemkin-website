#!/usr/bin/env python3
"""
Complete workflow demonstration for lemkin-ner.

This script demonstrates the full pipeline:
1. Entity extraction from legal documents
2. Cross-document entity linking
3. Quality validation and human review workflows
4. Export and analysis

Usage:
    python examples/complete_workflow.py
"""

import json
import tempfile
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Import lemkin-ner components
from lemkin_ner import (
    LegalNERProcessor,
    NERConfig,
    EntityValidator,
    LanguageCode,
    EntityType,
    create_default_config
)

console = Console()


def main():
    """Run the complete NER workflow demonstration."""
    console.print("[bold blue]Lemkin NER - Complete Workflow Demonstration[/bold blue]\n")
    
    # Step 1: Setup configuration
    console.print("[bold]Step 1: Configuration Setup[/bold]")
    
    config = create_default_config()
    config.min_confidence = 0.6
    config.entity_types = [
        EntityType.PERSON,
        EntityType.ORGANIZATION,
        EntityType.LOCATION,
        EntityType.DATE,
        EntityType.LEGAL_ENTITY,
        EntityType.COURT,
        EntityType.CASE_NAME
    ]
    config.enable_legal_ner = True
    config.enable_entity_linking = True
    config.enable_validation = True
    
    console.print(f"✓ Primary language: {config.primary_language.value}")
    console.print(f"✓ Entity types: {len(config.entity_types)}")
    console.print(f"✓ Minimum confidence: {config.min_confidence}")
    console.print()
    
    # Step 2: Initialize processor
    console.print("[bold]Step 2: Initialize NER Processor[/bold]")
    processor = LegalNERProcessor(config)
    validator = EntityValidator(config)
    console.print("✓ Processor initialized")
    console.print()
    
    # Step 3: Prepare sample legal documents
    console.print("[bold]Step 3: Sample Legal Documents[/bold]")
    
    sample_documents = {
        "case_filing": """
        Case No. 2023-CV-12345
        
        SUPERIOR COURT OF CALIFORNIA
        COUNTY OF LOS ANGELES
        
        JOHN SMITH, an individual,
                                                    Plaintiff,
        v.                                          
        
        ABC CORPORATION, a Delaware corporation,
        and DOES 1-10, inclusive,
                                                    Defendants.
        
        Filed: January 15, 2023
        
        TO: ABC CORPORATION and its attorneys of record:
        
        PLEASE TAKE NOTICE that plaintiff John Smith, through his attorney 
        Michael Johnson of Johnson & Associates, hereby files this complaint
        for breach of contract and seeks damages in the amount of $500,000.
        
        The contract in question was executed on March 10, 2022, at the
        defendant's headquarters in San Francisco, California.
        """,
        
        "court_order": """
        UNITED STATES DISTRICT COURT
        SOUTHERN DISTRICT OF NEW YORK
        
        Case No. 1:23-cv-03456-JMW
        
        SMITH v. ABC CORPORATION
        
        ORDER GRANTING MOTION FOR SUMMARY JUDGMENT
        
        Before the Court is Defendant ABC Corporation's Motion for Summary
        Judgment filed on June 1, 2023. Having considered the parties' 
        submissions, the Court GRANTS the motion for the following reasons:
        
        1. Plaintiff John Smith failed to establish a genuine issue of 
           material fact regarding the contract interpretation.
        
        2. The contract dated March 10, 2022, clearly states the limitations
           of liability in Section 12.
        
        Judge Patricia Williams presiding.
        
        Dated: August 15, 2023
        """,
        
        "appeal_brief": """
        COURT OF APPEALS
        SECOND CIRCUIT
        
        Case No. 23-4567
        
        JOHN SMITH,
                                                    Appellant,
        v.
        
        ABC CORPORATION,
                                                    Appellee.
        
        APPELLANT'S OPENING BRIEF
        
        Respectfully submitted by:
        Sarah Davis, Esq.
        Davis & Partners LLP
        123 Legal Street
        New York, NY 10001
        
        TO THE HONORABLE COURT:
        
        Appellant John Smith, by and through undersigned counsel, respectfully
        submits this opening brief in support of his appeal from the judgment
        entered by the Honorable Patricia Williams in the Southern District 
        of New York on August 15, 2023.
        
        The district court erred in granting summary judgment because
        genuine issues of material fact exist regarding the contract
        interpretation under California law.
        """
    }
    
    console.print(f"✓ {len(sample_documents)} sample documents prepared")
    console.print()
    
    # Step 4: Extract entities from each document
    console.print("[bold]Step 4: Entity Extraction[/bold]")
    
    extraction_results = []
    
    with Progress() as progress:
        task = progress.add_task("Extracting entities...", total=len(sample_documents))
        
        for doc_name, content in sample_documents.items():
            result = processor.process_text(content, doc_name)
            extraction_results.append(result)
            
            entities_found = len(result['entities'])
            progress.console.print(f"✓ {doc_name}: {entities_found} entities")
            progress.update(task, advance=1)
    
    total_entities = sum(len(r['entities']) for r in extraction_results)
    console.print(f"\n✓ Total entities extracted: {total_entities}")
    console.print()
    
    # Step 5: Display extracted entities
    console.print("[bold]Step 5: Extracted Entities Summary[/bold]")
    
    for i, result in enumerate(extraction_results):
        doc_name = list(sample_documents.keys())[i]
        entities = result['entities']
        
        if entities:
            console.print(f"\n[cyan]Document: {doc_name}[/cyan]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Entity", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Confidence", style="yellow")
            table.add_column("Language", style="blue")
            
            # Show top entities by confidence
            sorted_entities = sorted(entities, key=lambda x: x['confidence'], reverse=True)
            for entity in sorted_entities[:5]:  # Top 5 entities
                table.add_row(
                    entity['text'],
                    entity['entity_type'],
                    f"{entity['confidence']:.3f}",
                    entity['language']
                )
            
            console.print(table)
            
            if len(entities) > 5:
                console.print(f"... and {len(entities) - 5} more entities")
    
    console.print()
    
    # Step 6: Cross-document entity linking
    console.print("[bold]Step 6: Cross-Document Entity Linking[/bold]")
    
    entity_graph = processor.link_entities_across_documents(extraction_results)
    
    console.print(f"✓ Entity graph created:")
    console.print(f"  - Entities: {len(entity_graph.entities)}")
    console.print(f"  - Relationships: {len(entity_graph.relationships)}")
    
    # Display some relationships
    if entity_graph.relationships:
        console.print("\n[cyan]Sample Entity Relationships:[/cyan]")
        
        rel_table = Table(show_header=True, header_style="bold magenta")
        rel_table.add_column("Source Entity", style="cyan")
        rel_table.add_column("Relationship", style="green")
        rel_table.add_column("Target Entity", style="cyan")
        rel_table.add_column("Confidence", style="yellow")
        
        for rel in entity_graph.relationships[:5]:  # Show top 5 relationships
            source = entity_graph.entities[rel['source_id']]
            target = entity_graph.entities[rel['target_id']]
            
            rel_table.add_row(
                source.text,
                rel['relationship_type'],
                target.text,
                f"{rel['confidence']:.3f}"
            )
        
        console.print(rel_table)
    
    console.print()
    
    # Step 7: Entity validation
    console.print("[bold]Step 7: Entity Validation & Quality Assessment[/bold]")
    
    # Collect all entities for validation
    all_entities = []
    for result in extraction_results:
        from lemkin_ner import Entity
        for entity_dict in result['entities']:
            entity = Entity.model_validate(entity_dict)
            all_entities.append(entity)
    
    # Validate entities
    validation_results = validator.validate_batch(all_entities)
    
    # Generate quality report
    quality_report = validator.generate_quality_report(validation_results)
    
    console.print("✓ Validation completed")
    console.print(f"  - Total entities validated: {quality_report['summary']['total_entities']}")
    console.print(f"  - Valid entities: {quality_report['summary']['valid_entities']}")
    console.print(f"  - Validity rate: {quality_report['summary']['validity_rate']:.2%}")
    console.print(f"  - Average confidence: {quality_report['summary']['average_confidence']:.3f}")
    
    # Quality distribution
    console.print("\n[cyan]Quality Distribution:[/cyan]")
    quality_table = Table(show_header=True, header_style="bold magenta")
    quality_table.add_column("Quality Level", style="green")
    quality_table.add_column("Count", style="yellow")
    quality_table.add_column("Percentage", style="blue")
    
    total_entities = quality_report['summary']['total_entities']
    for level, count in quality_report['quality_distribution'].items():
        percentage = (count / total_entities) * 100 if total_entities > 0 else 0
        quality_table.add_row(level.title(), str(count), f"{percentage:.1f}%")
    
    console.print(quality_table)
    console.print()
    
    # Step 8: Create human review tasks
    console.print("[bold]Step 8: Human Review Workflow[/bold]")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        review_summary = validator.create_human_review_tasks(
            validation_results,
            output_dir=temp_dir
        )
        
        console.print(f"✓ Human review tasks created in: {temp_dir}")
        console.print(f"  - Total entities needing review: {review_summary['review_needed']}")
        console.print(f"  - High priority: {review_summary['high_priority']}")
        console.print(f"  - Medium priority: {review_summary['medium_priority']}")
        console.print(f"  - Low priority: {review_summary['low_priority']}")
        
        # List created files
        temp_path = Path(temp_dir)
        created_files = list(temp_path.glob("*"))
        
        if created_files:
            console.print("\n[cyan]Review files created:[/cyan]")
            for file_path in created_files:
                file_size = file_path.stat().st_size
                console.print(f"  - {file_path.name} ({file_size} bytes)")
        
        # Show sample of high priority review tasks
        high_priority_file = temp_path / "high_priority_review.json"
        if high_priority_file.exists():
            with open(high_priority_file, 'r') as f:
                review_data = json.load(f)
            
            if review_data.get('tasks'):
                console.print(f"\n[cyan]Sample High Priority Review Tasks:[/cyan]")
                
                review_table = Table(show_header=True, header_style="bold red")
                review_table.add_column("Entity", style="cyan")
                review_table.add_column("Type", style="green")
                review_table.add_column("Issues", style="red")
                review_table.add_column("Confidence", style="yellow")
                
                for task in review_data['tasks'][:3]:  # Show first 3 tasks
                    issues = "; ".join(task.get('issues', [])[:2])  # First 2 issues
                    if len(task.get('issues', [])) > 2:
                        issues += "..."
                    
                    review_table.add_row(
                        task['text'],
                        task['entity_type'],
                        issues or "Quality review needed",
                        f"{task['validation_confidence']:.3f}"
                    )
                
                console.print(review_table)
    
    console.print()
    
    # Step 9: Export results
    console.print("[bold]Step 9: Export Results[/bold]")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Export extraction results
        extraction_file = temp_path / "extraction_results.json"
        processor.export_results(extraction_results, extraction_file, "json")
        console.print(f"✓ Extraction results: {extraction_file.name}")
        
        # Export entity graph
        graph_file = temp_path / "entity_graph.json"
        graph_data = {
            "graph_id": entity_graph.graph_id,
            "entities": [entity.to_dict() for entity in entity_graph.entities.values()],
            "relationships": entity_graph.relationships,
            "metadata": entity_graph.metadata,
            "created_at": entity_graph.created_at.isoformat()
        }
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        console.print(f"✓ Entity graph: {graph_file.name}")
        
        # Export quality report
        quality_file = temp_path / "quality_report.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        console.print(f"✓ Quality report: {quality_file.name}")
        
        # Show file sizes
        console.print("\n[cyan]Export Summary:[/cyan]")
        export_table = Table(show_header=True, header_style="bold magenta")
        export_table.add_column("File", style="cyan")
        export_table.add_column("Size", style="yellow")
        export_table.add_column("Content", style="green")
        
        files_info = [
            (extraction_file, "Entity extraction results"),
            (graph_file, "Entity relationship graph"),
            (quality_file, "Validation & quality metrics")
        ]
        
        for file_path, description in files_info:
            size = file_path.stat().st_size
            size_str = f"{size:,} bytes"
            if size > 1024:
                size_str = f"{size/1024:.1f} KB"
            
            export_table.add_row(file_path.name, size_str, description)
        
        console.print(export_table)
    
    console.print()
    
    # Step 10: Analysis and insights
    console.print("[bold]Step 10: Analysis & Insights[/bold]")
    
    # Entity type distribution
    entity_type_counts = {}
    for result in extraction_results:
        for entity in result['entities']:
            entity_type = entity['entity_type']
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
    
    console.print("[cyan]Entity Type Distribution:[/cyan]")
    type_table = Table(show_header=True, header_style="bold magenta")
    type_table.add_column("Entity Type", style="green")
    type_table.add_column("Count", style="yellow")
    type_table.add_column("Percentage", style="blue")
    
    total_entities = sum(entity_type_counts.values())
    for entity_type, count in sorted(entity_type_counts.items()):
        percentage = (count / total_entities) * 100 if total_entities > 0 else 0
        type_table.add_row(entity_type, str(count), f"{percentage:.1f}%")
    
    console.print(type_table)
    
    # Language distribution
    language_counts = {}
    for result in extraction_results:
        for entity in result['entities']:
            language = entity['language']
            language_counts[language] = language_counts.get(language, 0) + 1
    
    if len(language_counts) > 1:
        console.print("\n[cyan]Language Distribution:[/cyan]")
        for language, count in sorted(language_counts.items()):
            percentage = (count / total_entities) * 100 if total_entities > 0 else 0
            console.print(f"  - {language}: {count} ({percentage:.1f}%)")
    
    # Top entities by confidence
    all_entities_with_conf = []
    for result in extraction_results:
        for entity in result['entities']:
            all_entities_with_conf.append((entity['text'], entity['entity_type'], entity['confidence']))
    
    console.print(f"\n[cyan]Top 10 Entities by Confidence:[/cyan]")
    top_entities = sorted(all_entities_with_conf, key=lambda x: x[2], reverse=True)[:10]
    
    top_table = Table(show_header=True, header_style="bold magenta")
    top_table.add_column("Rank", style="blue")
    top_table.add_column("Entity", style="cyan")
    top_table.add_column("Type", style="green")
    top_table.add_column("Confidence", style="yellow")
    
    for i, (text, entity_type, confidence) in enumerate(top_entities, 1):
        top_table.add_row(str(i), text, entity_type, f"{confidence:.3f}")
    
    console.print(top_table)
    console.print()
    
    # Summary and recommendations
    console.print("[bold green]Workflow Summary[/bold green]")
    console.print("✓ Successfully processed 3 legal documents")
    console.print(f"✓ Extracted {total_entities} entities across multiple types")
    console.print(f"✓ Created entity graph with {len(entity_graph.relationships)} relationships")
    console.print(f"✓ Achieved {quality_report['summary']['validity_rate']:.2%} validation success rate")
    console.print(f"✓ Generated human review tasks for quality assurance")
    console.print("✓ Exported results in multiple formats")
    
    console.print("\n[bold blue]Recommendations:[/bold blue]")
    
    if quality_report.get('recommendations'):
        for recommendation in quality_report['recommendations'][:3]:
            console.print(f"• {recommendation}")
    else:
        console.print("• Consider increasing minimum confidence threshold for higher precision")
        console.print("• Review entities with low validation scores manually")
        console.print("• Add domain-specific terminology for better legal entity recognition")
    
    console.print(f"\n[bold]🎉 Lemkin NER workflow demonstration completed successfully![/bold]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during demo: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())