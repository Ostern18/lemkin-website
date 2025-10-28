"""
Example: Using the Multi-Format Document Parser Agent

This example demonstrates how to use the Document Parser to process
various types of documents for legal and investigative purposes.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from document_parser.agent import DocumentParserAgent
from document_parser.config import DEFAULT_CONFIG, HIGH_ACCURACY_CONFIG
import json


def basic_example():
    """
    Basic document parsing example.
    """
    print("=== Basic Document Parsing Example ===\n")

    # Initialize agent with default configuration
    agent = DocumentParserAgent(config=DEFAULT_CONFIG)

    # Example: Parse a PDF document
    # Note: In real usage, replace with actual file path
    example_doc = Path("sample_evidence/witness_statement.pdf")

    if not example_doc.exists():
        print(f"Sample document not found: {example_doc}")
        print("Please provide a valid document path.\n")
        return

    # Parse the document
    result = agent.parse_document(
        file_path=example_doc,
        source="Police Department - District 5",
        case_id="CASE-2024-001",
        tags=["witness", "statement", "robbery"],
        custodian="Detective Sarah Johnson"
    )

    # Display results
    print(f"Evidence ID: {result['evidence_id']}")
    print(f"Document Type: {result.get('document_type', 'N/A')}")
    print(f"Language: {result.get('language', 'N/A')}")
    print(f"Overall Confidence: {result.get('confidence_scores', {}).get('overall', 'N/A')}")
    print(f"\nExtracted Text Preview:")
    print(result.get('extracted_text', {}).get('full_text', '')[:500] + "...\n")

    # Check for quality issues
    quality_flags = result.get('quality_flags', [])
    if quality_flags:
        print("Quality Issues Detected:")
        for flag in quality_flags:
            print(f"  - {flag.get('severity').upper()}: {flag.get('description')}")
    else:
        print("No quality issues detected.")

    print()


def high_accuracy_example():
    """
    Example using high-accuracy configuration for critical evidence.
    """
    print("=== High-Accuracy Processing Example ===\n")

    # Initialize with high-accuracy configuration
    agent = DocumentParserAgent(config=HIGH_ACCURACY_CONFIG)

    # Example: Parse critical evidence
    critical_doc = Path("sample_evidence/confession.pdf")

    if not critical_doc.exists():
        print(f"Sample document not found: {critical_doc}")
        print("This example requires a document at: {critical_doc}\n")
        return

    result = agent.parse_document(
        file_path=critical_doc,
        source="Interrogation Room A",
        case_id="CASE-2024-002",
        tags=["confession", "critical", "high-priority"],
        custodian="Lead Investigator"
    )

    print(f"Evidence ID: {result['evidence_id']}")

    # Check if human review was triggered
    if 'human_review_requested' in result:
        print("HUMAN REVIEW REQUESTED")
        print(f"  Review ID: {result['human_review_requested']['review_request_id']}")
        print(f"  Priority: {result['human_review_requested']['priority']}")
        print(f"  Reason: Low confidence ({result.get('confidence_scores', {}).get('overall', 'N/A')})")
    else:
        print("Automatic processing approved")
        print(f"Confidence: {result.get('confidence_scores', {}).get('overall', 'N/A')}")

    print()


def batch_processing_example():
    """
    Example of batch processing multiple documents.
    """
    print("=== Batch Processing Example ===\n")

    agent = DocumentParserAgent()

    # List of documents to process
    documents = [
        "sample_evidence/doc1.pdf",
        "sample_evidence/doc2.pdf",
        "sample_evidence/photo1.jpg",
        "sample_evidence/doc3.pdf"
    ]

    # Check if files exist
    existing_docs = [doc for doc in documents if Path(doc).exists()]

    if not existing_docs:
        print("No sample documents found.")
        print("Expected documents in: sample_evidence/")
        print()
        return

    print(f"Processing {len(existing_docs)} documents...\n")

    # Batch process
    results = agent.batch_process(
        file_paths=existing_docs,
        source="Evidence Archive",
        case_id="CASE-2024-003",
        tags=["batch", "archive"]
    )

    # Summary
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful

    print(f"Processed: {successful} successful, {failed} failed\n")

    # Details
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"{i}. FAILED: {result.get('file_path')}")
            print(f"   Error: {result.get('error')}")
        else:
            doc_type = result.get('document_type', 'Unknown')
            confidence = result.get('confidence_scores', {}).get('overall', 'N/A')
            print(f"{i}. SUCCESS: {result['evidence_id']}")
            print(f"   Type: {doc_type}, Confidence: {confidence}")

    print()


def chain_of_custody_example():
    """
    Example demonstrating chain-of-custody tracking.
    """
    print("=== Chain-of-Custody Example ===\n")

    agent = DocumentParserAgent()

    # Parse a document
    example_doc = Path("sample_evidence/evidence1.pdf")

    if not example_doc.exists():
        print("Sample document not found.\n")
        return

    result = agent.parse_document(
        file_path=example_doc,
        source="Crime Scene",
        case_id="CASE-2024-004"
    )

    evidence_id = result['evidence_id']

    print(f"Evidence ID: {evidence_id}\n")

    # Retrieve chain of custody
    chain = agent.get_chain_of_custody(evidence_id)

    print(f"Chain-of-Custody Events: {len(chain)}\n")

    for i, event in enumerate(chain, 1):
        print(f"{i}. {event['event_type'].upper()}")
        print(f"   Timestamp: {event['timestamp']}")
        print(f"   Agent: {event['agent_id']}")
        if event.get('details'):
            print(f"   Details: {json.dumps(event['details'], indent=6)}")
        print()

    # Verify integrity
    integrity_ok = agent.verify_integrity()
    print(f"Chain Integrity Verified: {'✓ YES' if integrity_ok else '✗ NO'}")
    print()


def export_results_example():
    """
    Example of exporting results in different formats.
    """
    print("=== Export Results Example ===\n")

    from shared import OutputFormatter

    agent = DocumentParserAgent()

    # Parse document
    example_doc = Path("sample_evidence/report.pdf")

    if not example_doc.exists():
        print("Sample document not found.\n")
        return

    result = agent.parse_document(
        file_path=example_doc,
        source="Investigation Team"
    )

    # Export as structured JSON
    json_output = OutputFormatter.format_structured_json(result)
    print("JSON Export (first 500 chars):")
    print(json_output[:500] + "...\n")

    # Export as evidence analysis report
    analysis_report = OutputFormatter.format_evidence_analysis(
        evidence_id=result['evidence_id'],
        evidence_type=result.get('document_type', 'Unknown'),
        analysis_type="document_parsing",
        findings=result.get('key_fields', {}),
        confidence_level=result.get('confidence_scores', {}).get('overall', 0)
    )

    print("Evidence Analysis Report (first 500 chars):")
    print(analysis_report[:500] + "...\n")


def main():
    """
    Run all examples.
    """
    print("=" * 60)
    print("MULTI-FORMAT DOCUMENT PARSER - EXAMPLES")
    print("=" * 60)
    print()

    # Note: These examples assume sample documents exist
    # In practice, you would provide your own documents

    print("NOTE: These examples require sample documents in 'sample_evidence/'")
    print("      Create this directory and add test documents to run examples.\n")

    # Run examples
    # Uncomment the examples you want to run:

    # basic_example()
    # high_accuracy_example()
    # batch_processing_example()
    # chain_of_custody_example()
    # export_results_example()

    print("Examples completed. Update paths and uncomment function calls to run.")
    print()


if __name__ == "__main__":
    main()
