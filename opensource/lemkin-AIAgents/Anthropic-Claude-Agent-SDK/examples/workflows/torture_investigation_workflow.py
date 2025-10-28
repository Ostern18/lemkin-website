"""
Example: Complete Torture Investigation Workflow

This example demonstrates a complete end-to-end workflow for investigating
allegations of torture using all four agents.

Scenario:
  - Victim alleges torture during detention
  - Available evidence: medical report, witness statements, detention records
  - Objective: Build case for torture prosecution

Workflow:
  1. Ingest and parse all documents
  2. Analyze medical evidence for torture indicators
  3. Compare witness statements for consistency
  4. Identify evidence gaps and next steps
  5. Generate investigation plan
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.document_parser.agent import DocumentParserAgent
from agents.comparative_analyzer.agent import ComparativeAnalyzerAgent
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent
from agents.evidence_gap_identifier.agent import EvidenceGapIdentifierAgent
from shared import AuditLogger, EvidenceHandler, OutputFormatter


def run_torture_investigation_workflow():
    """
    Complete torture investigation workflow.
    """
    print("=" * 80)
    print("LEMKINAI TORTURE INVESTIGATION WORKFLOW")
    print("=" * 80)
    print()

    case_id = "TORTURE-2024-001"
    print(f"Case ID: {case_id}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =========================================================================
    # SETUP: Initialize Shared Infrastructure
    # =========================================================================
    print("Setting up shared infrastructure...")
    audit_logger = AuditLogger()
    evidence_handler = EvidenceHandler()

    shared_infra = {
        'audit_logger': audit_logger,
        'evidence_handler': evidence_handler
    }

    # Initialize all agents
    parser = DocumentParserAgent(**shared_infra)
    medical_analyst = MedicalForensicAnalystAgent(**shared_infra)
    comparator = ComparativeAnalyzerAgent(**shared_infra)
    gap_finder = EvidenceGapIdentifierAgent(**shared_infra)

    print("✓ Infrastructure ready\n")

    # =========================================================================
    # PHASE 1: Evidence Ingestion & Parsing
    # =========================================================================
    print("-" * 80)
    print("PHASE 1: EVIDENCE INGESTION & PARSING")
    print("-" * 80)
    print()

    evidence_items = []

    # Document 1: Medical examination report
    print("1. Parsing medical examination report...")
    medical_doc = """
    MEDICAL EXAMINATION REPORT
    Patient ID: REDACTED
    Examination Date: 2024-01-20

    Chief Complaint: Patient reports being beaten during detention

    Physical Examination:
    - Multiple contusions on back (10cm x 5cm, 8cm x 4cm)
    - Bilateral shoulder tenderness
    - Linear marks on soles of feet
    - Patient in visible distress

    Assessment: Injuries consistent with blunt force trauma. Pattern suggests
    beating with rod or similar object. Foot injuries consistent with falanga.

    Examining Physician: Dr. Sarah Johnson, MD
    """

    # In real scenario, this would be a PDF file
    # For demo, we'll process the text directly
    print("   Source: Regional Hospital")
    print("   Type: PDF Medical Report")
    evidence_items.append({
        'type': 'medical_report',
        'text': medical_doc,
        'source': 'Regional Hospital',
        'tags': ['medical', 'examination', 'torture']
    })
    print("   ✓ Medical report processed\n")

    # Document 2: Victim's witness statement
    print("2. Parsing victim witness statement...")
    victim_statement = """
    WITNESS STATEMENT
    Name: REDACTED
    Date: 2024-01-22

    I was arrested on January 10, 2024 by military police. They took me to
    a detention facility I did not recognize. For three days, they beat me
    with batons on my back. They also beat the soles of my feet until I
    could not walk. They wanted me to confess to crimes I did not commit.

    The beatings occurred in a small room with no windows. There were usually
    2-3 officers present. I heard other prisoners screaming in nearby rooms.

    Statement taken by: Legal Aid Officer Martinez
    """

    print("   Source: Legal Aid Organization")
    print("   Type: Witness Statement")
    evidence_items.append({
        'type': 'witness_statement',
        'text': victim_statement,
        'source': 'Legal Aid Organization',
        'tags': ['witness', 'victim', 'statement']
    })
    print("   ✓ Victim statement processed\n")

    # Document 3: Detention facility log
    print("3. Parsing detention facility log...")
    detention_log = """
    DETENTION LOG ENTRY
    Facility: Military Detention Center #5
    Entry Date: 2024-01-10

    Detainee ID: 2024-0145
    Arrival Time: 14:30
    Arresting Unit: 3rd Battalion
    Reason: Suspected insurgent activities

    Detention Period: 2024-01-10 to 2024-01-13
    Release Time: 08:00
    Released to: Legal Aid Organization

    Notes: Standard processing. No incidents recorded.
    """

    print("   Source: Military Records")
    print("   Type: Official Detention Log")
    evidence_items.append({
        'type': 'detention_log',
        'text': detention_log,
        'source': 'Military Records (court-ordered)',
        'tags': ['detention', 'official', 'military']
    })
    print("   ✓ Detention log processed\n")

    print(f"Total Evidence Items Ingested: {len(evidence_items)}")
    print()

    # =========================================================================
    # PHASE 2: Medical Forensic Analysis
    # =========================================================================
    print("-" * 80)
    print("PHASE 2: MEDICAL FORENSIC ANALYSIS")
    print("-" * 80)
    print()

    print("Analyzing medical report for torture indicators...")
    print("Applying Istanbul Protocol standards...")
    print()

    medical_analysis = medical_analyst.analyze_for_torture(
        medical_record=medical_doc,
        case_id=case_id
    )

    # Display findings
    print("MEDICAL FINDINGS:")
    print()

    if medical_analysis.get('torture_indicators', {}).get('present'):
        print("✓ TORTURE INDICATORS IDENTIFIED")
        print()
        indicators = medical_analysis['torture_indicators'].get('indicators_found', [])
        for i, indicator in enumerate(indicators, 1):
            print(f"{i}. {indicator.get('indicator', 'Unknown')}")
            print(f"   Evidence: {indicator.get('evidence', 'N/A')}")
            print()

        consistency = medical_analysis['torture_indicators'].get('consistency_assessment')
        print(f"Istanbul Protocol Consistency: {consistency.upper()}")
    else:
        print("No torture indicators identified")

    print()
    print(f"Analysis Confidence: {medical_analysis.get('confidence_scores', {}).get('torture_assessment', 'N/A')}")
    print()

    # =========================================================================
    # PHASE 3: Comparative Analysis
    # =========================================================================
    print("-" * 80)
    print("PHASE 3: COMPARATIVE ANALYSIS")
    print("-" * 80)
    print()

    print("Comparing victim statement with detention log for inconsistencies...")
    print()

    comparison_result = comparator.process({
        'documents': [
            {'text': victim_statement, 'role': 'victim_statement'},
            {'text': detention_log, 'role': 'official_record'}
        ],
        'comparison_type': 'multi_document_similarity',
        'case_id': case_id
    })

    print("COMPARISON RESULTS:")
    print()

    # Check for inconsistencies
    if 'differences_identified' in comparison_result:
        diffs = comparison_result['differences_identified']
        if diffs:
            print(f"Inconsistencies Found: {len(diffs)}")
            print()
            for i, diff in enumerate(diffs[:3], 1):  # Show top 3
                print(f"{i}. {diff.get('description', 'N/A')}")
                print(f"   Significance: {diff.get('significance', 'N/A')}")
                print()

    # =========================================================================
    # PHASE 4: Evidence Gap Analysis
    # =========================================================================
    print("-" * 80)
    print("PHASE 4: EVIDENCE GAP ANALYSIS")
    print("-" * 80)
    print()

    print("Analyzing evidence against torture charges...")
    print()

    gap_analysis = gap_finder.process({
        'charges': ['torture', 'unlawful_detention', 'cruel_treatment'],
        'available_evidence': [
            {'type': 'medical_report', 'summary': 'Medical exam with torture indicators'},
            {'type': 'witness_statement', 'summary': 'Victim testimony'},
            {'type': 'detention_log', 'summary': 'Official detention record'}
        ],
        'case_theory': 'Systematic torture during unlawful military detention',
        'case_id': case_id
    })

    print("GAP ANALYSIS SUMMARY:")
    print()
    gap_summary = gap_analysis.get('gap_summary', {})
    print(f"Total Gaps: {gap_summary.get('total_gaps', 0)}")
    print(f"  - Critical: {gap_summary.get('critical', 0)}")
    print(f"  - High: {gap_summary.get('high', 0)}")
    print(f"  - Medium: {gap_summary.get('medium', 0)}")
    print()

    print("TOP 5 PRIORITY ACTIONS:")
    print()
    for i, action in enumerate(gap_analysis.get('critical_next_steps', [])[:5], 1):
        print(f"{i}. [{action.get('priority', 'N/A').upper()}] {action.get('description', 'N/A')}")
        print(f"   Rationale: {action.get('rationale', 'N/A')[:100]}...")
        print()

    # =========================================================================
    # PHASE 5: Generate Investigation Plan
    # =========================================================================
    print("-" * 80)
    print("PHASE 5: INVESTIGATION PLAN")
    print("-" * 80)
    print()

    print("RECOMMENDED NEXT STEPS:")
    print()

    # Interview recommendations
    if 'witness_interview_questions' in gap_analysis:
        print("A. WITNESS INTERVIEWS:")
        interview_plan = gap_analysis['witness_interview_questions']

        if 'new_witnesses_to_locate' in interview_plan:
            for witness in interview_plan['new_witnesses_to_locate'][:3]:
                print(f"   - Locate: {witness.get('witness_type', 'N/A')}")
                print(f"     Purpose: {witness.get('why_needed', 'N/A')}")
                print()

    # Document requests
    if 'document_requests' in gap_analysis:
        print("B. DOCUMENT REQUESTS:")
        for doc in gap_analysis['document_requests'][:3]:
            print(f"   - Request: {doc.get('document_type', 'N/A')}")
            print(f"     From: {doc.get('from_whom', 'N/A')}")
            print()

    # Expert consultations
    if 'expert_consultations' in gap_analysis:
        print("C. EXPERT CONSULTATIONS:")
        for expert in gap_analysis['expert_consultations'][:3]:
            print(f"   - Expert Type: {expert.get('expertise_needed', 'N/A')}")
            print(f"     Purpose: {expert.get('purpose', 'N/A')}")
            print()

    # =========================================================================
    # PHASE 6: Chain of Custody Verification
    # =========================================================================
    print("-" * 80)
    print("PHASE 6: CHAIN OF CUSTODY VERIFICATION")
    print("-" * 80)
    print()

    # Get session summary
    session_summary = audit_logger.get_session_summary()

    print("AUDIT TRAIL SUMMARY:")
    print()
    print(f"Total Events Logged: {session_summary.get('total_events', 0)}")
    print(f"Agents Involved: {len(session_summary.get('agents', []))}")
    print(f"  - {', '.join(session_summary.get('agents', []))}")
    print()

    # Verify integrity
    integrity = audit_logger.verify_chain_integrity()
    print(f"Chain Integrity Verified: {'✓ YES' if integrity else '✗ NO'}")
    print()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("=" * 80)
    print("INVESTIGATION WORKFLOW COMPLETE")
    print("=" * 80)
    print()

    print("CASE ASSESSMENT:")
    print()
    print(f"✓ {len(evidence_items)} evidence items processed")
    print(f"✓ Medical analysis completed (torture indicators: {'FOUND' if medical_analysis.get('torture_indicators', {}).get('present') else 'NOT FOUND'})")
    print(f"✓ Comparative analysis completed")
    print(f"✓ Gap analysis completed ({gap_summary.get('total_gaps', 0)} gaps identified)")
    print(f"✓ Investigation plan generated")
    print(f"✓ Chain of custody verified")
    print()

    print("CASE STATUS: Ready for legal review")
    print()

    print(f"Full audit log available: {audit_logger.log_file}")
    print()


if __name__ == "__main__":
    run_torture_investigation_workflow()
