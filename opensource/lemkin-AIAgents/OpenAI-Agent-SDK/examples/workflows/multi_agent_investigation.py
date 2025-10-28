"""
Multi-Agent Investigation Workflow

Demonstrates how multiple LemkinAI agents work together using
OpenAI Agents SDK handoffs for a complete investigation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import LemkinAgent, AuditLogger, EvidenceHandler
from agents.osint_synthesis.agent import OSINTSynthesisAgent
from agents.legal_advisor.agent import LegalAdvisorAgent
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent
from agents.evidence_gap_identifier.agent import EvidenceGapIdentifierAgent


def main():
    """
    Complete multi-agent investigation workflow.

    Workflow:
    1. OSINT analysis verifies claims
    2. Medical analysis reviews evidence
    3. Legal analysis determines applicable law
    4. Gap analysis identifies missing evidence
    """

    print("=" * 60)
    print("LemkinAI Multi-Agent Investigation Workflow")
    print("OpenAI Agents SDK Implementation")
    print("=" * 60)
    print()

    # Initialize shared infrastructure
    print("Initializing shared infrastructure...")
    audit_logger = AuditLogger()
    evidence_handler = EvidenceHandler()

    shared_infra = {
        'audit_logger': audit_logger,
        'evidence_handler': evidence_handler
    }

    # Initialize agents
    print("Initializing agents...")
    osint = OSINTSynthesisAgent(**shared_infra)
    medical = MedicalForensicAnalystAgent(**shared_infra)
    legal = LegalAdvisorAgent(**shared_infra)
    gap_finder = EvidenceGapIdentifierAgent(**shared_infra)

    print("✓ 4 agents initialized\n")

    # Case details
    case_id = "EXAMPLE-2025-001"

    # Step 1: OSINT Analysis
    print("Step 1: OSINT Analysis")
    print("-" * 40)

    osint_result = osint.verify_claim(
        claim="Torture occurred at detention facility between Jan-Mar 2025",
        sources=[
            {
                'url': 'https://example.com/source1',
                'content': 'Social media post describing abuse...',
                'date': '2025-02-15'
            },
            {
                'url': 'https://example.com/source2',
                'content': 'News report mentioning conditions...',
                'date': '2025-03-01'
            }
        ],
        case_id=case_id
    )

    print(f"✓ OSINT analysis complete")
    print(f"  Evidence IDs tracked: {osint_result.get('_metadata', {}).get('evidence_ids', [])}")
    print()

    # Step 2: Medical Analysis
    print("Step 2: Medical Forensic Analysis")
    print("-" * 40)

    medical_result = medical.process({
        'message': '''Analyze this medical record for Istanbul Protocol indicators:

        Patient presents with:
        - Multiple contusions on torso and limbs
        - Bilateral wrist scarring consistent with restraints
        - Psychological trauma symptoms

        Date of examination: 2025-03-05
        ''',
        'case_id': case_id
    })

    print(f"✓ Medical analysis complete")
    print(f"  Audit events logged: {len(audit_logger.get_session_summary()['event_type_counts'])}")
    print()

    # Step 3: Legal Analysis
    print("Step 3: Legal Framework Analysis")
    print("-" * 40)

    legal_result = legal.process({
        'message': '''Based on the following evidence:
        - OSINT verification of torture claims (see prior analysis)
        - Medical evidence of torture (Istanbul Protocol indicators present)

        Question: What international law applies and what charges are supported?
        Jurisdiction: International Criminal Court
        ''',
        'case_id': case_id
    })

    print(f"✓ Legal analysis complete")
    print()

    # Step 4: Gap Analysis
    print("Step 4: Evidence Gap Analysis")
    print("-" * 40)

    gap_result = gap_finder.process({
        'message': '''Analyze evidence gaps for torture prosecution:

        Available evidence:
        - OSINT verification (medium credibility)
        - Medical record with Istanbul Protocol indicators
        - Legal analysis confirming applicable law

        Target charges: Torture (Article 7(1)(f) Rome Statute)
        ''',
        'case_id': case_id
    })

    print(f"✓ Gap analysis complete")
    print()

    # Verify Audit Trail
    print("=" * 60)
    print("Audit Trail Verification")
    print("=" * 60)

    # Check integrity
    integrity = audit_logger.verify_chain_integrity()
    print(f"Chain integrity verified: {integrity}")

    # Get session summary
    summary = audit_logger.get_session_summary()
    print(f"Total audit events: {summary['total_events']}")
    print(f"Agents involved: {', '.join(summary['agents'])}")
    print(f"Evidence items tracked: {summary['unique_evidence_items']}")

    print()
    print("=" * 60)
    print("Investigation Complete")
    print("=" * 60)

    print("""
Summary:
- 4 specialized agents collaborated
- Complete audit trail maintained
- Chain-of-custody verified
- Evidence gaps identified for next steps

Next Actions:
1. Review gap analysis recommendations
2. Collect additional evidence as needed
3. Prepare case file for prosecution
    """)

    return {
        'osint': osint_result,
        'medical': medical_result,
        'legal': legal_result,
        'gaps': gap_result,
        'audit_summary': summary,
        'integrity_verified': integrity
    }


if __name__ == "__main__":
    results = main()
    print(f"\n✓ Workflow complete. Integrity verified: {results['integrity_verified']}")
