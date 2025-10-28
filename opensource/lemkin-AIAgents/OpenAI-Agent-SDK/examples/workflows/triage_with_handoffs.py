"""
Triage Agent with Handoffs Example

Demonstrates OpenAI Agents SDK's built-in handoff feature where a triage
agent automatically delegates to specialized agents.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import LemkinAgent, AuditLogger, EvidenceHandler
from agents.document_parser.agent import DocumentParserAgent
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent
from agents.legal_advisor.agent import LegalAdvisorAgent


def main():
    """
    Triage agent example with automatic handoffs.
    """

    print("=" * 60)
    print("LemkinAI Triage Agent with Handoffs")
    print("OpenAI Agents SDK Built-in Feature")
    print("=" * 60)
    print()

    # Initialize shared infrastructure
    audit_logger = AuditLogger()
    evidence_handler = EvidenceHandler()

    shared_infra = {
        'audit_logger': audit_logger,
        'evidence_handler': evidence_handler
    }

    # Create specialized agents
    print("Creating specialized agents...")

    document_parser = DocumentParserAgent(**shared_infra)
    medical_analyst = MedicalForensicAnalystAgent(**shared_infra)
    legal_advisor = LegalAdvisorAgent(**shared_infra)

    print("✓ 3 specialist agents created")
    print()

    # Create triage agent with handoffs
    print("Creating triage agent with handoff capabilities...")

    triage_agent = LemkinAgent(
        agent_id="investigation_triage",
        name="Investigation Triage Agent",
        instructions="""
        You are an investigation triage agent that routes requests to appropriate specialists.

        Your role:
        - Analyze incoming evidence and requests
        - Determine which specialist should handle the case
        - Handoff to the appropriate agent

        Specialists available:
        - Document Parser: For PDFs, images, scanned documents that need parsing
        - Medical Forensic Analyst: For medical records, autopsy reports, torture evidence
        - Legal Advisor: For legal questions, jurisdiction issues, applicable law

        Always explain your reasoning before handing off.
        """,
        handoffs=[
            document_parser.agent.agent,  # Access underlying OpenAI Agent
            medical_analyst.agent.agent,
            legal_advisor.agent.agent
        ],
        audit_logger=audit_logger
    )

    print("✓ Triage agent created with 3 handoff targets")
    print()

    # Test Cases
    test_cases = [
        {
            "name": "Medical Record Analysis",
            "message": """I have a medical record from a detention facility that shows:
            - Multiple injuries consistent with beatings
            - Psychological trauma symptoms
            - Patient reports abuse by guards

            Please analyze for torture indicators.""",
            "expected_agent": "Medical Forensic Analyst"
        },
        {
            "name": "Legal Question",
            "message": """What international law applies to crimes committed during:
            - Armed conflict
            - Against civilians
            - In 2025
            - In Eastern Europe

            Need to determine jurisdiction.""",
            "expected_agent": "Legal Framework Advisor"
        },
        {
            "name": "Document Parsing",
            "message": """I have a scanned PDF document in Arabic that needs to be:
            - Parsed and translated
            - Key information extracted
            - Dates and names identified

            Can you help process this?""",
            "expected_agent": "Document Parser"
        }
    ]

    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 40)
        print(f"Request: {test_case['message'][:60]}...")
        print()

        result = triage_agent.run(
            message=test_case['message'],
            evidence_ids=[f"TEST-{i}"]
        )

        print(f"✓ Request handled")
        print(f"  Final agent: {result['final_agent']}")
        print(f"  Expected: {test_case['expected_agent']}")
        print(f"  Messages exchanged: {len(result.get('messages', []))}")
        print()

    # Verify audit trail
    print("=" * 60)
    print("Audit Trail Verification")
    print("=" * 60)

    integrity = audit_logger.verify_chain_integrity()
    summary = audit_logger.get_session_summary()

    print(f"Chain integrity verified: {integrity}")
    print(f"Total events: {summary['total_events']}")
    print(f"Agents involved: {', '.join(summary['agents'])}")
    print()

    print("=" * 60)
    print("Handoff Demonstration Complete")
    print("=" * 60)

    print("""
Key Features Demonstrated:
✓ Automatic agent handoffs based on request content
✓ Triage agent reasoning before delegation
✓ Complete audit trail across all agents
✓ Chain-of-custody maintained through handoffs

This pattern allows:
- Intelligent routing of investigation requests
- Specialization without manual orchestration
- Seamless collaboration between agents
- Full evidentiary compliance
    """)


if __name__ == "__main__":
    main()
