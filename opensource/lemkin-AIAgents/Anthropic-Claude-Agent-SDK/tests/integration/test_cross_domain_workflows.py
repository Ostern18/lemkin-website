"""
Integration Tests: Cross-Domain Workflows

Tests combining Domain 1 (Investigative Research) with Domain 2 (Document Processing)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.document_parser.agent import DocumentParserAgent
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent
from agents.evidence_gap_identifier.agent import EvidenceGapIdentifierAgent
from agents.osint_synthesis.agent import OSINTSynthesisAgent
from agents.satellite_imagery_analyst.agent import SatelliteImageryAnalystAgent
from shared import AuditLogger, EvidenceHandler


class TestCrossDomainWorkflows:
    """Test workflows combining multiple agent domains."""

    @pytest.fixture
    def shared_infrastructure(self):
        """Create shared infrastructure."""
        return {
            'audit_logger': AuditLogger(),
            'evidence_handler': EvidenceHandler()
        }

    @pytest.fixture
    def all_agents(self, shared_infrastructure):
        """Create all agents with shared infrastructure."""
        return {
            # Domain 2: Document Processing
            'parser': DocumentParserAgent(**shared_infrastructure),
            'medical': MedicalForensicAnalystAgent(**shared_infrastructure),
            'gap_finder': EvidenceGapIdentifierAgent(**shared_infrastructure),

            # Domain 1: Investigative Research
            'osint': OSINTSynthesisAgent(**shared_infrastructure),
            'imagery': SatelliteImageryAnalystAgent(**shared_infrastructure)
        }

    def test_osint_to_document_workflow(self, all_agents, shared_infrastructure):
        """
        Test: OSINT finding leads to document analysis

        Workflow:
        1. OSINT agent finds social media claims about torture
        2. Verify claims against parsed medical documents
        3. Identify evidence gaps
        """
        osint = all_agents['osint']
        parser = all_agents['parser']
        gap_finder = all_agents['gap_finder']

        # Step 1: OSINT monitoring finds claims
        osint_result = osint.verify_claim(
            claim="Detainees at Facility X were tortured on January 15",
            context="Social media reports from multiple sources",
            case_id="CROSS-001"
        )

        assert 'executive_summary' in osint_result

        # Step 2: Parse medical document related to claim
        medical_doc = b"Medical report documenting injuries"
        parse_result = parser.process({
            'file_data': medical_doc,
            'file_type': 'pdf',
            'source': 'Hospital',
            'case_id': 'CROSS-001',
            'tags': ['medical', 'torture-allegation']
        })

        # Step 3: Gap analysis combining OSINT and document evidence
        gap_result = gap_finder.process({
            'charges': ['torture'],
            'available_evidence': [
                {
                    'type': 'osint_report',
                    'summary': osint_result.get('executive_summary', 'OSINT findings')
                },
                {
                    'type': 'medical_report',
                    'evidence_id': parse_result['evidence_id']
                }
            ],
            'case_id': 'CROSS-001'
        })

        assert 'evidence_gaps' in gap_result
        assert 'priority_actions' in gap_result

        # Verify complete chain of custody
        session_summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert session_summary['unique_agents'] >= 3

    def test_imagery_to_medical_workflow(self, all_agents, shared_infrastructure):
        """
        Test: Satellite imagery analysis combined with medical analysis

        Workflow:
        1. Analyze satellite imagery of detention facility
        2. Parse medical records from detainees
        3. Cross-reference findings for consistency
        """
        imagery = all_agents['imagery']
        medical = all_agents['medical']
        gap_finder = all_agents['gap_finder']

        # Step 1: Analyze satellite imagery
        mock_satellite_image = b"Mock satellite image data"

        imagery_result = imagery.assess_site(
            image_data=mock_satellite_image,
            site_type='detention_facility',
            case_id='CROSS-002'
        )

        assert 'summary' in imagery_result

        # Step 2: Medical analysis
        medical_result = medical.analyze_for_torture(
            medical_record="Patient shows injuries consistent with detention abuse...",
            case_id='CROSS-002'
        )

        # Step 3: Gap analysis with both types of evidence
        gap_result = gap_finder.process({
            'charges': ['unlawful_detention', 'torture'],
            'available_evidence': [
                {
                    'type': 'satellite_imagery',
                    'summary': 'Detention facility identified'
                },
                {
                    'type': 'medical_report',
                    'summary': 'Torture indicators present'
                }
            ],
            'case_id': 'CROSS-002'
        })

        assert 'priority_actions' in gap_result

    def test_complete_investigation_workflow(self, all_agents, shared_infrastructure):
        """
        Test: Complete multi-domain investigation

        Workflow:
        1. OSINT monitoring detects event
        2. Satellite imagery confirms location
        3. Parse witness documents
        4. Medical analysis of injuries
        5. Comprehensive gap analysis
        """
        osint = all_agents['osint']
        imagery = all_agents['imagery']
        parser = all_agents['parser']
        medical = all_agents['medical']
        gap_finder = all_agents['gap_finder']

        case_id = 'CROSS-COMPLETE-001'
        evidence_summary = []

        # Step 1: OSINT - Event detection
        osint_result = osint.analyze_event(
            event_description="Detention facility raid",
            location="City Center",
            date="2024-01-15",
            case_id=case_id
        )

        evidence_summary.append({
            'type': 'osint_report',
            'summary': osint_result.get('executive_summary', 'Event detected via OSINT')
        })

        # Step 2: Imagery - Location verification
        mock_image = b"Mock aerial image"
        imagery_result = imagery.assess_site(
            image_data=mock_image,
            site_type='detention_facility',
            case_id=case_id
        )

        evidence_summary.append({
            'type': 'satellite_imagery',
            'summary': 'Facility confirmed via imagery'
        })

        # Step 3: Document parsing - Witness statement
        witness_doc = b"Witness statement document"
        parse_result = parser.process({
            'file_data': witness_doc,
            'file_type': 'pdf',
            'source': 'Legal Aid',
            'case_id': case_id,
            'tags': ['witness', 'statement']
        })

        evidence_summary.append({
            'type': 'witness_statement',
            'evidence_id': parse_result['evidence_id']
        })

        # Step 4: Medical analysis
        medical_result = medical.analyze_for_torture(
            medical_record="Medical examination findings...",
            case_id=case_id
        )

        evidence_summary.append({
            'type': 'medical_report',
            'summary': 'Medical evidence analyzed'
        })

        # Step 5: Comprehensive gap analysis
        gap_result = gap_finder.process({
            'charges': ['unlawful_detention', 'torture', 'assault'],
            'available_evidence': evidence_summary,
            'case_theory': 'Systematic abuse at detention facility',
            'case_id': case_id
        })

        # Verify results
        assert len(evidence_summary) == 4
        assert 'evidence_gaps' in gap_result
        assert 'priority_actions' in gap_result
        assert 'critical_next_steps' in gap_result

        # Verify all agents logged to shared audit
        session = shared_infrastructure['audit_logger'].get_session_summary()
        assert session['unique_agents'] >= 5
        assert session['total_events'] > 10
        assert session['chain_integrity_verified'] is True

    def test_osint_verification_with_documents(self, all_agents):
        """
        Test: Use documents to verify OSINT claims
        """
        osint = all_agents['osint']
        parser = all_agents['parser']

        # OSINT finds claim
        osint_result = osint.monitor_keywords(
            keywords=["detention", "abuse"],
            time_period_days=7
        )

        # Parse official document that might verify/contradict
        doc = b"Official report"
        parse_result = parser.process({
            'file_data': doc,
            'file_type': 'pdf',
            'source': 'Government',
            'tags': ['official', 'report']
        })

        # Both results available for cross-referencing
        assert 'executive_summary' in osint_result
        assert 'evidence_id' in parse_result


@pytest.mark.integration
class TestAgentInteroperabilityExtended:
    """Extended interoperability tests."""

    def test_all_agents_share_infrastructure(self):
        """Test all 6 agents can share same infrastructure."""
        shared_audit = AuditLogger()
        shared_evidence = EvidenceHandler()

        agents = [
            DocumentParserAgent(audit_logger=shared_audit, evidence_handler=shared_evidence),
            MedicalForensicAnalystAgent(audit_logger=shared_audit, evidence_handler=shared_evidence),
            EvidenceGapIdentifierAgent(audit_logger=shared_audit, evidence_handler=shared_evidence),
            OSINTSynthesisAgent(audit_logger=shared_audit, evidence_handler=shared_evidence),
            SatelliteImageryAnalystAgent(audit_logger=shared_audit, evidence_handler=shared_evidence),
        ]

        # All agents should use same infrastructure
        for agent in agents:
            assert agent.audit_logger is shared_audit
            assert agent.evidence_handler is shared_evidence

    def test_evidence_type_support(self):
        """Test that new evidence types are supported."""
        from shared.evidence_handler import EvidenceType

        # Verify new types exist
        assert hasattr(EvidenceType, 'OSINT_REPORT')
        assert hasattr(EvidenceType, 'SATELLITE_IMAGERY')
        assert hasattr(EvidenceType, 'INTELLIGENCE_BRIEF')
        assert hasattr(EvidenceType, 'IMAGERY_ANALYSIS')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
