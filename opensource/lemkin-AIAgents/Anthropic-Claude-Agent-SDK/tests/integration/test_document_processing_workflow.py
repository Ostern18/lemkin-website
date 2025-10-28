"""
Integration Tests: Document Processing Workflow

Tests multi-agent workflows for complete document processing pipeline.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.document_parser.agent import DocumentParserAgent
from agents.comparative_analyzer.agent import ComparativeAnalyzerAgent
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent
from agents.evidence_gap_identifier.agent import EvidenceGapIdentifierAgent
from shared import AuditLogger, EvidenceHandler


class TestDocumentProcessingWorkflow:
    """Test complete document processing workflows."""

    @pytest.fixture
    def shared_infrastructure(self):
        """Create shared infrastructure for all agents."""
        audit_logger = AuditLogger()
        evidence_handler = EvidenceHandler()
        return {
            'audit_logger': audit_logger,
            'evidence_handler': evidence_handler
        }

    @pytest.fixture
    def all_agents(self, shared_infrastructure):
        """Create all agents with shared infrastructure."""
        return {
            'parser': DocumentParserAgent(**shared_infrastructure),
            'comparator': ComparativeAnalyzerAgent(**shared_infrastructure),
            'medical': MedicalForensicAnalystAgent(**shared_infrastructure),
            'gap_finder': EvidenceGapIdentifierAgent(**shared_infrastructure)
        }

    def test_single_document_workflow(self, all_agents, shared_infrastructure):
        """
        Test: Parse document -> Extract medical info -> Identify gaps

        Workflow:
        1. Parse medical report
        2. Analyze for medical findings
        3. Identify what's missing for torture charges
        """
        # Mock document data
        medical_report = b"Mock medical report PDF data"

        # Step 1: Parse document
        parser = all_agents['parser']
        parse_result = parser.process({
            'file_data': medical_report,
            'file_type': 'pdf',
            'source': 'Hospital',
            'case_id': 'TEST-001'
        })

        assert 'evidence_id' in parse_result
        evidence_id = parse_result['evidence_id']

        # Step 2: Medical analysis
        # In real scenario, we'd extract text from parse_result
        medical_analyst = all_agents['medical']
        medical_result = medical_analyst.process({
            'record_text': "Patient presents with multiple contusions...",
            'record_type': 'medical_report',
            'evidence_id': evidence_id,
            'case_id': 'TEST-001'
        })

        assert 'key_findings' in medical_result

        # Step 3: Gap analysis
        gap_finder = all_agents['gap_finder']
        gap_result = gap_finder.process({
            'charges': ['torture'],
            'available_evidence': [
                {'evidence_id': evidence_id, 'type': 'medical_report'}
            ],
            'case_id': 'TEST-001'
        })

        assert 'evidence_gaps' in gap_result
        assert 'priority_actions' in gap_result

        # Verify chain of custody
        chain = shared_infrastructure['evidence_handler'].get_chain_of_custody(evidence_id)
        assert len(chain) > 0  # Should have processing events

        # Verify audit integrity
        integrity = shared_infrastructure['audit_logger'].verify_chain_integrity()
        assert integrity is True

    def test_version_comparison_workflow(self, all_agents):
        """
        Test: Parse two documents -> Compare versions -> Identify suspicious changes

        Workflow:
        1. Parse original contract
        2. Parse modified contract
        3. Compare to find differences
        4. Flag suspicious modifications
        """
        parser = all_agents['parser']
        comparator = all_agents['comparator']

        # Parse original
        original_result = parser.process({
            'file_data': b"Original contract PDF",
            'file_type': 'pdf',
            'source': 'Legal Archive',
            'tags': ['contract', 'original']
        })

        # Parse modified
        modified_result = parser.process({
            'file_data': b"Modified contract PDF",
            'file_type': 'pdf',
            'source': 'Defendant',
            'tags': ['contract', 'modified']
        })

        # Compare versions
        # In real scenario, we'd use extracted text
        comparison_result = comparator.compare_versions(
            original_doc={'text': 'Original text...', 'evidence_id': original_result['evidence_id']},
            modified_doc={'text': 'Modified text...', 'evidence_id': modified_result['evidence_id']}
        )

        assert 'comparison_results' in comparison_result
        assert 'differences_identified' in comparison_result
        assert 'red_flags' in comparison_result

    def test_multiple_medical_records_workflow(self, all_agents):
        """
        Test: Parse multiple medical records -> Analyze each -> Compare for consistency

        Workflow:
        1. Parse 3 medical records from same patient
        2. Analyze each for medical findings
        3. Compare for timeline consistency
        4. Identify gaps for torture case
        """
        parser = all_agents['parser']
        medical_analyst = all_agents['medical']
        comparator = all_agents['comparator']
        gap_finder = all_agents['gap_finder']

        # Parse multiple records
        records = []
        for i in range(3):
            result = parser.process({
                'file_data': f"Medical record {i+1}".encode(),
                'file_type': 'pdf',
                'source': f'Hospital Visit {i+1}',
                'tags': ['medical', 'patient-123']
            })
            records.append(result)

        # Analyze each
        medical_analyses = []
        for record in records:
            analysis = medical_analyst.process({
                'record_text': f"Mock medical text for {record['evidence_id']}",
                'record_type': 'medical_report',
                'evidence_id': record['evidence_id']
            })
            medical_analyses.append(analysis)

        # Compare for consistency
        comparison = comparator.process({
            'documents': [
                {'text': f"Record {i}", 'evidence_id': r['evidence_id']}
                for i, r in enumerate(records)
            ],
            'comparison_type': 'multi_document_similarity'
        })

        # Gap analysis
        gaps = gap_finder.process({
            'charges': ['torture'],
            'available_evidence': [
                {'evidence_id': r['evidence_id'], 'type': 'medical_report'}
                for r in records
            ]
        })

        assert len(medical_analyses) == 3
        assert 'comparison_results' in comparison
        assert 'evidence_gaps' in gaps

    def test_complete_investigation_workflow(self, all_agents, shared_infrastructure):
        """
        Test: Complete investigation from document ingestion to gap identification

        Workflow:
        1. Ingest multiple evidence types
        2. Parse and analyze each
        3. Compare related documents
        4. Perform gap analysis
        5. Generate action plan
        6. Verify complete audit trail
        """
        parser = all_agents['parser']
        medical_analyst = all_agents['medical']
        comparator = all_agents['comparator']
        gap_finder = all_agents['gap_finder']

        case_id = 'TEST-COMPLETE-001'
        evidence_items = []

        # 1. Ingest evidence
        # Medical report
        med_report = parser.process({
            'file_data': b"Medical examination report",
            'file_type': 'pdf',
            'source': 'Hospital',
            'case_id': case_id,
            'tags': ['medical']
        })
        evidence_items.append(med_report)

        # Witness statement
        witness_stmt = parser.process({
            'file_data': b"Witness statement",
            'file_type': 'pdf',
            'source': 'Police',
            'case_id': case_id,
            'tags': ['witness', 'statement']
        })
        evidence_items.append(witness_stmt)

        # Detention order
        detention_order = parser.process({
            'file_data': b"Detention order",
            'file_type': 'pdf',
            'source': 'Military',
            'case_id': case_id,
            'tags': ['official', 'detention']
        })
        evidence_items.append(detention_order)

        # 2. Specialized analysis
        medical_analysis = medical_analyst.analyze_for_torture(
            medical_record="Mock medical examination with injury documentation",
            case_id=case_id
        )

        # 3. Compare witness statement with detention records
        comparison = comparator.detect_patterns(
            documents=[
                {'text': 'Witness statement text', 'evidence_id': witness_stmt['evidence_id']},
                {'text': 'Detention order text', 'evidence_id': detention_order['evidence_id']}
            ]
        )

        # 4. Comprehensive gap analysis
        gap_analysis = gap_finder.process({
            'charges': ['torture', 'unlawful_detention'],
            'available_evidence': [
                {'evidence_id': e['evidence_id'], 'description': 'Evidence item'}
                for e in evidence_items
            ],
            'case_theory': 'Systematic torture during unlawful detention',
            'case_id': case_id
        })

        # 5. Verify results
        assert len(evidence_items) == 3
        assert 'torture_indicators' in medical_analysis
        assert 'patterns_detected' in comparison
        assert 'priority_actions' in gap_analysis
        assert 'critical_next_steps' in gap_analysis

        # 6. Verify audit trail
        session_summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert session_summary['total_events'] > 10  # Multiple processing events
        assert session_summary['chain_integrity_verified'] is True
        assert session_summary['unique_evidence_items'] >= 3

        # Verify each evidence item has chain of custody
        for evidence in evidence_items:
            chain = shared_infrastructure['evidence_handler'].get_chain_of_custody(
                evidence['evidence_id']
            )
            assert len(chain) > 0


@pytest.mark.integration
class TestAgentInteroperability:
    """Test that agents work together correctly."""

    def test_shared_audit_logger(self):
        """Test that all agents can share same audit logger."""
        shared_audit = AuditLogger()

        parser = DocumentParserAgent(audit_logger=shared_audit)
        medical = MedicalForensicAnalystAgent(audit_logger=shared_audit)

        # Both agents should use same logger
        assert parser.audit_logger is shared_audit
        assert medical.audit_logger is shared_audit

        # Verify session summary includes both agents
        summary = shared_audit.get_session_summary()
        assert summary['unique_agents'] >= 2

    def test_shared_evidence_handler(self):
        """Test that all agents can share same evidence handler."""
        shared_evidence = EvidenceHandler()

        parser = DocumentParserAgent(evidence_handler=shared_evidence)
        comparator = ComparativeAnalyzerAgent(evidence_handler=shared_evidence)

        assert parser.evidence_handler is shared_evidence
        assert comparator.evidence_handler is shared_evidence

    def test_evidence_portability(self):
        """Test that evidence from one agent can be used by another."""
        shared_infra = {
            'audit_logger': AuditLogger(),
            'evidence_handler': EvidenceHandler()
        }

        parser = DocumentParserAgent(**shared_infra)
        medical = MedicalForensicAnalystAgent(**shared_infra)

        # Parser creates evidence
        parse_result = parser.process({
            'file_data': b"Medical report",
            'file_type': 'pdf',
            'source': 'Hospital'
        })

        evidence_id = parse_result['evidence_id']

        # Medical analyst can access same evidence
        metadata = shared_infra['evidence_handler'].get_metadata(evidence_id)
        assert metadata is not None
        assert metadata.source == 'Hospital'

        # Medical analyst can process it
        medical_result = medical.process({
            'record_text': 'Medical text',
            'evidence_id': evidence_id
        })

        assert medical_result['evidence_id'] == evidence_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
