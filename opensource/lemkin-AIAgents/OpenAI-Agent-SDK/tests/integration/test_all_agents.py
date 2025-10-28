"""
Integration Tests: All OpenAI SDK Agents

Tests that all 18 agents can be initialized and run basic operations.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import AuditLogger, EvidenceHandler


class TestAllAgents:
    """Test basic functionality of all agents."""

    @pytest.fixture
    def shared_infrastructure(self):
        """Create shared infrastructure for all agents."""
        return {
            'audit_logger': AuditLogger(),
            'evidence_handler': EvidenceHandler()
        }

    def test_osint_synthesis_agent(self, shared_infrastructure):
        """Test OSINT Synthesis Agent initialization and basic operation."""
        from agents.osint_synthesis.agent import OSINTSynthesisAgent

        agent = OSINTSynthesisAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Test OSINT analysis',
            'sources': [{'url': 'test', 'content': 'test'}],
            'case_id': 'TEST-001'
        })

        assert result is not None
        assert 'analysis' in result
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_satellite_imagery_analyst(self, shared_infrastructure):
        """Test Satellite Imagery Analyst initialization."""
        from agents.satellite_imagery_analyst.agent import SatelliteImageryAnalystAgent

        agent = SatelliteImageryAnalystAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Analyze satellite image for infrastructure',
            'case_id': 'TEST-002'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_social_media_harvester(self, shared_infrastructure):
        """Test Social Media Harvester initialization."""
        from agents.social_media_harvester.agent import SocialMediaHarvesterAgent

        agent = SocialMediaHarvesterAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Analyze social media post authenticity',
            'case_id': 'TEST-003'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_historical_researcher(self, shared_infrastructure):
        """Test Historical Researcher initialization."""
        from agents.historical_researcher.agent import HistoricalResearcherAgent

        agent = HistoricalResearcherAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Provide historical context for region',
            'case_id': 'TEST-004'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_legal_advisor(self, shared_infrastructure):
        """Test Legal Advisor initialization."""
        from agents.legal_advisor.agent import LegalAdvisorAgent

        agent = LegalAdvisorAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'What international law applies?',
            'case_id': 'TEST-005'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_document_parser(self, shared_infrastructure):
        """Test Document Parser initialization."""
        from agents.document_parser.agent import DocumentParserAgent

        agent = DocumentParserAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Parse this document',
            'case_id': 'TEST-006'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_comparative_analyzer(self, shared_infrastructure):
        """Test Comparative Analyzer initialization."""
        from agents.comparative_analyzer.agent import ComparativeAnalyzerAgent

        agent = ComparativeAnalyzerAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Compare two documents',
            'case_id': 'TEST-007'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_medical_forensic_analyst(self, shared_infrastructure):
        """Test Medical Forensic Analyst initialization."""
        from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent

        agent = MedicalForensicAnalystAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Analyze medical record for torture indicators',
            'case_id': 'TEST-008'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_evidence_gap_identifier(self, shared_infrastructure):
        """Test Evidence Gap Identifier initialization."""
        from agents.evidence_gap_identifier.agent import EvidenceGapIdentifierAgent

        agent = EvidenceGapIdentifierAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Identify evidence gaps',
            'case_id': 'TEST-009'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_torture_analyst(self, shared_infrastructure):
        """Test Torture Analyst initialization."""
        from agents.torture_analyst.agent import TortureAnalystAgent

        agent = TortureAnalystAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Analyze for torture evidence',
            'case_id': 'TEST-010'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_genocide_intent_analyzer(self, shared_infrastructure):
        """Test Genocide Intent Analyzer initialization."""
        from agents.genocide_intent_analyzer.agent import GenocideIntentAnalyzerAgent

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Analyze for genocidal intent',
            'case_id': 'TEST-011'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_disappearance_investigator(self, shared_infrastructure):
        """Test Disappearance Investigator initialization."""
        from agents.disappearance_investigator.agent import DisappearanceInvestigatorAgent

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Investigate disappearance case',
            'case_id': 'TEST-012'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_siege_starvation_analyst(self, shared_infrastructure):
        """Test Siege & Starvation Analyst initialization."""
        from agents.siege_starvation_analyst.agent import SiegeStarvationAnalystAgent

        agent = SiegeStarvationAnalystAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Analyze siege tactics',
            'case_id': 'TEST-013'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_digital_forensics_analyst(self, shared_infrastructure):
        """Test Digital Forensics Analyst initialization."""
        from agents.digital_forensics_analyst.agent import DigitalForensicsAnalystAgent

        agent = DigitalForensicsAnalystAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Extract digital metadata',
            'case_id': 'TEST-014'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_forensic_analysis_reviewer(self, shared_infrastructure):
        """Test Forensic Analysis Reviewer initialization."""
        from agents.forensic_analysis_reviewer.agent import ForensicAnalysisReviewerAgent

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Review forensic analysis',
            'case_id': 'TEST-015'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_ballistics_weapons_identifier(self, shared_infrastructure):
        """Test Ballistics & Weapons Identifier initialization."""
        from agents.ballistics_weapons_identifier.agent import BallisticsWeaponsIdentifierAgent

        agent = BallisticsWeaponsIdentifierAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Identify weapon type',
            'case_id': 'TEST-016'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_military_structure_analyst(self, shared_infrastructure):
        """Test Military Structure Analyst initialization."""
        from agents.military_structure_analyst.agent import MilitaryStructureAnalystAgent

        agent = MilitaryStructureAnalystAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Analyze command structure',
            'case_id': 'TEST-017'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_ngo_un_reporter(self, shared_infrastructure):
        """Test NGO & UN Reporter initialization."""
        from agents.ngo_un_reporter.agent import NGOUNReporterAgent

        agent = NGOUNReporterAgent(**shared_infrastructure)

        result = agent.process({
            'message': 'Format for UN submission',
            'case_id': 'TEST-018'
        })

        assert result is not None
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

    def test_audit_chain_verification(self, shared_infrastructure):
        """Test that audit chain remains intact across multiple agents."""
        from agents.document_parser.agent import DocumentParserAgent
        from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent

        parser = DocumentParserAgent(**shared_infrastructure)
        analyst = MedicalForensicAnalystAgent(**shared_infrastructure)

        # Run both agents
        parser_result = parser.process({
            'message': 'Parse document',
            'case_id': 'TEST-CHAIN'
        })

        analyst_result = analyst.process({
            'message': 'Analyze medical record',
            'case_id': 'TEST-CHAIN'
        })

        # Verify integrity maintained across agents
        assert shared_infrastructure['audit_logger'].verify_chain_integrity()

        # Check session summary
        summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert summary['total_events'] > 0
        assert summary['chain_integrity_verified'] is True
