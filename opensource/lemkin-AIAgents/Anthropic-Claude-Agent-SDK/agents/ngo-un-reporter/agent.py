"""
NGO & UN Reporting Specialist Agent

Creates professional reports for international organizations and advocacy groups.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import BaseAgent, AuditLogger, AuditEventType, EvidenceHandler, OutputFormatter
from .system_prompt import SYSTEM_PROMPT
from .config import NGOUNReporterConfig, DEFAULT_CONFIG


class NGOUNReporterAgent(BaseAgent):
    """
    Agent for creating professional reports for NGOs and UN mechanisms.

    Handles:
    - UN mechanism submissions (UPR, treaty bodies, special procedures)
    - NGO documentation and advocacy materials
    - Professional report formatting and compliance
    - Legal framework integration and citation
    - Source protection and confidentiality protocols
    - Multi-audience adaptation and messaging
    """

    def __init__(
        self,
        config: Optional[NGOUNReporterConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize NGO & UN Reporting Specialist agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="ngo_un_reporter",
            system_prompt=SYSTEM_PROMPT,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            audit_logger=audit_logger,
            **kwargs
        )

        self.evidence_handler = evidence_handler or EvidenceHandler()

    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate professional reports for international organizations.

        Args:
            input_data: Dictionary containing:
                - evidence_materials: Evidence and documentation to analyze
                - report_type: Type of report to generate (UPR, shadow report, etc.)
                - target_audience: Intended audience for the report
                - submission_requirements: Specific formatting and content requirements
                - legal_frameworks: Relevant legal frameworks to apply
                - case_background: Background information and context
                - advocacy_objectives: Goals and messaging objectives
                - confidentiality_requirements: Source protection needs
                - word_limit: Maximum word count for report
                - deadline: Submission deadline
                - case_id: Optional case ID

        Returns:
            Professional report formatted for specified audience
        """
        evidence_materials = input_data.get('evidence_materials', {})
        report_type = input_data.get('report_type', 'general_report')
        target_audience = input_data.get('target_audience', 'ngo_networks')
        submission_requirements = input_data.get('submission_requirements', {})
        legal_frameworks = input_data.get('legal_frameworks', [])
        case_background = input_data.get('case_background', {})
        advocacy_objectives = input_data.get('advocacy_objectives', [])
        confidentiality_requirements = input_data.get('confidentiality_requirements', {})
        word_limit = input_data.get('word_limit')
        deadline = input_data.get('deadline')
        case_id = input_data.get('case_id')

        if not evidence_materials and not case_background:
            raise ValueError("Either evidence materials or case background required for report generation")

        # Generate report ID
        report_id = str(uuid.uuid4())

        # Log report generation (with confidentiality awareness)
        self.audit_logger.log_event(
            event_type=AuditEventType.REPORT_GENERATED,
            agent_id=self.agent_id,
            details={
                "report_id": report_id,
                "case_id": case_id,
                "report_type": report_type,
                "target_audience": target_audience,
                "word_limit": word_limit,
                "deadline": deadline,
                "confidentiality_protected": bool(confidentiality_requirements),
                "source_protection_enabled": self.config.protect_witness_identity
            }
        )

        # Generate professional report
        result = self._generate_professional_report(
            evidence_materials=evidence_materials,
            report_type=report_type,
            target_audience=target_audience,
            submission_requirements=submission_requirements,
            legal_frameworks=legal_frameworks,
            case_background=case_background,
            advocacy_objectives=advocacy_objectives,
            confidentiality_requirements=confidentiality_requirements,
            word_limit=word_limit,
            deadline=deadline,
            report_id=report_id,
            case_id=case_id
        )

        return self.generate_output(
            output_data=result,
            output_type="professional_report",
            evidence_ids=[report_id]
        )

    def _generate_professional_report(
        self,
        evidence_materials: Dict[str, Any],
        report_type: str,
        target_audience: str,
        submission_requirements: Dict[str, Any],
        legal_frameworks: List[str],
        case_background: Dict[str, Any],
        advocacy_objectives: List[str],
        confidentiality_requirements: Dict[str, Any],
        word_limit: Optional[int],
        deadline: Optional[str],
        report_id: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive professional report."""

        # Build report generation prompt
        report_prompt = self._build_report_generation_prompt(
            evidence_materials=evidence_materials,
            report_type=report_type,
            target_audience=target_audience,
            submission_requirements=submission_requirements,
            legal_frameworks=legal_frameworks,
            case_background=case_background,
            advocacy_objectives=advocacy_objectives,
            confidentiality_requirements=confidentiality_requirements,
            word_limit=word_limit,
            deadline=deadline,
            report_id=report_id,
            case_id=case_id
        )

        # Perform Claude analysis
        messages = [
            {
                "role": "user",
                "content": report_prompt
            }
        ]

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=messages
        )

        # Parse response
        response_text = response.content[0].text if response.content else ""

        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
            else:
                # Fallback if no JSON found
                result = {
                    "report_id": report_id,
                    "error": "Could not parse report generation results",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            result = {
                "report_id": report_id,
                "error": "Invalid JSON in report generation results",
                "raw_response": response_text
            }

        # Add metadata
        result["report_metadata"] = {
            "report_id": report_id,
            "generation_date": datetime.now().isoformat(),
            "generator": self.agent_id,
            "case_id": case_id,
            "report_type": report_type,
            "target_audience": target_audience,
            "word_limit": word_limit,
            "deadline": deadline,
            "configuration_used": {
                "languages_supported": self.config.supported_languages,
                "source_protection": self.config.protect_witness_identity,
                "fact_checking": self.config.fact_check_requirements,
                "legal_review": self.config.legal_review_needed,
                "diplomatic_protocols": self.config.follow_diplomatic_protocols,
                "accessibility_compliance": self.config.accessibility_compliance
            }
        }

        return result

    def _build_report_generation_prompt(
        self,
        evidence_materials: Dict[str, Any],
        report_type: str,
        target_audience: str,
        submission_requirements: Dict[str, Any],
        legal_frameworks: List[str],
        case_background: Dict[str, Any],
        advocacy_objectives: List[str],
        confidentiality_requirements: Dict[str, Any],
        word_limit: Optional[int],
        deadline: Optional[str],
        report_id: str,
        case_id: Optional[str] = None
    ) -> str:
        """Build the report generation prompt for Claude."""

        prompt = f"""Please generate a professional report for international organizations and advocacy groups.

REPORT METADATA:
- Report ID: {report_id}
- Generation Date: {datetime.now().isoformat()}
- Report Type: {report_type}
- Target Audience: {target_audience}"""

        if case_id:
            prompt += f"\n- Case ID: {case_id}"

        if word_limit:
            prompt += f"\n- Word Limit: {word_limit}"

        if deadline:
            prompt += f"\n- Submission Deadline: {deadline}"

        if submission_requirements:
            prompt += f"\n\nSUBMISSION REQUIREMENTS:\n{json.dumps(submission_requirements, indent=2)}"

        if legal_frameworks:
            prompt += f"\n\nLEGAL FRAMEWORKS TO APPLY: {', '.join(legal_frameworks)}"

        if case_background:
            prompt += f"\n\nCASE BACKGROUND:\n{json.dumps(case_background, indent=2)}"

        if evidence_materials:
            prompt += f"\n\nEVIDENCE MATERIALS:\n{json.dumps(evidence_materials, indent=2)}"

        if advocacy_objectives:
            prompt += f"\n\nADVOCACY OBJECTIVES: {', '.join(advocacy_objectives)}"

        if confidentiality_requirements:
            prompt += f"\n\nCONFIDENTIALITY REQUIREMENTS:\n{json.dumps(confidentiality_requirements, indent=2)}"

        prompt += f"""

REPORT GENERATION CONFIGURATION:
- UN mechanism submissions: {self.config.upr_submissions or self.config.treaty_body_reports or self.config.special_procedures_communications}
- NGO documentation: {self.config.shadow_reports or self.config.advocacy_materials or self.config.fact_finding_reports}
- Source protection: {self.config.protect_witness_identity}
- Fact checking: {self.config.fact_check_requirements}
- Legal review required: {self.config.legal_review_needed}
- Diplomatic protocols: {self.config.follow_diplomatic_protocols}
- Word limit compliance: {self.config.comply_with_word_limits}
- Official citation format: {self.config.use_official_citation_format}
- Executive summary: {self.config.include_executive_summary}
- Annexes: {self.config.create_annexes}
- Accessibility compliance: {self.config.accessibility_compliance}
- Supported languages: {', '.join(self.config.supported_languages)}

Please provide a comprehensive professional report following the JSON format in your system prompt. Focus on:

1. Executive summary with key findings and recommendations
2. Detailed factual narrative organized thematically or chronologically
3. Legal framework analysis and violation allegations
4. State obligation assessment and compliance evaluation
5. Evidence documentation with source protection measures
6. Clear, actionable recommendations for different stakeholders
7. Professional formatting that meets submission requirements
8. Appropriate annexes and supporting documentation
9. Quality assurance including factual accuracy and legal soundness
10. Advocacy effectiveness and strategic messaging considerations

Ensure the report maintains professional standards while protecting sources and meeting the specific requirements of the target audience and submission mechanism."""

        return prompt

    def generate_upr_submission(
        self,
        country_assessment: Dict[str, Any],
        legal_framework_analysis: Dict[str, Any],
        civil_society_input: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate Universal Periodic Review submission.

        Args:
            country_assessment: Assessment of country's human rights situation
            legal_framework_analysis: Analysis of legal frameworks and gaps
            civil_society_input: Input from civil society organizations
            case_id: Optional case ID

        Returns:
            UPR submission document
        """
        return self.process({
            'evidence_materials': {
                'country_assessment': country_assessment,
                'legal_analysis': legal_framework_analysis,
                'civil_society': civil_society_input
            },
            'report_type': 'upr_submission',
            'target_audience': 'un_mechanisms',
            'submission_requirements': {
                'word_limit': 2815,
                'format': 'stakeholder_report',
                'citation_style': 'un_official',
                'paragraph_numbering': True
            },
            'legal_frameworks': ['UDHR', 'ICCPR', 'ICESCR', 'CAT', 'CEDAW', 'CRC', 'CERD'],
            'advocacy_objectives': ['systematic_review', 'state_accountability', 'civil_society_engagement'],
            'case_id': case_id
        })

    def create_shadow_report(
        self,
        treaty_monitoring: str,
        state_report_analysis: Dict[str, Any],
        independent_evidence: Dict[str, Any],
        civil_society_documentation: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create shadow report for treaty body monitoring.

        Args:
            treaty_monitoring: Which treaty body (CCPR, CESCR, CAT, etc.)
            state_report_analysis: Analysis of state's official report
            independent_evidence: Independent evidence contradicting state claims
            civil_society_documentation: Documentation from civil society
            case_id: Optional case ID

        Returns:
            Shadow report for treaty body
        """
        return self.process({
            'evidence_materials': {
                'state_report': state_report_analysis,
                'independent_evidence': independent_evidence,
                'civil_society': civil_society_documentation
            },
            'report_type': 'shadow_report',
            'target_audience': 'treaty_bodies',
            'submission_requirements': {
                'treaty_body': treaty_monitoring,
                'format': 'alternative_report',
                'citation_style': 'international_law'
            },
            'legal_frameworks': [treaty_monitoring, 'customary_international_law'],
            'advocacy_objectives': ['state_accountability', 'treaty_compliance', 'victim_rights'],
            'case_id': case_id
        })

    def draft_special_procedures_communication(
        self,
        violation_allegations: Dict[str, Any],
        victim_information: Dict[str, Any],
        state_response_request: Dict[str, Any],
        urgency_level: str = "standard",
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Draft communication to UN Special Procedures.

        Args:
            violation_allegations: Specific human rights violations alleged
            victim_information: Information about victims (protected)
            state_response_request: Specific information requested from state
            urgency_level: Level of urgency (urgent, standard)
            case_id: Optional case ID

        Returns:
            Special Procedures communication
        """
        return self.process({
            'evidence_materials': {
                'violations': violation_allegations,
                'victims': victim_information,
                'state_request': state_response_request
            },
            'report_type': 'special_procedures_communication',
            'target_audience': 'special_procedures',
            'submission_requirements': {
                'urgency': urgency_level,
                'format': 'individual_communication',
                'response_timeline': '30_days' if urgency_level == 'standard' else '5_days'
            },
            'confidentiality_requirements': {
                'protect_victims': True,
                'anonymize_sources': True,
                'redact_locations': True
            },
            'advocacy_objectives': ['state_response', 'victim_protection', 'accountability'],
            'case_id': case_id
        })

    def create_advocacy_materials(
        self,
        campaign_objectives: List[str],
        target_audiences: List[str],
        key_messages: Dict[str, Any],
        evidence_highlights: Dict[str, Any],
        call_to_action: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create advocacy campaign materials.

        Args:
            campaign_objectives: Goals of advocacy campaign
            target_audiences: Intended audiences for materials
            key_messages: Core messages to communicate
            evidence_highlights: Key evidence to highlight
            call_to_action: Specific actions requested
            case_id: Optional case ID

        Returns:
            Advocacy campaign materials
        """
        return self.process({
            'evidence_materials': evidence_highlights,
            'report_type': 'advocacy_materials',
            'target_audience': 'multiple',
            'submission_requirements': {
                'audiences': target_audiences,
                'format': 'campaign_package',
                'accessibility': True
            },
            'advocacy_objectives': campaign_objectives,
            'case_background': {
                'key_messages': key_messages,
                'call_to_action': call_to_action
            },
            'case_id': case_id
        })

    def generate_press_release(
        self,
        news_event: Dict[str, Any],
        organizational_response: Dict[str, Any],
        quotes_and_statements: Dict[str, Any],
        contact_information: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate press release for media engagement.

        Args:
            news_event: Event or development being responded to
            organizational_response: Organization's position and response
            quotes_and_statements: Quotes from officials or experts
            contact_information: Media contact information
            case_id: Optional case ID

        Returns:
            Professional press release
        """
        return self.process({
            'case_background': news_event,
            'evidence_materials': {
                'response': organizational_response,
                'quotes': quotes_and_statements,
                'contacts': contact_information
            },
            'report_type': 'press_release',
            'target_audience': 'media_outlets',
            'submission_requirements': {
                'format': 'media_advisory',
                'length': 'one_page',
                'urgency': 'immediate'
            },
            'advocacy_objectives': ['media_engagement', 'public_awareness', 'pressure_response'],
            'case_id': case_id
        })

    def compile_fact_finding_report(
        self,
        investigation_findings: Dict[str, Any],
        witness_testimonies: Dict[str, Any],
        expert_analysis: Dict[str, Any],
        recommendations: Dict[str, Any],
        methodology: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compile comprehensive fact-finding mission report.

        Args:
            investigation_findings: Findings from investigation
            witness_testimonies: Protected witness testimonies
            expert_analysis: Expert analysis and opinions
            recommendations: Recommendations for action
            methodology: Investigation methodology used
            case_id: Optional case ID

        Returns:
            Comprehensive fact-finding report
        """
        return self.process({
            'evidence_materials': {
                'findings': investigation_findings,
                'testimonies': witness_testimonies,
                'expert_analysis': expert_analysis,
                'methodology': methodology
            },
            'case_background': {'recommendations': recommendations},
            'report_type': 'fact_finding_report',
            'target_audience': 'multiple',
            'submission_requirements': {
                'format': 'comprehensive_report',
                'annexes': True,
                'executive_summary': True
            },
            'confidentiality_requirements': {
                'protect_witnesses': True,
                'anonymize_testimonies': True,
                'secure_handling': True
            },
            'advocacy_objectives': ['documentation', 'accountability', 'victim_rights'],
            'case_id': case_id
        })