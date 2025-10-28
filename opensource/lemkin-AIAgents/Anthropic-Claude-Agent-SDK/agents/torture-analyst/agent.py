"""
Torture & Ill-Treatment Analyst Agent

Documents and analyzes torture evidence according to international standards.
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
from .config import TortureAnalystConfig, DEFAULT_CONFIG


class TortureAnalystAgent(BaseAgent):
    """
    Agent for analyzing torture and ill-treatment evidence according to international standards.

    Handles:
    - Istanbul Protocol application for torture documentation
    - Legal element analysis for torture definitions
    - Medical evidence interpretation and consistency assessment
    - Pattern recognition and systematic torture analysis
    - Detention conditions evaluation
    - Perpetrator analysis and command responsibility assessment
    """

    def __init__(
        self,
        config: Optional[TortureAnalystConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Torture & Ill-Treatment Analyst agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="torture_analyst",
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
        Analyze torture and ill-treatment evidence.

        Args:
            input_data: Dictionary containing:
                - medical_evidence: Medical reports, examinations, expert opinions
                - witness_testimony: Victim and witness statements
                - detention_conditions: Information about detention conditions
                - alleged_methods: Specific torture methods alleged
                - perpetrator_information: Information about alleged perpetrators
                - case_id: Optional case ID
                - victim_id: Anonymized victim identifier
                - analysis_focus: Specific aspects to focus on

        Returns:
            Comprehensive torture analysis according to international standards
        """
        medical_evidence = input_data.get('medical_evidence', {})
        witness_testimony = input_data.get('witness_testimony', {})
        detention_conditions = input_data.get('detention_conditions', {})
        alleged_methods = input_data.get('alleged_methods', [])
        perpetrator_information = input_data.get('perpetrator_information', {})
        case_id = input_data.get('case_id')
        victim_id = input_data.get('victim_id', 'anonymous')
        analysis_focus = input_data.get('analysis_focus', [])

        if not medical_evidence and not witness_testimony:
            raise ValueError("Either medical evidence or witness testimony required for torture analysis")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Log torture analysis (with special sensitivity)
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            details={
                "analysis_id": analysis_id,
                "case_id": case_id,
                "victim_id": victim_id,
                "analysis_type": "torture_assessment",
                "has_medical_evidence": bool(medical_evidence),
                "has_witness_testimony": bool(witness_testimony),
                "methods_analyzed": len(alleged_methods),
                "confidentiality_maintained": self.config.maintain_confidentiality
            }
        )

        # Perform torture analysis
        result = self._conduct_torture_analysis(
            medical_evidence=medical_evidence,
            witness_testimony=witness_testimony,
            detention_conditions=detention_conditions,
            alleged_methods=alleged_methods,
            perpetrator_information=perpetrator_information,
            analysis_focus=analysis_focus,
            analysis_id=analysis_id,
            victim_id=victim_id,
            case_id=case_id
        )

        return self.generate_output(
            output_data=result,
            output_type="torture_analysis_report",
            evidence_ids=[]
        )

    def _conduct_torture_analysis(
        self,
        medical_evidence: Dict[str, Any],
        witness_testimony: Dict[str, Any],
        detention_conditions: Dict[str, Any],
        alleged_methods: List[str],
        perpetrator_information: Dict[str, Any],
        analysis_focus: List[str],
        analysis_id: str,
        victim_id: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive torture analysis according to international standards."""

        # Build torture analysis prompt
        analysis_prompt = self._build_torture_analysis_prompt(
            medical_evidence=medical_evidence,
            witness_testimony=witness_testimony,
            detention_conditions=detention_conditions,
            alleged_methods=alleged_methods,
            perpetrator_information=perpetrator_information,
            analysis_focus=analysis_focus,
            analysis_id=analysis_id,
            victim_id=victim_id,
            case_id=case_id
        )

        # Perform Claude analysis
        messages = [
            {
                "role": "user",
                "content": analysis_prompt
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
                    "torture_analysis_id": analysis_id,
                    "error": "Could not parse torture analysis results",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            result = {
                "torture_analysis_id": analysis_id,
                "error": "Invalid JSON in torture analysis results",
                "raw_response": response_text
            }

        # Add metadata
        result["analysis_metadata"] = {
            "torture_analysis_id": analysis_id,
            "analysis_date": datetime.now().isoformat(),
            "analyst": self.agent_id,
            "case_id": case_id,
            "victim_identifier": victim_id,
            "analysis_scope": "individual_case",
            "standards_applied": self.config.legal_frameworks,
            "configuration_used": {
                "istanbul_protocol_applied": self.config.apply_istanbul_protocol,
                "medical_analysis": self.config.assess_medical_evidence,
                "systematic_analysis": self.config.identify_systematic_patterns,
                "command_responsibility": self.config.assess_command_responsibility,
                "victim_centered": self.config.prioritize_victim_welfare
            }
        }

        return result

    def _build_torture_analysis_prompt(
        self,
        medical_evidence: Dict[str, Any],
        witness_testimony: Dict[str, Any],
        detention_conditions: Dict[str, Any],
        alleged_methods: List[str],
        perpetrator_information: Dict[str, Any],
        analysis_focus: List[str],
        analysis_id: str,
        victim_id: str,
        case_id: Optional[str] = None
    ) -> str:
        """Build the torture analysis prompt for Claude."""

        prompt = f"""Please conduct comprehensive torture and ill-treatment analysis according to international standards.

ANALYSIS METADATA:
- Analysis ID: {analysis_id}
- Analysis Date: {datetime.now().isoformat()}
- Victim ID: {victim_id}"""

        if case_id:
            prompt += f"\n- Case ID: {case_id}"

        if medical_evidence:
            prompt += f"\n\nMEDICAL EVIDENCE:\n{json.dumps(medical_evidence, indent=2)}"

        if witness_testimony:
            prompt += f"\n\nWITNESS TESTIMONY:\n{json.dumps(witness_testimony, indent=2)}"

        if detention_conditions:
            prompt += f"\n\nDETENTION CONDITIONS:\n{json.dumps(detention_conditions, indent=2)}"

        if alleged_methods:
            prompt += f"\n\nALLEGED TORTURE METHODS: {', '.join(alleged_methods)}"

        if perpetrator_information:
            prompt += f"\n\nPERPETRATOR INFORMATION:\n{json.dumps(perpetrator_information, indent=2)}"

        if analysis_focus:
            prompt += f"\n\nANALYSIS FOCUS: {', '.join(analysis_focus)}"

        prompt += f"""

ANALYSIS CONFIGURATION:
- Apply Istanbul Protocol: {self.config.apply_istanbul_protocol}
- Analyze legal elements: {self.config.analyze_legal_elements}
- Assess medical evidence: {self.config.assess_medical_evidence}
- Evaluate detention conditions: {self.config.evaluate_detention_conditions}
- Analyze perpetrator responsibility: {self.config.analyze_perpetrator_responsibility}
- Identify systematic patterns: {self.config.identify_systematic_patterns}
- Legal frameworks: {', '.join(self.config.legal_frameworks)}
- Assess command responsibility: {self.config.assess_command_responsibility}
- Evaluate state responsibility: {self.config.evaluate_state_responsibility}
- Prioritize victim welfare: {self.config.prioritize_victim_welfare}
- Minimum medical confidence: {self.config.min_medical_confidence}
- Minimum evidence confidence: {self.config.min_evidence_confidence}

Please provide comprehensive torture analysis following the JSON format in your system prompt. Focus on:

1. Legal classification according to international definitions (CAT Article 1)
2. Istanbul Protocol application and medical evidence assessment
3. Torture methods analysis and consistency evaluation
4. Detention conditions assessment against international standards
5. Perpetrator analysis and command responsibility assessment
6. Pattern and systematic analysis if applicable
7. Witness testimony evaluation and corroboration
8. Legal implications and victim rights
9. Evidence quality assessment and recommendations

Ensure all analysis maintains victim sensitivity while meeting rigorous international standards for torture documentation."""

        return prompt

    def assess_individual_case(
        self,
        medical_reports: List[Dict[str, Any]],
        victim_statement: str,
        alleged_methods: List[str],
        case_id: Optional[str] = None,
        victim_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess individual torture case.

        Args:
            medical_reports: Medical examination reports
            victim_statement: Victim's testimony about torture
            alleged_methods: Specific torture methods alleged
            case_id: Optional case ID
            victim_id: Optional victim identifier

        Returns:
            Individual torture case analysis
        """
        return self.process({
            'medical_evidence': {'reports': medical_reports},
            'witness_testimony': {'victim_statement': victim_statement},
            'alleged_methods': alleged_methods,
            'analysis_focus': ['individual_assessment', 'istanbul_protocol', 'legal_elements'],
            'case_id': case_id,
            'victim_id': victim_id or 'individual_case'
        })

    def analyze_systematic_torture(
        self,
        multiple_cases: List[Dict[str, Any]],
        institutional_information: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze patterns of systematic torture.

        Args:
            multiple_cases: Multiple individual torture cases
            institutional_information: Information about institutions involved
            case_id: Optional case ID

        Returns:
            Systematic torture pattern analysis
        """
        return self.process({
            'medical_evidence': {'multiple_cases': multiple_cases},
            'perpetrator_information': institutional_information,
            'analysis_focus': ['systematic_analysis', 'pattern_recognition', 'institutional_practices'],
            'case_id': case_id,
            'victim_id': 'systematic_analysis'
        })

    def evaluate_detention_conditions(
        self,
        facility_conditions: Dict[str, Any],
        witness_accounts: List[str],
        duration_of_detention: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate detention conditions for torture/ill-treatment.

        Args:
            facility_conditions: Description of facility conditions
            witness_accounts: Witness accounts of conditions
            duration_of_detention: How long victims were detained
            case_id: Optional case ID

        Returns:
            Detention conditions analysis
        """
        return self.process({
            'detention_conditions': {
                'facility_conditions': facility_conditions,
                'duration': duration_of_detention
            },
            'witness_testimony': {'accounts': witness_accounts},
            'analysis_focus': ['detention_conditions', 'international_standards', 'cumulative_impact'],
            'case_id': case_id,
            'victim_id': 'detention_conditions_analysis'
        })

    def assess_command_responsibility(
        self,
        torture_incidents: List[Dict[str, Any]],
        command_structure: Dict[str, Any],
        superior_knowledge: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess command responsibility for torture.

        Args:
            torture_incidents: Individual torture incidents
            command_structure: Military/police command structure
            superior_knowledge: Evidence of superior knowledge
            case_id: Optional case ID

        Returns:
            Command responsibility analysis
        """
        return self.process({
            'medical_evidence': {'incidents': torture_incidents},
            'perpetrator_information': {
                'command_structure': command_structure,
                'superior_knowledge': superior_knowledge
            },
            'analysis_focus': ['command_responsibility', 'superior_liability', 'institutional_responsibility'],
            'case_id': case_id,
            'victim_id': 'command_responsibility_analysis'
        })

    def generate_expert_opinion(
        self,
        medical_evidence: Dict[str, Any],
        legal_question: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate expert opinion for legal proceedings.

        Args:
            medical_evidence: Medical evidence to analyze
            legal_question: Specific legal question to address
            case_id: Optional case ID

        Returns:
            Expert opinion suitable for legal proceedings
        """
        return self.process({
            'medical_evidence': medical_evidence,
            'analysis_focus': ['expert_opinion', 'legal_standards', 'medical_consistency'],
            'case_id': case_id,
            'victim_id': 'expert_opinion'
        })

    def compare_torture_methods(
        self,
        cases_data: List[Dict[str, Any]],
        analysis_scope: str = "institutional",
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare torture methods across multiple cases.

        Args:
            cases_data: Multiple torture cases with method information
            analysis_scope: Scope of comparison analysis
            case_id: Optional case ID

        Returns:
            Comparative analysis of torture methods
        """
        return self.process({
            'medical_evidence': {'cases': cases_data},
            'analysis_focus': ['method_comparison', 'pattern_analysis', 'training_indicators'],
            'case_id': case_id,
            'victim_id': 'method_comparison_analysis'
        })