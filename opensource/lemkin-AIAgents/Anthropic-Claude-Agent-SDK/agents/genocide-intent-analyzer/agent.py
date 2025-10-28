"""
Genocide Intent Analyzer Agent

Evaluates evidence of genocidal intent according to international legal standards.
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
from .config import GenocideIntentAnalyzerConfig, DEFAULT_CONFIG


class GenocideIntentAnalyzerAgent(BaseAgent):
    """
    Agent for analyzing genocide intent according to international legal standards.

    Handles:
    - Intent evidence analysis (direct and circumstantial)
    - Protected group identification and targeting pattern assessment
    - Contextual evidence evaluation and historical analysis
    - Precedent and jurisprudence application
    - Perpetrator intent assessment and institutional analysis
    - Prevention risk assessment and recommendations
    """

    def __init__(
        self,
        config: Optional[GenocideIntentAnalyzerConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Genocide Intent Analyzer agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="genocide_intent_analyzer",
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
        Analyze evidence for genocidal intent.

        Args:
            input_data: Dictionary containing:
                - direct_evidence: Statements, documents, orders showing intent
                - circumstantial_evidence: Targeting patterns, scale, discrimination
                - protected_groups: Groups allegedly targeted for destruction
                - geographic_scope: Areas where alleged genocide occurred
                - temporal_scope: Time period of alleged genocide
                - perpetrator_information: Information about alleged perpetrators
                - contextual_information: Historical, political, social context
                - case_id: Optional case ID
                - analysis_focus: Specific aspects to focus on

        Returns:
            Comprehensive genocide intent analysis
        """
        direct_evidence = input_data.get('direct_evidence', {})
        circumstantial_evidence = input_data.get('circumstantial_evidence', {})
        protected_groups = input_data.get('protected_groups', [])
        geographic_scope = input_data.get('geographic_scope', [])
        temporal_scope = input_data.get('temporal_scope', {})
        perpetrator_information = input_data.get('perpetrator_information', {})
        contextual_information = input_data.get('contextual_information', {})
        case_id = input_data.get('case_id')
        analysis_focus = input_data.get('analysis_focus', [])

        if not direct_evidence and not circumstantial_evidence:
            raise ValueError("Either direct or circumstantial evidence required for genocide intent analysis")

        if not protected_groups:
            raise ValueError("Protected groups must be specified for genocide analysis")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Log genocide intent analysis (with special sensitivity)
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            details={
                "analysis_id": analysis_id,
                "case_id": case_id,
                "analysis_type": "genocide_intent_assessment",
                "protected_groups_count": len(protected_groups),
                "geographic_scope": geographic_scope,
                "has_direct_evidence": bool(direct_evidence),
                "has_circumstantial_evidence": bool(circumstantial_evidence),
                "high_sensitivity_analysis": True
            }
        )

        # Perform genocide intent analysis
        result = self._conduct_genocide_intent_analysis(
            direct_evidence=direct_evidence,
            circumstantial_evidence=circumstantial_evidence,
            protected_groups=protected_groups,
            geographic_scope=geographic_scope,
            temporal_scope=temporal_scope,
            perpetrator_information=perpetrator_information,
            contextual_information=contextual_information,
            analysis_focus=analysis_focus,
            analysis_id=analysis_id,
            case_id=case_id
        )

        return self.generate_output(
            output_data=result,
            output_type="genocide_intent_analysis",
            evidence_ids=[]
        )

    def _conduct_genocide_intent_analysis(
        self,
        direct_evidence: Dict[str, Any],
        circumstantial_evidence: Dict[str, Any],
        protected_groups: List[str],
        geographic_scope: List[str],
        temporal_scope: Dict[str, Any],
        perpetrator_information: Dict[str, Any],
        contextual_information: Dict[str, Any],
        analysis_focus: List[str],
        analysis_id: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive genocide intent analysis."""

        # Build genocide intent analysis prompt
        analysis_prompt = self._build_genocide_analysis_prompt(
            direct_evidence=direct_evidence,
            circumstantial_evidence=circumstantial_evidence,
            protected_groups=protected_groups,
            geographic_scope=geographic_scope,
            temporal_scope=temporal_scope,
            perpetrator_information=perpetrator_information,
            contextual_information=contextual_information,
            analysis_focus=analysis_focus,
            analysis_id=analysis_id,
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
                    "intent_analysis_id": analysis_id,
                    "error": "Could not parse genocide intent analysis results",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            result = {
                "intent_analysis_id": analysis_id,
                "error": "Invalid JSON in genocide intent analysis results",
                "raw_response": response_text
            }

        # Add metadata
        result["analysis_metadata"] = {
            "intent_analysis_id": analysis_id,
            "analysis_date": datetime.now().isoformat(),
            "analyst": self.agent_id,
            "case_id": case_id,
            "geographic_scope": geographic_scope,
            "temporal_scope": temporal_scope,
            "protected_groups_analyzed": protected_groups,
            "configuration_used": {
                "legal_frameworks": self.config.legal_frameworks,
                "jurisprudence_applied": {
                    "icty": self.config.apply_icty_jurisprudence,
                    "ictr": self.config.apply_ictr_jurisprudence,
                    "icj": self.config.apply_icj_jurisprudence
                },
                "minimum_intent_confidence": self.config.min_intent_confidence,
                "prevention_analysis": self.config.assess_ongoing_risk
            }
        }

        return result

    def _build_genocide_analysis_prompt(
        self,
        direct_evidence: Dict[str, Any],
        circumstantial_evidence: Dict[str, Any],
        protected_groups: List[str],
        geographic_scope: List[str],
        temporal_scope: Dict[str, Any],
        perpetrator_information: Dict[str, Any],
        contextual_information: Dict[str, Any],
        analysis_focus: List[str],
        analysis_id: str,
        case_id: Optional[str] = None
    ) -> str:
        """Build the genocide intent analysis prompt for Claude."""

        prompt = f"""Please conduct comprehensive genocide intent analysis according to international legal standards.

ANALYSIS METADATA:
- Analysis ID: {analysis_id}
- Analysis Date: {datetime.now().isoformat()}"""

        if case_id:
            prompt += f"\n- Case ID: {case_id}"

        prompt += f"\n\nPROTECTED GROUPS ANALYZED: {', '.join(protected_groups)}"

        if geographic_scope:
            prompt += f"\nGEOGRAPHIC SCOPE: {', '.join(geographic_scope)}"

        if temporal_scope:
            prompt += f"\nTEMPORAL SCOPE: {json.dumps(temporal_scope, indent=2)}"

        if direct_evidence:
            prompt += f"\n\nDIRECT EVIDENCE OF INTENT:\n{json.dumps(direct_evidence, indent=2)}"

        if circumstantial_evidence:
            prompt += f"\n\nCIRCUMSTANTIAL EVIDENCE:\n{json.dumps(circumstantial_evidence, indent=2)}"

        if perpetrator_information:
            prompt += f"\n\nPERPETRATOR INFORMATION:\n{json.dumps(perpetrator_information, indent=2)}"

        if contextual_information:
            prompt += f"\n\nCONTEXTUAL INFORMATION:\n{json.dumps(contextual_information, indent=2)}"

        if analysis_focus:
            prompt += f"\n\nANALYSIS FOCUS: {', '.join(analysis_focus)}"

        prompt += f"""

ANALYSIS CONFIGURATION:
- Legal frameworks: {', '.join(self.config.legal_frameworks)}
- Analyze direct evidence: {self.config.analyze_direct_evidence}
- Analyze circumstantial evidence: {self.config.analyze_circumstantial_evidence}
- Assess targeting patterns: {self.config.assess_targeting_patterns}
- Evaluate contextual factors: {self.config.evaluate_contextual_factors}
- Apply jurisprudence: {self.config.apply_jurisprudence}
- ICTY jurisprudence: {self.config.apply_icty_jurisprudence}
- ICTR jurisprudence: {self.config.apply_ictr_jurisprudence}
- ICJ jurisprudence: {self.config.apply_icj_jurisprudence}
- Analyze group identity: {self.config.analyze_group_identity}
- Assess substantial part: {self.config.assess_substantial_part}
- Minimum intent confidence: {self.config.min_intent_confidence}
- Require corroborating evidence: {self.config.require_corroborating_evidence}
- Assess alternative explanations: {self.config.assess_alternative_explanations}
- Maximum precedent cases: {self.config.max_precedent_cases}
- Include comparative analysis: {self.config.include_comparative_analysis}
- Assess ongoing risk: {self.config.assess_ongoing_risk}

Please provide comprehensive genocide intent analysis following the JSON format in your system prompt. Focus on:

1. Protected group analysis and identification under Genocide Convention
2. Intent evidence analysis (direct statements, circumstantial patterns)
3. Targeting pattern assessment and systematic destruction indicators
4. Contextual analysis including historical, political, and social factors
5. Perpetrator analysis and institutional involvement assessment
6. Legal analysis applying Genocide Convention and international jurisprudence
7. Evidence assessment including strengths, gaps, and corroboration
8. Comparative analysis with relevant genocide precedents
9. Risk assessment for ongoing genocidal intent and prevention needs
10. Conclusions and recommendations for legal action and prevention

Ensure analysis maintains the highest legal standards for genocide intent assessment while considering prevention implications."""

        return prompt

    def analyze_direct_intent_evidence(
        self,
        statements: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        protected_groups: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze direct evidence of genocidal intent.

        Args:
            statements: Direct statements showing intent
            documents: Documentary evidence of intent
            protected_groups: Groups allegedly targeted
            case_id: Optional case ID

        Returns:
            Direct intent evidence analysis
        """
        return self.process({
            'direct_evidence': {
                'statements': statements,
                'documents': documents
            },
            'protected_groups': protected_groups,
            'analysis_focus': ['direct_evidence', 'intent_assessment', 'legal_elements'],
            'case_id': case_id
        })

    def assess_targeting_patterns(
        self,
        targeting_data: Dict[str, Any],
        victim_demographics: Dict[str, Any],
        protected_groups: List[str],
        geographic_scope: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess targeting patterns for genocidal intent.

        Args:
            targeting_data: Data on how victims were selected
            victim_demographics: Demographic information about victims
            protected_groups: Groups allegedly targeted
            geographic_scope: Geographic areas involved
            case_id: Optional case ID

        Returns:
            Targeting pattern analysis for genocide intent
        """
        return self.process({
            'circumstantial_evidence': {
                'targeting_patterns': targeting_data,
                'victim_demographics': victim_demographics
            },
            'protected_groups': protected_groups,
            'geographic_scope': geographic_scope,
            'analysis_focus': ['targeting_patterns', 'systematic_destruction', 'group_identity'],
            'case_id': case_id
        })

    def evaluate_systematic_destruction(
        self,
        destruction_evidence: Dict[str, Any],
        institutional_involvement: Dict[str, Any],
        protected_groups: List[str],
        temporal_scope: Dict[str, str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate evidence of systematic destruction.

        Args:
            destruction_evidence: Evidence of systematic destruction
            institutional_involvement: Information about institutional participation
            protected_groups: Groups allegedly targeted
            temporal_scope: Time period of alleged genocide
            case_id: Optional case ID

        Returns:
            Systematic destruction analysis
        """
        return self.process({
            'circumstantial_evidence': {
                'systematic_destruction': destruction_evidence,
                'institutional_involvement': institutional_involvement
            },
            'protected_groups': protected_groups,
            'temporal_scope': temporal_scope,
            'analysis_focus': ['systematic_destruction', 'institutional_involvement', 'escalation_patterns'],
            'case_id': case_id
        })

    def analyze_propaganda_incitement(
        self,
        propaganda_materials: List[Dict[str, Any]],
        dissemination_data: Dict[str, Any],
        protected_groups: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze propaganda and incitement for genocidal intent.

        Args:
            propaganda_materials: Propaganda content and materials
            dissemination_data: Information about how propaganda was spread
            protected_groups: Groups allegedly targeted
            case_id: Optional case ID

        Returns:
            Propaganda and incitement analysis
        """
        return self.process({
            'direct_evidence': {
                'propaganda_incitement': propaganda_materials,
                'dissemination': dissemination_data
            },
            'protected_groups': protected_groups,
            'analysis_focus': ['propaganda_analysis', 'incitement_assessment', 'intent_indicators'],
            'case_id': case_id
        })

    def assess_prevention_indicators(
        self,
        current_situation: Dict[str, Any],
        risk_factors: List[str],
        vulnerable_groups: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess indicators for genocide prevention.

        Args:
            current_situation: Current situation assessment
            risk_factors: Identified risk factors
            vulnerable_groups: Groups at risk
            case_id: Optional case ID

        Returns:
            Prevention indicator analysis
        """
        return self.process({
            'contextual_information': {
                'current_situation': current_situation,
                'risk_factors': risk_factors
            },
            'protected_groups': vulnerable_groups,
            'analysis_focus': ['prevention_indicators', 'risk_assessment', 'early_warning'],
            'case_id': case_id
        })

    def compare_genocide_precedents(
        self,
        current_case_facts: Dict[str, Any],
        comparison_cases: List[str],
        protected_groups: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare current case with genocide precedents.

        Args:
            current_case_facts: Facts of current case
            comparison_cases: Names of cases to compare with
            protected_groups: Groups involved
            case_id: Optional case ID

        Returns:
            Comparative genocide analysis
        """
        return self.process({
            'circumstantial_evidence': current_case_facts,
            'protected_groups': protected_groups,
            'contextual_information': {
                'comparison_cases': comparison_cases
            },
            'analysis_focus': ['comparative_analysis', 'precedent_application', 'legal_standards'],
            'case_id': case_id
        })

    def assess_command_responsibility_genocide(
        self,
        command_structure: Dict[str, Any],
        superior_knowledge: Dict[str, Any],
        genocidal_acts: List[Dict[str, Any]],
        protected_groups: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess command responsibility for genocide.

        Args:
            command_structure: Military/civilian command structure
            superior_knowledge: Evidence of superior knowledge
            genocidal_acts: Specific genocidal acts committed
            protected_groups: Groups targeted
            case_id: Optional case ID

        Returns:
            Command responsibility analysis for genocide
        """
        return self.process({
            'perpetrator_information': {
                'command_structure': command_structure,
                'superior_knowledge': superior_knowledge,
                'genocidal_acts': genocidal_acts
            },
            'protected_groups': protected_groups,
            'analysis_focus': ['command_responsibility', 'superior_intent', 'institutional_genocide'],
            'case_id': case_id
        })