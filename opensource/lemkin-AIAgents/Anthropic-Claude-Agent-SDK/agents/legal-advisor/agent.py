"""
Legal Framework & Jurisdiction Advisor Agent

Clarifies applicable law and jurisdictional questions for investigations.
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
from .config import LegalAdvisorConfig, DEFAULT_CONFIG


class LegalAdvisorAgent(BaseAgent):
    """
    Agent for legal framework analysis and jurisdictional advice.

    Handles:
    - International and domestic law analysis
    - Jurisdictional assessment and forum selection
    - Legal element mapping and evidence requirements
    - Procedural analysis and admissibility standards
    - Legal strategy development and risk assessment
    - Precedent analysis and comparative law
    """

    def __init__(
        self,
        config: Optional[LegalAdvisorConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Legal Framework & Jurisdiction Advisor agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="legal_advisor",
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
        Analyze legal framework and jurisdictional questions.

        Args:
            input_data: Dictionary containing:
                - legal_question: Primary legal question to address
                - factual_scenario: Description of relevant facts
                - potential_charges: Potential criminal charges to analyze
                - jurisdictions_of_interest: Specific jurisdictions to consider
                - case_id: Optional case ID
                - analysis_priorities: Specific areas to focus on

        Returns:
            Comprehensive legal analysis and recommendations
        """
        legal_question = input_data.get('legal_question')
        factual_scenario = input_data.get('factual_scenario')
        potential_charges = input_data.get('potential_charges', [])
        jurisdictions_of_interest = input_data.get('jurisdictions_of_interest', [])
        case_id = input_data.get('case_id')
        analysis_priorities = input_data.get('analysis_priorities', [])

        if not legal_question and not factual_scenario:
            raise ValueError("Either legal question or factual scenario required")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Log legal analysis
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            details={
                "analysis_id": analysis_id,
                "case_id": case_id,
                "legal_question": legal_question,
                "charges_analyzed": len(potential_charges),
                "jurisdictions_considered": len(jurisdictions_of_interest)
            }
        )

        # Perform legal analysis
        result = self._conduct_legal_analysis(
            legal_question=legal_question,
            factual_scenario=factual_scenario,
            potential_charges=potential_charges,
            jurisdictions_of_interest=jurisdictions_of_interest,
            analysis_priorities=analysis_priorities,
            analysis_id=analysis_id,
            case_id=case_id
        )

        return self.generate_output(
            output_data=result,
            output_type="legal_analysis_report",
            evidence_ids=[]
        )

    def _conduct_legal_analysis(
        self,
        legal_question: Optional[str],
        factual_scenario: Optional[str],
        potential_charges: List[str],
        jurisdictions_of_interest: List[str],
        analysis_priorities: List[str],
        analysis_id: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive legal framework analysis."""

        # Build legal analysis prompt
        analysis_prompt = self._build_legal_analysis_prompt(
            legal_question=legal_question,
            factual_scenario=factual_scenario,
            potential_charges=potential_charges,
            jurisdictions_of_interest=jurisdictions_of_interest,
            analysis_priorities=analysis_priorities,
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
                    "legal_analysis_id": analysis_id,
                    "error": "Could not parse legal analysis results",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            result = {
                "legal_analysis_id": analysis_id,
                "error": "Invalid JSON in legal analysis results",
                "raw_response": response_text
            }

        # Add metadata
        result["analysis_metadata"] = {
            "legal_analysis_id": analysis_id,
            "analysis_date": datetime.now().isoformat(),
            "analyst": self.agent_id,
            "legal_question": legal_question or "General legal framework analysis",
            "case_id": case_id,
            "configuration_used": {
                "frameworks_analyzed": self.config.legal_frameworks,
                "icc_analysis": self.config.consider_icc_jurisdiction,
                "domestic_analysis": self.config.assess_domestic_courts,
                "precedent_analysis": self.config.analyze_precedents,
                "strategic_development": self.config.develop_strategies
            }
        }

        return result

    def _build_legal_analysis_prompt(
        self,
        legal_question: Optional[str],
        factual_scenario: Optional[str],
        potential_charges: List[str],
        jurisdictions_of_interest: List[str],
        analysis_priorities: List[str],
        analysis_id: str,
        case_id: Optional[str] = None
    ) -> str:
        """Build the legal analysis prompt for Claude."""

        prompt = f"""Please conduct comprehensive legal framework and jurisdictional analysis.

ANALYSIS METADATA:
- Analysis ID: {analysis_id}
- Analysis Date: {datetime.now().isoformat()}"""

        if case_id:
            prompt += f"\n- Case ID: {case_id}"

        if legal_question:
            prompt += f"\n\nPRIMARY LEGAL QUESTION: {legal_question}"

        if factual_scenario:
            prompt += f"\n\nFACTUAL SCENARIO: {factual_scenario}"

        if potential_charges:
            prompt += f"\n\nPOTENTIAL CHARGES TO ANALYZE: {', '.join(potential_charges)}"

        if jurisdictions_of_interest:
            prompt += f"\n\nJURISDICTIONS OF INTEREST: {', '.join(jurisdictions_of_interest)}"

        if analysis_priorities:
            prompt += f"\n\nANALYSIS PRIORITIES: {', '.join(analysis_priorities)}"

        prompt += f"""

ANALYSIS CONFIGURATION:
- Legal frameworks to analyze: {', '.join(self.config.legal_frameworks)}
- Consider ICC jurisdiction: {self.config.consider_icc_jurisdiction}
- Analyze universal jurisdiction: {self.config.analyze_universal_jurisdiction}
- Assess domestic courts: {self.config.assess_domestic_courts}
- Examine regional courts: {self.config.examine_regional_courts}
- Map legal elements: {self.config.map_legal_elements}
- Analyze precedents: {self.config.analyze_precedents}
- Develop strategies: {self.config.develop_strategies}
- Include procedural analysis: {self.config.include_procedural_analysis}
- Consider political factors: {self.config.include_political_considerations}
- Assess evidence admissibility: {self.config.assess_evidence_admissibility}
- Analyze victim participation: {self.config.analyze_victim_participation}
- Consider immunity issues: {self.config.consider_immunity_issues}
- Maximum precedents to analyze: {self.config.max_precedents_analyzed}

Please provide comprehensive legal analysis following the JSON format in your system prompt. Focus on:

1. Applicable international and domestic legal frameworks
2. Jurisdictional analysis for all relevant forums
3. Legal element mapping for potential charges
4. Procedural considerations and evidence requirements
5. Precedent analysis and case law review
6. Alternative legal strategies and approaches
7. Risk assessment and practical considerations
8. Strategic recommendations with implementation steps

Ensure all legal analysis is precise, well-sourced, and includes confidence assessments for all conclusions."""

        return prompt

    def analyze_icc_jurisdiction(
        self,
        factual_scenario: str,
        potential_charges: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze ICC jurisdiction and admissibility.

        Args:
            factual_scenario: Description of relevant facts
            potential_charges: Potential ICC charges
            case_id: Optional case ID

        Returns:
            ICC jurisdiction and admissibility analysis
        """
        return self.process({
            'legal_question': 'ICC jurisdiction and admissibility analysis',
            'factual_scenario': factual_scenario,
            'potential_charges': potential_charges,
            'jurisdictions_of_interest': ['ICC'],
            'analysis_priorities': ['icc_jurisdiction', 'complementarity', 'admissibility'],
            'case_id': case_id
        })

    def assess_universal_jurisdiction(
        self,
        factual_scenario: str,
        potential_charges: List[str],
        target_states: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess universal jurisdiction opportunities.

        Args:
            factual_scenario: Description of relevant facts
            potential_charges: Potential charges under universal jurisdiction
            target_states: States where prosecution might be possible
            case_id: Optional case ID

        Returns:
            Universal jurisdiction analysis
        """
        return self.process({
            'legal_question': 'Universal jurisdiction prosecution opportunities',
            'factual_scenario': factual_scenario,
            'potential_charges': potential_charges,
            'jurisdictions_of_interest': target_states,
            'analysis_priorities': ['universal_jurisdiction', 'domestic_implementation', 'practical_feasibility'],
            'case_id': case_id
        })

    def map_legal_elements(
        self,
        charges: List[str],
        factual_scenario: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Map legal elements for specific charges.

        Args:
            charges: Specific criminal charges to analyze
            factual_scenario: Relevant factual background
            case_id: Optional case ID

        Returns:
            Detailed legal element analysis
        """
        return self.process({
            'legal_question': 'Legal element mapping and evidence requirements',
            'factual_scenario': factual_scenario,
            'potential_charges': charges,
            'analysis_priorities': ['legal_elements', 'evidence_requirements', 'evidentiary_challenges'],
            'case_id': case_id
        })

    def analyze_immunity_issues(
        self,
        potential_defendants: List[str],
        their_positions: List[str],
        factual_scenario: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze immunity issues for potential defendants.

        Args:
            potential_defendants: Names of potential defendants
            their_positions: Their official positions/roles
            factual_scenario: Relevant factual context
            case_id: Optional case ID

        Returns:
            Comprehensive immunity analysis
        """
        immunity_scenario = f"{factual_scenario}\n\nPOTENTIAL DEFENDANTS:\n"
        for defendant, position in zip(potential_defendants, their_positions):
            immunity_scenario += f"- {defendant}: {position}\n"

        return self.process({
            'legal_question': 'Immunity analysis for potential defendants',
            'factual_scenario': immunity_scenario,
            'analysis_priorities': ['immunity_issues', 'exceptions_to_immunity', 'jurisdictional_implications'],
            'case_id': case_id
        })

    def develop_prosecution_strategy(
        self,
        factual_scenario: str,
        available_evidence: List[str],
        constraints: List[str],
        objectives: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Develop comprehensive prosecution strategy.

        Args:
            factual_scenario: Description of case facts
            available_evidence: Types of evidence available
            constraints: Legal/practical constraints
            objectives: Prosecution objectives
            case_id: Optional case ID

        Returns:
            Strategic prosecution plan
        """
        strategy_scenario = f"{factual_scenario}\n\nAVAILABLE EVIDENCE: {', '.join(available_evidence)}\n"
        strategy_scenario += f"CONSTRAINTS: {', '.join(constraints)}\n"
        strategy_scenario += f"OBJECTIVES: {', '.join(objectives)}"

        return self.process({
            'legal_question': 'Comprehensive prosecution strategy development',
            'factual_scenario': strategy_scenario,
            'analysis_priorities': ['strategic_planning', 'forum_selection', 'charge_selection', 'risk_assessment'],
            'case_id': case_id
        })

    def compare_legal_forums(
        self,
        factual_scenario: str,
        potential_forums: List[str],
        evaluation_criteria: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare different legal forums for prosecution.

        Args:
            factual_scenario: Description of case facts
            potential_forums: Forums to compare (ICC, domestic courts, etc.)
            evaluation_criteria: Criteria for comparison
            case_id: Optional case ID

        Returns:
            Comparative forum analysis
        """
        return self.process({
            'legal_question': 'Comparative analysis of legal forums',
            'factual_scenario': factual_scenario,
            'jurisdictions_of_interest': potential_forums,
            'analysis_priorities': ['forum_comparison', 'advantages_disadvantages', 'strategic_considerations'],
            'case_id': case_id
        })