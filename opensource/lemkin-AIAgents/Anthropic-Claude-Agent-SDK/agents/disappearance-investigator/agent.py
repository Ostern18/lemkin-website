"""
Enforced Disappearance Investigator Agent

Documents patterns of disappearances according to international legal standards.
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
from .config import DisappearanceInvestigatorConfig, DEFAULT_CONFIG


class DisappearanceInvestigatorAgent(BaseAgent):
    """
    Agent for investigating enforced disappearances according to international standards.

    Handles:
    - Legal element analysis for enforced disappearance definition
    - Pattern recognition and systematic analysis across cases
    - State responsibility and obligation assessment
    - Family impact and rights analysis
    - Search and investigation documentation and evaluation
    - Institutional analysis and command responsibility
    """

    def __init__(
        self,
        config: Optional[DisappearanceInvestigatorConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Enforced Disappearance Investigator agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="disappearance_investigator",
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
        Analyze enforced disappearance cases and patterns.

        Args:
            input_data: Dictionary containing:
                - disappearance_cases: Individual disappearance cases to analyze
                - missing_persons_reports: Missing persons reports and data
                - state_response_data: Information about state responses
                - family_accounts: Family testimonies and experiences
                - investigation_records: Official investigation records
                - institutional_information: Information about institutions involved
                - geographic_scope: Geographic areas involved
                - temporal_scope: Time period of disappearances
                - case_id: Optional case ID
                - analysis_focus: Specific aspects to focus on

        Returns:
            Comprehensive enforced disappearance analysis
        """
        disappearance_cases = input_data.get('disappearance_cases', [])
        missing_persons_reports = input_data.get('missing_persons_reports', {})
        state_response_data = input_data.get('state_response_data', {})
        family_accounts = input_data.get('family_accounts', [])
        investigation_records = input_data.get('investigation_records', {})
        institutional_information = input_data.get('institutional_information', {})
        geographic_scope = input_data.get('geographic_scope', [])
        temporal_scope = input_data.get('temporal_scope', {})
        case_id = input_data.get('case_id')
        analysis_focus = input_data.get('analysis_focus', [])

        if not disappearance_cases and not missing_persons_reports:
            raise ValueError("Either disappearance cases or missing persons reports required")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Log disappearance analysis (with special sensitivity)
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            details={
                "analysis_id": analysis_id,
                "case_id": case_id,
                "analysis_type": "enforced_disappearance_investigation",
                "cases_analyzed": len(disappearance_cases),
                "geographic_scope": geographic_scope,
                "family_accounts_included": len(family_accounts),
                "high_sensitivity_analysis": True,
                "family_rights_prioritized": self.config.prioritize_family_rights
            }
        )

        # Perform disappearance analysis
        result = self._conduct_disappearance_analysis(
            disappearance_cases=disappearance_cases,
            missing_persons_reports=missing_persons_reports,
            state_response_data=state_response_data,
            family_accounts=family_accounts,
            investigation_records=investigation_records,
            institutional_information=institutional_information,
            geographic_scope=geographic_scope,
            temporal_scope=temporal_scope,
            analysis_focus=analysis_focus,
            analysis_id=analysis_id,
            case_id=case_id
        )

        return self.generate_output(
            output_data=result,
            output_type="enforced_disappearance_analysis",
            evidence_ids=[]
        )

    def _conduct_disappearance_analysis(
        self,
        disappearance_cases: List[Dict[str, Any]],
        missing_persons_reports: Dict[str, Any],
        state_response_data: Dict[str, Any],
        family_accounts: List[Dict[str, Any]],
        investigation_records: Dict[str, Any],
        institutional_information: Dict[str, Any],
        geographic_scope: List[str],
        temporal_scope: Dict[str, Any],
        analysis_focus: List[str],
        analysis_id: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive enforced disappearance analysis."""

        # Build disappearance analysis prompt
        analysis_prompt = self._build_disappearance_analysis_prompt(
            disappearance_cases=disappearance_cases,
            missing_persons_reports=missing_persons_reports,
            state_response_data=state_response_data,
            family_accounts=family_accounts,
            investigation_records=investigation_records,
            institutional_information=institutional_information,
            geographic_scope=geographic_scope,
            temporal_scope=temporal_scope,
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
                    "disappearance_analysis_id": analysis_id,
                    "error": "Could not parse disappearance analysis results",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            result = {
                "disappearance_analysis_id": analysis_id,
                "error": "Invalid JSON in disappearance analysis results",
                "raw_response": response_text
            }

        # Add metadata
        result["analysis_metadata"] = {
            "disappearance_analysis_id": analysis_id,
            "analysis_date": datetime.now().isoformat(),
            "analyst": self.agent_id,
            "case_id": case_id,
            "geographic_scope": geographic_scope,
            "temporal_scope": temporal_scope,
            "cases_analyzed": len(disappearance_cases),
            "analysis_scope": "pattern_analysis" if len(disappearance_cases) > 1 else "individual_case",
            "configuration_used": {
                "legal_frameworks": self.config.legal_frameworks,
                "family_rights_prioritized": self.config.prioritize_family_rights,
                "pattern_analysis": self.config.assess_pattern_analysis,
                "state_obligations": self.config.evaluate_state_obligations,
                "institutional_analysis": self.config.assess_institutional_involvement
            }
        }

        return result

    def _build_disappearance_analysis_prompt(
        self,
        disappearance_cases: List[Dict[str, Any]],
        missing_persons_reports: Dict[str, Any],
        state_response_data: Dict[str, Any],
        family_accounts: List[Dict[str, Any]],
        investigation_records: Dict[str, Any],
        institutional_information: Dict[str, Any],
        geographic_scope: List[str],
        temporal_scope: Dict[str, Any],
        analysis_focus: List[str],
        analysis_id: str,
        case_id: Optional[str] = None
    ) -> str:
        """Build the disappearance analysis prompt for Claude."""

        prompt = f"""Please conduct comprehensive enforced disappearance analysis according to international legal standards.

ANALYSIS METADATA:
- Analysis ID: {analysis_id}
- Analysis Date: {datetime.now().isoformat()}"""

        if case_id:
            prompt += f"\n- Case ID: {case_id}"

        if geographic_scope:
            prompt += f"\nGEOGRAPHIC SCOPE: {', '.join(geographic_scope)}"

        if temporal_scope:
            prompt += f"\nTEMPORAL SCOPE: {json.dumps(temporal_scope, indent=2)}"

        if disappearance_cases:
            prompt += f"\n\nDISAPPEARANCE CASES ({len(disappearance_cases)} cases):\n{json.dumps(disappearance_cases, indent=2)}"

        if missing_persons_reports:
            prompt += f"\n\nMISSING PERSONS REPORTS:\n{json.dumps(missing_persons_reports, indent=2)}"

        if state_response_data:
            prompt += f"\n\nSTATE RESPONSE DATA:\n{json.dumps(state_response_data, indent=2)}"

        if family_accounts:
            prompt += f"\n\nFAMILY ACCOUNTS ({len(family_accounts)} accounts):\n{json.dumps(family_accounts, indent=2)}"

        if investigation_records:
            prompt += f"\n\nINVESTIGATION RECORDS:\n{json.dumps(investigation_records, indent=2)}"

        if institutional_information:
            prompt += f"\n\nINSTITUTIONAL INFORMATION:\n{json.dumps(institutional_information, indent=2)}"

        if analysis_focus:
            prompt += f"\n\nANALYSIS FOCUS: {', '.join(analysis_focus)}"

        prompt += f"""

ANALYSIS CONFIGURATION:
- Legal frameworks: {', '.join(self.config.legal_frameworks)}
- Analyze legal elements: {self.config.analyze_legal_elements}
- Assess pattern analysis: {self.config.assess_pattern_analysis}
- Evaluate state obligations: {self.config.evaluate_state_obligations}
- Analyze family rights: {self.config.analyze_family_rights}
- Assess institutional involvement: {self.config.assess_institutional_involvement}
- Evaluate search investigation: {self.config.evaluate_search_investigation}
- Identify systematic patterns: {self.config.identify_systematic_patterns}
- Prioritize family rights: {self.config.prioritize_family_rights}
- Assess prevention obligations: {self.config.assess_prevention_obligations}
- Evaluate investigation obligations: {self.config.evaluate_investigation_obligations}
- Minimum confidence threshold: {self.config.min_confidence_threshold}

Please provide comprehensive enforced disappearance analysis following the JSON format in your system prompt. Focus on:

1. Legal classification according to international definitions (ICPPED Article 2)
2. Individual case analysis with detailed element assessment
3. Pattern analysis including temporal, geographic, and demographic patterns
4. Institutional analysis and command responsibility assessment
5. State obligation analysis (prevention, investigation, information, remedy)
6. Family rights analysis and impact assessment
7. Search and investigation evaluation
8. Legal implications and criminal/state responsibility
9. Evidence assessment including strengths, gaps, and recommendations
10. Family support and protection needs

Ensure analysis maintains family-centered approach while meeting rigorous international standards for enforced disappearance documentation."""

        return prompt

    def analyze_individual_case(
        self,
        case_details: Dict[str, Any],
        family_account: Dict[str, Any],
        state_response: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze individual enforced disappearance case.

        Args:
            case_details: Details of the disappearance incident
            family_account: Family testimony and experience
            state_response: State response to the disappearance
            case_id: Optional case ID

        Returns:
            Individual case analysis
        """
        return self.process({
            'disappearance_cases': [case_details],
            'family_accounts': [family_account],
            'state_response_data': state_response,
            'analysis_focus': ['legal_elements', 'family_rights', 'state_obligations'],
            'case_id': case_id
        })

    def assess_systematic_disappearances(
        self,
        multiple_cases: List[Dict[str, Any]],
        institutional_data: Dict[str, Any],
        geographic_scope: List[str],
        temporal_scope: Dict[str, str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess systematic patterns of enforced disappearances.

        Args:
            multiple_cases: Multiple disappearance cases
            institutional_data: Information about institutions involved
            geographic_scope: Geographic areas where disappearances occurred
            temporal_scope: Time period of disappearances
            case_id: Optional case ID

        Returns:
            Systematic disappearance pattern analysis
        """
        return self.process({
            'disappearance_cases': multiple_cases,
            'institutional_information': institutional_data,
            'geographic_scope': geographic_scope,
            'temporal_scope': temporal_scope,
            'analysis_focus': ['pattern_analysis', 'systematic_practice', 'institutional_involvement'],
            'case_id': case_id
        })

    def evaluate_state_compliance(
        self,
        disappearance_data: Dict[str, Any],
        investigation_records: Dict[str, Any],
        family_experiences: List[Dict[str, Any]],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate state compliance with international obligations.

        Args:
            disappearance_data: Data about disappearances
            investigation_records: Records of official investigations
            family_experiences: Family experiences with authorities
            case_id: Optional case ID

        Returns:
            State compliance assessment
        """
        return self.process({
            'missing_persons_reports': disappearance_data,
            'investigation_records': investigation_records,
            'family_accounts': family_experiences,
            'analysis_focus': ['state_obligations', 'investigation_quality', 'prevention_measures'],
            'case_id': case_id
        })

    def analyze_family_rights_violations(
        self,
        family_testimonies: List[Dict[str, Any]],
        state_interactions: Dict[str, Any],
        information_provided: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze violations of family rights in disappearance cases.

        Args:
            family_testimonies: Family testimonies about their experiences
            state_interactions: Information about family interactions with state
            information_provided: Information provided (or denied) to families
            case_id: Optional case ID

        Returns:
            Family rights violation analysis
        """
        return self.process({
            'family_accounts': family_testimonies,
            'state_response_data': {
                'interactions': state_interactions,
                'information_provided': information_provided
            },
            'analysis_focus': ['family_rights', 'right_to_truth', 'information_obligations'],
            'case_id': case_id
        })

    def assess_search_investigation_adequacy(
        self,
        search_records: Dict[str, Any],
        investigation_procedures: Dict[str, Any],
        resources_allocated: Dict[str, Any],
        obstacles_encountered: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess adequacy of search and investigation efforts.

        Args:
            search_records: Records of search efforts
            investigation_procedures: Investigation procedures used
            resources_allocated: Resources dedicated to search/investigation
            obstacles_encountered: Obstacles that hindered efforts
            case_id: Optional case ID

        Returns:
            Search and investigation adequacy assessment
        """
        return self.process({
            'investigation_records': {
                'search_records': search_records,
                'procedures': investigation_procedures,
                'resources': resources_allocated,
                'obstacles': obstacles_encountered
            },
            'analysis_focus': ['investigation_quality', 'search_adequacy', 'state_obligations'],
            'case_id': case_id
        })

    def identify_prevention_failures(
        self,
        pre_disappearance_context: Dict[str, Any],
        warning_signs: List[str],
        prevention_measures: Dict[str, Any],
        institutional_responses: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identify failures in prevention of enforced disappearances.

        Args:
            pre_disappearance_context: Context before disappearances occurred
            warning_signs: Warning signs that were present
            prevention_measures: Prevention measures that were (not) taken
            institutional_responses: How institutions responded to risks
            case_id: Optional case ID

        Returns:
            Prevention failure analysis
        """
        return self.process({
            'institutional_information': {
                'context': pre_disappearance_context,
                'warning_signs': warning_signs,
                'prevention_measures': prevention_measures,
                'responses': institutional_responses
            },
            'analysis_focus': ['prevention_obligations', 'institutional_responsibility', 'early_warning'],
            'case_id': case_id
        })

    def map_command_responsibility(
        self,
        disappearance_incidents: List[Dict[str, Any]],
        command_structure: Dict[str, Any],
        superior_knowledge: Dict[str, Any],
        prevention_measures: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Map command responsibility for enforced disappearances.

        Args:
            disappearance_incidents: Specific disappearance incidents
            command_structure: Military/police command structure
            superior_knowledge: Evidence of superior knowledge
            prevention_measures: Measures taken (or not taken) by superiors
            case_id: Optional case ID

        Returns:
            Command responsibility analysis
        """
        return self.process({
            'disappearance_cases': disappearance_incidents,
            'institutional_information': {
                'command_structure': command_structure,
                'superior_knowledge': superior_knowledge,
                'prevention_measures': prevention_measures
            },
            'analysis_focus': ['command_responsibility', 'superior_liability', 'institutional_involvement'],
            'case_id': case_id
        })