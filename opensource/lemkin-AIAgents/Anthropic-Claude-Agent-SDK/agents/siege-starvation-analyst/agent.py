"""
Siege & Starvation Warfare Analyst Agent

Documents crimes related to blockades, sieges, and starvation tactics.
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
from .config import SiegeStarvationAnalystConfig, DEFAULT_CONFIG


class SiegeStarvationAnalystAgent(BaseAgent):
    """
    Agent for analyzing siege warfare and starvation tactics according to international law.

    Handles:
    - Supply flow and humanitarian access analysis
    - Population impact assessment (nutrition, health, mortality)
    - Siege infrastructure mapping
    - Legal element analysis (IHL violations, crimes against humanity)
    - Pattern and systematic analysis
    - Command responsibility assessment
    """

    def __init__(
        self,
        config: Optional[SiegeStarvationAnalystConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Siege & Starvation Warfare Analyst agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="siege_starvation_analyst",
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
        Analyze siege and starvation warfare evidence.

        Args:
            input_data: Dictionary containing:
                - location: Besieged area name/coordinates
                - siege_start_date: Start date of siege
                - siege_end_date: End date or 'ongoing'
                - population_affected: Estimated number
                - supply_data: Supply flow information
                - humanitarian_access_reports: Access restriction documentation
                - health_nutrition_data: Population health and nutrition data
                - geographic_data: Maps, siege lines, checkpoints
                - witness_testimony: Testimony from affected population
                - policy_documents: Orders, policies related to siege
                - case_id: Optional case ID
                - analysis_focus: Specific aspects to analyze

        Returns:
            Comprehensive siege and starvation analysis
        """
        location = input_data.get('location')
        siege_start_date = input_data.get('siege_start_date')
        siege_end_date = input_data.get('siege_end_date', 'ongoing')
        population_affected = input_data.get('population_affected')
        supply_data = input_data.get('supply_data', {})
        humanitarian_access_reports = input_data.get('humanitarian_access_reports', [])
        health_nutrition_data = input_data.get('health_nutrition_data', {})
        geographic_data = input_data.get('geographic_data', {})
        witness_testimony = input_data.get('witness_testimony', [])
        policy_documents = input_data.get('policy_documents', [])
        case_id = input_data.get('case_id')
        analysis_focus = input_data.get('analysis_focus', [])

        if not location:
            raise ValueError("Location is required for siege analysis")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Log analysis start
        self.audit_logger.log_analysis_start(
            evidence_id=analysis_id,
            agent_id=self.agent_id,
            metadata={
                'analysis_type': 'siege_starvation',
                'location': location,
                'case_id': case_id
            }
        )

        # Prepare input for Claude
        user_message = self._prepare_input_message(
            location=location,
            siege_start_date=siege_start_date,
            siege_end_date=siege_end_date,
            population_affected=population_affected,
            supply_data=supply_data,
            humanitarian_access_reports=humanitarian_access_reports,
            health_nutrition_data=health_nutrition_data,
            geographic_data=geographic_data,
            witness_testimony=witness_testimony,
            policy_documents=policy_documents,
            analysis_focus=analysis_focus
        )

        # Call Claude for analysis
        result = self._call_claude(user_message=user_message)

        # Parse and validate result
        analysis_result = self._parse_result(result, analysis_id, case_id, location)

        # Check for human review if needed
        if self._requires_human_review(analysis_result):
            analysis_result['human_review_required'] = True
            analysis_result['review_reason'] = self._get_review_reason(analysis_result)

            if self.enable_human_review:
                review_result = self._request_human_review(
                    analysis_result,
                    reason=analysis_result['review_reason']
                )
                if review_result:
                    analysis_result['human_review_completed'] = True
                    analysis_result['review_outcome'] = review_result

        # Log completion
        self.audit_logger.log_analysis_complete(
            evidence_id=analysis_id,
            agent_id=self.agent_id,
            result_summary={
                'location': location,
                'legal_violations_identified': len(analysis_result.get('legal_analysis', {}).get('ihl_violations', {})),
                'confidence_score': analysis_result.get('confidence_assessment', {}).get('overall_confidence')
            }
        )

        return analysis_result

    def _prepare_input_message(
        self,
        location: str,
        siege_start_date: str,
        siege_end_date: str,
        population_affected: str,
        supply_data: Dict,
        humanitarian_access_reports: List,
        health_nutrition_data: Dict,
        geographic_data: Dict,
        witness_testimony: List,
        policy_documents: List,
        analysis_focus: List
    ) -> str:
        """Prepare detailed input message for Claude."""

        message_parts = [
            f"# Siege Analysis Request",
            f"",
            f"## Siege Context",
            f"**Location:** {location}",
            f"**Siege Start Date:** {siege_start_date}",
            f"**Siege End Date:** {siege_end_date}",
            f"**Population Affected:** {population_affected}",
            f""
        ]

        if analysis_focus:
            message_parts.extend([
                f"## Analysis Focus",
                f"Please focus on the following aspects: {', '.join(analysis_focus)}",
                f""
            ])

        if supply_data:
            message_parts.extend([
                f"## Supply Flow Data",
                f"```json",
                json.dumps(supply_data, indent=2),
                f"```",
                f""
            ])

        if humanitarian_access_reports:
            message_parts.extend([
                f"## Humanitarian Access Reports",
                f""
            ])
            for i, report in enumerate(humanitarian_access_reports, 1):
                message_parts.append(f"### Report {i}")
                if isinstance(report, dict):
                    message_parts.append(f"```json\n{json.dumps(report, indent=2)}\n```")
                else:
                    message_parts.append(str(report))
                message_parts.append("")

        if health_nutrition_data:
            message_parts.extend([
                f"## Health and Nutrition Data",
                f"```json",
                json.dumps(health_nutrition_data, indent=2),
                f"```",
                f""
            ])

        if geographic_data:
            message_parts.extend([
                f"## Geographic and Infrastructure Data",
                f"```json",
                json.dumps(geographic_data, indent=2),
                f"```",
                f""
            ])

        if witness_testimony:
            message_parts.extend([
                f"## Witness Testimony",
                f""
            ])
            for i, testimony in enumerate(witness_testimony, 1):
                message_parts.append(f"### Testimony {i}")
                message_parts.append(str(testimony))
                message_parts.append("")

        if policy_documents:
            message_parts.extend([
                f"## Policy Documents",
                f""
            ])
            for i, doc in enumerate(policy_documents, 1):
                message_parts.append(f"### Document {i}")
                message_parts.append(str(doc))
                message_parts.append("")

        message_parts.extend([
            f"## Instructions",
            f"",
            f"Please provide a comprehensive analysis of this siege situation according to your capabilities.",
            f"Focus on:",
            f"1. Documenting humanitarian access restrictions and patterns",
            f"2. Assessing population impact (nutrition, health, mortality)",
            f"3. Mapping siege infrastructure and territorial control",
            f"4. Analyzing violations of international humanitarian law",
            f"5. Identifying patterns indicating systematic or deliberate starvation",
            f"6. Assessing command responsibility",
            f"",
            f"Provide your analysis in the structured JSON format specified in your system prompt.",
            f"Be thorough, evidence-based, and precise in your legal analysis."
        ])

        return "\n".join(message_parts)

    def _parse_result(self, result: str, analysis_id: str, case_id: Optional[str], location: str) -> Dict[str, Any]:
        """Parse and structure the analysis result."""
        try:
            # Try to parse as JSON
            if "```json" in result:
                json_start = result.find("```json") + 7
                json_end = result.find("```", json_start)
                json_str = result[json_start:json_end].strip()
                analysis = json.loads(json_str)
            elif result.strip().startswith("{"):
                analysis = json.loads(result)
            else:
                # Fallback: structure the text response
                analysis = {
                    "siege_analysis_id": analysis_id,
                    "analysis_metadata": {
                        "analysis_date": datetime.utcnow().isoformat(),
                        "analyst": self.agent_id,
                        "case_id": case_id,
                        "location": location
                    },
                    "analysis_text": result,
                    "parsing_note": "Structured JSON not found, returning text analysis"
                }

            # Ensure required fields
            if "siege_analysis_id" not in analysis:
                analysis["siege_analysis_id"] = analysis_id
            if "case_id" not in analysis.get("analysis_metadata", {}):
                if "analysis_metadata" not in analysis:
                    analysis["analysis_metadata"] = {}
                analysis["analysis_metadata"]["case_id"] = case_id

            return analysis

        except json.JSONDecodeError as e:
            # Return structured error response
            return {
                "siege_analysis_id": analysis_id,
                "analysis_metadata": {
                    "analysis_date": datetime.utcnow().isoformat(),
                    "analyst": self.agent_id,
                    "case_id": case_id,
                    "location": location
                },
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": result
            }

    def _requires_human_review(self, analysis: Dict[str, Any]) -> bool:
        """Determine if human review is required based on analysis."""
        # Require review for high-stakes findings
        legal_analysis = analysis.get('legal_analysis', {})

        # Review if starvation as warfare method identified with high confidence
        ihl_violations = legal_analysis.get('ihl_violations', {})
        starvation_violation = ihl_violations.get('starvation_as_warfare_method', {})
        if starvation_violation.get('violation_identified') and \
           starvation_violation.get('confidence_level', 0) >= 0.7:
            return True

        # Review if crimes against humanity identified
        crimes_against_humanity = legal_analysis.get('crimes_against_humanity', {})
        if crimes_against_humanity.get('assessment', {}).get('confidence_level', 0) >= 0.7:
            return True

        # Review if genocide indicators identified
        genocide = legal_analysis.get('genocide_indicators', {})
        if genocide.get('assessment', {}).get('confidence_level', 0) >= 0.5:
            return True

        # Review if systematic denial pattern identified with high confidence
        access_analysis = analysis.get('humanitarian_access_analysis', {})
        systematic_denial = access_analysis.get('systematic_denial_assessment', {})
        if systematic_denial.get('pattern_identified') and \
           systematic_denial.get('confidence_level', 0) >= 0.7:
            return True

        return False

    def _get_review_reason(self, analysis: Dict[str, Any]) -> str:
        """Get reason for human review requirement."""
        reasons = []

        legal_analysis = analysis.get('legal_analysis', {})

        if legal_analysis.get('ihl_violations', {}).get('starvation_as_warfare_method', {}).get('violation_identified'):
            reasons.append("Starvation as method of warfare identified")

        if legal_analysis.get('crimes_against_humanity', {}).get('assessment', {}).get('confidence_level', 0) >= 0.7:
            reasons.append("High-confidence crimes against humanity assessment")

        if legal_analysis.get('genocide_indicators', {}).get('assessment', {}).get('confidence_level', 0) >= 0.5:
            reasons.append("Genocide indicators identified")

        if analysis.get('humanitarian_access_analysis', {}).get('systematic_denial_assessment', {}).get('pattern_identified'):
            reasons.append("Systematic denial of humanitarian access pattern identified")

        return "; ".join(reasons) if reasons else "High-stakes legal findings"

    def analyze_siege_conditions(
        self,
        location: str,
        siege_data: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method for analyzing siege conditions.

        Args:
            location: Besieged area name
            siege_data: All siege-related data
            case_id: Optional case ID

        Returns:
            Comprehensive siege analysis
        """
        return self.process({
            'location': location,
            'case_id': case_id,
            **siege_data
        })

    def assess_humanitarian_impact(
        self,
        location: str,
        population_data: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method for assessing humanitarian impact.

        Args:
            location: Affected area
            population_data: Health, nutrition, and population data
            case_id: Optional case ID

        Returns:
            Humanitarian impact assessment
        """
        return self.process({
            'location': location,
            'health_nutrition_data': population_data,
            'analysis_focus': ['population_impact', 'humanitarian'],
            'case_id': case_id
        })


# Convenience function for quick analysis
def analyze_siege(
    location: str,
    siege_data: Dict[str, Any],
    case_id: Optional[str] = None,
    config: Optional[SiegeStarvationAnalystConfig] = None
) -> Dict[str, Any]:
    """
    Quick analysis function for siege situations.

    Args:
        location: Besieged area
        siege_data: All siege-related data
        case_id: Optional case ID
        config: Optional custom configuration

    Returns:
        Comprehensive siege analysis
    """
    agent = SiegeStarvationAnalystAgent(config=config)
    return agent.analyze_siege_conditions(location, siege_data, case_id)
