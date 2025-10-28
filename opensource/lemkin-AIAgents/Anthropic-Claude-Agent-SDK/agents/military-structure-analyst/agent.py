"""
Military Structure & Tactics Analyst Agent

Provides military expertise on operations and command structures.
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
from .config import MilitaryStructureAnalystConfig, DEFAULT_CONFIG


class MilitaryStructureAnalystAgent(BaseAgent):
    """
    Agent for analyzing military structures, tactics, and command responsibility.

    Handles:
    - Military unit organization and hierarchy analysis
    - Command structure mapping
    - Tactical operations analysis
    - Military doctrine assessment
    - IHL compliance evaluation
    - Command responsibility assessment
    """

    def __init__(
        self,
        config: Optional[MilitaryStructureAnalystConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Military Structure & Tactics Analyst agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="military_structure_analyst",
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
        Analyze military structures and operations.

        Args:
            input_data: Dictionary containing:
                - military_force: Name/description of force to analyze
                - organizational_data: Unit structure information
                - command_data: Command hierarchy information
                - operational_data: Information on operations and tactics
                - doctrine_documents: Military doctrine and training materials
                - case_id: Optional case ID
                - analysis_focus: Specific aspects to analyze

        Returns:
            Comprehensive military analysis
        """
        military_force = input_data.get('military_force')
        organizational_data = input_data.get('organizational_data', {})
        command_data = input_data.get('command_data', {})
        operational_data = input_data.get('operational_data', {})
        doctrine_documents = input_data.get('doctrine_documents', [])
        case_id = input_data.get('case_id')
        analysis_focus = input_data.get('analysis_focus', [])

        if not military_force:
            raise ValueError("Military force name/description is required")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Log analysis start
        self.audit_logger.log_analysis_start(
            evidence_id=analysis_id,
            agent_id=self.agent_id,
            metadata={'analysis_type': 'military_structure', 'military_force': military_force, 'case_id': case_id}
        )

        # Prepare input
        user_message = self._prepare_input_message(
            military_force, organizational_data, command_data, operational_data, doctrine_documents, analysis_focus
        )

        # Call Claude
        result = self._call_claude(user_message=user_message)

        # Parse result
        analysis_result = self._parse_result(result, analysis_id, case_id, military_force)

        # Log completion
        self.audit_logger.log_analysis_complete(
            evidence_id=analysis_id,
            agent_id=self.agent_id,
            result_summary={'military_force': military_force}
        )

        return analysis_result

    def _prepare_input_message(
        self, military_force: str, organizational_data: Dict, command_data: Dict,
        operational_data: Dict, doctrine_documents: List, analysis_focus: List
    ) -> str:
        """Prepare input message for Claude."""
        message_parts = [
            f"# Military Structure & Tactics Analysis Request",
            f"",
            f"**Military Force:** {military_force}",
            f""
        ]

        if analysis_focus:
            message_parts.extend([f"**Analysis Focus:** {', '.join(analysis_focus)}", ""])

        if organizational_data:
            message_parts.extend([
                "## Organizational Data",
                f"```json\n{json.dumps(organizational_data, indent=2)}\n```",
                ""
            ])

        if command_data:
            message_parts.extend([
                "## Command Data",
                f"```json\n{json.dumps(command_data, indent=2)}\n```",
                ""
            ])

        if operational_data:
            message_parts.extend([
                "## Operational Data",
                f"```json\n{json.dumps(operational_data, indent=2)}\n```",
                ""
            ])

        if doctrine_documents:
            message_parts.extend(["## Doctrine Documents", ""])
            for i, doc in enumerate(doctrine_documents, 1):
                message_parts.append(f"### Document {i}")
                message_parts.append(str(doc))
                message_parts.append("")

        message_parts.extend([
            "## Instructions",
            "",
            "Please provide comprehensive military analysis according to your capabilities.",
            "Focus on unit structure, command relationships, tactical operations, and command responsibility.",
            "",
            "Provide your analysis in the structured JSON format specified in your system prompt."
        ])

        return "\n".join(message_parts)

    def _parse_result(self, result: str, analysis_id: str, case_id: Optional[str], military_force: str) -> Dict[str, Any]:
        """Parse and structure the analysis result."""
        try:
            if "```json" in result:
                json_start = result.find("```json") + 7
                json_end = result.find("```", json_start)
                json_str = result[json_start:json_end].strip()
                analysis = json.loads(json_str)
            elif result.strip().startswith("{"):
                analysis = json.loads(result)
            else:
                analysis = {
                    "military_analysis_id": analysis_id,
                    "analysis_metadata": {
                        "analysis_date": datetime.utcnow().isoformat(),
                        "case_id": case_id,
                        "military_force_analyzed": military_force
                    },
                    "analysis_text": result
                }

            if "military_analysis_id" not in analysis:
                analysis["military_analysis_id"] = analysis_id

            return analysis

        except json.JSONDecodeError:
            return {
                "military_analysis_id": analysis_id,
                "error": "JSON parsing error",
                "raw_response": result
            }

    def analyze_command_structure(
        self, military_force: str, command_data: Dict, case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convenience method for command structure analysis."""
        return self.process({
            'military_force': military_force,
            'command_data': command_data,
            'analysis_focus': ['command_structure', 'command_responsibility'],
            'case_id': case_id
        })


def analyze_military_operations(
    military_force: str, military_data: Dict[str, Any], case_id: Optional[str] = None,
    config: Optional[MilitaryStructureAnalystConfig] = None
) -> Dict[str, Any]:
    """Quick military analysis function."""
    agent = MilitaryStructureAnalystAgent(config=config)
    return agent.process({'military_force': military_force, **military_data, 'case_id': case_id})
