"""
Ballistics & Weapons Identifier Agent

Analyzes evidence of weapons and ammunition.
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
from .config import BallisticsWeaponsIdentifierConfig, DEFAULT_CONFIG


class BallisticsWeaponsIdentifierAgent(BaseAgent):
    """
    Agent for identifying weapons, analyzing ammunition, and reviewing ballistics evidence.

    Handles:
    - Weapon identification from photos and descriptions
    - Ammunition analysis and origin tracing
    - Ballistics report interpretation
    - Wound pattern analysis
    - Weapon-to-incident attribution
    - IHL compliance assessment for weapons
    """

    def __init__(
        self,
        config: Optional[BallisticsWeaponsIdentifierConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Ballistics & Weapons Identifier agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="ballistics_weapons_identifier",
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
        Identify weapons and analyze ballistics evidence.

        Args:
            input_data: Dictionary containing:
                - weapon_images: Images or descriptions of weapons
                - ammunition_data: Ammunition markings, photos, descriptions
                - ballistics_reports: Ballistics analysis reports
                - wound_patterns: Medical descriptions of injuries
                - case_id: Optional case ID
                - analysis_focus: Specific aspects to analyze

        Returns:
            Comprehensive weapons and ballistics analysis
        """
        weapon_images = input_data.get('weapon_images', [])
        ammunition_data = input_data.get('ammunition_data', [])
        ballistics_reports = input_data.get('ballistics_reports', [])
        wound_patterns = input_data.get('wound_patterns', [])
        case_id = input_data.get('case_id')
        analysis_focus = input_data.get('analysis_focus', [])

        if not any([weapon_images, ammunition_data, ballistics_reports, wound_patterns]):
            raise ValueError("At least one type of weapons evidence is required")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Log analysis start
        self.audit_logger.log_analysis_start(
            evidence_id=analysis_id,
            agent_id=self.agent_id,
            metadata={'analysis_type': 'weapons_ballistics', 'case_id': case_id}
        )

        # Prepare input
        user_message = self._prepare_input_message(
            weapon_images, ammunition_data, ballistics_reports, wound_patterns, analysis_focus
        )

        # Call Claude
        result = self._call_claude(user_message=user_message)

        # Parse result
        analysis_result = self._parse_result(result, analysis_id, case_id)

        # Log completion
        self.audit_logger.log_analysis_complete(
            evidence_id=analysis_id,
            agent_id=self.agent_id,
            result_summary={'weapons_identified': len(analysis_result.get('weapon_identification', []))}
        )

        return analysis_result

    def _prepare_input_message(
        self, weapon_images: List, ammunition_data: List, ballistics_reports: List, wound_patterns: List, analysis_focus: List
    ) -> str:
        """Prepare input message for Claude."""
        message_parts = ["# Weapons & Ballistics Analysis Request", ""]

        if analysis_focus:
            message_parts.extend([f"**Analysis Focus:** {', '.join(analysis_focus)}", ""])

        if weapon_images:
            message_parts.extend(["## Weapon Images/Descriptions", ""])
            for i, weapon in enumerate(weapon_images, 1):
                message_parts.append(f"### Weapon {i}")
                if isinstance(weapon, dict):
                    message_parts.append(f"```json\n{json.dumps(weapon, indent=2)}\n```")
                else:
                    message_parts.append(str(weapon))
                message_parts.append("")

        if ammunition_data:
            message_parts.extend(["## Ammunition Data", ""])
            for i, ammo in enumerate(ammunition_data, 1):
                message_parts.append(f"### Ammunition {i}")
                if isinstance(ammo, dict):
                    message_parts.append(f"```json\n{json.dumps(ammo, indent=2)}\n```")
                else:
                    message_parts.append(str(ammo))
                message_parts.append("")

        if ballistics_reports:
            message_parts.extend(["## Ballistics Reports", ""])
            for i, report in enumerate(ballistics_reports, 1):
                message_parts.append(f"### Report {i}")
                message_parts.append(str(report))
                message_parts.append("")

        if wound_patterns:
            message_parts.extend(["## Wound Patterns", ""])
            for i, pattern in enumerate(wound_patterns, 1):
                message_parts.append(f"### Pattern {i}")
                message_parts.append(str(pattern))
                message_parts.append("")

        message_parts.extend([
            "## Instructions",
            "",
            "Please provide comprehensive weapons and ballistics analysis according to your capabilities.",
            "Include weapon identification, ammunition analysis, ballistics findings, and attribution analysis.",
            "",
            "Provide your analysis in the structured JSON format specified in your system prompt."
        ])

        return "\n".join(message_parts)

    def _parse_result(self, result: str, analysis_id: str, case_id: Optional[str]) -> Dict[str, Any]:
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
                    "weapon_analysis_id": analysis_id,
                    "analysis_metadata": {
                        "analysis_date": datetime.utcnow().isoformat(),
                        "case_id": case_id
                    },
                    "analysis_text": result
                }

            if "weapon_analysis_id" not in analysis:
                analysis["weapon_analysis_id"] = analysis_id

            return analysis

        except json.JSONDecodeError:
            return {
                "weapon_analysis_id": analysis_id,
                "error": "JSON parsing error",
                "raw_response": result
            }

    def identify_weapon(self, weapon_image_or_description: str, case_id: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method for weapon identification."""
        return self.process({'weapon_images': [weapon_image_or_description], 'case_id': case_id})


def identify_weapons(
    weapons_evidence: Dict[str, Any], case_id: Optional[str] = None, config: Optional[BallisticsWeaponsIdentifierConfig] = None
) -> Dict[str, Any]:
    """Quick weapons identification function."""
    agent = BallisticsWeaponsIdentifierAgent(config=config)
    return agent.process({**weapons_evidence, 'case_id': case_id})
