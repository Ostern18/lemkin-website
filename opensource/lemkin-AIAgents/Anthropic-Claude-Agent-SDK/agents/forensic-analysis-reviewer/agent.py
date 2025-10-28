"""
Forensic Analysis Reviewer Agent

Interprets forensic reports for legal teams.
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
from .config import ForensicAnalysisReviewerConfig, DEFAULT_CONFIG


class ForensicAnalysisReviewerAgent(BaseAgent):
    """
    Agent for reviewing and interpreting forensic reports for legal proceedings.

    Handles:
    - DNA, ballistics, autopsy, toxicology, and trace evidence reports
    - Translation of technical findings to legal context
    - Methodology assessment and quality review
    - Expert opinion evaluation
    - Non-expert summaries and explanations
    """

    def __init__(
        self,
        config: Optional[ForensicAnalysisReviewerConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Forensic Analysis Reviewer agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="forensic_analysis_reviewer",
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
        Review and interpret forensic reports.

        Args:
            input_data: Dictionary containing:
                - forensic_report: Report content or file path
                - report_type: DNA|ballistics|autopsy|toxicology|trace|multiple
                - case_id: Optional case ID
                - charges: Charges to map evidence to
                - review_focus: Specific aspects to focus on

        Returns:
            Comprehensive forensic review with legal analysis
        """
        forensic_report = input_data.get('forensic_report')
        report_type = input_data.get('report_type', 'multiple')
        case_id = input_data.get('case_id')
        charges = input_data.get('charges', [])
        review_focus = input_data.get('review_focus', [])

        if not forensic_report:
            raise ValueError("Forensic report is required")

        # Generate review ID
        review_id = str(uuid.uuid4())

        # Log analysis start
        self.audit_logger.log_analysis_start(
            evidence_id=review_id,
            agent_id=self.agent_id,
            metadata={'analysis_type': 'forensic_review', 'report_type': report_type, 'case_id': case_id}
        )

        # Prepare input
        user_message = self._prepare_input_message(
            forensic_report, report_type, charges, review_focus
        )

        # Call Claude
        result = self._call_claude(user_message=user_message)

        # Parse result
        analysis_result = self._parse_result(result, review_id, case_id, report_type)

        # Log completion
        self.audit_logger.log_analysis_complete(
            evidence_id=review_id,
            agent_id=self.agent_id,
            result_summary={'report_type': report_type, 'key_findings_count': len(analysis_result.get('key_findings', []))}
        )

        return analysis_result

    def _prepare_input_message(
        self, forensic_report: str, report_type: str, charges: List[str], review_focus: List[str]
    ) -> str:
        """Prepare input message for Claude."""
        message_parts = [
            f"# Forensic Report Review Request",
            f"",
            f"**Report Type:** {report_type}",
            f""
        ]

        if charges:
            message_parts.extend([
                f"**Charges to Consider:** {', '.join(charges)}",
                f""
            ])

        if review_focus:
            message_parts.extend([
                f"**Review Focus:** {', '.join(review_focus)}",
                f""
            ])

        message_parts.extend([
            f"## Forensic Report",
            f"",
            forensic_report,
            f"",
            f"## Instructions",
            f"",
            f"Please provide a comprehensive review of this forensic report according to your capabilities.",
            f"Focus on:",
            f"1. Identifying key findings and their legal relevance",
            f"2. Assessing methodology and evidence quality",
            f"3. Evaluating expert opinions and conclusions",
            f"4. Mapping findings to legal elements",
            f"5. Generating accessible summaries for non-experts",
            f"",
            f"Provide your analysis in the structured JSON format specified in your system prompt."
        ])

        return "\n".join(message_parts)

    def _parse_result(self, result: str, review_id: str, case_id: Optional[str], report_type: str) -> Dict[str, Any]:
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
                    "forensic_review_id": review_id,
                    "report_metadata": {
                        "report_type": report_type,
                        "review_date": datetime.utcnow().isoformat(),
                        "case_id": case_id
                    },
                    "analysis_text": result
                }

            if "forensic_review_id" not in analysis:
                analysis["forensic_review_id"] = review_id

            return analysis

        except json.JSONDecodeError:
            return {
                "forensic_review_id": review_id,
                "error": "JSON parsing error",
                "raw_response": result
            }

    def review_dna_report(self, report: str, case_id: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method for DNA report review."""
        return self.process({'forensic_report': report, 'report_type': 'DNA', 'case_id': case_id})

    def review_autopsy_report(self, report: str, case_id: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method for autopsy report review."""
        return self.process({'forensic_report': report, 'report_type': 'autopsy', 'case_id': case_id})


def review_forensic_report(
    report: str, report_type: str = 'multiple', case_id: Optional[str] = None, config: Optional[ForensicAnalysisReviewerConfig] = None
) -> Dict[str, Any]:
    """Quick forensic report review function."""
    agent = ForensicAnalysisReviewerAgent(config=config)
    return agent.process({'forensic_report': report, 'report_type': report_type, 'case_id': case_id})
