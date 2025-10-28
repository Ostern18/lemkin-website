"""
Medical & Forensic Record Analyst Agent

Interprets medical and forensic documentation for legal purposes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import BaseAgent, AuditLogger, AuditEventType, EvidenceHandler
from .system_prompt import SYSTEM_PROMPT
from .config import MedicalForensicConfig, DEFAULT_CONFIG


class MedicalForensicAnalystAgent(BaseAgent):
    """
    Agent for analyzing medical and forensic records.

    Handles:
    - Medical report interpretation
    - Torture indicator assessment (Istanbul Protocol)
    - Autopsy/forensic report analysis
    - Consistency checking
    - Legal element mapping
    """

    def __init__(
        self,
        config: Optional[MedicalForensicConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize medical/forensic analyst agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="medical_forensic_analyst",
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
        Analyze medical/forensic record.

        Args:
            input_data: Dictionary containing:
                - record_text: Text of medical/forensic record
                - record_type: Type of record
                - evidence_id: Optional evidence ID
                - **metadata: Additional metadata

        Returns:
            Medical analysis results
        """
        record_text = input_data.get('record_text')
        record_type = input_data.get('record_type', 'medical_report')
        evidence_id = input_data.get('evidence_id')

        if not record_text:
            raise ValueError("record_text is required")

        # Log analysis
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            evidence_ids=[evidence_id] if evidence_id else [],
            details={
                "record_type": record_type,
                "record_length": len(record_text)
            }
        )

        # Perform analysis
        result = self._analyze_record(record_text, record_type, evidence_id)

        # Check if human review needed
        if self._requires_human_review(result):
            review_request = self.request_human_review(
                item_for_review=result,
                review_type=self._determine_review_type(result),
                priority="high"
            )
            result['human_review_requested'] = review_request

        return self.generate_output(
            output_data=result,
            output_type="medical_forensic_analysis",
            evidence_ids=[evidence_id] if evidence_id else []
        )

    def _analyze_record(
        self,
        record_text: str,
        record_type: str,
        evidence_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Analyze medical/forensic record using Claude.

        Args:
            record_text: Record text
            record_type: Type of record
            evidence_id: Evidence ID

        Returns:
            Analysis results
        """
        prompt = self._build_analysis_prompt(record_text, record_type)

        messages = [{"role": "user", "content": prompt}]
        response = self.call_claude(messages)

        return self._parse_analysis_response(response, evidence_id)

    def _build_analysis_prompt(
        self,
        record_text: str,
        record_type: str
    ) -> str:
        """Build analysis prompt."""
        prompt = f"""Please analyze this {record_type.replace('_', ' ')}:

--- MEDICAL/FORENSIC RECORD ---
{record_text}
--- END RECORD ---

Provide comprehensive analysis according to your system prompt, including:
1. Key medical findings (injuries, diagnoses, treatments)
2. Torture indicators (if applicable, using Istanbul Protocol)
3. Temporal and narrative consistency
4. Legal relevance
5. Plain language summary for non-experts

"""
        if self.config.apply_istanbul_protocol:
            prompt += "Apply Istanbul Protocol standards systematically.\n"

        if self.config.redact_patient_identifiers:
            prompt += "Redact patient identifying information in your output.\n"

        prompt += "\nProvide results in JSON format as specified in your system prompt."

        return prompt

    def _parse_analysis_response(
        self,
        response: Dict[str, Any],
        evidence_id: Optional[str]
    ) -> Dict[str, Any]:
        """Parse Claude's analysis response."""
        content = response.get('content', [])

        text_content = None
        for block in content:
            if hasattr(block, 'text'):
                text_content = block.text
                break
            elif isinstance(block, dict) and 'text' in block:
                text_content = block['text']
                break

        if not text_content:
            raise ValueError("No content in response")

        try:
            json_start = text_content.find('{')
            json_end = text_content.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                result = json.loads(text_content[json_start:json_end])
            else:
                result = {"analysis_text": text_content}

        except json.JSONDecodeError:
            result = {"analysis_text": text_content}

        if evidence_id:
            result['evidence_id'] = evidence_id

        result['_processing'] = {
            "model": response.get('model'),
            "tokens_used": response.get('usage')
        }

        return result

    def _requires_human_review(self, result: Dict[str, Any]) -> bool:
        """Determine if human review is required."""
        # Torture findings
        if self.config.require_review_for_torture_findings:
            torture_indicators = result.get('torture_indicators', {})
            if torture_indicators.get('present'):
                return True

        # Death cases
        if self.config.require_review_for_death_cases:
            if result.get('key_findings', {}).get('cause_of_death'):
                return True

        # High-severity inconsistencies
        if self.config.require_review_for_inconsistencies:
            inconsistencies = result.get('inconsistencies_identified', [])
            for inc in inconsistencies:
                if inc.get('severity') == 'high':
                    return True

        return False

    def _determine_review_type(self, result: Dict[str, Any]) -> str:
        """Determine type of review needed."""
        if result.get('torture_indicators', {}).get('present'):
            return "torture_assessment"
        elif result.get('key_findings', {}).get('cause_of_death'):
            return "death_investigation"
        else:
            return "medical_forensic_review"

    def analyze_for_torture(
        self,
        medical_record: str,
        **metadata
    ) -> Dict[str, Any]:
        """
        Analyze medical record specifically for torture indicators.

        Args:
            medical_record: Medical examination text
            **metadata: Additional metadata

        Returns:
            Torture assessment results
        """
        from .config import TORTURE_ASSESSMENT_CONFIG

        # Temporarily use torture-specific config
        original_config = self.config
        self.config = TORTURE_ASSESSMENT_CONFIG

        result = self.process({
            'record_text': medical_record,
            'record_type': 'medical_examination',
            **metadata
        })

        self.config = original_config
        return result

    def analyze_autopsy(
        self,
        autopsy_report: str,
        **metadata
    ) -> Dict[str, Any]:
        """
        Analyze autopsy/pathology report.

        Args:
            autopsy_report: Autopsy report text
            **metadata: Additional metadata

        Returns:
            Autopsy analysis results
        """
        return self.process({
            'record_text': autopsy_report,
            'record_type': 'autopsy',
            **metadata
        })
