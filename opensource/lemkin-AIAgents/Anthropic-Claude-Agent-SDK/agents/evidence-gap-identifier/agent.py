"""
Evidence Gap & Next Steps Identifier Agent

Analyzes investigations to identify missing evidence and recommend next steps.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import BaseAgent, AuditLogger, AuditEventType, EvidenceHandler, OutputFormatter
from .system_prompt import SYSTEM_PROMPT
from .config import GapIdentifierConfig, DEFAULT_CONFIG


class EvidenceGapIdentifierAgent(BaseAgent):
    """
    Agent for identifying evidence gaps and recommending next steps.

    Handles:
    - Gap analysis against legal elements
    - Next step prioritization
    - Interview question generation
    - Alternative strategy identification
    - Risk assessment
    """

    def __init__(
        self,
        config: Optional[GapIdentifierConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize gap identifier agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="evidence_gap_identifier",
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
        Analyze case and identify evidence gaps.

        Args:
            input_data: Dictionary containing:
                - charges: List of charges/claims being pursued
                - available_evidence: Summary of available evidence
                - case_theory: Description of case theory
                - witnesses: Information about witnesses
                - **metadata: Additional case information

        Returns:
            Gap analysis and recommendations
        """
        charges = input_data.get('charges', [])
        available_evidence = input_data.get('available_evidence', {})
        case_id = input_data.get('case_id')

        if not charges:
            raise ValueError("charges list is required")

        # Log analysis
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            details={
                "case_id": case_id,
                "charges_count": len(charges),
                "evidence_items_count": len(available_evidence) if isinstance(available_evidence, list) else 0
            }
        )

        # Perform gap analysis
        result = self._analyze_gaps(
            charges=charges,
            available_evidence=available_evidence,
            case_context=input_data
        )

        # Generate formatted output
        formatted_output = self._format_gap_analysis(result, case_id)

        return self.generate_output(
            output_data=formatted_output,
            output_type="gap_analysis",
            evidence_ids=self._extract_evidence_ids(available_evidence)
        )

    def _analyze_gaps(
        self,
        charges: List[Any],
        available_evidence: Any,
        case_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform gap analysis using Claude.

        Args:
            charges: Charges being pursued
            available_evidence: Available evidence summary
            case_context: Full case context

        Returns:
            Gap analysis results
        """
        prompt = self._build_gap_analysis_prompt(charges, available_evidence, case_context)

        messages = [{"role": "user", "content": prompt}]
        response = self.call_claude(messages)

        return self._parse_gap_response(response)

    def _build_gap_analysis_prompt(
        self,
        charges: List[Any],
        available_evidence: Any,
        case_context: Dict[str, Any]
    ) -> str:
        """Build prompt for gap analysis."""
        prompt = "Please perform a comprehensive evidence gap analysis for this case:\n\n"

        prompt += "--- CHARGES/CLAIMS ---\n"
        for i, charge in enumerate(charges, 1):
            if isinstance(charge, dict):
                prompt += f"{i}. {charge.get('charge', charge.get('name', str(charge)))}\n"
            else:
                prompt += f"{i}. {charge}\n"

        prompt += "\n--- AVAILABLE EVIDENCE ---\n"
        if isinstance(available_evidence, list):
            for i, evidence in enumerate(available_evidence, 1):
                prompt += f"{i}. {evidence}\n"
        elif isinstance(available_evidence, dict):
            prompt += json.dumps(available_evidence, indent=2)
        else:
            prompt += str(available_evidence)

        prompt += "\n--- CASE CONTEXT ---\n"
        for key, value in case_context.items():
            if key not in ['charges', 'available_evidence']:
                prompt += f"{key}: {value}\n"

        prompt += "\n--- ANALYSIS REQUEST ---\n"
        prompt += "Please analyze:\n"
        prompt += "1. What evidence is needed for each charge?\n"
        prompt += "2. What critical gaps exist?\n"
        prompt += "3. What specific actions should investigators take next?\n"

        if self.config.generate_interview_questions:
            prompt += "4. What follow-up questions should be asked?\n"

        if self.config.suggest_document_requests:
            prompt += "5. What documents should be requested?\n"

        if self.config.recommend_expert_consultations:
            prompt += "6. What expert consultations are needed?\n"

        if self.config.identify_alternative_strategies:
            prompt += "7. What alternative strategies could work?\n"

        if self.config.assess_risks:
            prompt += "8. What risks should investigators be aware of?\n"

        prompt += f"\nPrioritize the top {self.config.max_priority_actions} most important actions.\n"
        prompt += "\nProvide results in JSON format as specified in your system prompt."

        return prompt

    def _parse_gap_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Claude's gap analysis response."""
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

        result['_processing'] = {
            "model": response.get('model'),
            "tokens_used": response.get('usage')
        }

        return result

    def _format_gap_analysis(
        self,
        result: Dict[str, Any],
        case_id: Optional[str]
    ) -> Dict[str, Any]:
        """Format gap analysis for output."""
        # Add case ID
        if case_id:
            result['case_id'] = case_id

        # Extract priorities for quick reference
        priority_actions = result.get('priority_actions', [])
        result['critical_next_steps'] = [
            action for action in priority_actions
            if action.get('priority') in ['critical', 'high']
        ][:5]

        # Count gaps by severity
        gaps = result.get('evidence_gaps', [])
        result['gap_summary'] = {
            'total_gaps': len(gaps),
            'critical': sum(1 for g in gaps if g.get('severity') == 'critical'),
            'high': sum(1 for g in gaps if g.get('severity') == 'high'),
            'medium': sum(1 for g in gaps if g.get('severity') == 'medium'),
            'low': sum(1 for g in gaps if g.get('severity') == 'low')
        }

        return result

    def _extract_evidence_ids(self, available_evidence: Any) -> List[str]:
        """Extract evidence IDs from available evidence."""
        evidence_ids = []

        if isinstance(available_evidence, list):
            for item in available_evidence:
                if isinstance(item, dict) and 'evidence_id' in item:
                    evidence_ids.append(item['evidence_id'])

        return evidence_ids

    def quick_gap_check(
        self,
        charge: str,
        available_evidence_summary: str,
        **metadata
    ) -> Dict[str, Any]:
        """
        Quick gap check for a single charge.

        Args:
            charge: Charge to assess
            available_evidence_summary: Brief summary of evidence
            **metadata: Additional metadata

        Returns:
            Quick gap assessment
        """
        return self.process({
            'charges': [charge],
            'available_evidence': available_evidence_summary,
            **metadata
        })

    def generate_interview_plan(
        self,
        witness_id: str,
        case_context: Dict[str, Any]
    ) -> List[str]:
        """
        Generate follow-up questions for a witness.

        Args:
            witness_id: Witness identifier
            case_context: Case information

        Returns:
            List of follow-up questions
        """
        result = self.process({
            'charges': case_context.get('charges', []),
            'available_evidence': case_context.get('available_evidence', {}),
            'witnesses': [{
                'witness_id': witness_id,
                'prior_statement': case_context.get('prior_statement', '')
            }]
        })

        # Extract questions for this witness
        interview_questions = result.get('witness_interview_questions', {})
        for witness in interview_questions.get('existing_witnesses', []):
            if witness.get('witness_id') == witness_id:
                return witness.get('follow_up_questions', [])

        return []
