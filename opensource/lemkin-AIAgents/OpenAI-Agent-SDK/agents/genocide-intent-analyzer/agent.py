"""
Genocide Intent Analyzer Agent (OpenAI Agents SDK Implementation)

Analyzes evidence for genocide intent.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    LemkinAgent,
    AuditLogger,
    AuditEventType,
    EvidenceHandler,
    EvidenceType,
    OutputFormatter
)
from .system_prompt import SYSTEM_PROMPT
from .config import GenocideConfig, DEFAULT_CONFIG


class GenocideIntentAnalyzerAgent:
    """
    Agent for genocide intent analysis.

    Built on OpenAI Agents SDK with evidentiary compliance.
    """

    def __init__(
        self,
        config: Optional[GenocideConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """
        Initialize Genocide Intent Analyzer.

        Args:
            config: Agent configuration
            audit_logger: Audit logger instance
            evidence_handler: Evidence handler instance
            **kwargs: Additional arguments for LemkinAgent
        """
        self.config = config or DEFAULT_CONFIG
        self.audit_logger = audit_logger or AuditLogger()
        self.evidence_handler = evidence_handler or EvidenceHandler()

        # Create the underlying LemkinAgent (wraps OpenAI Agent)
        self.agent = LemkinAgent(
            agent_id="genocide_intent_analyzer",
            name="Genocide Intent Analyzer",
            instructions=SYSTEM_PROMPT,
            model=self.config.model,
            audit_logger=self.audit_logger,
            enable_human_review=True,
            **kwargs
        )

    def process(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process input and generate analysis.

        Args:
            input_data: Dictionary containing:
                - message: Analysis request
                - case_id: Optional case identifier
                - evidence_ids: Optional list of evidence IDs
                - Additional agent-specific parameters

        Returns:
            Dictionary containing analysis results
        """
        message = input_data.get('message')
        case_id = input_data.get('case_id')
        evidence_ids = input_data.get('evidence_ids', [])

        if not message:
            raise ValueError("message is required")

        # Log processing start
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="genocide_intent_analyzer",
            evidence_ids=evidence_ids,
            details={
                "case_id": case_id,
                "operation": "process"
            }
        )

        # Prepare context
        context_variables = {
            "case_id": case_id,
            **{k: v for k, v in input_data.items() if k not in ['message', 'case_id', 'evidence_ids']}
        }

        # Run the agent using OpenAI Agents SDK
        result = self.agent.run(
            message=message,
            context_variables=context_variables,
            evidence_ids=evidence_ids
        )

        # Parse output if JSON
        try:
            analysis = json.loads(result['final_output'])
        except (json.JSONDecodeError, KeyError):
            analysis = {
                "output": result.get('final_output', ''),
                "messages": result.get('messages', [])
            }

        # Generate formatted output
        output = self.agent.generate_output(
            output_data={
                "analysis": analysis,
                "case_id": case_id,
                "messages": result.get('messages', [])
            },
            output_type="genocide_analysis",
            evidence_ids=evidence_ids
        )

        return output
