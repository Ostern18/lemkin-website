"""
OSINT Synthesis Agent (OpenAI Agents SDK Implementation)

Aggregates and analyzes publicly available information for investigations.
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
from .config import OSINTConfig, DEFAULT_CONFIG


class OSINTSynthesisAgent:
    """
    Agent for synthesizing and verifying open-source intelligence.

    Built on OpenAI Agents SDK with evidentiary compliance.

    Capabilities:
    - Multi-source information aggregation
    - Claim verification and source assessment
    - Pattern detection (coordinated campaigns, temporal trends)
    - Intelligence brief generation
    - Geographic/temporal heat mapping
    """

    def __init__(
        self,
        config: Optional[OSINTConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """
        Initialize OSINT Synthesis agent.

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
            agent_id="osint_synthesis",
            name="OSINT Synthesis Agent",
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
        Process OSINT data and generate synthesis report.

        Args:
            input_data: Dictionary containing:
                - message: Analysis request or claim to verify
                - sources: Optional list of source data
                - case_id: Optional case identifier
                - evidence_ids: Optional list of evidence IDs
                - context: Additional context information

        Returns:
            Dictionary containing synthesis results
        """
        message = input_data.get('message')
        sources = input_data.get('sources', [])
        case_id = input_data.get('case_id')
        evidence_ids = input_data.get('evidence_ids', [])

        if not message:
            raise ValueError("message is required")

        # Log processing start
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="osint_synthesis",
            evidence_ids=evidence_ids,
            details={
                "case_id": case_id,
                "source_count": len(sources)
            }
        )

        # Prepare context with sources
        context_variables = {
            "sources": sources,
            "case_id": case_id,
            "config": {
                "min_sources": self.config.minimum_sources_for_verification,
                "credibility_threshold": self.config.high_credibility_threshold
            }
        }

        # Run the agent using OpenAI Agents SDK
        result = self.agent.run(
            message=message,
            context_variables=context_variables,
            evidence_ids=evidence_ids
        )

        # Parse the output (expect JSON from agent)
        try:
            analysis = json.loads(result['final_output'])
        except json.JSONDecodeError:
            # If not JSON, wrap in structure
            analysis = {
                "executive_summary": result['final_output'],
                "findings": [],
                "credibility": "unknown"
            }

        # Check if human review is needed
        if self._needs_human_review(analysis):
            review_request = self.agent.request_human_review(
                item_for_review=analysis,
                review_type="low_credibility_osint",
                priority="normal"
            )
            analysis['human_review_requested'] = review_request

        # Generate formatted output
        output = self.agent.generate_output(
            output_data={
                "analysis": analysis,
                "case_id": case_id,
                "sources_analyzed": len(sources),
                "messages": result.get('messages', [])
            },
            output_type="osint_synthesis",
            evidence_ids=evidence_ids
        )

        return output

    def verify_claim(
        self,
        claim: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        case_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Verify a specific claim using OSINT methods.

        Args:
            claim: Claim to verify
            sources: Optional source data
            case_id: Optional case identifier

        Returns:
            Verification results
        """
        return self.process({
            'message': f"Verify this claim using OSINT methods: {claim}",
            'sources': sources or [],
            'case_id': case_id
        })

    def assess_source(
        self,
        source_url: str,
        source_content: str,
        case_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Assess credibility of a specific source.

        Args:
            source_url: URL of the source
            source_content: Content from the source
            case_id: Optional case identifier

        Returns:
            Credibility assessment
        """
        return self.process({
            'message': f"Assess the credibility of this source: {source_url}",
            'sources': [{'url': source_url, 'content': source_content}],
            'case_id': case_id
        })

    def _needs_human_review(self, analysis: Dict[str, Any]) -> bool:
        """
        Determine if analysis requires human review.

        Args:
            analysis: Analysis results

        Returns:
            True if human review needed
        """
        # Check credibility level
        credibility = analysis.get('credibility', 'unknown')

        if credibility == 'low':
            return True

        # Check if credibility score is below threshold
        credibility_score = analysis.get('credibility_score', 1.0)
        if credibility_score < self.config.require_human_review_below_credibility:
            return True

        # Check if insufficient sources
        source_count = analysis.get('sources_verified', 0)
        if source_count < self.config.minimum_sources_for_verification:
            return True

        return False
