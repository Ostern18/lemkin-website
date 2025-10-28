"""
Base Agent Class for LemkinAI Agents
Provides common functionality for all investigative agents.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from anthropic import Anthropic
from .audit_logger import AuditLogger, AuditEventType


class BaseAgent(ABC):
    """
    Abstract base class for all LemkinAI agents.

    Provides:
    - Claude API client management
    - Audit logging integration
    - Human-in-the-loop gates
    - Standard configuration patterns
    - Error handling and recovery
    """

    def __init__(
        self,
        agent_id: str,
        system_prompt: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        audit_logger: Optional[AuditLogger] = None,
        enable_human_review: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique identifier for this agent
            system_prompt: System prompt defining agent role and capabilities
            model: Claude model to use (default: Sonnet 4.5)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower = more focused)
            audit_logger: AuditLogger instance (creates new one if not provided)
            enable_human_review: Whether to enable human-in-the-loop gates
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_human_review = enable_human_review

        # Initialize Claude client
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided or set in environment")
        self.client = Anthropic(api_key=api_key)

        # Initialize audit logging
        self.audit_logger = audit_logger or AuditLogger()
        self._log_initialization()

    def _log_initialization(self):
        """Log agent initialization."""
        self.audit_logger.log_event(
            event_type=AuditEventType.AGENT_INITIALIZED,
            agent_id=self.agent_id,
            details={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "human_review_enabled": self.enable_human_review
            }
        )

    @abstractmethod
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process input and return results.

        This is the main entry point for each agent and must be implemented
        by subclasses.

        Args:
            input_data: Input data for the agent to process
            **kwargs: Additional parameters

        Returns:
            Dictionary containing processing results
        """
        pass

    def call_claude(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a call to Claude API with standard error handling and logging.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional override of default system prompt
            temperature: Optional override of default temperature
            max_tokens: Optional override of default max_tokens
            **kwargs: Additional parameters to pass to Claude API

        Returns:
            Dictionary containing Claude's response and metadata
        """
        # Log the API call
        self.audit_logger.log_event(
            event_type=AuditEventType.TOOL_EXECUTED,
            agent_id=self.agent_id,
            details={
                "tool": "claude_api",
                "message_count": len(messages),
                "system_prompt_override": system_prompt is not None
            }
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system_prompt or self.system_prompt,
                messages=messages,
                **kwargs
            )

            # Log successful completion
            self.audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=self.agent_id,
                details={
                    "status": "success",
                    "tokens_used": {
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens
                    },
                    "stop_reason": response.stop_reason
                }
            )

            return {
                "content": response.content,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "stop_reason": response.stop_reason,
                "model": response.model
            }

        except Exception as e:
            # Log error
            self.audit_logger.log_event(
                event_type=AuditEventType.ERROR_OCCURRED,
                agent_id=self.agent_id,
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "operation": "claude_api_call"
                }
            )
            raise

    def request_human_review(
        self,
        item_for_review: Dict[str, Any],
        review_type: str,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Request human review for high-stakes decisions.

        Args:
            item_for_review: Data requiring review
            review_type: Type of review needed (e.g., "evidence_verification", "legal_analysis")
            priority: Priority level ("low", "normal", "high", "critical")

        Returns:
            Dictionary containing review request details
        """
        # Log review request
        event_id = self.audit_logger.log_event(
            event_type=AuditEventType.HUMAN_REVIEW_REQUESTED,
            agent_id=self.agent_id,
            details={
                "review_type": review_type,
                "priority": priority,
                "item": item_for_review
            }
        )

        # In production, this would integrate with a review queue system
        # For now, return the review request details
        return {
            "review_request_id": event_id,
            "status": "pending_review",
            "review_type": review_type,
            "priority": priority,
            "requested_at": self.audit_logger.last_event_hash
        }

    def complete_human_review(
        self,
        review_request_id: str,
        decision: str,
        reviewer_notes: Optional[str] = None,
        reviewer_id: Optional[str] = None
    ):
        """
        Log completion of human review.

        Args:
            review_request_id: ID from request_human_review()
            decision: Review decision ("approved", "rejected", "needs_modification")
            reviewer_notes: Optional notes from reviewer
            reviewer_id: Optional reviewer identifier
        """
        self.audit_logger.log_event(
            event_type=AuditEventType.HUMAN_REVIEW_COMPLETED,
            agent_id=self.agent_id,
            details={
                "review_request_id": review_request_id,
                "decision": decision,
                "notes": reviewer_notes
            },
            metadata={"reviewer_id": reviewer_id}
        )

    def generate_output(
        self,
        output_data: Dict[str, Any],
        output_type: str,
        evidence_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate and log final output.

        Args:
            output_data: Output data to generate
            output_type: Type of output (e.g., "report", "memo", "analysis")
            evidence_ids: List of evidence IDs used in this output

        Returns:
            Output data with metadata
        """
        # Log output generation
        event_id = self.audit_logger.log_event(
            event_type=AuditEventType.OUTPUT_GENERATED,
            agent_id=self.agent_id,
            evidence_ids=evidence_ids or [],
            details={
                "output_type": output_type,
                "output_size": len(str(output_data))
            }
        )

        # Add metadata to output
        output_with_metadata = {
            **output_data,
            "_metadata": {
                "agent_id": self.agent_id,
                "output_id": event_id,
                "output_type": output_type,
                "evidence_ids": evidence_ids or [],
                "audit_session": self.audit_logger.session_id
            }
        }

        return output_with_metadata

    def get_chain_of_custody(self, evidence_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve complete chain-of-custody for evidence.

        Args:
            evidence_id: Evidence identifier

        Returns:
            List of all audit events for this evidence
        """
        return self.audit_logger.get_evidence_chain(evidence_id)

    def verify_integrity(self) -> bool:
        """
        Verify integrity of audit trail.

        Returns:
            True if audit chain is intact, False otherwise
        """
        return self.audit_logger.verify_chain_integrity()

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of agent's session activity.

        Returns:
            Session summary including all events and statistics
        """
        return self.audit_logger.get_session_summary()


class VisionCapableAgent(BaseAgent):
    """
    Extended base class for agents that process images and PDFs.

    Adds helper methods for vision-based processing using Claude's native
    vision capabilities.
    """

    def process_image(
        self,
        image_data: bytes,
        prompt: str,
        media_type: str = "image/jpeg",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image using Claude's vision capabilities.

        Args:
            image_data: Image bytes
            prompt: Processing instructions
            media_type: MIME type of image (image/jpeg, image/png, image/webp, image/gif)
            **kwargs: Additional parameters

        Returns:
            Processing results
        """
        import base64

        # Encode image
        image_base64 = base64.standard_b64encode(image_data).decode("utf-8")

        # Create message with image
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]

        return self.call_claude(messages, **kwargs)

    def process_pdf(
        self,
        pdf_data: bytes,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a PDF using Claude's native PDF processing.

        Args:
            pdf_data: PDF bytes
            prompt: Processing instructions
            **kwargs: Additional parameters

        Returns:
            Processing results
        """
        import base64

        # Encode PDF
        pdf_base64 = base64.standard_b64encode(pdf_data).decode("utf-8")

        # Create message with PDF
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_base64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]

        return self.call_claude(messages, **kwargs)
