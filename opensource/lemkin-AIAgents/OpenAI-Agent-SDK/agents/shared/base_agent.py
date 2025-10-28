"""
Base Agent Wrapper for LemkinAI Agents (OpenAI Agents SDK Implementation)
Wraps the OpenAI Agents SDK with evidentiary compliance features.
"""

import os
import base64
from typing import Dict, Any, List, Optional, Callable
from agents import Agent, Runner, handoff
from .audit_logger import AuditLogger, AuditEventType


class LemkinAgent:
    """
    Wrapper around OpenAI Agents SDK Agent class with evidentiary compliance.

    This class integrates OpenAI's Agent framework with LemkinAI's
    chain-of-custody tracking, audit logging, and evidence handling.

    Provides:
    - Audit logging integration
    - Human-in-the-loop gates
    - Evidence tracking
    - Standard configuration patterns
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        instructions: str,
        model: str = "gpt-4o",
        tools: Optional[List[Callable]] = None,
        handoffs: Optional[List] = None,
        audit_logger: Optional[AuditLogger] = None,
        enable_human_review: bool = True,
        **agent_kwargs
    ):
        """
        Initialize LemkinAI agent wrapper.

        Args:
            agent_id: Unique identifier for this agent (for audit logs)
            name: Agent name (used by OpenAI Agents SDK)
            instructions: System instructions defining agent role and capabilities
            model: Model to use (default: gpt-4o)
            tools: Optional list of function tools the agent can use
            handoffs: Optional list of agents this agent can handoff to
            audit_logger: AuditLogger instance (creates new one if not provided)
            enable_human_review: Whether to enable human-in-the-loop gates
            **agent_kwargs: Additional arguments passed to Agent constructor
        """
        self.agent_id = agent_id
        self.enable_human_review = enable_human_review

        # Initialize audit logging
        self.audit_logger = audit_logger or AuditLogger()

        # Create the underlying OpenAI Agent
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
            tools=tools or [],
            handoffs=handoffs or [],
            **agent_kwargs
        )

        self._log_initialization(model)

    def _log_initialization(self, model: str):
        """Log agent initialization."""
        self.audit_logger.log_event(
            event_type=AuditEventType.AGENT_INITIALIZED,
            agent_id=self.agent_id,
            details={
                "name": self.agent.name,
                "model": model,
                "human_review_enabled": self.enable_human_review
            }
        )

    def run(
        self,
        message: str,
        context_variables: Optional[Dict[str, Any]] = None,
        evidence_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the agent with a message and track in audit log.

        Args:
            message: User message to process
            context_variables: Optional context variables for the agent
            evidence_ids: Optional list of evidence IDs being processed
            **kwargs: Additional arguments passed to Runner

        Returns:
            Dictionary containing agent response and metadata
        """
        # Log the execution start
        self.audit_logger.log_event(
            event_type=AuditEventType.TOOL_EXECUTED,
            agent_id=self.agent_id,
            evidence_ids=evidence_ids or [],
            details={
                "tool": "openai_agents_sdk",
                "message_length": len(message)
            }
        )

        try:
            # Run the agent using OpenAI Agents SDK Runner
            result = Runner.run_sync(
                agent=self.agent,
                messages=message,
                context_variables=context_variables or {},
                **kwargs
            )

            # Log successful completion
            self.audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=self.agent_id,
                evidence_ids=evidence_ids or [],
                details={
                    "status": "success",
                    "final_agent": result.final_agent,
                    "message_count": len(result.messages)
                }
            )

            return {
                "final_output": result.final_output,
                "messages": result.messages,
                "final_agent": result.final_agent,
                "context_variables": result.context_variables,
                "_audit_session": self.audit_logger.session_id
            }

        except Exception as e:
            # Log error
            self.audit_logger.log_event(
                event_type=AuditEventType.ERROR_OCCURRED,
                agent_id=self.agent_id,
                evidence_ids=evidence_ids or [],
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "operation": "agent_run"
                }
            )
            raise

    async def run_async(
        self,
        message: str,
        context_variables: Optional[Dict[str, Any]] = None,
        evidence_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the agent asynchronously with audit logging.

        Args:
            message: User message to process
            context_variables: Optional context variables for the agent
            evidence_ids: Optional list of evidence IDs being processed
            **kwargs: Additional arguments passed to Runner

        Returns:
            Dictionary containing agent response and metadata
        """
        # Log the execution start
        self.audit_logger.log_event(
            event_type=AuditEventType.TOOL_EXECUTED,
            agent_id=self.agent_id,
            evidence_ids=evidence_ids or [],
            details={
                "tool": "openai_agents_sdk_async",
                "message_length": len(message)
            }
        )

        try:
            # Run the agent using OpenAI Agents SDK Runner (async)
            result = await Runner.run(
                agent=self.agent,
                messages=message,
                context_variables=context_variables or {},
                **kwargs
            )

            # Log successful completion
            self.audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=self.agent_id,
                evidence_ids=evidence_ids or [],
                details={
                    "status": "success",
                    "final_agent": result.final_agent,
                    "message_count": len(result.messages)
                }
            )

            return {
                "final_output": result.final_output,
                "messages": result.messages,
                "final_agent": result.final_agent,
                "context_variables": result.context_variables,
                "_audit_session": self.audit_logger.session_id
            }

        except Exception as e:
            # Log error
            self.audit_logger.log_event(
                event_type=AuditEventType.ERROR_OCCURRED,
                agent_id=self.agent_id,
                evidence_ids=evidence_ids or [],
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "operation": "agent_run_async"
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


# Backward compatibility aliases
BaseAgent = LemkinAgent
VisionCapableAgent = LemkinAgent  # OpenAI Agents SDK has built-in vision support
