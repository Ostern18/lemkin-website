"""
Audit Logger for LemkinAI Agents
Provides immutable audit trails for chain-of-custody and evidentiary compliance.
"""

import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum


class AuditEventType(Enum):
    """Types of auditable events in the evidence processing pipeline."""
    AGENT_INITIALIZED = "agent_initialized"
    EVIDENCE_INGESTED = "evidence_ingested"
    EVIDENCE_PROCESSED = "evidence_processed"
    ANALYSIS_PERFORMED = "analysis_performed"
    OUTPUT_GENERATED = "output_generated"
    TOOL_EXECUTED = "tool_executed"
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"
    ERROR_OCCURRED = "error_occurred"
    CHAIN_VERIFIED = "chain_verified"


class AuditLogger:
    """
    Immutable audit logging system for maintaining chain-of-custody.

    Each audit entry includes:
    - Unique event ID
    - Timestamp (UTC)
    - Event type
    - Agent identifier
    - Evidence identifier(s)
    - Action details
    - Hash of previous event (blockchain-style linking)
    - Digital signature/hash of current event
    """

    def __init__(self, log_directory: Optional[Path] = None, session_id: Optional[str] = None):
        """
        Initialize audit logger.

        Args:
            log_directory: Directory to store audit logs (defaults to ./audit_logs)
            session_id: Unique session identifier (auto-generated if not provided)
        """
        self.log_directory = log_directory or Path("./audit_logs")
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or str(uuid.uuid4())
        self.log_file = self.log_directory / f"audit_{self.session_id}.jsonl"

        # Track last event hash for blockchain-style linking
        self.last_event_hash: Optional[str] = None

        # Initialize session
        self._log_session_start()

    def _log_session_start(self):
        """Log session initialization."""
        self.log_event(
            event_type=AuditEventType.AGENT_INITIALIZED,
            agent_id="audit_logger",
            details={"session_id": self.session_id}
        )

    def log_event(
        self,
        event_type: AuditEventType,
        agent_id: str,
        evidence_ids: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an auditable event with full chain-of-custody information.

        Args:
            event_type: Type of event being logged
            agent_id: Identifier of the agent performing the action
            evidence_ids: List of evidence identifiers affected by this event
            details: Detailed information about the event
            metadata: Additional metadata (user, permissions, etc.)

        Returns:
            Event ID of the logged event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build event record
        event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_type": event_type.value,
            "agent_id": agent_id,
            "evidence_ids": evidence_ids or [],
            "details": details or {},
            "metadata": metadata or {},
            "previous_event_hash": self.last_event_hash,
        }

        # Calculate hash of this event for chain integrity
        event_hash = self._hash_event(event)
        event["event_hash"] = event_hash

        # Write to append-only log file
        self._write_event(event)

        # Update chain
        self.last_event_hash = event_hash

        return event_id

    def _hash_event(self, event: Dict[str, Any]) -> str:
        """
        Generate cryptographic hash of event for chain integrity.

        Args:
            event: Event dictionary to hash

        Returns:
            SHA-256 hash of the event
        """
        # Create deterministic JSON representation
        event_json = json.dumps(event, sort_keys=True, default=str)
        return hashlib.sha256(event_json.encode()).hexdigest()

    def _write_event(self, event: Dict[str, Any]):
        """
        Write event to append-only log file.

        Args:
            event: Event dictionary to write
        """
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event, default=str) + '\n')

    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the audit chain by checking all event hashes.

        Returns:
            True if chain is intact, False if tampering detected
        """
        if not self.log_file.exists():
            return True

        previous_hash = None

        with open(self.log_file, 'r') as f:
            for line in f:
                event = json.loads(line)

                # Verify previous hash matches
                if event["previous_event_hash"] != previous_hash:
                    return False

                # Verify event hash is correct
                stored_hash = event.pop("event_hash")
                calculated_hash = self._hash_event(event)

                if stored_hash != calculated_hash:
                    return False

                previous_hash = stored_hash

        return True

    def get_evidence_chain(self, evidence_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve complete chain-of-custody for a specific piece of evidence.

        Args:
            evidence_id: Identifier of the evidence

        Returns:
            List of all audit events involving this evidence
        """
        chain = []

        if not self.log_file.exists():
            return chain

        with open(self.log_file, 'r') as f:
            for line in f:
                event = json.loads(line)
                if evidence_id in event.get("evidence_ids", []):
                    chain.append(event)

        return chain

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Generate summary of all activity in this session.

        Returns:
            Summary statistics and key events
        """
        if not self.log_file.exists():
            return {"session_id": self.session_id, "total_events": 0}

        events = []
        event_counts = {}
        evidence_set = set()
        agent_set = set()

        with open(self.log_file, 'r') as f:
            for line in f:
                event = json.loads(line)
                events.append(event)

                # Count event types
                event_type = event["event_type"]
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

                # Track unique evidence and agents
                evidence_set.update(event.get("evidence_ids", []))
                agent_set.add(event["agent_id"])

        return {
            "session_id": self.session_id,
            "total_events": len(events),
            "event_type_counts": event_counts,
            "unique_evidence_items": len(evidence_set),
            "unique_agents": len(agent_set),
            "agents": list(agent_set),
            "start_time": events[0]["timestamp"] if events else None,
            "end_time": events[-1]["timestamp"] if events else None,
            "chain_integrity_verified": self.verify_chain_integrity()
        }


class AuditDecorator:
    """
    Decorator for automatically logging agent operations.

    Usage:
        @AuditDecorator.log_operation(AuditEventType.ANALYSIS_PERFORMED)
        def analyze_document(self, doc):
            # ... analysis code
            return results
    """

    @staticmethod
    def log_operation(event_type: AuditEventType):
        """
        Decorator to automatically log agent operations.

        Args:
            event_type: Type of event to log
        """
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                # Assume self has audit_logger and agent_id attributes
                if not hasattr(self, 'audit_logger'):
                    return func(self, *args, **kwargs)

                agent_id = getattr(self, 'agent_id', self.__class__.__name__)

                # Extract evidence IDs if present
                evidence_ids = kwargs.get('evidence_ids', [])

                # Log start of operation
                self.audit_logger.log_event(
                    event_type=event_type,
                    agent_id=agent_id,
                    evidence_ids=evidence_ids,
                    details={
                        "operation": func.__name__,
                        "started": True
                    }
                )

                try:
                    result = func(self, *args, **kwargs)

                    # Log successful completion
                    self.audit_logger.log_event(
                        event_type=event_type,
                        agent_id=agent_id,
                        evidence_ids=evidence_ids,
                        details={
                            "operation": func.__name__,
                            "completed": True,
                            "status": "success"
                        }
                    )

                    return result

                except Exception as e:
                    # Log error
                    self.audit_logger.log_event(
                        event_type=AuditEventType.ERROR_OCCURRED,
                        agent_id=agent_id,
                        evidence_ids=evidence_ids,
                        details={
                            "operation": func.__name__,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                    raise

            return wrapper
        return decorator
