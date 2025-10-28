"""
Social Media Evidence Harvester Agent

Collects and contextualizes social media posts as legal evidence.
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import BaseAgent, AuditLogger, AuditEventType, EvidenceHandler, OutputFormatter
from .system_prompt import SYSTEM_PROMPT
from .config import SocialMediaHarvesterConfig, DEFAULT_CONFIG


class SocialMediaHarvesterAgent(BaseAgent):
    """
    Agent for harvesting and analyzing social media content as legal evidence.

    Handles:
    - Screenshot analysis of social media posts
    - Metadata extraction and documentation
    - Authenticity assessment
    - Context preservation for legal use
    - Chain of custody documentation
    - Network analysis of interactions
    """

    def __init__(
        self,
        config: Optional[SocialMediaHarvesterConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Social Media Evidence Harvester agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="social_media_harvester",
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
        Analyze social media content and create legal evidence documentation.

        Args:
            input_data: Dictionary containing:
                - screenshot_data: Base64 encoded screenshot or file path
                - screenshot_filename: Original filename
                - case_id: Optional case ID
                - collector_info: Information about who collected the evidence
                - collection_context: Context of the collection
                - analysis_focus: Specific aspects to analyze

        Returns:
            Comprehensive evidence documentation
        """
        screenshot_data = input_data.get('screenshot_data')
        screenshot_filename = input_data.get('screenshot_filename')
        case_id = input_data.get('case_id')
        collector_info = input_data.get('collector_info', 'Unknown')
        collection_context = input_data.get('collection_context', '')
        analysis_focus = input_data.get('analysis_focus', [])

        if not screenshot_data:
            raise ValueError("Screenshot data required for analysis")

        # Generate evidence ID and calculate hash
        evidence_id = str(uuid.uuid4())
        if isinstance(screenshot_data, str) and screenshot_data.startswith('data:'):
            # Base64 data
            file_hash = hashlib.sha256(screenshot_data.encode()).hexdigest()
        else:
            # File path or binary data
            file_hash = hashlib.sha256(str(screenshot_data).encode()).hexdigest()

        # Log evidence collection
        self.audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_PROCESSED,
            agent_id=self.agent_id,
            details={
                "evidence_id": evidence_id,
                "case_id": case_id,
                "filename": screenshot_filename,
                "file_hash": file_hash,
                "collector": collector_info,
                "context": collection_context
            }
        )

        # Perform analysis
        result = self._analyze_social_media_content(
            screenshot_data=screenshot_data,
            screenshot_filename=screenshot_filename,
            evidence_id=evidence_id,
            file_hash=file_hash,
            collector_info=collector_info,
            case_id=case_id,
            analysis_focus=analysis_focus
        )

        # Store evidence
        if self.config.preserve_chain_of_custody:
            self.evidence_handler.store_evidence(
                evidence_id=evidence_id,
                evidence_type="social_media_post",
                content=result,
                source_hash=file_hash,
                metadata={
                    "original_filename": screenshot_filename,
                    "collector": collector_info,
                    "case_id": case_id
                }
            )

        return self.generate_output(
            output_data=result,
            output_type="social_media_evidence",
            evidence_ids=[evidence_id]
        )

    def _analyze_social_media_content(
        self,
        screenshot_data: Any,
        screenshot_filename: str,
        evidence_id: str,
        file_hash: str,
        collector_info: str,
        case_id: Optional[str] = None,
        analysis_focus: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze social media screenshot and extract evidence."""

        # Prepare analysis prompt
        analysis_prompt = self._build_analysis_prompt(
            screenshot_filename=screenshot_filename,
            evidence_id=evidence_id,
            file_hash=file_hash,
            collector_info=collector_info,
            case_id=case_id,
            analysis_focus=analysis_focus
        )

        # Perform Claude analysis with vision
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": analysis_prompt
                    }
                ]
            }
        ]

        # Add image data if available
        if isinstance(screenshot_data, str) and screenshot_data.startswith('data:'):
            # Base64 encoded image
            media_type = "image/png"  # Default, could be detected
            if "data:image/jpeg" in screenshot_data:
                media_type = "image/jpeg"
            elif "data:image/png" in screenshot_data:
                media_type = "image/png"

            base64_data = screenshot_data.split(',')[1] if ',' in screenshot_data else screenshot_data

            messages[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data
                }
            })

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=messages
        )

        # Parse response
        response_text = response.content[0].text if response.content else ""

        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
            else:
                # Fallback if no JSON found
                result = {
                    "evidence_id": evidence_id,
                    "error": "Could not parse analysis results",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            result = {
                "evidence_id": evidence_id,
                "error": "Invalid JSON in analysis results",
                "raw_response": response_text
            }

        # Add metadata
        result["analysis_metadata"] = {
            "analyzed_by": self.agent_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "model_used": self.model,
            "config_settings": {
                "authenticity_assessment": self.config.perform_authenticity_assessment,
                "network_analysis": self.config.analyze_network_patterns,
                "legal_assessment": self.config.assess_legal_admissibility
            }
        }

        return result

    def _build_analysis_prompt(
        self,
        screenshot_filename: str,
        evidence_id: str,
        file_hash: str,
        collector_info: str,
        case_id: Optional[str] = None,
        analysis_focus: Optional[List[str]] = None
    ) -> str:
        """Build the analysis prompt for Claude."""

        prompt = f"""Please analyze this social media screenshot as legal evidence.

EVIDENCE METADATA:
- Evidence ID: {evidence_id}
- Original Filename: {screenshot_filename}
- File Hash: {file_hash}
- Collected by: {collector_info}
- Collection Date: {datetime.now().isoformat()}"""

        if case_id:
            prompt += f"\n- Case ID: {case_id}"

        if analysis_focus:
            prompt += f"\n\nFOCUS AREAS: {', '.join(analysis_focus)}"

        prompt += f"""

CONFIGURATION:
- Perform authenticity assessment: {self.config.perform_authenticity_assessment}
- Analyze network patterns: {self.config.analyze_network_patterns}
- Extract conversation context: {self.config.extract_conversation_context}
- Assess legal admissibility: {self.config.assess_legal_admissibility}
- Minimum authenticity confidence: {self.config.min_authenticity_confidence}

Please provide a complete analysis following the specified JSON format in your system prompt. Pay special attention to:

1. Extracting all visible text and metadata accurately
2. Assessing authenticity and potential manipulation
3. Documenting chain of custody considerations
4. Identifying any red flags or suspicious indicators
5. Providing legal admissibility assessment
6. Preserving conversation context if present

Analyze the image thoroughly and provide the structured JSON output."""

        return prompt

    def process_batch(self, screenshots: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple social media screenshots in batch.

        Args:
            screenshots: List of screenshot data dictionaries

        Returns:
            List of analysis results
        """
        results = []

        for i, screenshot_data in enumerate(screenshots):
            try:
                result = self.process(screenshot_data, **kwargs)
                results.append(result)

                # Log batch progress
                self.audit_logger.log_event(
                    event_type=AuditEventType.ANALYSIS_PERFORMED,
                    agent_id=self.agent_id,
                    details={
                        "batch_item": i + 1,
                        "total_items": len(screenshots),
                        "success": True
                    }
                )

            except Exception as e:
                error_result = {
                    "error": str(e),
                    "screenshot_index": i,
                    "filename": screenshot_data.get('screenshot_filename', 'unknown')
                }
                results.append(error_result)

                # Log batch error
                self.audit_logger.log_event(
                    event_type=AuditEventType.ERROR_OCCURRED,
                    agent_id=self.agent_id,
                    details={
                        "batch_item": i + 1,
                        "total_items": len(screenshots),
                        "error": str(e)
                    }
                )

        return results

    def verify_evidence_integrity(self, evidence_id: str) -> Dict[str, Any]:
        """
        Verify the integrity of previously processed evidence.

        Args:
            evidence_id: ID of evidence to verify

        Returns:
            Verification results
        """
        return self.evidence_handler.verify_evidence_integrity(evidence_id)

    def get_evidence_chain_of_custody(self, evidence_id: str) -> List[Dict[str, Any]]:
        """
        Get complete chain of custody for evidence.

        Args:
            evidence_id: ID of evidence

        Returns:
            Chain of custody log
        """
        return self.audit_logger.get_evidence_chain(evidence_id)