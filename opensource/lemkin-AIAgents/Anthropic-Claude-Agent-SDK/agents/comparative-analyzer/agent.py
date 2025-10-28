"""
Comparative Document Analyzer Agent

Identifies similarities, differences, and patterns across multiple documents.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import BaseAgent, AuditLogger, AuditEventType, EvidenceHandler, OutputFormatter
from .system_prompt import SYSTEM_PROMPT
from .config import ComparativeAnalyzerConfig, DEFAULT_CONFIG


class ComparativeAnalyzerAgent(BaseAgent):
    """
    Agent for comparing and analyzing multiple documents.

    Handles:
    - Version comparison
    - Multi-document similarity analysis
    - Pattern detection
    - Forgery indicators
    - Redaction analysis
    """

    def __init__(
        self,
        config: Optional[ComparativeAnalyzerConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize comparative analyzer agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="comparative_analyzer",
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
        Compare multiple documents and identify patterns/differences.

        Args:
            input_data: Dictionary containing:
                - documents: List of document data (text or evidence_ids)
                - comparison_type: Type of comparison to perform
                - **metadata: Additional metadata

        Returns:
            Comparison results
        """
        documents = input_data.get('documents', [])
        comparison_type = input_data.get('comparison_type', 'multi_document_similarity')

        if len(documents) < 2:
            raise ValueError("At least 2 documents required for comparison")

        evidence_ids = self._prepare_documents(documents)

        # Log analysis start
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            evidence_ids=evidence_ids,
            details={
                "comparison_type": comparison_type,
                "document_count": len(documents)
            }
        )

        # Perform comparison
        result = self._compare_documents(documents, comparison_type, evidence_ids)

        # Check for high-severity red flags
        if self._has_critical_red_flags(result):
            review_request = self.request_human_review(
                item_for_review=result,
                review_type="critical_red_flags",
                priority="high"
            )
            result['human_review_requested'] = review_request

        return self.generate_output(
            output_data=result,
            output_type="comparative_analysis",
            evidence_ids=evidence_ids
        )

    def _prepare_documents(self, documents: List[Any]) -> List[str]:
        """
        Prepare documents for comparison and return evidence IDs.

        Args:
            documents: List of document data

        Returns:
            List of evidence IDs
        """
        evidence_ids = []

        for doc in documents:
            if isinstance(doc, str):
                # Assume it's an evidence ID
                evidence_ids.append(doc)
            elif isinstance(doc, dict) and 'evidence_id' in doc:
                evidence_ids.append(doc['evidence_id'])

        return evidence_ids

    def _compare_documents(
        self,
        documents: List[Any],
        comparison_type: str,
        evidence_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Perform document comparison using Claude.

        Args:
            documents: Document data
            comparison_type: Type of comparison
            evidence_ids: Evidence identifiers

        Returns:
            Comparison results
        """
        # Build comparison prompt
        prompt = self._build_comparison_prompt(documents, comparison_type)

        # Call Claude
        messages = [{"role": "user", "content": prompt}]
        response = self.call_claude(messages)

        # Parse response
        result = self._parse_comparison_response(response, evidence_ids)

        return result

    def _build_comparison_prompt(
        self,
        documents: List[Any],
        comparison_type: str
    ) -> str:
        """
        Build prompt for document comparison.

        Args:
            documents: Documents to compare
            comparison_type: Type of comparison

        Returns:
            Prompt string
        """
        prompt = f"Please perform a {comparison_type.replace('_', ' ')} of the following documents:\n\n"

        for i, doc in enumerate(documents, 1):
            prompt += f"--- DOCUMENT {i} ---\n"

            if isinstance(doc, dict):
                if 'text' in doc:
                    prompt += doc['text']
                elif 'content' in doc:
                    prompt += doc['content']
                else:
                    prompt += str(doc)
            else:
                prompt += str(doc)

            prompt += "\n\n"

        prompt += """Analyze these documents according to your system prompt, focusing on:
1. Similarities and differences
2. Patterns and recurring content
3. Timeline consistency
4. Metadata analysis
5. Any red flags or suspicious indicators

Provide detailed results in JSON format as specified in your system prompt."""

        return prompt

    def _parse_comparison_response(
        self,
        response: Dict[str, Any],
        evidence_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Parse Claude's comparison response.

        Args:
            response: Claude response
            evidence_ids: Evidence IDs

        Returns:
            Parsed comparison results
        """
        content = response.get('content', [])

        # Extract text
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

        # Try to parse JSON
        try:
            json_start = text_content.find('{')
            json_end = text_content.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text_content[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = {
                    "comparison_results": {"overall_similarity": 0.5},
                    "analysis_text": text_content
                }

        except json.JSONDecodeError:
            result = {
                "comparison_results": {"overall_similarity": 0.5},
                "analysis_text": text_content,
                "parsing_error": "Failed to parse JSON response"
            }

        # Add metadata
        result['documents_compared_ids'] = evidence_ids
        result['_processing'] = {
            "model": response.get('model'),
            "tokens_used": response.get('usage')
        }

        return result

    def _has_critical_red_flags(self, result: Dict[str, Any]) -> bool:
        """
        Check if result contains critical red flags requiring human review.

        Args:
            result: Comparison result

        Returns:
            True if critical red flags present
        """
        if not self.config.human_review_high_severity_flags:
            return False

        red_flags = result.get('red_flags', [])

        for flag in red_flags:
            if flag.get('severity') in ['high', 'critical']:
                return True

        return False

    def compare_versions(
        self,
        original_doc: Any,
        modified_doc: Any,
        **metadata
    ) -> Dict[str, Any]:
        """
        Compare two versions of the same document.

        Args:
            original_doc: Original document
            modified_doc: Modified document
            **metadata: Additional metadata

        Returns:
            Version comparison results
        """
        return self.process({
            'documents': [original_doc, modified_doc],
            'comparison_type': 'version_comparison',
            **metadata
        })

    def detect_patterns(
        self,
        documents: List[Any],
        **metadata
    ) -> Dict[str, Any]:
        """
        Detect patterns across multiple documents.

        Args:
            documents: List of documents to analyze
            **metadata: Additional metadata

        Returns:
            Pattern analysis results
        """
        return self.process({
            'documents': documents,
            'comparison_type': 'pattern_analysis',
            **metadata
        })

    def analyze_similarity(
        self,
        documents: List[Any],
        **metadata
    ) -> Dict[str, Any]:
        """
        Analyze similarity across multiple documents.

        Args:
            documents: List of documents
            **metadata: Additional metadata

        Returns:
            Similarity analysis results
        """
        return self.process({
            'documents': documents,
            'comparison_type': 'multi_document_similarity',
            **metadata
        })
