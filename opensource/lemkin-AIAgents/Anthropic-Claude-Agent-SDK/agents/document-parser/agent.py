"""
Multi-Format Document Parser Agent

Extracts and structures content from PDFs, images, and scanned documents
for legal and investigative purposes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    VisionCapableAgent,
    AuditLogger,
    AuditEventType,
    EvidenceHandler,
    EvidenceType,
    OutputFormatter
)
from .system_prompt import SYSTEM_PROMPT
from .config import ParserConfig, DEFAULT_CONFIG


class DocumentParserAgent(VisionCapableAgent):
    """
    Agent for parsing and extracting structured content from documents.

    Handles:
    - PDFs (native and scanned)
    - Images (photos of documents)
    - Multi-page documents
    - Mixed content (text + images + tables)
    - Multiple languages
    - Handwritten content
    """

    def __init__(
        self,
        config: Optional[ParserConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """
        Initialize document parser agent.

        Args:
            config: Parser configuration
            audit_logger: Audit logger instance
            evidence_handler: Evidence handler instance
            **kwargs: Additional arguments for base agent
        """
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="document_parser",
            system_prompt=SYSTEM_PROMPT,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            audit_logger=audit_logger,
            **kwargs
        )

        self.evidence_handler = evidence_handler or EvidenceHandler()

    def process(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document and extract structured content.

        Args:
            input_data: Dictionary containing:
                - file_data: bytes of the document
                - file_type: 'pdf' or 'image'
                - evidence_id: optional evidence ID (will ingest if not provided)
                - source: source of the document
                - case_id: optional case ID
                - **metadata: additional metadata

        Returns:
            Dictionary containing parsed document data
        """
        file_data = input_data.get('file_data')
        file_type = input_data.get('file_type', 'pdf')
        evidence_id = input_data.get('evidence_id')

        if not file_data:
            raise ValueError("file_data is required")

        # Ingest evidence if not already done
        if not evidence_id:
            evidence_id = self._ingest_document(file_data, input_data)

        # Log processing start
        self.audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_PROCESSED,
            agent_id=self.agent_id,
            evidence_ids=[evidence_id],
            details={
                "file_type": file_type,
                "file_size": len(file_data)
            }
        )

        # Process based on file type
        if file_type.lower() == 'pdf':
            result = self._process_pdf(file_data, evidence_id)
        elif file_type.lower() in ['image', 'jpg', 'jpeg', 'png']:
            result = self._process_image(file_data, evidence_id, file_type)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Add evidence ID to result
        result['evidence_id'] = evidence_id

        # Check if human review is needed
        if self._needs_human_review(result):
            review_request = self.request_human_review(
                item_for_review=result,
                review_type="low_confidence_extraction",
                priority="normal"
            )
            result['human_review_requested'] = review_request

        # Update evidence status
        self.evidence_handler.update_status(
            evidence_id,
            self.evidence_handler.get_metadata(evidence_id).status.__class__.PROCESSED,
            notes=f"Processed by document parser. Overall confidence: {result.get('confidence_scores', {}).get('overall', 'N/A')}"
        )

        # Generate output
        return self.generate_output(
            output_data=result,
            output_type="document_analysis",
            evidence_ids=[evidence_id]
        )

    def _ingest_document(
        self,
        file_data: bytes,
        input_data: Dict[str, Any]
    ) -> str:
        """
        Ingest document as evidence.

        Args:
            file_data: Document bytes
            input_data: Input metadata

        Returns:
            Evidence ID
        """
        file_type = input_data.get('file_type', 'pdf')

        # Determine evidence type
        evidence_type_map = {
            'pdf': EvidenceType.DOCUMENT_PDF,
            'image': EvidenceType.DOCUMENT_IMAGE,
            'jpg': EvidenceType.DOCUMENT_IMAGE,
            'jpeg': EvidenceType.DOCUMENT_IMAGE,
            'png': EvidenceType.DOCUMENT_IMAGE
        }

        evidence_type = evidence_type_map.get(file_type.lower(), EvidenceType.DOCUMENT_PDF)

        # Ingest
        evidence_id = self.evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=evidence_type,
            source=input_data.get('source', 'unknown'),
            original_filename=input_data.get('filename'),
            case_id=input_data.get('case_id'),
            tags=input_data.get('tags', []),
            mime_type=input_data.get('mime_type'),
            custodian=input_data.get('custodian')
        )

        self.audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_INGESTED,
            agent_id=self.agent_id,
            evidence_ids=[evidence_id],
            details={
                "source": input_data.get('source'),
                "file_type": file_type,
                "file_size": len(file_data)
            }
        )

        return evidence_id

    def _process_pdf(
        self,
        pdf_data: bytes,
        evidence_id: str
    ) -> Dict[str, Any]:
        """
        Process a PDF document.

        Args:
            pdf_data: PDF bytes
            evidence_id: Evidence identifier

        Returns:
            Parsed document data
        """
        # Create processing prompt
        prompt = self._create_processing_prompt()

        # Process with Claude
        response = self.process_pdf(pdf_data, prompt)

        # Parse response
        return self._parse_response(response, evidence_id)

    def _process_image(
        self,
        image_data: bytes,
        evidence_id: str,
        file_type: str
    ) -> Dict[str, Any]:
        """
        Process an image of a document.

        Args:
            image_data: Image bytes
            evidence_id: Evidence identifier
            file_type: Image file type

        Returns:
            Parsed document data
        """
        # Determine media type
        media_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'webp': 'image/webp',
            'gif': 'image/gif'
        }

        media_type = media_type_map.get(file_type.lower(), 'image/jpeg')

        # Create processing prompt
        prompt = self._create_processing_prompt()

        # Process with Claude
        response = self.process_image(image_data, prompt, media_type=media_type)

        # Parse response
        return self._parse_response(response, evidence_id)

    def _create_processing_prompt(self) -> str:
        """
        Create the processing prompt based on configuration.

        Returns:
            Processing instructions
        """
        prompt = """Please analyze this document and extract all relevant information according to your system prompt.

Pay special attention to:
1. Document type and classification
2. All readable text (preserve formatting)
3. Key fields: dates, names, signatures, identifiers
4. Document structure and layout
5. Any quality issues or illegible sections
"""

        if self.config.enable_table_extraction:
            prompt += "\n6. Any tables present (extract data in structured format)"

        if self.config.enable_handwriting_detection:
            prompt += "\n7. Presence and legibility of handwritten content"

        prompt += "\n\nProvide your analysis in the JSON format specified in your system prompt."

        return prompt

    def _parse_response(
        self,
        response: Dict[str, Any],
        evidence_id: str
    ) -> Dict[str, Any]:
        """
        Parse Claude's response into structured output.

        Args:
            response: Response from Claude
            evidence_id: Evidence identifier

        Returns:
            Structured parsed data
        """
        # Extract text content
        content = response.get('content', [])

        if not content:
            raise ValueError("No content in response")

        # Get the text from the first content block
        text_content = None
        for block in content:
            if hasattr(block, 'text'):
                text_content = block.text
                break
            elif isinstance(block, dict) and 'text' in block:
                text_content = block['text']
                break

        if not text_content:
            raise ValueError("No text content found in response")

        # Try to parse as JSON
        try:
            # Find JSON in response (might be wrapped in markdown code blocks)
            json_start = text_content.find('{')
            json_end = text_content.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text_content[json_start:json_end]
                parsed_data = json.loads(json_str)
            else:
                # No JSON found, wrap raw text
                parsed_data = {
                    "extracted_text": {"full_text": text_content},
                    "confidence_scores": {"overall": 0.5},
                    "processing_notes": ["Response not in expected JSON format"]
                }

        except json.JSONDecodeError as e:
            # Failed to parse JSON, return raw text with error
            parsed_data = {
                "extracted_text": {"full_text": text_content},
                "confidence_scores": {"overall": 0.3},
                "processing_notes": [f"JSON parsing error: {str(e)}"]
            }

        # Add processing metadata
        parsed_data['_processing'] = {
            "model": response.get('model'),
            "tokens_used": response.get('usage'),
            "evidence_id": evidence_id
        }

        return parsed_data

    def _needs_human_review(self, result: Dict[str, Any]) -> bool:
        """
        Determine if result needs human review.

        Args:
            result: Processing result

        Returns:
            True if human review is needed
        """
        if not self.config.human_review_low_confidence:
            return False

        # Check overall confidence
        overall_confidence = result.get('confidence_scores', {}).get('overall', 1.0)

        if overall_confidence < self.config.human_review_threshold:
            return True

        # Check for high-severity quality flags
        quality_flags = result.get('quality_flags', [])
        for flag in quality_flags:
            if flag.get('severity') == 'high':
                return True

        return False

    def parse_document(
        self,
        file_path: Union[str, Path],
        **metadata
    ) -> Dict[str, Any]:
        """
        Convenience method to parse a document from file path.

        Args:
            file_path: Path to document file
            **metadata: Additional metadata

        Returns:
            Parsed document data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file
        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Determine file type
        suffix = file_path.suffix.lower()
        file_type_map = {
            '.pdf': 'pdf',
            '.jpg': 'jpg',
            '.jpeg': 'jpeg',
            '.png': 'png'
        }

        file_type = file_type_map.get(suffix, 'pdf')

        # Process
        return self.process({
            'file_data': file_data,
            'file_type': file_type,
            'filename': file_path.name,
            **metadata
        })

    def batch_process(
        self,
        file_paths: list[Union[str, Path]],
        **shared_metadata
    ) -> list[Dict[str, Any]]:
        """
        Process multiple documents in batch.

        Args:
            file_paths: List of file paths
            **shared_metadata: Metadata to apply to all documents

        Returns:
            List of parsed document data
        """
        results = []

        for file_path in file_paths:
            try:
                result = self.parse_document(file_path, **shared_metadata)
                results.append(result)
            except Exception as e:
                # Log error and continue
                self.audit_logger.log_event(
                    event_type=AuditEventType.ERROR_OCCURRED,
                    agent_id=self.agent_id,
                    details={
                        "file_path": str(file_path),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                results.append({
                    "file_path": str(file_path),
                    "error": str(e),
                    "status": "failed"
                })

        return results
