"""
Digital Forensics & Metadata Analyst Agent

Analyzes digital evidence and metadata for legal investigations.
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
from .config import DigitalForensicsAnalystConfig, DEFAULT_CONFIG


class DigitalForensicsAnalystAgent(BaseAgent):
    """
    Agent for digital forensics and metadata analysis.

    Handles:
    - Metadata extraction and analysis from digital files
    - File authentication and integrity verification
    - Digital communication analysis
    - Timeline reconstruction from digital evidence
    - Pattern analysis across multiple digital artifacts
    - Legal admissibility assessment of digital evidence
    """

    def __init__(
        self,
        config: Optional[DigitalForensicsAnalystConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Digital Forensics & Metadata Analyst agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="digital_forensics_analyst",
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
        Analyze digital evidence and metadata.

        Args:
            input_data: Dictionary containing:
                - digital_files: Digital files to analyze (file paths, metadata, or base64 data)
                - file_metadata: Known metadata about files
                - communication_data: Email headers, messaging data
                - timeline_data: Timestamp information for correlation
                - comparison_files: Files to compare for similarities/differences
                - case_id: Optional case ID
                - analysis_focus: Specific aspects to focus on
                - chain_of_custody_info: Chain of custody information

        Returns:
            Comprehensive digital forensics analysis
        """
        digital_files = input_data.get('digital_files', [])
        file_metadata = input_data.get('file_metadata', {})
        communication_data = input_data.get('communication_data', {})
        timeline_data = input_data.get('timeline_data', {})
        comparison_files = input_data.get('comparison_files', [])
        case_id = input_data.get('case_id')
        analysis_focus = input_data.get('analysis_focus', [])
        chain_of_custody_info = input_data.get('chain_of_custody_info', {})

        if not digital_files and not file_metadata and not communication_data:
            raise ValueError("Digital files, metadata, or communication data required for analysis")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())

        # Log digital forensics analysis
        self.audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_PROCESSED,
            agent_id=self.agent_id,
            details={
                "analysis_id": analysis_id,
                "case_id": case_id,
                "analysis_type": "digital_forensics_analysis",
                "files_analyzed": len(digital_files),
                "chain_of_custody_maintained": self.config.maintain_chain_of_custody,
                "hash_verification": self.config.require_hash_verification,
                "legal_admissibility_assessed": self.config.assess_legal_admissibility
            }
        )

        # Perform digital forensics analysis
        result = self._conduct_digital_forensics_analysis(
            digital_files=digital_files,
            file_metadata=file_metadata,
            communication_data=communication_data,
            timeline_data=timeline_data,
            comparison_files=comparison_files,
            analysis_focus=analysis_focus,
            chain_of_custody_info=chain_of_custody_info,
            analysis_id=analysis_id,
            case_id=case_id
        )

        # Store evidence if configured
        if self.config.maintain_chain_of_custody:
            for i, file_data in enumerate(digital_files):
                file_id = file_data.get('file_id', f"{analysis_id}_file_{i}")
                self.evidence_handler.store_evidence(
                    evidence_id=file_id,
                    evidence_type="digital_file",
                    content=file_data,
                    source_hash=self._calculate_content_hash(file_data),
                    metadata={
                        "analysis_id": analysis_id,
                        "case_id": case_id,
                        "forensics_analysis": True
                    }
                )

        return self.generate_output(
            output_data=result,
            output_type="digital_forensics_analysis",
            evidence_ids=[analysis_id]
        )

    def _conduct_digital_forensics_analysis(
        self,
        digital_files: List[Dict[str, Any]],
        file_metadata: Dict[str, Any],
        communication_data: Dict[str, Any],
        timeline_data: Dict[str, Any],
        comparison_files: List[Dict[str, Any]],
        analysis_focus: List[str],
        chain_of_custody_info: Dict[str, Any],
        analysis_id: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive digital forensics analysis."""

        # Build digital forensics analysis prompt
        analysis_prompt = self._build_forensics_analysis_prompt(
            digital_files=digital_files,
            file_metadata=file_metadata,
            communication_data=communication_data,
            timeline_data=timeline_data,
            comparison_files=comparison_files,
            analysis_focus=analysis_focus,
            chain_of_custody_info=chain_of_custody_info,
            analysis_id=analysis_id,
            case_id=case_id
        )

        # Perform Claude analysis
        messages = [
            {
                "role": "user",
                "content": analysis_prompt
            }
        ]

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
                    "digital_forensics_id": analysis_id,
                    "error": "Could not parse digital forensics analysis results",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            result = {
                "digital_forensics_id": analysis_id,
                "error": "Invalid JSON in digital forensics analysis results",
                "raw_response": response_text
            }

        # Add metadata
        result["analysis_metadata"] = {
            "digital_forensics_id": analysis_id,
            "analysis_date": datetime.now().isoformat(),
            "analyst": self.agent_id,
            "case_id": case_id,
            "evidence_items_analyzed": len(digital_files),
            "analysis_type": "comprehensive_forensics" if len(digital_files) > 1 else "single_file_analysis",
            "chain_of_custody_maintained": self.config.maintain_chain_of_custody,
            "configuration_used": {
                "hash_algorithms": self.config.hash_algorithms,
                "metadata_extraction": self.config.extract_metadata,
                "authenticity_verification": self.config.verify_authenticity,
                "timeline_reconstruction": self.config.reconstruct_timeline,
                "communication_analysis": self.config.analyze_communications,
                "legal_admissibility": self.config.assess_legal_admissibility
            }
        }

        return result

    def _build_forensics_analysis_prompt(
        self,
        digital_files: List[Dict[str, Any]],
        file_metadata: Dict[str, Any],
        communication_data: Dict[str, Any],
        timeline_data: Dict[str, Any],
        comparison_files: List[Dict[str, Any]],
        analysis_focus: List[str],
        chain_of_custody_info: Dict[str, Any],
        analysis_id: str,
        case_id: Optional[str] = None
    ) -> str:
        """Build the digital forensics analysis prompt for Claude."""

        prompt = f"""Please conduct comprehensive digital forensics and metadata analysis.

ANALYSIS METADATA:
- Analysis ID: {analysis_id}
- Analysis Date: {datetime.now().isoformat()}"""

        if case_id:
            prompt += f"\n- Case ID: {case_id}"

        if digital_files:
            prompt += f"\n\nDIGITAL FILES TO ANALYZE ({len(digital_files)} files):\n{json.dumps(digital_files, indent=2)}"

        if file_metadata:
            prompt += f"\n\nKNOWN FILE METADATA:\n{json.dumps(file_metadata, indent=2)}"

        if communication_data:
            prompt += f"\n\nCOMMUNICATION DATA:\n{json.dumps(communication_data, indent=2)}"

        if timeline_data:
            prompt += f"\n\nTIMELINE DATA:\n{json.dumps(timeline_data, indent=2)}"

        if comparison_files:
            prompt += f"\n\nCOMPARISON FILES:\n{json.dumps(comparison_files, indent=2)}"

        if chain_of_custody_info:
            prompt += f"\n\nCHAIN OF CUSTODY INFO:\n{json.dumps(chain_of_custody_info, indent=2)}"

        if analysis_focus:
            prompt += f"\n\nANALYSIS FOCUS: {', '.join(analysis_focus)}"

        prompt += f"""

ANALYSIS CONFIGURATION:
- Extract metadata: {self.config.extract_metadata}
- Verify authenticity: {self.config.verify_authenticity}
- Detect manipulation: {self.config.detect_manipulation}
- Reconstruct timeline: {self.config.reconstruct_timeline}
- Analyze communications: {self.config.analyze_communications}
- Perform comparison analysis: {self.config.perform_comparison_analysis}
- Hash algorithms: {', '.join(self.config.hash_algorithms)}
- Calculate file hashes: {self.config.calculate_file_hashes}
- Verify digital signatures: {self.config.verify_digital_signatures}
- Detect steganography: {self.config.detect_steganography}
- Assess legal admissibility: {self.config.assess_legal_admissibility}
- Maintain chain of custody: {self.config.maintain_chain_of_custody}
- Minimum confidence threshold: {self.config.min_confidence_threshold}

Please provide comprehensive digital forensics analysis following the JSON format in your system prompt. Focus on:

1. Metadata extraction and analysis for all digital files
2. File authentication and integrity verification using hash analysis
3. Manipulation detection and authenticity assessment
4. Timeline reconstruction from digital timestamps and metadata
5. Communication analysis including email headers and routing
6. Pattern analysis across multiple files and data sources
7. Technical analysis of file structures and digital signatures
8. Comparison analysis between related files
9. Expert findings and technical conclusions
10. Legal admissibility assessment and expert testimony recommendations
11. Chain of custody documentation and integrity verification

Ensure all analysis meets forensic standards and maintains evidence integrity for legal proceedings."""

        return prompt

    def _calculate_content_hash(self, content: Any) -> str:
        """Calculate hash of content for integrity verification."""
        content_str = json.dumps(content, sort_keys=True) if isinstance(content, dict) else str(content)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def analyze_file_metadata(
        self,
        file_data: Dict[str, Any],
        focus_areas: Optional[List[str]] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze metadata from a single digital file.

        Args:
            file_data: File information and metadata
            focus_areas: Specific metadata areas to focus on
            case_id: Optional case ID

        Returns:
            File metadata analysis
        """
        return self.process({
            'digital_files': [file_data],
            'analysis_focus': focus_areas or ['metadata_extraction', 'authenticity_assessment'],
            'case_id': case_id
        })

    def verify_file_authenticity(
        self,
        file_data: Dict[str, Any],
        known_hashes: Optional[Dict[str, str]] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify authenticity and integrity of digital file.

        Args:
            file_data: File to verify
            known_hashes: Known good hashes for comparison
            case_id: Optional case ID

        Returns:
            File authenticity verification results
        """
        analysis_data = {'digital_files': [file_data]}
        if known_hashes:
            analysis_data['file_metadata'] = {'known_hashes': known_hashes}

        return self.process({
            **analysis_data,
            'analysis_focus': ['authenticity_verification', 'integrity_check', 'manipulation_detection'],
            'case_id': case_id
        })

    def reconstruct_digital_timeline(
        self,
        files_and_data: List[Dict[str, Any]],
        external_timeline: Optional[Dict[str, Any]] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct timeline from digital evidence.

        Args:
            files_and_data: Digital files and data with timestamps
            external_timeline: External timeline data for correlation
            case_id: Optional case ID

        Returns:
            Digital timeline reconstruction
        """
        analysis_data = {'digital_files': files_and_data}
        if external_timeline:
            analysis_data['timeline_data'] = external_timeline

        return self.process({
            **analysis_data,
            'analysis_focus': ['timeline_reconstruction', 'timestamp_analysis', 'event_correlation'],
            'case_id': case_id
        })

    def analyze_email_communications(
        self,
        email_data: Dict[str, Any],
        additional_files: Optional[List[Dict[str, Any]]] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze email communications and headers.

        Args:
            email_data: Email headers, content, and metadata
            additional_files: Additional related files
            case_id: Optional case ID

        Returns:
            Email communication analysis
        """
        return self.process({
            'communication_data': email_data,
            'digital_files': additional_files or [],
            'analysis_focus': ['communication_analysis', 'header_verification', 'routing_analysis'],
            'case_id': case_id
        })

    def compare_digital_files(
        self,
        files_to_compare: List[Dict[str, Any]],
        comparison_type: str = "similarity_analysis",
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple digital files for similarities and differences.

        Args:
            files_to_compare: Files to compare
            comparison_type: Type of comparison to perform
            case_id: Optional case ID

        Returns:
            File comparison analysis
        """
        return self.process({
            'digital_files': files_to_compare,
            'comparison_files': files_to_compare,
            'analysis_focus': ['comparison_analysis', comparison_type, 'pattern_detection'],
            'case_id': case_id
        })

    def detect_manipulation(
        self,
        suspicious_files: List[Dict[str, Any]],
        reference_files: Optional[List[Dict[str, Any]]] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect digital manipulation in files.

        Args:
            suspicious_files: Files suspected of manipulation
            reference_files: Reference files for comparison
            case_id: Optional case ID

        Returns:
            Manipulation detection analysis
        """
        analysis_data = {'digital_files': suspicious_files}
        if reference_files:
            analysis_data['comparison_files'] = reference_files

        return self.process({
            **analysis_data,
            'analysis_focus': ['manipulation_detection', 'authenticity_assessment', 'technical_analysis'],
            'case_id': case_id
        })

    def analyze_mobile_device_evidence(
        self,
        mobile_data: Dict[str, Any],
        location_data: Optional[Dict[str, Any]] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze evidence from mobile devices.

        Args:
            mobile_data: Mobile device files and data
            location_data: GPS and location information
            case_id: Optional case ID

        Returns:
            Mobile device evidence analysis
        """
        analysis_data = {'digital_files': mobile_data.get('files', [])}
        if location_data:
            analysis_data['file_metadata'] = {'location_data': location_data}

        return self.process({
            **analysis_data,
            'analysis_focus': ['metadata_extraction', 'gps_analysis', 'device_identification', 'timeline_reconstruction'],
            'case_id': case_id
        })

    def verify_chain_of_custody(
        self,
        evidence_items: List[Dict[str, Any]],
        custody_records: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify digital chain of custody.

        Args:
            evidence_items: Digital evidence items
            custody_records: Chain of custody records
            case_id: Optional case ID

        Returns:
            Chain of custody verification
        """
        return self.process({
            'digital_files': evidence_items,
            'chain_of_custody_info': custody_records,
            'analysis_focus': ['chain_of_custody', 'integrity_verification', 'access_control'],
            'case_id': case_id
        })