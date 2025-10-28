"""
Satellite Imagery Analyst Agent
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import VisionCapableAgent, AuditLogger, AuditEventType, EvidenceHandler
from .system_prompt import SYSTEM_PROMPT
from .config import ImageryAnalystConfig, DEFAULT_CONFIG


class SatelliteImageryAnalystAgent(VisionCapableAgent):
    """Agent for analyzing satellite and aerial imagery."""

    def __init__(
        self,
        config: Optional[ImageryAnalystConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="satellite_imagery_analyst",
            system_prompt=SYSTEM_PROMPT,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            audit_logger=audit_logger,
            **kwargs
        )

        self.evidence_handler = evidence_handler or EvidenceHandler()

    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Analyze satellite/aerial imagery."""
        image_data = input_data.get('image_data')
        analysis_type = input_data.get('analysis_type', 'feature_identification')
        evidence_id = input_data.get('evidence_id')

        if not image_data:
            raise ValueError("image_data is required")

        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            evidence_ids=[evidence_id] if evidence_id else [],
            details={"analysis_type": analysis_type}
        )

        result = self._analyze_imagery(image_data, analysis_type, evidence_id)

        return self.generate_output(
            output_data=result,
            output_type="imagery_analysis",
            evidence_ids=[evidence_id] if evidence_id else []
        )

    def _analyze_imagery(
        self,
        image_data: bytes,
        analysis_type: str,
        evidence_id: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze imagery using Claude Vision."""
        prompt = self._build_analysis_prompt(analysis_type)

        response = self.process_image(
            image_data=image_data,
            prompt=prompt,
            media_type="image/jpeg"
        )

        return self._parse_response(response, evidence_id)

    def _build_analysis_prompt(self, analysis_type: str) -> str:
        """Build imagery analysis prompt."""
        prompt = f"Please perform {analysis_type.replace('_', ' ')} on this satellite/aerial image.\n\n"

        if analysis_type == 'change_detection':
            prompt += "Compare this image to identify changes over time.\n"
        elif analysis_type == 'site_assessment':
            prompt += "Assess this site for indicators of mass graves, detention facilities, or other significant features.\n"
        elif analysis_type == 'damage_assessment':
            prompt += "Assess damage levels and patterns of destruction.\n"

        prompt += "\nProvide comprehensive analysis in JSON format per your system prompt."

        return prompt

    def _parse_response(self, response: Dict[str, Any], evidence_id: Optional[str]) -> Dict[str, Any]:
        """Parse Claude's imagery analysis response."""
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
                result = {"summary": text_content[:500]}
        except json.JSONDecodeError:
            result = {"summary": text_content[:500]}

        if evidence_id:
            result['evidence_id'] = evidence_id

        return result

    def assess_site(self, image_data: bytes, site_type: str, **metadata) -> Dict[str, Any]:
        """Assess a specific type of site."""
        return self.process({
            'image_data': image_data,
            'analysis_type': 'site_assessment',
            'site_type': site_type,
            **metadata
        })

    def compare_images(self, before_image: bytes, after_image: bytes, **metadata) -> Dict[str, Any]:
        """Compare before/after images for changes."""
        # For simplicity, analyze after image with change detection context
        return self.process({
            'image_data': after_image,
            'analysis_type': 'change_detection',
            **metadata
        })
