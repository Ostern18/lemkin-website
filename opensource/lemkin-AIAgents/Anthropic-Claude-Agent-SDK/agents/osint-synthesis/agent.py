"""
OSINT Synthesis Agent

Aggregates and analyzes publicly available information for investigations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import BaseAgent, AuditLogger, AuditEventType, EvidenceHandler, OutputFormatter
from .system_prompt import SYSTEM_PROMPT
from .config import OSINTConfig, DEFAULT_CONFIG


class OSINTSynthesisAgent(BaseAgent):
    """
    Agent for open-source intelligence gathering and analysis.

    Handles:
    - Multi-source monitoring
    - Claim extraction and verification
    - Source credibility assessment
    - Pattern and narrative analysis
    - Geographic and temporal analysis
    - Intelligence brief generation
    """

    def __init__(
        self,
        config: Optional[OSINTConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize OSINT synthesis agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="osint_synthesis",
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
        Analyze OSINT sources and generate intelligence brief.

        Args:
            input_data: Dictionary containing:
                - query: Search query or topic to investigate
                - keywords: List of keywords/hashtags to monitor
                - time_period: Time period to analyze
                - source_data: Optional pre-gathered source data
                - case_id: Optional case ID

        Returns:
            Intelligence brief with findings
        """
        query = input_data.get('query')
        keywords = input_data.get('keywords', [])
        case_id = input_data.get('case_id')
        source_data = input_data.get('source_data')

        if not query and not keywords:
            raise ValueError("Either query or keywords required")

        # Log analysis
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            details={
                "case_id": case_id,
                "query": query,
                "keywords_count": len(keywords)
            }
        )

        # Perform OSINT analysis
        result = self._analyze_osint(
            query=query,
            keywords=keywords,
            source_data=source_data,
            case_id=case_id
        )

        return self.generate_output(
            output_data=result,
            output_type="osint_intelligence_brief",
            evidence_ids=[]
        )

    def _analyze_osint(
        self,
        query: Optional[str],
        keywords: List[str],
        source_data: Optional[Any],
        case_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Perform OSINT analysis using Claude with web search.

        Args:
            query: Search query
            keywords: Keywords to monitor
            source_data: Pre-gathered source data
            case_id: Case ID

        Returns:
            OSINT analysis results
        """
        prompt = self._build_osint_prompt(query, keywords, source_data)

        # Call Claude with web search capability
        messages = [{"role": "user", "content": prompt}]
        response = self.call_claude(messages)

        return self._parse_osint_response(response, case_id)

    def _build_osint_prompt(
        self,
        query: Optional[str],
        keywords: List[str],
        source_data: Optional[Any]
    ) -> str:
        """Build OSINT analysis prompt."""
        prompt = "Please perform OSINT analysis on the following:\n\n"

        if query:
            prompt += f"--- INVESTIGATION QUERY ---\n{query}\n\n"

        if keywords:
            prompt += f"--- KEYWORDS TO MONITOR ---\n"
            for kw in keywords:
                prompt += f"  - {kw}\n"
            prompt += "\n"

        if source_data:
            prompt += f"--- PRE-GATHERED SOURCE DATA ---\n"
            if isinstance(source_data, dict) or isinstance(source_data, list):
                prompt += json.dumps(source_data, indent=2)
            else:
                prompt += str(source_data)
            prompt += "\n\n"

        prompt += "--- ANALYSIS REQUEST ---\n"
        prompt += "Please analyze available open-source information and provide:\n\n"
        prompt += "1. Key findings with credibility assessment\n"
        prompt += "2. Claims identified and verification status\n"

        if self.config.perform_credibility_assessment:
            prompt += "3. Source credibility assessment\n"

        if self.config.identify_narratives:
            prompt += "4. Narrative analysis and dominant themes\n"

        if self.config.detect_coordination:
            prompt += "5. Coordinated activity or information operations\n"

        if self.config.analyze_temporal_patterns:
            prompt += "6. Temporal patterns and activity spikes\n"

        if self.config.generate_geographic_heat_map:
            prompt += "7. Geographic distribution of information\n"

        prompt += "\nProvide a comprehensive intelligence brief in JSON format as specified in your system prompt.\n"

        if self.config.flag_bot_activity:
            prompt += "\nFlag any suspected bot activity or artificial amplification.\n"

        return prompt

    def _parse_osint_response(
        self,
        response: Dict[str, Any],
        case_id: Optional[str]
    ) -> Dict[str, Any]:
        """Parse Claude's OSINT response."""
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
                result = {
                    "executive_summary": text_content[:500],
                    "full_analysis": text_content
                }

        except json.JSONDecodeError:
            result = {
                "executive_summary": "Analysis completed",
                "full_analysis": text_content
            }

        if case_id:
            result['case_id'] = case_id

        result['_processing'] = {
            "model": response.get('model'),
            "tokens_used": response.get('usage'),
            "analysis_timestamp": datetime.now().isoformat()
        }

        return result

    def monitor_keywords(
        self,
        keywords: List[str],
        time_period_days: int = 7,
        **metadata
    ) -> Dict[str, Any]:
        """
        Monitor specific keywords across sources.

        Args:
            keywords: Keywords/hashtags to monitor
            time_period_days: How many days to analyze
            **metadata: Additional metadata

        Returns:
            Monitoring results
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)

        return self.process({
            'keywords': keywords,
            'time_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            **metadata
        })

    def verify_claim(
        self,
        claim: str,
        context: Optional[str] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Verify a specific claim using OSINT.

        Args:
            claim: Claim to verify
            context: Optional context about the claim
            **metadata: Additional metadata

        Returns:
            Verification results
        """
        query = f"Verify the following claim: {claim}"
        if context:
            query += f"\n\nContext: {context}"

        result = self.process({
            'query': query,
            **metadata
        })

        # Extract verification assessment
        claims = result.get('claims_identified', [])
        if claims:
            verification_status = claims[0].get('verification', {}).get('status')
            confidence = claims[0].get('verification', {}).get('confidence')

            result['verification_summary'] = {
                "claim": claim,
                "status": verification_status,
                "confidence": confidence,
                "assessment": claims[0]
            }

        return result

    def analyze_event(
        self,
        event_description: str,
        location: Optional[str] = None,
        date: Optional[str] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Analyze OSINT coverage of a specific event.

        Args:
            event_description: Description of event
            location: Event location
            date: Event date
            **metadata: Additional metadata

        Returns:
            Event analysis
        """
        query = f"Analyze open-source information about: {event_description}"

        if location:
            query += f"\nLocation: {location}"

        if date:
            query += f"\nDate: {date}"

        return self.process({
            'query': query,
            **metadata
        })

    def track_actor(
        self,
        actor_name: str,
        actor_type: str = "individual",
        **metadata
    ) -> Dict[str, Any]:
        """
        Track an actor's online presence and activities.

        Args:
            actor_name: Name of actor to track
            actor_type: Type (individual, organization, group)
            **metadata: Additional metadata

        Returns:
            Actor tracking results
        """
        query = f"Track {actor_type}: {actor_name}"

        return self.process({
            'query': query,
            'keywords': [actor_name],
            **metadata
        })
