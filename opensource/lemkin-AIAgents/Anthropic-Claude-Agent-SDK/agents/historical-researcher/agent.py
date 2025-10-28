"""
Historical Context & Background Researcher Agent

Provides deep background on conflicts, actors, and regions for investigations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import BaseAgent, AuditLogger, AuditEventType, EvidenceHandler, OutputFormatter
from .system_prompt import SYSTEM_PROMPT
from .config import HistoricalResearcherConfig, DEFAULT_CONFIG


class HistoricalResearcherAgent(BaseAgent):
    """
    Agent for historical context and background research for legal investigations.

    Handles:
    - Historical context research and analysis
    - Political dynamics assessment
    - Key actor profiling and network analysis
    - Cultural and social context evaluation
    - Regional context and cross-border analysis
    - Legal and institutional background research
    """

    def __init__(
        self,
        config: Optional[HistoricalResearcherConfig] = None,
        audit_logger: Optional[AuditLogger] = None,
        evidence_handler: Optional[EvidenceHandler] = None,
        **kwargs
    ):
        """Initialize Historical Context & Background Researcher agent."""
        self.config = config or DEFAULT_CONFIG

        super().__init__(
            agent_id="historical_researcher",
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
        Conduct historical research and analysis.

        Args:
            input_data: Dictionary containing:
                - research_question: Primary research question or focus
                - geographic_scope: Countries/regions to research
                - time_period: Temporal scope for research
                - actors_of_interest: Specific actors to research
                - case_id: Optional case ID
                - research_priorities: Specific areas to prioritize

        Returns:
            Comprehensive historical analysis and context
        """
        research_question = input_data.get('research_question')
        geographic_scope = input_data.get('geographic_scope', [])
        time_period = input_data.get('time_period', {})
        actors_of_interest = input_data.get('actors_of_interest', [])
        case_id = input_data.get('case_id')
        research_priorities = input_data.get('research_priorities', [])

        if not research_question and not geographic_scope:
            raise ValueError("Either research question or geographic scope required")

        # Generate research ID
        research_id = str(uuid.uuid4())

        # Log research initiation
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            details={
                "research_id": research_id,
                "case_id": case_id,
                "research_question": research_question,
                "geographic_scope": geographic_scope,
                "actors_count": len(actors_of_interest)
            }
        )

        # Conduct historical research
        result = self._conduct_historical_research(
            research_question=research_question,
            geographic_scope=geographic_scope,
            time_period=time_period,
            actors_of_interest=actors_of_interest,
            research_priorities=research_priorities,
            research_id=research_id,
            case_id=case_id
        )

        return self.generate_output(
            output_data=result,
            output_type="historical_research_report",
            evidence_ids=[]
        )

    def _conduct_historical_research(
        self,
        research_question: Optional[str],
        geographic_scope: List[str],
        time_period: Dict[str, Any],
        actors_of_interest: List[str],
        research_priorities: List[str],
        research_id: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive historical research and analysis."""

        # Build research prompt
        research_prompt = self._build_research_prompt(
            research_question=research_question,
            geographic_scope=geographic_scope,
            time_period=time_period,
            actors_of_interest=actors_of_interest,
            research_priorities=research_priorities,
            research_id=research_id,
            case_id=case_id
        )

        # Perform web search if enabled
        web_search_results = None
        if self.config.enable_web_search:
            web_search_results = self._perform_web_research(
                research_question, geographic_scope, actors_of_interest
            )

        # Perform Claude analysis
        messages = [
            {
                "role": "user",
                "content": research_prompt
            }
        ]

        # Add web search results if available
        if web_search_results:
            messages[0]["content"] += f"\n\nWEB SEARCH RESULTS:\n{web_search_results}"

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
                    "research_id": research_id,
                    "error": "Could not parse research results",
                    "raw_response": response_text
                }
        except json.JSONDecodeError:
            result = {
                "research_id": research_id,
                "error": "Invalid JSON in research results",
                "raw_response": response_text
            }

        # Add metadata
        result["research_metadata"] = {
            "research_id": research_id,
            "research_date": datetime.now().isoformat(),
            "researcher": self.agent_id,
            "research_focus": research_question or f"Geographic analysis: {', '.join(geographic_scope)}",
            "case_id": case_id,
            "configuration_used": {
                "web_search_enabled": self.config.enable_web_search,
                "max_sources": self.config.max_web_sources,
                "historical_period_years": self.config.max_historical_period_years,
                "include_minor_actors": self.config.include_minor_actors
            }
        }

        return result

    def _build_research_prompt(
        self,
        research_question: Optional[str],
        geographic_scope: List[str],
        time_period: Dict[str, Any],
        actors_of_interest: List[str],
        research_priorities: List[str],
        research_id: str,
        case_id: Optional[str] = None
    ) -> str:
        """Build the research prompt for Claude."""

        prompt = f"""Please conduct comprehensive historical research and analysis.

RESEARCH METADATA:
- Research ID: {research_id}
- Research Date: {datetime.now().isoformat()}"""

        if case_id:
            prompt += f"\n- Case ID: {case_id}"

        if research_question:
            prompt += f"\n\nPRIMARY RESEARCH QUESTION: {research_question}"

        if geographic_scope:
            prompt += f"\n\nGEOGRAPHIC SCOPE: {', '.join(geographic_scope)}"

        if time_period:
            prompt += f"\n\nTIME PERIOD:"
            if 'start_date' in time_period:
                prompt += f"\n- Start Date: {time_period['start_date']}"
            if 'end_date' in time_period:
                prompt += f"\n- End Date: {time_period['end_date']}"

        if actors_of_interest:
            prompt += f"\n\nACTORS OF INTEREST: {', '.join(actors_of_interest)}"

        if research_priorities:
            prompt += f"\n\nRESEARCH PRIORITIES: {', '.join(research_priorities)}"

        prompt += f"""

RESEARCH CONFIGURATION:
- Include historical context: {self.config.include_historical_context}
- Analyze political dynamics: {self.config.analyze_political_dynamics}
- Profile key actors: {self.config.profile_key_actors}
- Assess cultural factors: {self.config.assess_cultural_factors}
- Examine regional context: {self.config.examine_regional_context}
- Research legal background: {self.config.research_legal_background}
- Maximum historical period: {self.config.max_historical_period_years} years
- Maximum actors to profile: {self.config.max_actors_to_profile}
- Focus on legal relevance: {self.config.focus_on_legal_relevance}

Please provide comprehensive research following the JSON format in your system prompt. Focus on:

1. Historical context and background that explains current situation
2. Key actors and their roles, relationships, and evolution over time
3. Political dynamics and institutional analysis
4. Cultural, ethnic, and religious factors affecting the situation
5. Regional context and cross-border influences
6. Legal and institutional background
7. Analogous cases and precedents
8. Information gaps and research recommendations

Ensure all information is well-sourced and includes confidence assessments."""

        return prompt

    def _perform_web_research(
        self,
        research_question: Optional[str],
        geographic_scope: List[str],
        actors_of_interest: List[str]
    ) -> str:
        """Perform web research to gather additional context."""

        # This is a placeholder for web search integration
        # In a real implementation, this would use web search tools
        # For now, we'll return a placeholder indicating web research capability

        search_terms = []

        if research_question:
            search_terms.append(research_question)

        if geographic_scope:
            search_terms.extend([f"{region} history conflict" for region in geographic_scope])
            search_terms.extend([f"{region} political situation" for region in geographic_scope])

        if actors_of_interest:
            search_terms.extend([f"{actor} biography background" for actor in actors_of_interest])

        # Placeholder web search results
        web_results = f"""
WEB SEARCH PERFORMED FOR: {', '.join(search_terms[:10])}

Note: This implementation includes placeholder for web search integration.
In production, this would integrate with:
- Academic databases (JSTOR, Google Scholar)
- News archives (BBC, Reuters, AP)
- Government databases and reports
- UN and NGO report repositories
- Legal databases (ICJ, ECHR, etc.)
- Historical archives and libraries

Search would focus on:
- Academic articles and research papers
- Government documents and statements
- UN reports and resolutions
- NGO documentation and analysis
- News coverage and investigative reporting
- Legal proceedings and court documents

Results would be filtered by:
- Source credibility (minimum {self.config.min_source_credibility})
- Relevance to research question
- Temporal relevance to case
- Geographic relevance to scope
"""

        return web_results

    def research_specific_actor(
        self,
        actor_name: str,
        focus_areas: Optional[List[str]] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conduct focused research on a specific actor.

        Args:
            actor_name: Name of actor to research
            focus_areas: Specific aspects to focus on
            case_id: Optional case ID

        Returns:
            Detailed actor profile and analysis
        """
        return self.process({
            'research_question': f'Comprehensive background research on {actor_name}',
            'actors_of_interest': [actor_name],
            'research_priorities': focus_areas or ['background', 'roles', 'relationships', 'legal_exposure'],
            'case_id': case_id
        })

    def research_conflict_background(
        self,
        conflict_name: str,
        geographic_scope: List[str],
        time_period: Optional[Dict[str, str]] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Research background of a specific conflict.

        Args:
            conflict_name: Name or description of conflict
            geographic_scope: Countries/regions involved
            time_period: Temporal scope for research
            case_id: Optional case ID

        Returns:
            Comprehensive conflict background analysis
        """
        return self.process({
            'research_question': f'Historical background and context of {conflict_name}',
            'geographic_scope': geographic_scope,
            'time_period': time_period or {},
            'research_priorities': ['historical_context', 'root_causes', 'key_events', 'actors'],
            'case_id': case_id
        })

    def identify_analogous_cases(
        self,
        current_situation: str,
        violation_types: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identify analogous legal cases and precedents.

        Args:
            current_situation: Description of current situation
            violation_types: Types of violations involved
            case_id: Optional case ID

        Returns:
            Analysis of analogous cases and precedents
        """
        return self.process({
            'research_question': f'Analogous cases and legal precedents for: {current_situation}',
            'research_priorities': ['analogous_cases', 'legal_precedents', 'jurisdictional_issues'],
            'case_id': case_id
        })

    def assess_institutional_capacity(
        self,
        country: str,
        institutions: List[str],
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess capacity of domestic institutions.

        Args:
            country: Country to assess
            institutions: Specific institutions to evaluate
            case_id: Optional case ID

        Returns:
            Assessment of institutional capacity and effectiveness
        """
        return self.process({
            'research_question': f'Institutional capacity assessment for {country}',
            'geographic_scope': [country],
            'research_priorities': ['institutional_capacity', 'judicial_independence', 'rule_of_law'],
            'case_id': case_id
        })