"""
Legal research aggregation and synthesis module.

This module provides comprehensive research compilation from multiple sources,
synthesis of legal findings, and generation of legal research memoranda.
Integrates case law search, precedent analysis, and citation processing.
"""

import asyncio
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from collections import defaultdict
import hashlib

from loguru import logger

from .core import (
    ResearchSummary, LegalMemo, CaseLawResults, CaseOpinion, Precedent,
    SimilarCase, PrecedentMatch, CitationMatch, SearchQuery, LegalCitation,
    ResearchConfig, DatabaseType, CitationStyle, JurisdictionType,
    RelevanceScore, ResearchStatus
)


class ResearchSource:
    """Represents a research source with metadata"""
    
    def __init__(
        self,
        source_id: str,
        source_type: str,
        name: str,
        description: str = "",
        reliability_score: float = 1.0,
        last_updated: Optional[datetime] = None
    ):
        self.source_id = source_id
        self.source_type = source_type  # 'database', 'precedent', 'citation', 'manual'
        self.name = name
        self.description = description
        self.reliability_score = max(0.0, min(1.0, reliability_score))
        self.last_updated = last_updated or datetime.now()
        self.results_count = 0
        self.processing_time = 0.0


class ResearchKnowledgeBase:
    """Knowledge base for legal research findings"""
    
    def __init__(self):
        # Core research data
        self.cases = {}  # case_id -> CaseOpinion
        self.precedents = {}  # precedent_id -> Precedent
        self.citations = {}  # citation_id -> LegalCitation
        
        # Relationship mappings
        self.case_to_precedents = defaultdict(list)  # case_id -> [precedent_id]
        self.precedent_to_cases = defaultdict(list)  # precedent_id -> [case_id]
        self.citation_to_cases = defaultdict(list)   # citation_id -> [case_id]
        
        # Source tracking
        self.sources = {}  # source_id -> ResearchSource
        self.source_to_results = defaultdict(list)  # source_id -> [result_id]
        
        # Analysis cache
        self._analysis_cache = {}
        
    def add_case_law_results(self, results: CaseLawResults, source: ResearchSource):
        """Add case law search results to knowledge base"""
        logger.info(f"Adding case law results from {source.name}")
        
        source.results_count = len(results.aggregated_cases)
        source.processing_time = results.search_duration
        self.sources[source.source_id] = source
        
        # Add cases
        for case in results.aggregated_cases:
            case_id = case.id
            self.cases[case_id] = case
            self.source_to_results[source.source_id].append(case_id)
            
            # Add precedents if available
            for precedent in results.precedents:
                precedent_id = precedent.id
                self.precedents[precedent_id] = precedent
                self.case_to_precedents[case_id].append(precedent_id)
                self.precedent_to_cases[precedent_id].append(case_id)
    
    def add_precedent_matches(self, matches: List[PrecedentMatch], source: ResearchSource):
        """Add precedent analysis results to knowledge base"""
        logger.info(f"Adding precedent matches from {source.name}")
        
        source.results_count = len(matches)
        self.sources[source.source_id] = source
        
        for match in matches:
            precedent = match.matched_precedent
            precedent_id = precedent.id
            
            self.precedents[precedent_id] = precedent
            self.source_to_results[source.source_id].append(precedent_id)
            
            # Link to case if available
            case = precedent.case_opinion
            case_id = case.id
            
            if case_id not in self.cases:
                self.cases[case_id] = case
            
            self.precedent_to_cases[precedent_id].append(case_id)
            self.case_to_precedents[case_id].append(precedent_id)
    
    def add_citations(self, citation_match: CitationMatch, source: ResearchSource):
        """Add citation parsing results to knowledge base"""
        logger.info(f"Adding citations from {source.name}")
        
        source.results_count = len(citation_match.citations_found)
        self.sources[source.source_id] = source
        
        for citation in citation_match.citations_found:
            citation_id = citation.id
            self.citations[citation_id] = citation
            self.source_to_results[source.source_id].append(citation_id)
    
    def get_related_cases(self, case_id: str, max_depth: int = 2) -> List[CaseOpinion]:
        """Get cases related to a given case through precedent relationships"""
        if case_id not in self.cases:
            return []
        
        related_cases = set()
        visited = set()
        queue = [(case_id, 0)]
        
        while queue and max_depth > 0:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth >= max_depth:
                continue
            
            visited.add(current_id)
            
            # Get precedents for this case
            for precedent_id in self.case_to_precedents[current_id]:
                # Get other cases citing this precedent
                for related_case_id in self.precedent_to_cases[precedent_id]:
                    if related_case_id != case_id and related_case_id not in visited:
                        related_cases.add(related_case_id)
                        queue.append((related_case_id, depth + 1))
        
        return [self.cases[case_id] for case_id in related_cases if case_id in self.cases]
    
    def get_strongest_precedents(
        self, 
        jurisdiction: Optional[JurisdictionType] = None,
        min_precedential_value: float = 0.7,
        limit: int = 20
    ) -> List[Precedent]:
        """Get strongest precedents, optionally filtered by jurisdiction"""
        precedents = list(self.precedents.values())
        
        # Filter by jurisdiction if specified
        if jurisdiction:
            precedents = [
                p for p in precedents 
                if p.case_opinion.jurisdiction == jurisdiction
            ]
        
        # Filter by minimum precedential value
        precedents = [
            p for p in precedents 
            if p.precedential_value >= min_precedential_value
        ]
        
        # Sort by precedential value
        precedents.sort(key=lambda p: p.precedential_value, reverse=True)
        
        return precedents[:limit]
    
    def find_contradictory_authority(self, reference_case: CaseOpinion) -> List[CaseOpinion]:
        """Find cases that might contradict the reference case"""
        contradictory = []
        
        # Look for cases with opposing holdings in same subject areas
        if reference_case.subject_areas:
            for case in self.cases.values():
                if case.id == reference_case.id:
                    continue
                
                # Check for subject area overlap
                if case.subject_areas:
                    overlap = set(reference_case.subject_areas) & set(case.subject_areas)
                    if overlap:
                        # Simple heuristic: different outcomes might indicate contradiction
                        # This would be more sophisticated in practice
                        contradictory.append(case)
        
        return contradictory
    
    def get_citation_network(self, case_id: str) -> Dict[str, List[str]]:
        """Get citation network for a case"""
        if case_id not in self.cases:
            return {}
        
        case = self.cases[case_id]
        network = {
            'cited_by': case.citing_cases[:20] if case.citing_cases else [],
            'cites': case.cited_cases[:20] if case.cited_cases else [],
            'related_precedents': [
                p.id for p in self.precedents.values()
                if p.case_opinion.id == case_id
            ]
        }
        
        return network
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            'total_cases': len(self.cases),
            'total_precedents': len(self.precedents),
            'total_citations': len(self.citations),
            'total_sources': len(self.sources),
            'jurisdiction_breakdown': self._get_jurisdiction_breakdown(),
            'date_range': self._get_date_range(),
            'source_reliability': self._get_source_reliability_stats()
        }
    
    def _get_jurisdiction_breakdown(self) -> Dict[str, int]:
        """Get breakdown of cases by jurisdiction"""
        breakdown = defaultdict(int)
        for case in self.cases.values():
            if case.jurisdiction:
                breakdown[case.jurisdiction.value] += 1
        return dict(breakdown)
    
    def _get_date_range(self) -> Dict[str, Optional[str]]:
        """Get date range of cases in knowledge base"""
        dates = [
            case.date_decided for case in self.cases.values()
            if case.date_decided
        ]
        
        if not dates:
            return {'earliest': None, 'latest': None}
        
        return {
            'earliest': min(dates).isoformat(),
            'latest': max(dates).isoformat()
        }
    
    def _get_source_reliability_stats(self) -> Dict[str, float]:
        """Get source reliability statistics"""
        if not self.sources:
            return {}
        
        reliabilities = [source.reliability_score for source in self.sources.values()]
        return {
            'average': sum(reliabilities) / len(reliabilities),
            'min': min(reliabilities),
            'max': max(reliabilities)
        }


class LegalMemoGenerator:
    """Generates legal research memoranda from research findings"""
    
    def __init__(self, citation_style: CitationStyle = CitationStyle.BLUEBOOK):
        self.citation_style = citation_style
        
        # Memo templates
        self.templates = {
            'standard': self._get_standard_template(),
            'brief': self._get_brief_template(),
            'comprehensive': self._get_comprehensive_template()
        }
    
    def generate_memo(
        self,
        research_summary: ResearchSummary,
        memo_template: str = 'standard',
        client: Optional[str] = None,
        attorney: Optional[str] = None,
        custom_template: Optional[str] = None
    ) -> LegalMemo:
        """Generate a legal research memorandum"""
        logger.info(f"Generating legal memo using {memo_template} template")
        
        # Use custom template if provided
        if custom_template:
            template = custom_template
        else:
            template = self.templates.get(memo_template, self.templates['standard'])
        
        # Extract key components from research summary
        components = self._extract_memo_components(research_summary)
        
        # Generate memo sections
        memo = LegalMemo(
            title=self._generate_title(research_summary.research_question),
            client=client,
            attorney=attorney,
            research_question=research_summary.research_question,
            brief_answer=self._generate_brief_answer(research_summary),
            executive_summary=self._generate_executive_summary(research_summary),
            factual_background=self._generate_factual_background(research_summary),
            legal_analysis=self._generate_legal_analysis(research_summary),
            conclusion=self._generate_conclusion(research_summary),
            recommendations=research_summary.recommendations.copy(),
            supporting_authority=self._extract_supporting_authority(research_summary),
            contrary_authority=self._extract_contrary_authority(research_summary),
            research_sources=self._extract_research_sources(research_summary),
            formatting_style=self.citation_style,
            metadata={
                'template_used': memo_template,
                'generation_date': datetime.now().isoformat(),
                'research_confidence': research_summary.confidence_level
            }
        )
        
        # Calculate word count
        memo.word_count = self._calculate_word_count(memo)
        
        logger.info(f"Generated {memo.word_count} word legal memo")
        return memo
    
    def _extract_memo_components(self, summary: ResearchSummary) -> Dict[str, Any]:
        """Extract components needed for memo generation"""
        return {
            'key_findings': summary.key_findings,
            'precedents': summary.primary_precedents,
            'supporting_cases': summary.supporting_cases,
            'contrary_cases': summary.contradictory_authority,
            'jurisdictional_analysis': summary.jurisdictional_analysis,
            'confidence': summary.confidence_level,
            'gaps': summary.research_gaps
        }
    
    def _generate_title(self, research_question: str) -> str:
        """Generate memo title from research question"""
        # Extract key terms from question
        question_lower = research_question.lower()
        
        if 'liability' in question_lower:
            return f"Legal Memorandum: Liability Analysis - {research_question[:50]}..."
        elif 'contract' in question_lower:
            return f"Legal Memorandum: Contract Analysis - {research_question[:50]}..."
        elif 'constitutional' in question_lower:
            return f"Legal Memorandum: Constitutional Analysis - {research_question[:50]}..."
        else:
            return f"Legal Research Memorandum - {research_question[:60]}..."
    
    def _generate_brief_answer(self, summary: ResearchSummary) -> str:
        """Generate brief answer section"""
        if summary.confidence_level >= 0.8:
            confidence_phrase = "Based on the research, it appears likely that"
        elif summary.confidence_level >= 0.6:
            confidence_phrase = "The research suggests that"
        else:
            confidence_phrase = "Based on limited research, it is possible that"
        
        # Use first key finding as basis for brief answer
        if summary.key_findings:
            main_finding = summary.key_findings[0]
            return f"{confidence_phrase} {main_finding.lower()}. However, further analysis of specific facts and applicable law would be required for a definitive conclusion."
        else:
            return "Further research is needed to provide a definitive answer to this legal question."
    
    def _generate_executive_summary(self, summary: ResearchSummary) -> str:
        """Generate executive summary section"""
        parts = []
        
        # Research scope
        parts.append(f"This memorandum addresses {summary.research_question.lower()}")
        
        # Key findings summary
        if summary.key_findings:
            findings_text = ". ".join(summary.key_findings[:3])
            parts.append(f"The research revealed the following key points: {findings_text}")
        
        # Precedent summary
        if summary.primary_precedents:
            precedent_count = len(summary.primary_precedents)
            parts.append(f"Analysis of {precedent_count} primary precedents provides relevant guidance")
        
        # Confidence and limitations
        if summary.confidence_level >= 0.7:
            parts.append("The research provides a solid foundation for legal analysis")
        else:
            parts.append("Additional research may be needed for comprehensive analysis")
        
        if summary.research_gaps:
            parts.append(f"Areas requiring further investigation include: {', '.join(summary.research_gaps[:3])}")
        
        return ". ".join(parts) + "."
    
    def _generate_factual_background(self, summary: ResearchSummary) -> str:
        """Generate factual background section"""
        # Extract facts from supporting cases
        facts = []
        
        for case in summary.supporting_cases[:3]:
            if case.case_opinion.key_facts:
                facts.extend(case.case_opinion.key_facts[:2])
        
        if facts:
            return ("The following factual patterns emerged from the case law analysis: " + 
                   ". ".join(facts[:5]) + ".")
        else:
            return "The factual background for this analysis is based on the legal precedents identified in the research."
    
    def _generate_legal_analysis(self, summary: ResearchSummary) -> str:
        """Generate legal analysis section"""
        analysis_parts = []
        
        # Primary precedents analysis
        if summary.primary_precedents:
            analysis_parts.append("## Primary Authority\n")
            
            for i, precedent in enumerate(summary.primary_precedents[:5], 1):
                case = precedent.case_opinion
                case_cite = f"{case.case_name}, {case.citation}" if case.citation else case.case_name
                
                analysis_parts.append(f"{i}. **{case_cite}**")
                
                if precedent.supporting_rationale:
                    rationale = ". ".join(precedent.supporting_rationale[:2])
                    analysis_parts.append(f"   {rationale}")
                
                analysis_parts.append(f"   Precedential Value: {precedent.precedential_value:.2f}")
                analysis_parts.append("")
        
        # Supporting cases
        if summary.supporting_cases:
            analysis_parts.append("## Supporting Authority\n")
            
            for case in summary.supporting_cases[:3]:
                case_name = case.case_opinion.case_name
                similarity = case.similarity_score
                analysis_parts.append(f"- {case_name} (Similarity: {similarity:.2f})")
                
                if case.key_similarities:
                    similarities = ", ".join(case.key_similarities[:2])
                    analysis_parts.append(f"  Similarities: {similarities}")
                
                analysis_parts.append("")
        
        # Contradictory authority
        if summary.contradictory_authority:
            analysis_parts.append("## Contrary Authority\n")
            
            for case in summary.contradictory_authority[:3]:
                analysis_parts.append(f"- {case.case_name}")
                if case.holdings:
                    analysis_parts.append(f"  Holding: {case.holdings[0]}")
                analysis_parts.append("")
        
        # Jurisdictional analysis
        if summary.jurisdictional_analysis:
            analysis_parts.append("## Jurisdictional Considerations\n")
            for jurisdiction, analysis in summary.jurisdictional_analysis.items():
                analysis_parts.append(f"**{jurisdiction}**: {analysis}")
            analysis_parts.append("")
        
        return "\n".join(analysis_parts)
    
    def _generate_conclusion(self, summary: ResearchSummary) -> str:
        """Generate conclusion section"""
        conclusion_parts = []
        
        # Summarize key findings
        if summary.key_findings:
            conclusion_parts.append("Based on the legal research conducted:")
            for i, finding in enumerate(summary.key_findings[:3], 1):
                conclusion_parts.append(f"{i}. {finding}")
        
        # Confidence assessment
        if summary.confidence_level >= 0.8:
            conclusion_parts.append("\nThe research provides strong support for these conclusions.")
        elif summary.confidence_level >= 0.6:
            conclusion_parts.append("\nThe research provides reasonable support for these conclusions.")
        else:
            conclusion_parts.append("\nThese conclusions are tentative and require additional research.")
        
        # Next steps
        if summary.research_gaps:
            conclusion_parts.append(f"\nFurther research is recommended in the following areas: {', '.join(summary.research_gaps)}.")
        
        return "\n".join(conclusion_parts)
    
    def _extract_supporting_authority(self, summary: ResearchSummary) -> List[str]:
        """Extract supporting authority citations"""
        authorities = []
        
        # Primary precedents
        for precedent in summary.primary_precedents:
            case = precedent.case_opinion
            citation = f"{case.case_name}, {case.citation}" if case.citation else case.case_name
            authorities.append(citation)
        
        # Supporting cases
        for case in summary.supporting_cases:
            case_op = case.case_opinion
            citation = f"{case_op.case_name}, {case_op.citation}" if case_op.citation else case_op.case_name
            authorities.append(citation)
        
        return list(set(authorities))  # Remove duplicates
    
    def _extract_contrary_authority(self, summary: ResearchSummary) -> List[str]:
        """Extract contrary authority citations"""
        authorities = []
        
        for case in summary.contradictory_authority:
            citation = f"{case.case_name}, {case.citation}" if case.citation else case.case_name
            authorities.append(citation)
        
        return authorities
    
    def _extract_research_sources(self, summary: ResearchSummary) -> List[str]:
        """Extract research sources used"""
        sources = []
        
        # Database sources
        for db_type in summary.databases_searched:
            sources.append(f"{db_type.value.title()} Legal Database")
        
        # Add general sources
        sources.extend([
            "Legal precedent analysis",
            "Citation network analysis", 
            "Jurisdictional authority review"
        ])
        
        return sources
    
    def _calculate_word_count(self, memo: LegalMemo) -> int:
        """Calculate total word count of memo"""
        text_fields = [
            memo.brief_answer,
            memo.executive_summary, 
            memo.factual_background,
            memo.legal_analysis,
            memo.conclusion
        ]
        
        total_words = 0
        for field in text_fields:
            if field:
                total_words += len(field.split())
        
        return total_words
    
    def _get_standard_template(self) -> str:
        """Standard legal memo template"""
        return """
# Legal Research Memorandum

**To:** {client}
**From:** {attorney}
**Date:** {date}
**Re:** {title}

## Brief Answer
{brief_answer}

## Executive Summary  
{executive_summary}

## Factual Background
{factual_background}

## Legal Analysis
{legal_analysis}

## Conclusion
{conclusion}

## Recommendations
{recommendations}
"""
    
    def _get_brief_template(self) -> str:
        """Brief legal memo template"""
        return """
# Legal Memo: {title}

**Brief Answer:** {brief_answer}

**Analysis:** {legal_analysis}

**Conclusion:** {conclusion}
"""
    
    def _get_comprehensive_template(self) -> str:
        """Comprehensive legal memo template"""
        return """
# Comprehensive Legal Research Memorandum

**MEMORANDUM**

**TO:** {client}
**FROM:** {attorney}  
**DATE:** {date}
**RE:** {title}

---

## I. BRIEF ANSWER
{brief_answer}

## II. EXECUTIVE SUMMARY
{executive_summary}

## III. FACTUAL BACKGROUND
{factual_background}

## IV. LEGAL ANALYSIS
{legal_analysis}

## V. CONCLUSION
{conclusion}

## VI. RECOMMENDATIONS
{recommendations}

## VII. SUPPORTING AUTHORITY
{supporting_authority}

## VIII. CONTRARY AUTHORITY
{contrary_authority}

## IX. RESEARCH SOURCES
{research_sources}

---
*This memorandum was prepared using the Lemkin Legal Research Assistant.*
"""


class ResearchAggregator:
    """
    Main research aggregation engine.
    
    Coordinates multiple research sources, synthesizes findings,
    and generates comprehensive research summaries and legal memoranda.
    """
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.knowledge_base = ResearchKnowledgeBase()
        self.memo_generator = LegalMemoGenerator(config.default_citation_style)
        
        # Component references (will be injected)
        self._case_searcher = None
        self._precedent_analyzer = None
        self._citation_processor = None
        
        logger.info("Research aggregator initialized")
    
    def set_components(self, case_searcher, precedent_analyzer, citation_processor):
        """Set component references for integrated research"""
        self._case_searcher = case_searcher
        self._precedent_analyzer = precedent_analyzer
        self._citation_processor = citation_processor
    
    async def aggregate_research(
        self,
        research_question: str,
        sources: Optional[List[str]] = None,
        max_cases: int = 50,
        include_precedent_analysis: bool = True,
        include_citation_analysis: bool = True
    ) -> ResearchSummary:
        """
        Aggregate research from multiple sources and queries
        
        Args:
            research_question: Primary research question
            sources: Specific sources to include (default: all available)
            max_cases: Maximum number of cases to include
            include_precedent_analysis: Whether to perform precedent analysis
            include_citation_analysis: Whether to analyze citations
            
        Returns:
            ResearchSummary with comprehensive analysis
        """
        logger.info(f"Starting research aggregation for: {research_question}")
        start_time = datetime.now()
        
        # Initialize research summary
        summary = ResearchSummary(
            research_question=research_question,
            status=ResearchStatus.RUNNING,
            metadata={
                'start_time': start_time.isoformat(),
                'max_cases': max_cases,
                'include_precedent_analysis': include_precedent_analysis,
                'include_citation_analysis': include_citation_analysis
            }
        )
        
        try:
            # Phase 1: Case law search
            case_results = await self._perform_case_search(research_question, max_cases)
            if case_results:
                source = ResearchSource(
                    source_id="case_search",
                    source_type="database",
                    name="Case Law Search",
                    description="Multi-database case law search"
                )
                self.knowledge_base.add_case_law_results(case_results, source)
                summary.databases_searched.extend([
                    result.database for result in case_results.database_results
                ])
            
            # Phase 2: Precedent analysis
            if include_precedent_analysis and case_results:
                precedent_matches = await self._perform_precedent_analysis(case_results)
                if precedent_matches:
                    source = ResearchSource(
                        source_id="precedent_analysis",
                        source_type="precedent",
                        name="Precedent Analysis",
                        description="Semantic similarity precedent matching"
                    )
                    self.knowledge_base.add_precedent_matches(precedent_matches, source)
            
            # Phase 3: Citation analysis
            if include_citation_analysis:
                citation_results = await self._perform_citation_analysis(research_question)
                if citation_results:
                    source = ResearchSource(
                        source_id="citation_analysis", 
                        source_type="citation",
                        name="Citation Analysis",
                        description="Legal citation parsing and validation"
                    )
                    self.knowledge_base.add_citations(citation_results, source)
            
            # Phase 4: Research synthesis
            await self._synthesize_research(summary)
            
            # Update final status
            summary.status = ResearchStatus.COMPLETED
            processing_time = (datetime.now() - start_time).total_seconds()
            summary.metadata['processing_time'] = processing_time
            summary.metadata['end_time'] = datetime.now().isoformat()
            
            logger.info(f"Research aggregation completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Research aggregation failed: {e}")
            summary.status = ResearchStatus.FAILED
            summary.metadata['error'] = str(e)
        
        return summary
    
    async def _perform_case_search(
        self, 
        research_question: str, 
        max_cases: int
    ) -> Optional[CaseLawResults]:
        """Perform case law search"""
        if not self._case_searcher:
            logger.warning("Case searcher not available")
            return None
        
        try:
            query = SearchQuery(
                query_text=research_question,
                max_results=max_cases,
                include_related=True
            )
            
            return await self._case_searcher.search(query)
            
        except Exception as e:
            logger.error(f"Case search failed: {e}")
            return None
    
    async def _perform_precedent_analysis(
        self, 
        case_results: CaseLawResults
    ) -> Optional[List[PrecedentMatch]]:
        """Perform precedent analysis on search results"""
        if not self._precedent_analyzer or not case_results.aggregated_cases:
            return None
        
        try:
            # Add cases to precedent analyzer index
            self._precedent_analyzer.add_cases_to_index(case_results.aggregated_cases)
            
            # Find precedents for top cases
            precedent_matches = []
            for case in case_results.aggregated_cases[:10]:  # Analyze top 10 cases
                matches = await asyncio.to_thread(
                    self._precedent_analyzer.find_similar_precedents,
                    case, 
                    5  # Top 5 precedents per case
                )
                precedent_matches.extend(matches)
            
            return precedent_matches
            
        except Exception as e:
            logger.error(f"Precedent analysis failed: {e}")
            return None
    
    async def _perform_citation_analysis(
        self, 
        research_question: str
    ) -> Optional[CitationMatch]:
        """Perform citation analysis"""
        if not self._citation_processor:
            return None
        
        try:
            # Parse any citations in the research question
            return await asyncio.to_thread(
                self._citation_processor.parse,
                research_question
            )
            
        except Exception as e:
            logger.error(f"Citation analysis failed: {e}")
            return None
    
    async def _synthesize_research(self, summary: ResearchSummary):
        """Synthesize research findings into summary"""
        logger.info("Synthesizing research findings")
        
        # Get knowledge base stats
        kb_stats = self.knowledge_base.get_knowledge_base_stats()
        summary.citations_analyzed = kb_stats['total_citations']
        
        # Extract key findings
        summary.key_findings = self._extract_key_findings()
        
        # Get primary precedents
        summary.primary_precedents = self.knowledge_base.get_strongest_precedents(limit=10)
        
        # Get supporting cases
        summary.supporting_cases = self._get_supporting_cases(limit=10)
        
        # Find contradictory authority
        summary.contradictory_authority = self._find_contradictory_authority(limit=5)
        
        # Perform jurisdictional analysis
        summary.jurisdictional_analysis = self._analyze_jurisdictions()
        
        # Temporal analysis
        summary.temporal_analysis = self._analyze_temporal_trends()
        
        # Calculate confidence level
        summary.confidence_level = self._calculate_confidence_level()
        
        # Identify research gaps
        summary.research_gaps = self._identify_research_gaps()
        
        # Generate recommendations
        summary.recommendations = self._generate_recommendations()
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from research"""
        findings = []
        
        # Analyze most frequent holdings
        holdings_count = defaultdict(int)
        for case in self.knowledge_base.cases.values():
            for holding in case.holdings:
                holdings_count[holding] += 1
        
        # Top holdings become key findings
        for holding, count in sorted(holdings_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            if count > 1:  # Only include holdings from multiple cases
                findings.append(f"Multiple cases establish that {holding.lower()}")
        
        # Analyze precedent patterns
        if self.knowledge_base.precedents:
            high_value_precedents = [
                p for p in self.knowledge_base.precedents.values()
                if p.precedential_value >= 0.8
            ]
            
            if high_value_precedents:
                findings.append(f"{len(high_value_precedents)} high-value precedents support the analysis")
        
        # Jurisdictional patterns
        jurisdiction_stats = self.knowledge_base._get_jurisdiction_breakdown()
        if jurisdiction_stats:
            dominant_jurisdiction = max(jurisdiction_stats, key=jurisdiction_stats.get)
            findings.append(f"Majority of relevant cases are from {dominant_jurisdiction} jurisdiction")
        
        return findings or ["Research analysis is ongoing"]
    
    def _get_supporting_cases(self, limit: int = 10) -> List[SimilarCase]:
        """Get supporting cases from knowledge base"""
        supporting_cases = []
        
        # Convert precedents to similar cases for consistency
        for precedent in list(self.knowledge_base.precedents.values())[:limit]:
            similar_case = SimilarCase(
                case_opinion=precedent.case_opinion,
                similarity_score=precedent.precedential_value,
                relevance_score=RelevanceScore.HIGHLY_RELEVANT if precedent.precedential_value >= 0.8 else RelevanceScore.RELEVANT,
                similarity_reasons=precedent.supporting_rationale,
                key_similarities=precedent.supporting_rationale[:3],
                key_differences=precedent.distinguishable_factors[:3]
            )
            supporting_cases.append(similar_case)
        
        return supporting_cases
    
    def _find_contradictory_authority(self, limit: int = 5) -> List[CaseOpinion]:
        """Find contradictory authority"""
        contradictory = []
        
        # Simple approach: look for cases with different outcomes
        # In practice, this would use more sophisticated analysis
        cases_list = list(self.knowledge_base.cases.values())
        
        for case in cases_list[:limit]:
            # Placeholder logic - would be more sophisticated
            if case.overruled or case.reversed:
                contradictory.append(case)
        
        return contradictory
    
    def _analyze_jurisdictions(self) -> Dict[str, str]:
        """Analyze jurisdictional patterns"""
        analysis = {}
        
        jurisdiction_stats = self.knowledge_base._get_jurisdiction_breakdown()
        
        for jurisdiction, count in jurisdiction_stats.items():
            if count >= 3:
                analysis[jurisdiction] = f"Strong precedential support with {count} relevant cases"
            elif count >= 1:
                analysis[jurisdiction] = f"Limited precedential support with {count} case(s)"
        
        return analysis
    
    def _analyze_temporal_trends(self) -> Dict[str, str]:
        """Analyze temporal trends in case law"""
        analysis = {}
        
        cases_with_dates = [
            case for case in self.knowledge_base.cases.values()
            if case.date_decided
        ]
        
        if cases_with_dates:
            dates = [case.date_decided for case in cases_with_dates]
            earliest = min(dates)
            latest = max(dates)
            
            analysis['date_range'] = f"Cases span from {earliest} to {latest}"
            
            # Analyze decade distribution
            decade_counts = defaultdict(int)
            for case_date in dates:
                decade = (case_date.year // 10) * 10
                decade_counts[decade] += 1
            
            if decade_counts:
                most_common_decade = max(decade_counts, key=decade_counts.get)
                analysis['temporal_concentration'] = f"Most cases from {most_common_decade}s ({decade_counts[most_common_decade]} cases)"
        
        return analysis
    
    def _calculate_confidence_level(self) -> float:
        """Calculate overall confidence level in research"""
        factors = []
        
        # Number of sources
        source_count = len(self.knowledge_base.sources)
        source_factor = min(1.0, source_count / 3.0)  # Ideal: 3+ sources
        factors.append(source_factor)
        
        # Number of cases
        case_count = len(self.knowledge_base.cases)
        case_factor = min(1.0, case_count / 10.0)  # Ideal: 10+ cases
        factors.append(case_factor)
        
        # Precedent quality
        if self.knowledge_base.precedents:
            avg_precedent_value = sum(
                p.precedential_value for p in self.knowledge_base.precedents.values()
            ) / len(self.knowledge_base.precedents)
            factors.append(avg_precedent_value)
        
        # Source reliability
        if self.knowledge_base.sources:
            avg_reliability = sum(
                s.reliability_score for s in self.knowledge_base.sources.values()
            ) / len(self.knowledge_base.sources)
            factors.append(avg_reliability)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _identify_research_gaps(self) -> List[str]:
        """Identify gaps in research"""
        gaps = []
        
        # Check for missing jurisdictions
        jurisdiction_stats = self.knowledge_base._get_jurisdiction_breakdown()
        if JurisdictionType.FEDERAL.value not in jurisdiction_stats:
            gaps.append("Federal jurisdiction analysis")
        
        # Check for recent cases
        recent_cases = [
            case for case in self.knowledge_base.cases.values()
            if case.date_decided and case.date_decided.year >= datetime.now().year - 5
        ]
        
        if len(recent_cases) < 3:
            gaps.append("Recent case law (last 5 years)")
        
        # Check for citation analysis
        if not self.knowledge_base.citations:
            gaps.append("Legal citation analysis")
        
        # Check for secondary sources
        secondary_citations = [
            c for c in self.knowledge_base.citations.values()
            if c.citation_type.value == 'secondary'
        ]
        
        if not secondary_citations:
            gaps.append("Secondary authority (law reviews, treatises)")
        
        return gaps
    
    def _generate_recommendations(self) -> List[str]:
        """Generate research recommendations"""
        recommendations = []
        
        # Based on confidence level
        confidence = self._calculate_confidence_level()
        
        if confidence >= 0.8:
            recommendations.append("Research provides strong foundation for legal analysis")
        elif confidence >= 0.6:
            recommendations.append("Consider additional research for comprehensive analysis")
        else:
            recommendations.append("Significant additional research recommended")
        
        # Based on gaps
        gaps = self._identify_research_gaps()
        for gap in gaps[:3]:
            recommendations.append(f"Address research gap: {gap}")
        
        # Jurisdictional recommendations
        jurisdiction_stats = self.knowledge_base._get_jurisdiction_breakdown()
        if len(jurisdiction_stats) == 1:
            recommendations.append("Consider researching other jurisdictions for comparison")
        
        return recommendations
    
    def generate_memo(
        self,
        research_summary: ResearchSummary,
        memo_template: Optional[str] = None,
        client: Optional[str] = None,
        attorney: Optional[str] = None
    ) -> LegalMemo:
        """
        Generate a legal research memorandum
        
        Args:
            research_summary: Research summary to base memo on
            memo_template: Template to use ('standard', 'brief', 'comprehensive')
            client: Client name
            attorney: Attorney name
            
        Returns:
            LegalMemo formatted according to template
        """
        return self.memo_generator.generate_memo(
            research_summary,
            memo_template or 'standard',
            client,
            attorney
        )
    
    def export_research_data(
        self,
        format_type: str = "json",
        include_full_text: bool = False
    ) -> Dict[str, Any]:
        """Export research data in specified format"""
        export_data = {
            'knowledge_base_stats': self.knowledge_base.get_knowledge_base_stats(),
            'sources': {
                source_id: {
                    'name': source.name,
                    'type': source.source_type,
                    'results_count': source.results_count,
                    'reliability_score': source.reliability_score
                }
                for source_id, source in self.knowledge_base.sources.items()
            },
            'cases_summary': [],
            'precedents_summary': [],
            'export_metadata': {
                'format': format_type,
                'timestamp': datetime.now().isoformat(),
                'include_full_text': include_full_text
            }
        }
        
        # Add case summaries
        for case in self.knowledge_base.cases.values():
            case_summary = {
                'id': case.id,
                'case_name': case.case_name,
                'citation': case.citation,
                'court': case.court,
                'date_decided': case.date_decided.isoformat() if case.date_decided else None,
                'jurisdiction': case.jurisdiction.value if case.jurisdiction else None,
                'holdings': case.holdings,
                'legal_issues': case.legal_issues
            }
            
            if include_full_text:
                case_summary['full_text'] = case.full_text
                case_summary['summary'] = case.summary
            
            export_data['cases_summary'].append(case_summary)
        
        # Add precedent summaries
        for precedent in self.knowledge_base.precedents.values():
            precedent_summary = {
                'id': precedent.id,
                'case_name': precedent.case_opinion.case_name,
                'precedential_value': precedent.precedential_value,
                'binding_strength': precedent.binding_strength,
                'jurisdictional_weight': precedent.jurisdictional_weight,
                'supporting_rationale': precedent.supporting_rationale
            }
            
            export_data['precedents_summary'].append(precedent_summary)
        
        return export_data
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get comprehensive research statistics"""
        return {
            'knowledge_base': self.knowledge_base.get_knowledge_base_stats(),
            'processing_stats': {
                'total_sources': len(self.knowledge_base.sources),
                'avg_source_reliability': sum(
                    s.reliability_score for s in self.knowledge_base.sources.values()
                ) / len(self.knowledge_base.sources) if self.knowledge_base.sources else 0,
                'total_processing_time': sum(
                    s.processing_time for s in self.knowledge_base.sources.values()
                )
            }
        }


# Convenience function for direct module usage
async def aggregate_research(
    research_question: str,
    sources: Optional[List[str]] = None,
    config: Optional[ResearchConfig] = None
) -> ResearchSummary:
    """
    Convenience function to aggregate legal research
    
    Args:
        research_question: Primary research question
        sources: Specific sources to include
        config: Research configuration
        
    Returns:
        ResearchSummary with comprehensive analysis
    """
    if config is None:
        from .core import ResearchConfig
        config = ResearchConfig()
    
    aggregator = ResearchAggregator(config)
    return await aggregator.aggregate_research(research_question, sources)


# Export main classes and functions
__all__ = [
    'ResearchAggregator',
    'ResearchKnowledgeBase',
    'LegalMemoGenerator', 
    'ResearchSource',
    'aggregate_research'
]