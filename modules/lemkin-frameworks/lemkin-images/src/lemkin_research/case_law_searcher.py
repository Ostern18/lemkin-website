"""
Legal case law search module for multiple databases and sources.

This module provides comprehensive case law search capabilities across
various legal databases including Google Scholar, Westlaw API framework,
CourtListener, Justia, and other free legal databases.
"""

import asyncio
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Set
from urllib.parse import urljoin, quote_plus

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from loguru import logger

from .core import (
    CaseLawResults, CaseOpinion, DatabaseResult, SearchQuery,
    DatabaseType, JurisdictionType, ResearchConfig, ResearchStatus
)


class RateLimiter:
    """Rate limiting for API and web requests"""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove old requests outside time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)


class BaseSearchEngine:
    """Base class for legal database search engines"""
    
    def __init__(self, config: ResearchConfig, database_type: DatabaseType):
        self.config = config
        self.database_type = database_type
        self.rate_limiter = RateLimiter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Lemkin Legal Research Assistant)'
        })
    
    async def search(self, query: Union[str, SearchQuery]) -> DatabaseResult:
        """Search the database - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _create_case_opinion(self, case_data: Dict[str, Any]) -> CaseOpinion:
        """Create CaseOpinion from raw case data"""
        return CaseOpinion(
            case_name=case_data.get('case_name', ''),
            citation=case_data.get('citation', ''),
            court=case_data.get('court', ''),
            date_decided=case_data.get('date_decided'),
            judge=case_data.get('judge'),
            docket_number=case_data.get('docket_number'),
            opinion_type=case_data.get('opinion_type', 'majority'),
            full_text=case_data.get('full_text'),
            summary=case_data.get('summary'),
            key_facts=case_data.get('key_facts', []),
            legal_issues=case_data.get('legal_issues', []),
            holdings=case_data.get('holdings', []),
            jurisdiction=case_data.get('jurisdiction'),
            subject_areas=case_data.get('subject_areas', []),
            cited_cases=case_data.get('cited_cases', []),
            citing_cases=case_data.get('citing_cases', []),
            overruled=case_data.get('overruled', False),
            reversed=case_data.get('reversed', False),
            url=case_data.get('url')
        )


class GoogleScholarSearchEngine(BaseSearchEngine):
    """Google Scholar legal case search"""
    
    BASE_URL = "https://scholar.google.com/scholar"
    
    def __init__(self, config: ResearchConfig):
        super().__init__(config, DatabaseType.GOOGLE_SCHOLAR)
        self.search_params = {
            'hl': 'en',
            'as_sdt': '2006',  # Case law and opinions
            'as_vis': '1',     # Include citations
        }
    
    async def search(self, query: Union[str, SearchQuery]) -> DatabaseResult:
        """Search Google Scholar for legal cases"""
        await self.rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        if isinstance(query, str):
            search_query = SearchQuery(query_text=query)
        else:
            search_query = query
        
        # Build search parameters
        params = self.search_params.copy()
        params['q'] = self._build_query_string(search_query)
        params['num'] = min(search_query.max_results, 20)  # Google Scholar limit
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            cases = self._parse_scholar_results(response.text)
            
            return DatabaseResult(
                database=DatabaseType.GOOGLE_SCHOLAR,
                query=search_query.query_text,
                results_count=len(cases),
                cases=cases,
                search_time=time.time() - start_time,
                filters_applied=self._get_applied_filters(search_query)
            )
            
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
            return DatabaseResult(
                database=DatabaseType.GOOGLE_SCHOLAR,
                query=search_query.query_text,
                results_count=0,
                cases=[],
                search_time=time.time() - start_time
            )
    
    def _build_query_string(self, query: SearchQuery) -> str:
        """Build Google Scholar query string"""
        query_parts = [query.query_text]
        
        if query.case_name:
            query_parts.append(f'"{query.case_name}"')
        
        if query.court:
            query_parts.append(f'court:"{query.court}"')
        
        if query.date_range:
            start_year = query.date_range.get('start', {}).year if query.date_range.get('start') else None
            end_year = query.date_range.get('end', {}).year if query.date_range.get('end') else None
            if start_year and end_year:
                query_parts.append(f'after:{start_year} before:{end_year}')
        
        for exclude in query.exclude_terms:
            query_parts.append(f'-{exclude}')
        
        return ' '.join(query_parts)
    
    def _parse_scholar_results(self, html: str) -> List[CaseOpinion]:
        """Parse Google Scholar search results"""
        soup = BeautifulSoup(html, 'html.parser')
        cases = []
        
        for result in soup.find_all('div', class_='gs_r gs_or gs_scl'):
            try:
                case_data = self._extract_case_data(result)
                if case_data:
                    cases.append(self._create_case_opinion(case_data))
            except Exception as e:
                logger.warning(f"Failed to parse Google Scholar result: {e}")
        
        return cases
    
    def _extract_case_data(self, result_element) -> Optional[Dict[str, Any]]:
        """Extract case data from Google Scholar result element"""
        try:
            # Title and link
            title_element = result_element.find('h3', class_='gs_rt')
            if not title_element:
                return None
            
            case_name = title_element.get_text().strip()
            url = None
            link = title_element.find('a')
            if link and link.get('href'):
                url = link.get('href')
            
            # Citation and court info
            citation_element = result_element.find('div', class_='gs_a')
            citation_text = citation_element.get_text() if citation_element else ''
            
            # Extract court and citation
            court, citation = self._parse_citation_info(citation_text)
            
            # Summary/snippet
            summary_element = result_element.find('div', class_='gs_rs')
            summary = summary_element.get_text().strip() if summary_element else ''
            
            return {
                'case_name': case_name,
                'citation': citation,
                'court': court,
                'summary': summary,
                'url': url,
                'jurisdiction': self._infer_jurisdiction(court)
            }
            
        except Exception as e:
            logger.warning(f"Error extracting case data: {e}")
            return None
    
    def _parse_citation_info(self, citation_text: str) -> tuple:
        """Parse court and citation from citation text"""
        # Pattern to match court and citation info
        patterns = [
            r'(.+?)\s*-\s*(.+?)\s*,\s*(\d{4})',  # Court - Citation, Year
            r'(.+?)\s*,\s*(\d+\s+\w+\.?\s+\d+)',  # Case, Citation
            r'(.+?)\s*(\d+\s+\w+\.?\s+\w*\.?\s+\d+)'  # Case Citation
        ]
        
        court = ''
        citation = ''
        
        for pattern in patterns:
            match = re.search(pattern, citation_text)
            if match:
                if len(match.groups()) >= 2:
                    court = match.group(1).strip()
                    citation = match.group(2).strip()
                break
        
        return court, citation
    
    def _infer_jurisdiction(self, court: str) -> Optional[JurisdictionType]:
        """Infer jurisdiction type from court name"""
        court_lower = court.lower()
        
        if any(term in court_lower for term in ['supreme court', 'scotus', 'us']):
            return JurisdictionType.FEDERAL
        elif any(term in court_lower for term in ['circuit', 'federal', 'district']):
            return JurisdictionType.FEDERAL
        elif any(term in court_lower for term in ['state', 'commonwealth']):
            return JurisdictionType.STATE
        else:
            return JurisdictionType.STATE  # Default assumption
    
    def _get_applied_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Get applied search filters"""
        filters = {}
        if query.case_name:
            filters['case_name'] = query.case_name
        if query.court:
            filters['court'] = query.court
        if query.date_range:
            filters['date_range'] = query.date_range
        if query.exclude_terms:
            filters['excluded_terms'] = query.exclude_terms
        return filters


class CourtListenerSearchEngine(BaseSearchEngine):
    """CourtListener.com free legal database search"""
    
    BASE_URL = "https://www.courtlistener.com/api/rest/v3/search/"
    
    def __init__(self, config: ResearchConfig):
        super().__init__(config, DatabaseType.COURTLISTENER)
        # CourtListener API token if available
        self.api_token = config.database_configs.get('courtlistener', {}).get('api_key')
        if self.api_token:
            self.session.headers['Authorization'] = f'Token {self.api_token}'
    
    async def search(self, query: Union[str, SearchQuery]) -> DatabaseResult:
        """Search CourtListener database"""
        await self.rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        if isinstance(query, str):
            search_query = SearchQuery(query_text=query)
        else:
            search_query = query
        
        # Build API parameters
        params = {
            'q': search_query.query_text,
            'type': 'o',  # Opinions
            'format': 'json',
        }
        
        if search_query.court:
            params['court'] = search_query.court
        
        if search_query.date_range:
            if search_query.date_range.get('start'):
                params['filed_after'] = search_query.date_range['start'].isoformat()
            if search_query.date_range.get('end'):
                params['filed_before'] = search_query.date_range['end'].isoformat()
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            cases = self._parse_courtlistener_results(data)
            
            return DatabaseResult(
                database=DatabaseType.COURTLISTENER,
                query=search_query.query_text,
                results_count=len(cases),
                cases=cases,
                search_time=time.time() - start_time,
                total_estimated=data.get('count', len(cases)),
                next_page_token=data.get('next'),
                api_response=data
            )
            
        except Exception as e:
            logger.error(f"CourtListener search failed: {e}")
            return DatabaseResult(
                database=DatabaseType.COURTLISTENER,
                query=search_query.query_text,
                results_count=0,
                cases=[],
                search_time=time.time() - start_time
            )
    
    def _parse_courtlistener_results(self, data: Dict[str, Any]) -> List[CaseOpinion]:
        """Parse CourtListener API response"""
        cases = []
        
        for result in data.get('results', []):
            try:
                case_data = {
                    'case_name': result.get('caseName', ''),
                    'citation': self._build_citation(result),
                    'court': result.get('court', ''),
                    'date_decided': result.get('dateFiled'),
                    'judge': result.get('author_str', ''),
                    'docket_number': result.get('docketNumber', ''),
                    'full_text': result.get('text', ''),
                    'summary': result.get('snippet', ''),
                    'url': f"https://www.courtlistener.com{result.get('absolute_url', '')}",
                    'jurisdiction': self._map_court_to_jurisdiction(result.get('court', ''))
                }
                
                cases.append(self._create_case_opinion(case_data))
                
            except Exception as e:
                logger.warning(f"Failed to parse CourtListener result: {e}")
        
        return cases
    
    def _build_citation(self, result: Dict[str, Any]) -> str:
        """Build citation string from CourtListener result"""
        citations = []
        
        if result.get('citation'):
            citations.extend(result['citation'])
        
        # Fallback to building citation from available data
        if not citations and result.get('volume') and result.get('reporter'):
            citation = f"{result['volume']} {result['reporter']} {result.get('page', '')}"
            citations.append(citation.strip())
        
        return '; '.join(citations) if citations else ''
    
    def _map_court_to_jurisdiction(self, court: str) -> Optional[JurisdictionType]:
        """Map CourtListener court ID to jurisdiction type"""
        # This would typically use a mapping from CourtListener's court API
        if 'scotus' in court.lower():
            return JurisdictionType.FEDERAL
        elif any(term in court.lower() for term in ['ca', 'cit', 'uscfc']):
            return JurisdictionType.FEDERAL
        else:
            return JurisdictionType.STATE


class JustiaSearchEngine(BaseSearchEngine):
    """Justia.com free legal database search"""
    
    BASE_URL = "https://law.justia.com"
    SEARCH_URL = "https://law.justia.com/cases/search/"
    
    def __init__(self, config: ResearchConfig):
        super().__init__(config, DatabaseType.JUSTIA)
        self.driver = None
    
    def _setup_driver(self):
        """Setup Selenium WebDriver for JavaScript-heavy sites"""
        if self.driver is None:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--user-agent=Mozilla/5.0 (Lemkin Legal Research)')
            
            try:
                self.driver = webdriver.Chrome(options=options)
            except Exception as e:
                logger.warning(f"Could not setup Chrome driver: {e}")
                self.driver = None
    
    async def search(self, query: Union[str, SearchQuery]) -> DatabaseResult:
        """Search Justia database"""
        await self.rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        if isinstance(query, str):
            search_query = SearchQuery(query_text=query)
        else:
            search_query = query
        
        # Try web scraping approach first
        try:
            cases = await self._scrape_search_results(search_query)
            
            return DatabaseResult(
                database=DatabaseType.JUSTIA,
                query=search_query.query_text,
                results_count=len(cases),
                cases=cases,
                search_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Justia search failed: {e}")
            return DatabaseResult(
                database=DatabaseType.JUSTIA,
                query=search_query.query_text,
                results_count=0,
                cases=[],
                search_time=time.time() - start_time
            )
    
    async def _scrape_search_results(self, query: SearchQuery) -> List[CaseOpinion]:
        """Scrape Justia search results"""
        cases = []
        
        # Build search URL
        search_url = f"{self.SEARCH_URL}?q={quote_plus(query.query_text)}"
        
        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse search results
            for result in soup.find_all('div', class_='result'):
                case_data = self._extract_justia_case_data(result)
                if case_data:
                    cases.append(self._create_case_opinion(case_data))
            
        except Exception as e:
            logger.warning(f"Justia scraping failed: {e}")
        
        return cases
    
    def _extract_justia_case_data(self, result_element) -> Optional[Dict[str, Any]]:
        """Extract case data from Justia result element"""
        try:
            # Case name and link
            title_element = result_element.find('h3') or result_element.find('a')
            if not title_element:
                return None
            
            case_name = title_element.get_text().strip()
            url = title_element.get('href') if title_element.name == 'a' else None
            
            # Court and citation info
            meta_element = result_element.find('div', class_='meta')
            meta_text = meta_element.get_text() if meta_element else ''
            
            # Summary
            summary_element = result_element.find('div', class_='snippet')
            summary = summary_element.get_text().strip() if summary_element else ''
            
            return {
                'case_name': case_name,
                'citation': self._extract_citation(meta_text),
                'court': self._extract_court(meta_text),
                'summary': summary,
                'url': urljoin(self.BASE_URL, url) if url else None
            }
            
        except Exception as e:
            logger.warning(f"Error extracting Justia case data: {e}")
            return None
    
    def _extract_citation(self, meta_text: str) -> str:
        """Extract citation from meta text"""
        # Look for citation patterns
        citation_pattern = r'(\d+\s+\w+\.?\s+\w*\.?\s+\d+)'
        match = re.search(citation_pattern, meta_text)
        return match.group(1) if match else ''
    
    def _extract_court(self, meta_text: str) -> str:
        """Extract court from meta text"""
        # Extract court information - simplified approach
        parts = meta_text.split('|')
        for part in parts:
            if 'court' in part.lower() or 'supreme' in part.lower():
                return part.strip()
        return parts[0].strip() if parts else ''
    
    def __del__(self):
        """Cleanup WebDriver"""
        if self.driver:
            self.driver.quit()


class WestlawAPISearchEngine(BaseSearchEngine):
    """Westlaw API integration framework"""
    
    def __init__(self, config: ResearchConfig):
        super().__init__(config, DatabaseType.WESTLAW)
        self.api_config = config.database_configs.get('westlaw', {})
        self.api_key = self.api_config.get('api_key')
        self.base_url = self.api_config.get('base_url', 'https://api.westlaw.com/')
        
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
    
    async def search(self, query: Union[str, SearchQuery]) -> DatabaseResult:
        """Search Westlaw via API (requires subscription)"""
        if not self.api_key:
            logger.warning("Westlaw API key not configured")
            return DatabaseResult(
                database=DatabaseType.WESTLAW,
                query=query.query_text if isinstance(query, SearchQuery) else query,
                results_count=0,
                cases=[]
            )
        
        # Implementation would require actual Westlaw API documentation
        # This is a framework for when API access is available
        await self.rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        # Placeholder implementation
        logger.info("Westlaw API integration not fully implemented - requires subscription")
        
        return DatabaseResult(
            database=DatabaseType.WESTLAW,
            query=query.query_text if isinstance(query, SearchQuery) else query,
            results_count=0,
            cases=[],
            search_time=time.time() - start_time
        )


class CaseLawSearcher:
    """
    Main case law search coordinator.
    
    Manages searches across multiple legal databases and aggregates results.
    """
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        
        # Initialize search engines
        self.engines = {}
        
        if DatabaseType.GOOGLE_SCHOLAR in config.enabled_databases:
            self.engines[DatabaseType.GOOGLE_SCHOLAR] = GoogleScholarSearchEngine(config)
        
        if DatabaseType.COURTLISTENER in config.enabled_databases:
            self.engines[DatabaseType.COURTLISTENER] = CourtListenerSearchEngine(config)
        
        if DatabaseType.JUSTIA in config.enabled_databases:
            self.engines[DatabaseType.JUSTIA] = JustiaSearchEngine(config)
        
        if DatabaseType.WESTLAW in config.enabled_databases:
            self.engines[DatabaseType.WESTLAW] = WestlawAPISearchEngine(config)
        
        logger.info(f"Initialized {len(self.engines)} search engines")
    
    async def search(
        self, 
        query: Union[str, SearchQuery],
        databases: Optional[List[DatabaseType]] = None
    ) -> CaseLawResults:
        """
        Search for case law across specified databases
        
        Args:
            query: Search query string or SearchQuery object
            databases: Specific databases to search (default: all enabled)
            
        Returns:
            CaseLawResults with aggregated search results
        """
        start_time = time.time()
        
        if isinstance(query, str):
            search_query = SearchQuery(query_text=query)
        else:
            search_query = query
        
        # Determine which databases to search
        target_databases = databases or list(self.engines.keys())
        target_databases = [db for db in target_databases if db in self.engines]
        
        if not target_databases:
            logger.warning("No valid databases specified for search")
            return CaseLawResults(
                query=search_query.query_text,
                status=ResearchStatus.FAILED
            )
        
        logger.info(f"Starting search across {len(target_databases)} databases")
        
        # Execute searches concurrently
        tasks = []
        for db_type in target_databases:
            task = asyncio.create_task(
                self._search_database(db_type, search_query)
            )
            tasks.append(task)
        
        # Wait for all searches to complete
        database_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        all_cases = []
        total_results = 0
        jurisdiction_breakdown = {}
        
        for i, result in enumerate(database_results):
            if isinstance(result, Exception):
                logger.error(f"Database search failed: {result}")
                continue
            
            if isinstance(result, DatabaseResult):
                valid_results.append(result)
                all_cases.extend(result.cases)
                total_results += result.results_count
                
                # Update jurisdiction breakdown
                for case in result.cases:
                    if case.jurisdiction:
                        jurisdiction_breakdown[case.jurisdiction.value] = \
                            jurisdiction_breakdown.get(case.jurisdiction.value, 0) + 1
        
        # Remove duplicates based on citation or case name
        unique_cases = self._deduplicate_cases(all_cases)
        
        search_duration = time.time() - start_time
        
        results = CaseLawResults(
            query=search_query.query_text,
            total_results=len(unique_cases),
            database_results=valid_results,
            aggregated_cases=unique_cases,
            search_duration=search_duration,
            filters_applied=self._get_applied_filters(search_query),
            jurisdiction_breakdown=jurisdiction_breakdown,
            status=ResearchStatus.COMPLETED if valid_results else ResearchStatus.FAILED
        )
        
        logger.info(f"Search completed: {len(unique_cases)} unique cases from "
                   f"{len(valid_results)} databases in {search_duration:.2f}s")
        
        return results
    
    async def _search_database(
        self, 
        database_type: DatabaseType, 
        query: SearchQuery
    ) -> DatabaseResult:
        """Search a specific database"""
        try:
            engine = self.engines[database_type]
            result = await engine.search(query)
            logger.info(f"{database_type.value}: {result.results_count} results in "
                       f"{result.search_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Search failed for {database_type.value}: {e}")
            return DatabaseResult(
                database=database_type,
                query=query.query_text,
                results_count=0,
                cases=[]
            )
    
    def _deduplicate_cases(self, cases: List[CaseOpinion]) -> List[CaseOpinion]:
        """Remove duplicate cases based on citation and case name"""
        seen_citations = set()
        seen_names = set()
        unique_cases = []
        
        for case in cases:
            # Create identifier based on citation and case name
            citation_key = case.citation.lower().strip() if case.citation else ''
            name_key = case.case_name.lower().strip() if case.case_name else ''
            
            # Skip if we've seen this citation or very similar case name
            if citation_key and citation_key in seen_citations:
                continue
            
            if name_key and name_key in seen_names:
                continue
            
            # Add to unique cases
            unique_cases.append(case)
            
            if citation_key:
                seen_citations.add(citation_key)
            if name_key:
                seen_names.add(name_key)
        
        return unique_cases
    
    def _get_applied_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Get dictionary of applied search filters"""
        filters = {}
        
        if query.case_name:
            filters['case_name'] = query.case_name
        if query.court:
            filters['court'] = query.court
        if query.judge:
            filters['judge'] = query.judge
        if query.date_range:
            filters['date_range'] = query.date_range
        if query.jurisdiction:
            filters['jurisdiction'] = query.jurisdiction.value
        if query.subject_areas:
            filters['subject_areas'] = query.subject_areas
        if query.exclude_terms:
            filters['exclude_terms'] = query.exclude_terms
        
        return filters
    
    def get_available_databases(self) -> List[DatabaseType]:
        """Get list of available/enabled databases"""
        return list(self.engines.keys())
    
    def is_database_available(self, database_type: DatabaseType) -> bool:
        """Check if a specific database is available"""
        return database_type in self.engines
    
    async def test_database_connection(
        self, 
        database_type: DatabaseType
    ) -> Dict[str, Any]:
        """Test connection to a specific database"""
        if database_type not in self.engines:
            return {'available': False, 'error': 'Database not configured'}
        
        try:
            # Perform a simple test search
            test_query = SearchQuery(query_text="test", max_results=1)
            result = await self.engines[database_type].search(test_query)
            
            return {
                'available': True,
                'response_time': result.search_time,
                'test_results': result.results_count
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }


# Convenience function for direct module usage
async def search_case_law(
    query: Union[str, SearchQuery],
    databases: Optional[List[DatabaseType]] = None,
    config: Optional[ResearchConfig] = None
) -> CaseLawResults:
    """
    Convenience function to search case law
    
    Args:
        query: Search query string or SearchQuery object
        databases: Specific databases to search
        config: Research configuration (uses default if not provided)
        
    Returns:
        CaseLawResults with search results
    """
    if config is None:
        from .core import ResearchConfig
        config = ResearchConfig()
    
    searcher = CaseLawSearcher(config)
    return await searcher.search(query, databases)


# Export main classes and functions
__all__ = [
    'CaseLawSearcher',
    'search_case_law',
    'GoogleScholarSearchEngine',
    'CourtListenerSearchEngine', 
    'JustiaSearchEngine',
    'WestlawAPISearchEngine',
    'BaseSearchEngine',
    'RateLimiter'
]