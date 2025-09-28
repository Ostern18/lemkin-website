"""
Lemkin Image Verification Suite - Reverse Image Search Module

This module implements multi-engine reverse image search functionality for
tracking image distribution and identifying potential sources. It supports
multiple search engines and provides comprehensive result analysis.

Legal Compliance: Meets standards for digital evidence collection in legal proceedings
"""

import asyncio
import base64
import hashlib
import io
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
import cv2
import numpy as np

from .core import (
    SearchEngine,
    SearchResult,
    ReverseSearchResults,
    ImageAuthConfig
)

logger = logging.getLogger(__name__)


class ReverseImageSearcher:
    """
    Multi-engine reverse image search implementation for comprehensive
    image tracking and source identification.
    """
    
    def __init__(self, config: Optional[ImageAuthConfig] = None):
        """Initialize the reverse image searcher"""
        self.config = config or ImageAuthConfig()
        self.session = requests.Session()
        
        # Set a user agent to appear like a regular browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 2.0  # seconds between requests
        
        logger.info("Reverse image searcher initialized")
    
    def search_image(
        self,
        image_path: Path,
        engines: Optional[List[SearchEngine]] = None,
        max_results_per_engine: Optional[int] = None
    ) -> ReverseSearchResults:
        """
        Perform reverse image search across multiple engines
        
        Args:
            image_path: Path to the image file
            engines: List of search engines to use
            max_results_per_engine: Maximum results per engine
            
        Returns:
            ReverseSearchResults with combined results from all engines
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Calculate image hash
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_hash = hashlib.sha256(image_data).hexdigest()
        
        # Use configured engines if not specified
        search_engines = engines or self.config.search_engines
        max_results = max_results_per_engine or (self.config.max_search_results // len(search_engines))
        
        logger.info(f"Starting reverse search for {image_path.name} using {len(search_engines)} engines")
        
        # Initialize results
        results = ReverseSearchResults(
            image_hash=image_hash,
            engines_used=search_engines,
            total_results_found=0
        )
        
        # Search each engine
        for engine in search_engines:
            try:
                logger.info(f"Searching {engine.value}...")
                engine_results = self._search_engine(image_path, engine, max_results)
                results.results.extend(engine_results)
                results.total_results_found += len(engine_results)
                
                # Respect rate limits
                time.sleep(self.min_request_interval)
                
            except Exception as e:
                logger.error(f"Search failed for {engine.value}: {str(e)}")
                continue
        
        # Analyze results
        self._analyze_search_results(results)
        
        logger.info(f"Reverse search completed: {results.total_results_found} total results")
        
        return results
    
    def _search_engine(
        self,
        image_path: Path,
        engine: SearchEngine,
        max_results: int
    ) -> List[SearchResult]:
        """Search a specific engine for the image"""
        
        if engine == SearchEngine.GOOGLE:
            return self._search_google(image_path, max_results)
        elif engine == SearchEngine.TINEYE:
            return self._search_tineye(image_path, max_results)
        elif engine == SearchEngine.BING:
            return self._search_bing(image_path, max_results)
        elif engine == SearchEngine.YANDEX:
            return self._search_yandex(image_path, max_results)
        elif engine == SearchEngine.BAIDU:
            return self._search_baidu(image_path, max_results)
        else:
            logger.warning(f"Unsupported search engine: {engine}")
            return []
    
    def _search_google(self, image_path: Path, max_results: int) -> List[SearchResult]:
        """Search Google Images using reverse image search"""
        results = []
        
        try:
            # Prepare image for upload
            image_data = self._prepare_image_for_upload(image_path)
            
            # Google reverse image search endpoint
            search_url = "https://www.google.com/searchbyimage/upload"
            
            files = {'encoded_image': ('image.jpg', image_data, 'image/jpeg')}
            data = {'image_content': ''}
            
            response = self.session.post(
                search_url,
                files=files,
                data=data,
                timeout=self.config.search_timeout_seconds
            )
            
            if response.status_code == 200:
                results.extend(self._parse_google_results(response.text, max_results))
            
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
        
        return results
    
    def _search_tineye(self, image_path: Path, max_results: int) -> List[SearchResult]:
        """Search TinEye for the image"""
        results = []
        
        try:
            # TinEye search (Note: Requires API key for full functionality)
            # This is a simplified implementation
            search_url = "https://tineye.com/search"
            
            # Prepare image
            image_data = self._prepare_image_for_upload(image_path)
            
            files = {'image': ('image.jpg', image_data, 'image/jpeg')}
            
            response = self.session.post(
                search_url,
                files=files,
                timeout=self.config.search_timeout_seconds
            )
            
            if response.status_code == 200:
                results.extend(self._parse_tineye_results(response.text, max_results))
            
        except Exception as e:
            logger.error(f"TinEye search failed: {str(e)}")
        
        return results
    
    def _search_bing(self, image_path: Path, max_results: int) -> List[SearchResult]:
        """Search Bing Visual Search for the image"""
        results = []
        
        try:
            # Bing Visual Search endpoint
            search_url = "https://www.bing.com/images/search"
            
            # Prepare image
            image_data = self._prepare_image_for_upload(image_path)
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            params = {
                'view': 'detailv2',
                'iss': 'sbi',
                'FORM': 'SBIVSP',
                'sbisrc': 'ImgPicker',
                'q': 'imgurl:data:image/jpeg;base64,' + image_b64[:100]  # Truncated for URL length
            }
            
            response = self.session.get(
                search_url,
                params=params,
                timeout=self.config.search_timeout_seconds
            )
            
            if response.status_code == 200:
                results.extend(self._parse_bing_results(response.text, max_results))
            
        except Exception as e:
            logger.error(f"Bing search failed: {str(e)}")
        
        return results
    
    def _search_yandex(self, image_path: Path, max_results: int) -> List[SearchResult]:
        """Search Yandex Images for the image"""
        results = []
        
        try:
            # Yandex reverse image search
            search_url = "https://yandex.com/images/search"
            
            # Prepare image
            image_data = self._prepare_image_for_upload(image_path)
            
            files = {'upfile': ('image.jpg', image_data, 'image/jpeg')}
            data = {'rpt': 'imageview'}
            
            response = self.session.post(
                search_url,
                files=files,
                data=data,
                timeout=self.config.search_timeout_seconds
            )
            
            if response.status_code == 200:
                results.extend(self._parse_yandex_results(response.text, max_results))
            
        except Exception as e:
            logger.error(f"Yandex search failed: {str(e)}")
        
        return results
    
    def _search_baidu(self, image_path: Path, max_results: int) -> List[SearchResult]:
        """Search Baidu Images for the image"""
        results = []
        
        try:
            # Baidu reverse image search
            search_url = "http://image.baidu.com/pcdutu/a_upload"
            
            # Prepare image
            image_data = self._prepare_image_for_upload(image_path)
            
            files = {'upload': ('image.jpg', image_data, 'image/jpeg')}
            
            response = self.session.post(
                search_url,
                files=files,
                timeout=self.config.search_timeout_seconds
            )
            
            if response.status_code == 200:
                results.extend(self._parse_baidu_results(response.text, max_results))
            
        except Exception as e:
            logger.error(f"Baidu search failed: {str(e)}")
        
        return results
    
    def _prepare_image_for_upload(self, image_path: Path, max_size: Tuple[int, int] = (1024, 1024)) -> bytes:
        """Prepare image for upload by resizing if necessary"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=85)
                return img_bytes.getvalue()
                
        except Exception as e:
            logger.error(f"Failed to prepare image: {str(e)}")
            # Fallback: return original file data
            with open(image_path, 'rb') as f:
                return f.read()
    
    def _parse_google_results(self, html_content: str, max_results: int) -> List[SearchResult]:
        """Parse Google reverse image search results"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find search result elements (this is a simplified parser)
            result_elements = soup.find_all('div', class_='g')[:max_results]
            
            for element in result_elements:
                try:
                    # Extract URL
                    link_elem = element.find('a')
                    if not link_elem or 'href' not in link_elem.attrs:
                        continue
                    
                    url = link_elem['href']
                    if url.startswith('/url?q='):
                        # Extract actual URL from Google redirect
                        url = url.split('/url?q=')[1].split('&')[0]
                    
                    # Extract title
                    title_elem = element.find('h3')
                    title = title_elem.get_text(strip=True) if title_elem else None
                    
                    # Extract description
                    desc_elem = element.find('span', class_='st')
                    description = desc_elem.get_text(strip=True) if desc_elem else None
                    
                    # Extract domain
                    domain = urlparse(url).netloc
                    
                    result = SearchResult(
                        search_engine=SearchEngine.GOOGLE,
                        url=url,
                        title=title,
                        description=description,
                        domain=domain
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse Google result element: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to parse Google results: {str(e)}")
        
        return results
    
    def _parse_tineye_results(self, html_content: str, max_results: int) -> List[SearchResult]:
        """Parse TinEye search results"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find TinEye result elements
            result_elements = soup.find_all('div', class_='match')[:max_results]
            
            for element in result_elements:
                try:
                    # Extract URL
                    link_elem = element.find('a', class_='match-url')
                    if not link_elem:
                        continue
                    
                    url = link_elem.get('href', '')
                    domain = urlparse(url).netloc
                    
                    # Extract image details
                    img_elem = element.find('img')
                    thumbnail_url = img_elem.get('src') if img_elem else None
                    
                    # Extract date if available
                    date_elem = element.find('p', class_='match-date')
                    date_text = date_elem.get_text(strip=True) if date_elem else None
                    publication_date = self._parse_date_string(date_text) if date_text else None
                    
                    result = SearchResult(
                        search_engine=SearchEngine.TINEYE,
                        url=url,
                        domain=domain,
                        thumbnail_url=thumbnail_url,
                        publication_date=publication_date
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse TinEye result element: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to parse TinEye results: {str(e)}")
        
        return results
    
    def _parse_bing_results(self, html_content: str, max_results: int) -> List[SearchResult]:
        """Parse Bing Visual Search results"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find Bing result elements
            result_elements = soup.find_all('li', class_='dg_u')[:max_results]
            
            for element in result_elements:
                try:
                    # Extract URL
                    link_elem = element.find('a')
                    if not link_elem:
                        continue
                    
                    url = link_elem.get('href', '')
                    if url.startswith('//'):
                        url = 'https:' + url
                    
                    # Extract title
                    title_elem = element.find('div', class_='dg_title')
                    title = title_elem.get_text(strip=True) if title_elem else None
                    
                    # Extract domain
                    domain = urlparse(url).netloc
                    
                    result = SearchResult(
                        search_engine=SearchEngine.BING,
                        url=url,
                        title=title,
                        domain=domain
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse Bing result element: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to parse Bing results: {str(e)}")
        
        return results
    
    def _parse_yandex_results(self, html_content: str, max_results: int) -> List[SearchResult]:
        """Parse Yandex Images search results"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find Yandex result elements (simplified)
            result_elements = soup.find_all('div', class_='cbir-similar__thumb')[:max_results]
            
            for element in result_elements:
                try:
                    # Extract URL from parent link
                    link_elem = element.find_parent('a')
                    if not link_elem:
                        continue
                    
                    url = link_elem.get('href', '')
                    domain = urlparse(url).netloc
                    
                    # Extract thumbnail
                    img_elem = element.find('img')
                    thumbnail_url = img_elem.get('src') if img_elem else None
                    
                    result = SearchResult(
                        search_engine=SearchEngine.YANDEX,
                        url=url,
                        domain=domain,
                        thumbnail_url=thumbnail_url
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse Yandex result element: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to parse Yandex results: {str(e)}")
        
        return results
    
    def _parse_baidu_results(self, html_content: str, max_results: int) -> List[SearchResult]:
        """Parse Baidu Images search results"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find Baidu result elements (simplified)
            result_elements = soup.find_all('div', class_='imgpage')[:max_results]
            
            for element in result_elements:
                try:
                    # Extract URL
                    link_elem = element.find('a')
                    if not link_elem:
                        continue
                    
                    url = link_elem.get('href', '')
                    domain = urlparse(url).netloc
                    
                    # Extract title
                    title_elem = element.find('div', class_='imgTit')
                    title = title_elem.get_text(strip=True) if title_elem else None
                    
                    result = SearchResult(
                        search_engine=SearchEngine.BAIDU,
                        url=url,
                        title=title,
                        domain=domain
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse Baidu result element: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to parse Baidu results: {str(e)}")
        
        return results
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        if not date_str:
            return None
        
        # Common date patterns
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{2})/(\d{2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{2})-(\d{2})-(\d{4})',  # DD-MM-YYYY
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',  # DD Mon YYYY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        if pattern.endswith(r'(\d{4})'):  # Year at end
                            if 'Jan|Feb|Mar' in pattern:  # Month name format
                                day, month_str, year = groups
                                month_map = {
                                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                                }
                                month = month_map.get(month_str.lower()[:3])
                                if month:
                                    return datetime(int(year), month, int(day))
                            else:
                                # Assume MM/DD/YYYY or DD-MM-YYYY
                                part1, part2, year = groups
                                return datetime(int(year), int(part1), int(part2))
                        else:  # YYYY-MM-DD
                            year, month, day = groups
                            return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        return None
    
    def _analyze_search_results(self, results: ReverseSearchResults):
        """Analyze search results to extract patterns and insights"""
        
        if not results.results:
            return
        
        # Extract unique domains
        domains = set()
        countries = set()
        languages = set()
        dates = []
        
        for result in results.results:
            if result.domain:
                domains.add(result.domain)
            
            if result.publication_date:
                dates.append(result.publication_date)
            
            # Extract country from domain (simplified)
            if result.domain:
                domain_parts = result.domain.split('.')
                if len(domain_parts) > 1:
                    tld = domain_parts[-1].lower()
                    country_map = {
                        'uk': 'United Kingdom', 'de': 'Germany', 'fr': 'France',
                        'jp': 'Japan', 'cn': 'China', 'ru': 'Russia',
                        'it': 'Italy', 'es': 'Spain', 'br': 'Brazil',
                        'au': 'Australia', 'ca': 'Canada', 'in': 'India'
                    }
                    if tld in country_map:
                        countries.add(country_map[tld])
        
        # Update results with analysis
        results.unique_domains = list(domains)
        results.countries_found = list(countries)
        
        if dates:
            results.oldest_result_date = min(dates)
            results.most_recent_result_date = max(dates)
        
        # Identify potential authenticity indicators
        self._identify_authenticity_indicators(results)
    
    def _identify_authenticity_indicators(self, results: ReverseSearchResults):
        """Identify indicators of image authenticity or manipulation"""
        
        # Check for widespread distribution
        if len(results.unique_domains) > 20:
            results.widespread_usage = True
        
        # Check for stock photo indicators
        stock_domains = [
            'shutterstock.com', 'gettyimages.com', 'istockphoto.com',
            'depositphotos.com', 'unsplash.com', 'pexels.com', 'pixabay.com'
        ]
        
        for domain in results.unique_domains:
            if any(stock_domain in domain for stock_domain in stock_domains):
                results.stock_photo_indicators.append(f"Found on stock photo site: {domain}")
        
        # Check for social media presence
        social_domains = [
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'reddit.com', 'pinterest.com', 'tumblr.com'
        ]
        
        for domain in results.unique_domains:
            if any(social_domain in domain for social_domain in social_domains):
                results.social_media_presence = True
                break
        
        # Identify potential source URLs (high similarity or early dates)
        high_similarity_results = [
            r for r in results.results 
            if r.similarity_score and r.similarity_score > 0.9
        ]
        
        early_results = []
        if results.oldest_result_date:
            early_threshold = results.oldest_result_date
            early_results = [
                r for r in results.results 
                if r.publication_date and r.publication_date <= early_threshold
            ]
        
        # Combine high similarity and early results as potential sources
        potential_sources = set()
        for result in high_similarity_results + early_results:
            potential_sources.add(result.url)
        
        results.potential_source_urls = list(potential_sources)


def reverse_search_image(image_path: Path, config: Optional[ImageAuthConfig] = None) -> ReverseSearchResults:
    """
    Convenience function to perform reverse image search
    
    Args:
        image_path: Path to the image file
        config: Optional configuration
        
    Returns:
        ReverseSearchResults with search results from multiple engines
    """
    searcher = ReverseImageSearcher(config)
    return searcher.search_image(image_path)