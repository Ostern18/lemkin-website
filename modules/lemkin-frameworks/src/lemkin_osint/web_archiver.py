"""
Lemkin OSINT Collection Toolkit - Web Archiver

Web content preservation and archiving using Wayback Machine API and other services.
Implements responsible archiving practices for digital evidence preservation.

Compliance: Berkeley Protocol for Digital Investigations
"""

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse
import logging
import json
import re

import aiohttp
import requests
from bs4 import BeautifulSoup

from .core import (
    ArchiveCollection, ArchiveEntry, WebContent, CollectionStatus,
    OSINTConfig, ContentType
)

logger = logging.getLogger(__name__)


class ArchiveError(Exception):
    """Raised when archiving operations fail"""
    pass


class WebArchiver:
    """
    Web content archiver that preserves content using multiple archive services
    while respecting robots.txt and implementing ethical collection practices
    """
    
    def __init__(self, config: OSINTConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LemkinOSINT/1.0 (Digital Evidence Preservation Tool)'
        })
        
        # Archive service endpoints
        self.wayback_api = "http://web.archive.org"
        self.wayback_save_api = "https://web.archive.org/save"
        self.wayback_availability_api = "http://archive.org/wayback/available"
        
        # Rate limiting
        self._last_archive_time = 0
        self._archive_requests = []
        
        logger.info("Web archiver initialized")
    
    def _check_robots_txt(self, url: str) -> bool:
        """
        Check robots.txt compliance for the given URL
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if archiving is allowed
        """
        if not self.config.respect_robots_txt:
            return True
            
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                robots_content = response.text.lower()
                
                # Check for archive-specific rules
                user_agent_sections = re.findall(
                    r'user-agent:\s*\*.*?(?=user-agent:|$)', 
                    robots_content, 
                    re.DOTALL | re.IGNORECASE
                )
                
                for section in user_agent_sections:
                    if 'disallow: /' in section:
                        logger.warning(f"robots.txt disallows archiving for {url}")
                        return False
                        
        except Exception as e:
            logger.warning(f"Could not check robots.txt for {url}: {e}")
            # If we can't check, be conservative
            return False
            
        return True
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting for archive requests"""
        now = time.time()
        
        # Remove old requests (older than 1 minute)
        minute_ago = now - 60
        self._archive_requests = [
            req_time for req_time in self._archive_requests
            if req_time > minute_ago
        ]
        
        # Check rate limit (max 30 requests per minute for Wayback Machine)
        if len(self._archive_requests) >= 30:
            sleep_time = 60 - (now - min(self._archive_requests))
            if sleep_time > 0:
                logger.info(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Enforce minimum delay between requests
        if self._last_archive_time > 0:
            time_since_last = now - self._last_archive_time
            min_delay = 2.0  # 2 second minimum delay
            
            if time_since_last < min_delay:
                sleep_time = min_delay - time_since_last
                time.sleep(sleep_time)
        
        # Record this request
        self._archive_requests.append(time.time())
        self._last_archive_time = time.time()
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _extract_page_metadata(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract metadata from HTML content
        
        Args:
            html_content: HTML content to parse
            url: Original URL
            
        Returns:
            Dict containing extracted metadata
        """
        metadata = {
            'extraction_time': datetime.utcnow(),
            'original_url': url
        }
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Extract meta tags
            meta_tags = soup.find_all('meta')
            metadata['meta_tags'] = {}
            
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    metadata['meta_tags'][name] = content
            
            # Extract language
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                metadata['language'] = html_tag.get('lang')
            
            # Extract links
            links = soup.find_all('a', href=True)
            metadata['outbound_links'] = [
                urljoin(url, link.get('href')) 
                for link in links[:100]  # Limit to avoid huge metadata
            ]
            
            # Extract images
            images = soup.find_all('img', src=True)
            metadata['images'] = [
                urljoin(url, img.get('src')) 
                for img in images[:50]  # Limit to avoid huge metadata
            ]
            
        except Exception as e:
            logger.warning(f"Error extracting page metadata: {e}")
        
        return metadata
    
    async def archive_web_content(self, urls: List[str]) -> ArchiveCollection:
        """
        Archive web content from multiple URLs
        
        Args:
            urls: List of URLs to archive
            
        Returns:
            ArchiveCollection containing archive results
        """
        collection = ArchiveCollection(
            name=f"Web Archive Collection - {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            total_urls=len(urls)
        )
        
        collection.status = CollectionStatus.IN_PROGRESS
        
        try:
            for url in urls:
                try:
                    logger.info(f"Archiving URL: {url}")
                    
                    # Check robots.txt compliance
                    if not self._check_robots_txt(url):
                        logger.warning(f"Skipping {url} due to robots.txt restrictions")
                        collection.failed_archives += 1
                        continue
                    
                    # Archive the URL
                    archive_result = await self._archive_single_url(url)
                    
                    if archive_result:
                        collection.archives.append(archive_result['archive_entry'])
                        collection.web_content.append(archive_result['web_content'])
                        collection.successful_archives += 1
                        logger.info(f"Successfully archived: {url}")
                    else:
                        collection.failed_archives += 1
                        logger.error(f"Failed to archive: {url}")
                    
                except Exception as e:
                    logger.error(f"Error archiving {url}: {e}")
                    collection.failed_archives += 1
                
                # Rate limiting
                if len(urls) > 1:
                    await asyncio.sleep(2)  # 2 second delay between URLs
            
            collection.status = CollectionStatus.COMPLETED
            logger.info(
                f"Archive collection completed. "
                f"Success: {collection.successful_archives}, "
                f"Failed: {collection.failed_archives}"
            )
            
        except Exception as e:
            collection.status = CollectionStatus.FAILED
            logger.error(f"Archive collection failed: {e}")
        
        return collection
    
    async def _archive_single_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Archive a single URL using Wayback Machine
        
        Args:
            url: URL to archive
            
        Returns:
            Dict containing archive entry and web content, or None if failed
        """
        try:
            # First, fetch the current content
            web_content = await self._fetch_web_content(url)
            if not web_content:
                return None
            
            # Submit to Wayback Machine
            archive_entry = await self._submit_to_wayback(url)
            if not archive_entry:
                # If Wayback submission fails, still preserve the content
                archive_entry = ArchiveEntry(
                    original_url=url,
                    archived_url=url,  # Use original URL as fallback
                    archive_timestamp=datetime.utcnow(),
                    content_hash=web_content.content_hash,
                    archive_service="local_preservation"
                )
            
            return {
                'archive_entry': archive_entry,
                'web_content': web_content
            }
            
        except Exception as e:
            logger.error(f"Error archiving single URL {url}: {e}")
            return None
    
    async def _fetch_web_content(self, url: str) -> Optional[WebContent]:
        """
        Fetch web content from a URL
        
        Args:
            url: URL to fetch
            
        Returns:
            WebContent object or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        logger.error(f"HTTP {response.status} for {url}")
                        return None
                    
                    html_content = await response.text()
                    
                    # Extract text content
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text_content = soup.get_text(separator=' ', strip=True)
                    
                    # Calculate content hash
                    content_hash = self._calculate_content_hash(html_content)
                    
                    # Extract metadata
                    metadata = self._extract_page_metadata(html_content, url)
                    
                    # Create WebContent object
                    web_content = WebContent(
                        url=url,
                        title=metadata.get('title', ''),
                        html_content=html_content,
                        text_content=text_content,
                        content_hash=content_hash,
                        content_type=response.headers.get('content-type', 'text/html'),
                        content_length=len(html_content),
                        status_code=response.status,
                        headers=dict(response.headers),
                        last_modified=self._parse_last_modified(
                            response.headers.get('last-modified')
                        ),
                        robots_txt_compliant=self._check_robots_txt(url)
                    )
                    
                    return web_content
                    
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return None
    
    async def _submit_to_wayback(self, url: str) -> Optional[ArchiveEntry]:
        """
        Submit URL to Wayback Machine for archiving
        
        Args:
            url: URL to submit for archiving
            
        Returns:
            ArchiveEntry if successful, None otherwise
        """
        if not self.config.use_wayback_machine:
            return None
            
        try:
            # Enforce rate limiting
            self._enforce_rate_limit()
            
            # Submit to Wayback Machine save API
            save_url = f"{self.wayback_save_api}/{url}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    save_url,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        # Extract archived URL from response
                        archived_url = str(response.url)
                        
                        # Create archive entry
                        archive_entry = ArchiveEntry(
                            original_url=url,
                            archived_url=archived_url,
                            archive_timestamp=datetime.utcnow(),
                            content_hash="",  # Will be updated later
                            archive_service="wayback_machine"
                        )
                        
                        return archive_entry
                    else:
                        logger.error(f"Wayback submission failed: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error submitting to Wayback Machine: {e}")
            return None
    
    def _parse_last_modified(self, last_modified_header: Optional[str]) -> Optional[datetime]:
        """Parse Last-Modified header to datetime"""
        if not last_modified_header:
            return None
            
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(last_modified_header)
        except Exception:
            return None
    
    def check_archive_availability(self, url: str) -> Dict[str, Any]:
        """
        Check if a URL is already archived in Wayback Machine
        
        Args:
            url: URL to check
            
        Returns:
            Dict containing availability information
        """
        try:
            availability_url = f"{self.wayback_availability_api}?url={url}"
            
            response = self.session.get(availability_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                archived_snapshots = data.get('archived_snapshots', {})
                closest = archived_snapshots.get('closest', {})
                
                if closest.get('available'):
                    return {
                        'available': True,
                        'url': closest.get('url'),
                        'timestamp': closest.get('timestamp'),
                        'status': closest.get('status')
                    }
            
            return {'available': False}
            
        except Exception as e:
            logger.error(f"Error checking archive availability: {e}")
            return {'available': False, 'error': str(e)}
    
    def get_archived_versions(self, url: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of archived versions of a URL from Wayback Machine
        
        Args:
            url: URL to search for
            limit: Maximum number of versions to return
            
        Returns:
            List of archived version information
        """
        versions = []
        
        try:
            # Use Wayback Machine CDX API to get timeline
            cdx_url = f"{self.wayback_api}/cdx/search/cdx"
            params = {
                'url': url,
                'output': 'json',
                'limit': limit
            }
            
            response = self.session.get(cdx_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                # Skip header row
                for row in data[1:]:
                    if len(row) >= 3:
                        timestamp = row[1]
                        archived_url = f"{self.wayback_api}/web/{timestamp}/{url}"
                        
                        versions.append({
                            'timestamp': timestamp,
                            'archived_url': archived_url,
                            'status_code': row[4] if len(row) > 4 else None,
                            'mime_type': row[3] if len(row) > 3 else None
                        })
            
        except Exception as e:
            logger.error(f"Error getting archived versions: {e}")
        
        return versions
    
    def verify_archive_integrity(self, archive_entry: ArchiveEntry) -> bool:
        """
        Verify the integrity of an archived entry
        
        Args:
            archive_entry: Archive entry to verify
            
        Returns:
            bool: True if integrity is verified
        """
        try:
            # Fetch archived content
            response = self.session.get(archive_entry.archived_url, timeout=30)
            if response.status_code == 200:
                content_hash = self._calculate_content_hash(response.text)
                
                # Compare with stored hash
                if archive_entry.content_hash:
                    return content_hash == archive_entry.content_hash
                else:
                    # Update hash if not set
                    archive_entry.content_hash = content_hash
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verifying archive integrity: {e}")
            return False
    
    def search_archived_content(
        self, 
        query: str, 
        domain: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search archived content using Wayback Machine
        
        Args:
            query: Search query
            domain: Optional domain to limit search
            date_range: Optional date range tuple (start, end)
            
        Returns:
            List of search results
        """
        results = []
        
        try:
            # Note: This is a simplified implementation
            # Real implementation would use more sophisticated search APIs
            
            search_url = domain if domain else f"*.com/*{query}*"
            cdx_url = f"{self.wayback_api}/cdx/search/cdx"
            
            params = {
                'url': search_url,
                'output': 'json',
                'limit': 100,
                'collapse': 'urlkey'
            }
            
            # Add date range if specified
            if date_range:
                start_date = date_range[0].strftime('%Y%m%d')
                end_date = date_range[1].strftime('%Y%m%d')
                params['from'] = start_date
                params['to'] = end_date
            
            response = self.session.get(cdx_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                for row in data[1:]:  # Skip header
                    if len(row) >= 3:
                        results.append({
                            'url': row[2],
                            'timestamp': row[1],
                            'archived_url': f"{self.wayback_api}/web/{row[1]}/{row[2]}",
                            'status_code': row[4] if len(row) > 4 else None
                        })
            
        except Exception as e:
            logger.error(f"Error searching archived content: {e}")
        
        return results
    
    def create_preservation_package(
        self, 
        collection: ArchiveCollection,
        output_path: str
    ) -> bool:
        """
        Create a preservation package with all archived content and metadata
        
        Args:
            collection: Archive collection to package
            output_path: Path to save the package
            
        Returns:
            bool: True if package created successfully
        """
        try:
            import zipfile
            from pathlib import Path
            
            output_path = Path(output_path)
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add collection metadata
                collection_data = {
                    'collection': collection.dict(),
                    'created_at': datetime.utcnow().isoformat(),
                    'tool': 'lemkin-osint-web-archiver',
                    'version': '1.0'
                }
                
                zipf.writestr(
                    'collection_metadata.json',
                    json.dumps(collection_data, indent=2, default=str)
                )
                
                # Add individual web content files
                for i, content in enumerate(collection.web_content):
                    filename = f"content_{i:04d}.html"
                    zipf.writestr(filename, content.html_content or '')
                    
                    # Add metadata for each content
                    metadata_filename = f"content_{i:04d}_metadata.json"
                    zipf.writestr(
                        metadata_filename,
                        json.dumps(content.dict(), indent=2, default=str)
                    )
                
                # Add archive entries
                archive_data = [entry.dict() for entry in collection.archives]
                zipf.writestr(
                    'archive_entries.json',
                    json.dumps(archive_data, indent=2, default=str)
                )
            
            logger.info(f"Preservation package created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating preservation package: {e}")
            return False
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("Web archiver closed")