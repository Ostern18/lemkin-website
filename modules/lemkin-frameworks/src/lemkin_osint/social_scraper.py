"""
Lemkin OSINT Collection Toolkit - Social Media Scraper

Ethical social media data collection within platform Terms of Service limits.
Implements responsible scraping practices with rate limiting and ToS compliance.

Compliance: Berkeley Protocol for Digital Investigations
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, AsyncGenerator
import logging
import json
import re
from urllib.parse import urljoin, urlparse
import hashlib

import aiohttp
import requests
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry

from .core import (
    OSINTCollection, SocialMediaPost, Source, PlatformType, 
    ContentType, CollectionStatus, OSINTConfig
)

logger = logging.getLogger(__name__)


class ToSViolationError(Exception):
    """Raised when an operation would violate platform Terms of Service"""
    pass


class RateLimitExceededError(Exception):
    """Raised when rate limits are exceeded"""
    pass


class PlatformConfig:
    """Configuration for different social media platforms"""
    
    PLATFORM_CONFIGS = {
        PlatformType.TWITTER: {
            'base_url': 'https://twitter.com',
            'search_endpoint': '/search',
            'rate_limit_rpm': 30,
            'max_results_per_request': 20,
            'requires_auth': True,
            'public_data_only': True,
            'respect_robots': True
        },
        PlatformType.REDDIT: {
            'base_url': 'https://www.reddit.com',
            'search_endpoint': '/search.json',
            'rate_limit_rpm': 60,
            'max_results_per_request': 25,
            'requires_auth': False,
            'public_data_only': True,
            'respect_robots': True
        },
        PlatformType.YOUTUBE: {
            'base_url': 'https://www.youtube.com',
            'search_endpoint': '/results',
            'rate_limit_rpm': 100,
            'max_results_per_request': 50,
            'requires_auth': True,
            'public_data_only': True,
            'respect_robots': True
        },
        PlatformType.FACEBOOK: {
            'base_url': 'https://www.facebook.com',
            'search_endpoint': '/search',
            'rate_limit_rpm': 20,
            'max_results_per_request': 10,
            'requires_auth': True,
            'public_data_only': True,
            'respect_robots': True
        }
    }
    
    @classmethod
    def get_config(cls, platform: PlatformType) -> Dict[str, Any]:
        """Get configuration for a specific platform"""
        return cls.PLATFORM_CONFIGS.get(platform, {})


class SocialMediaScraper:
    """
    Ethical social media scraper that respects ToS and implements
    responsible collection practices
    """
    
    def __init__(self, config: OSINTConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LemkinOSINT/1.0 (Research Tool; Ethical Collection)'
        })
        
        # Rate limiting tracking
        self._last_request_time = {}
        self._request_counts = {}
        
        logger.info("Social media scraper initialized")
    
    def _check_robots_txt(self, platform: PlatformType) -> bool:
        """
        Check robots.txt compliance for the platform
        
        Args:
            platform: Platform to check
            
        Returns:
            bool: True if collection is allowed by robots.txt
        """
        if not self.config.respect_robots_txt:
            return True
            
        try:
            platform_config = PlatformConfig.get_config(platform)
            base_url = platform_config.get('base_url', '')
            robots_url = urljoin(base_url, '/robots.txt')
            
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                robots_content = response.text
                
                # Simple robots.txt parsing (in production, use robotparser)
                if 'Disallow: /' in robots_content or 'Disallow: /search' in robots_content:
                    logger.warning(f"robots.txt disallows collection for {platform}")
                    return False
                    
        except Exception as e:
            logger.warning(f"Could not check robots.txt for {platform}: {e}")
            # If we can't check, err on the side of caution
            return False
            
        return True
    
    def _check_rate_limit(self, platform: PlatformType) -> bool:
        """
        Check if we're within rate limits for the platform
        
        Args:
            platform: Platform to check
            
        Returns:
            bool: True if within rate limits
        """
        now = time.time()
        platform_config = PlatformConfig.get_config(platform)
        rpm_limit = platform_config.get('rate_limit_rpm', 30)
        
        # Initialize tracking if needed
        if platform not in self._request_counts:
            self._request_counts[platform] = []
            
        # Remove old requests (older than 1 minute)
        minute_ago = now - 60
        self._request_counts[platform] = [
            req_time for req_time in self._request_counts[platform]
            if req_time > minute_ago
        ]
        
        # Check if we're under the limit
        if len(self._request_counts[platform]) >= rpm_limit:
            logger.warning(f"Rate limit exceeded for {platform}")
            return False
            
        return True
    
    def _enforce_rate_limit(self, platform: PlatformType):
        """
        Enforce rate limiting with delays between requests
        
        Args:
            platform: Platform to enforce rate limiting for
        """
        now = time.time()
        
        # Add current request to tracking
        if platform not in self._request_counts:
            self._request_counts[platform] = []
        self._request_counts[platform].append(now)
        
        # Enforce minimum delay between requests
        if platform in self._last_request_time:
            time_since_last = now - self._last_request_time[platform]
            min_delay = self.config.rate_limit_delay_seconds
            
            if time_since_last < min_delay:
                sleep_time = min_delay - time_since_last
                logger.info(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self._last_request_time[platform] = time.time()
    
    def _validate_collection_request(
        self, 
        platform: PlatformType, 
        query: str
    ) -> bool:
        """
        Validate that a collection request complies with ToS and ethics
        
        Args:
            platform: Platform to collect from
            query: Search query
            
        Returns:
            bool: True if request is valid and ethical
        """
        # Check ToS compliance
        if self.config.tos_compliance_check:
            platform_config = PlatformConfig.get_config(platform)
            
            # Ensure we only collect public data
            if not platform_config.get('public_data_only', False):
                raise ToSViolationError(f"Platform {platform} requires private data access")
            
            # Check if platform requires authentication
            if platform_config.get('requires_auth', False):
                logger.warning(f"Platform {platform} requires authentication - using public endpoints only")
        
        # Check robots.txt
        if not self._check_robots_txt(platform):
            raise ToSViolationError(f"robots.txt disallows collection for {platform}")
        
        # Check rate limits
        if not self._check_rate_limit(platform):
            raise RateLimitExceededError(f"Rate limit exceeded for {platform}")
        
        # Validate query for potentially harmful content
        if self._contains_harmful_intent(query):
            logger.error(f"Query rejected due to harmful intent: {query}")
            return False
        
        return True
    
    def _contains_harmful_intent(self, query: str) -> bool:
        """
        Check if query contains potentially harmful search intent
        
        Args:
            query: Search query to validate
            
        Returns:
            bool: True if query appears harmful
        """
        harmful_patterns = [
            r'\b(doxx?ing|dox)\b',
            r'\b(stalk|stalking)\b',
            r'\b(harass|harassment)\b',
            r'\b(private.*info|personal.*data)\b',
            r'\b(revenge.*porn)\b'
        ]
        
        query_lower = query.lower()
        for pattern in harmful_patterns:
            if re.search(pattern, query_lower):
                return True
                
        return False
    
    def _extract_post_metadata(self, post_element, platform: PlatformType) -> Dict[str, Any]:
        """
        Extract metadata from a social media post element
        
        Args:
            post_element: BeautifulSoup element containing post
            platform: Platform the post is from
            
        Returns:
            Dict containing extracted metadata
        """
        metadata = {
            'platform': platform,
            'extracted_at': datetime.utcnow(),
            'extraction_method': 'html_parsing'
        }
        
        try:
            if platform == PlatformType.TWITTER:
                # Twitter-specific extraction (placeholder)
                metadata.update({
                    'post_id': post_element.get('data-tweet-id'),
                    'author': post_element.get('data-screen-name'),
                    'timestamp': post_element.get('data-time')
                })
            elif platform == PlatformType.REDDIT:
                # Reddit-specific extraction (placeholder)
                metadata.update({
                    'post_id': post_element.get('data-fullname'),
                    'subreddit': post_element.get('data-subreddit'),
                    'score': post_element.get('data-score')
                })
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _create_social_post(
        self, 
        post_data: Dict[str, Any], 
        source: Source
    ) -> SocialMediaPost:
        """
        Create a SocialMediaPost object from extracted data
        
        Args:
            post_data: Extracted post data
            source: Source information
            
        Returns:
            SocialMediaPost object
        """
        return SocialMediaPost(
            platform=post_data.get('platform'),
            platform_post_id=post_data.get('post_id', ''),
            text_content=post_data.get('text', ''),
            author_username=post_data.get('author'),
            author_display_name=post_data.get('author_display_name'),
            author_verified=post_data.get('verified', False),
            likes_count=post_data.get('likes', 0),
            shares_count=post_data.get('shares', 0),
            comments_count=post_data.get('comments', 0),
            published_at=post_data.get('timestamp'),
            location=post_data.get('location'),
            hashtags=post_data.get('hashtags', []),
            mentions=post_data.get('mentions', []),
            media_urls=post_data.get('media_urls', []),
            source=source,
            collection_method='ethical_scraping',
            tos_compliant=True
        )
    
    async def collect_social_media_evidence(
        self,
        query: str,
        platforms: List[PlatformType],
        max_results: Optional[int] = None
    ) -> OSINTCollection:
        """
        Collect social media evidence ethically across multiple platforms
        
        Args:
            query: Search query
            platforms: List of platforms to search
            max_results: Maximum results to collect
            
        Returns:
            OSINTCollection containing collected evidence
        """
        collection = OSINTCollection(
            name=f"Social Media Evidence: {query}",
            query=query,
            platforms=platforms,
            collection_config=self.config
        )
        
        collection.status = CollectionStatus.IN_PROGRESS
        collection.started_at = datetime.utcnow()
        collection.add_log_entry(f"Starting collection for query: {query}")
        
        max_results = max_results or self.config.max_results_per_query
        results_per_platform = max_results // len(platforms) if platforms else 0
        
        try:
            for platform in platforms:
                collection.add_log_entry(f"Collecting from {platform}")
                
                try:
                    # Validate collection request
                    self._validate_collection_request(platform, query)
                    
                    # Collect posts from platform
                    posts = await self._collect_from_platform(
                        platform, query, results_per_platform
                    )
                    
                    collection.social_posts.extend(posts)
                    collection.add_log_entry(f"Collected {len(posts)} posts from {platform}")
                    
                except ToSViolationError as e:
                    error_msg = f"ToS violation for {platform}: {e}"
                    collection.tos_violations.append(error_msg)
                    collection.add_log_entry(error_msg)
                    logger.error(error_msg)
                    
                except RateLimitExceededError as e:
                    error_msg = f"Rate limit exceeded for {platform}: {e}"
                    collection.rate_limit_hits += 1
                    collection.add_log_entry(error_msg)
                    logger.warning(error_msg)
                    
                    # Wait before continuing
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    error_msg = f"Error collecting from {platform}: {e}"
                    collection.add_log_entry(error_msg)
                    logger.error(error_msg)
            
            collection.total_items_collected = len(collection.social_posts)
            collection.status = CollectionStatus.COMPLETED
            collection.completed_at = datetime.utcnow()
            
            collection.add_log_entry(
                f"Collection completed. Total items: {collection.total_items_collected}"
            )
            
        except Exception as e:
            collection.status = CollectionStatus.FAILED
            collection.add_log_entry(f"Collection failed: {e}")
            logger.error(f"Collection failed: {e}")
        
        return collection
    
    async def _collect_from_platform(
        self,
        platform: PlatformType,
        query: str,
        max_results: int
    ) -> List[SocialMediaPost]:
        """
        Collect posts from a specific platform
        
        Args:
            platform: Platform to collect from
            query: Search query
            max_results: Maximum results to collect
            
        Returns:
            List of SocialMediaPost objects
        """
        posts = []
        
        try:
            if platform == PlatformType.REDDIT:
                posts = await self._collect_from_reddit(query, max_results)
            elif platform == PlatformType.TWITTER:
                posts = await self._collect_from_twitter(query, max_results)
            elif platform == PlatformType.YOUTUBE:
                posts = await self._collect_from_youtube(query, max_results)
            else:
                logger.warning(f"Collection not implemented for {platform}")
                
        except Exception as e:
            logger.error(f"Error collecting from {platform}: {e}")
            
        return posts
    
    async def _collect_from_reddit(
        self,
        query: str,
        max_results: int
    ) -> List[SocialMediaPost]:
        """
        Collect posts from Reddit using public API
        
        Args:
            query: Search query
            max_results: Maximum results to collect
            
        Returns:
            List of SocialMediaPost objects
        """
        posts = []
        
        try:
            # Enforce rate limiting
            self._enforce_rate_limit(PlatformType.REDDIT)
            
            # Use Reddit's public JSON API
            url = f"https://www.reddit.com/search.json"
            params = {
                'q': query,
                'limit': min(max_results, 25),
                'sort': 'relevance',
                't': 'all'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for post_data in data.get('data', {}).get('children', []):
                            post = post_data.get('data', {})
                            
                            # Create source
                            source = Source(
                                name=f"Reddit - r/{post.get('subreddit', 'unknown')}",
                                url=f"https://reddit.com{post.get('permalink', '')}",
                                platform=PlatformType.REDDIT
                            )
                            
                            # Create social media post
                            social_post = SocialMediaPost(
                                platform=PlatformType.REDDIT,
                                platform_post_id=post.get('id', ''),
                                text_content=post.get('selftext', post.get('title', '')),
                                author_username=post.get('author'),
                                likes_count=post.get('ups', 0),
                                comments_count=post.get('num_comments', 0),
                                published_at=datetime.fromtimestamp(
                                    post.get('created_utc', 0)
                                ) if post.get('created_utc') else None,
                                source=source,
                                collection_method='reddit_json_api',
                                tos_compliant=True
                            )
                            
                            posts.append(social_post)
                            
                            if len(posts) >= max_results:
                                break
                                
        except Exception as e:
            logger.error(f"Error collecting from Reddit: {e}")
            
        return posts
    
    async def _collect_from_twitter(
        self,
        query: str,
        max_results: int
    ) -> List[SocialMediaPost]:
        """
        Collect posts from Twitter (placeholder - would require API access)
        
        Args:
            query: Search query
            max_results: Maximum results to collect
            
        Returns:
            List of SocialMediaPost objects
        """
        posts = []
        
        # Note: This is a placeholder. Real Twitter collection would require
        # official API access and authentication
        logger.warning(
            "Twitter collection requires official API access. "
            "This is a placeholder implementation."
        )
        
        # Enforce rate limiting
        self._enforce_rate_limit(PlatformType.TWITTER)
        
        # In a real implementation, you would:
        # 1. Use Twitter API v2 with proper authentication
        # 2. Implement pagination for large result sets
        # 3. Handle Twitter-specific rate limits and quotas
        # 4. Parse Twitter's specific JSON response format
        
        return posts
    
    async def _collect_from_youtube(
        self,
        query: str,
        max_results: int
    ) -> List[SocialMediaPost]:
        """
        Collect content from YouTube (placeholder - would require API access)
        
        Args:
            query: Search query
            max_results: Maximum results to collect
            
        Returns:
            List of SocialMediaPost objects
        """
        posts = []
        
        # Note: This is a placeholder. Real YouTube collection would require
        # YouTube Data API access
        logger.warning(
            "YouTube collection requires official Data API access. "
            "This is a placeholder implementation."
        )
        
        # Enforce rate limiting
        self._enforce_rate_limit(PlatformType.YOUTUBE)
        
        # In a real implementation, you would:
        # 1. Use YouTube Data API v3 with API key
        # 2. Search for videos, channels, or playlists
        # 3. Extract video metadata, comments, etc.
        # 4. Handle YouTube's quota system
        
        return posts
    
    def search_public_posts(
        self,
        platform: PlatformType,
        query: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for public posts on a platform (synchronous method)
        
        Args:
            platform: Platform to search
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of post dictionaries
        """
        try:
            self._validate_collection_request(platform, query)
            
            if platform == PlatformType.REDDIT:
                return self._search_reddit_sync(query, limit)
            else:
                logger.warning(f"Synchronous search not implemented for {platform}")
                return []
                
        except Exception as e:
            logger.error(f"Error in public post search: {e}")
            return []
    
    def _search_reddit_sync(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Synchronous Reddit search using public API
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of post dictionaries
        """
        posts = []
        
        try:
            self._enforce_rate_limit(PlatformType.REDDIT)
            
            url = "https://www.reddit.com/search.json"
            params = {
                'q': query,
                'limit': min(limit, 100),
                'sort': 'relevance'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for post_data in data.get('data', {}).get('children', []):
                    post = post_data.get('data', {})
                    
                    post_dict = {
                        'id': post.get('id'),
                        'title': post.get('title'),
                        'text': post.get('selftext', ''),
                        'author': post.get('author'),
                        'subreddit': post.get('subreddit'),
                        'score': post.get('score', 0),
                        'comments': post.get('num_comments', 0),
                        'created_utc': post.get('created_utc'),
                        'url': f"https://reddit.com{post.get('permalink', '')}",
                        'platform': PlatformType.REDDIT
                    }
                    
                    posts.append(post_dict)
                    
        except Exception as e:
            logger.error(f"Error in Reddit search: {e}")
            
        return posts
    
    def validate_ethical_collection(self, collection_request: Dict[str, Any]) -> bool:
        """
        Validate that a collection request meets ethical standards
        
        Args:
            collection_request: Dictionary containing collection parameters
            
        Returns:
            bool: True if collection is ethical
        """
        try:
            query = collection_request.get('query', '')
            platforms = collection_request.get('platforms', [])
            purpose = collection_request.get('purpose', '')
            
            # Check for harmful intent
            if self._contains_harmful_intent(query):
                logger.error("Collection request rejected: harmful intent detected")
                return False
            
            # Validate purpose
            ethical_purposes = [
                'journalism', 'research', 'investigation', 'fact-checking',
                'academic', 'legal', 'law_enforcement', 'security'
            ]
            
            if purpose.lower() not in ethical_purposes:
                logger.warning(f"Purpose '{purpose}' may not be ethical")
            
            # Check platform compliance
            for platform in platforms:
                if not self._check_robots_txt(platform):
                    logger.error(f"robots.txt violation for {platform}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating ethical collection: {e}")
            return False
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("Social media scraper closed")