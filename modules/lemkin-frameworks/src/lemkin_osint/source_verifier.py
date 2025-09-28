"""
Lemkin OSINT Collection Toolkit - Source Verifier

Source credibility assessment and verification for digital investigations.
Implements multi-factor credibility scoring and verification checks.

Compliance: Berkeley Protocol for Digital Investigations
"""

import re
import ssl
import socket
import whois
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
import logging
import json
import requests

from .core import (
    Source, CredibilityAssessment, CredibilityLevel, PlatformType, OSINTConfig
)

logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised when source verification fails"""
    pass


class SourceVerifier:
    """
    Comprehensive source credibility assessor that evaluates sources
    using multiple verification factors and scoring algorithms.
    """
    
    def __init__(self, config: OSINTConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LemkinOSINT/1.0 (Source Verification Tool)'
        })
        
        # Credibility databases and APIs (would be configured in production)
        self.fact_check_apis = {
            'google_factcheck': 'https://factchecktools.googleapis.com/v1alpha1/claims:search',
            'snopes': 'https://www.snopes.com/api',  # Placeholder
            'politifact': 'https://www.politifact.com/api'  # Placeholder
        }
        
        # Known high-credibility domains
        self.trusted_domains = {
            'reuters.com': 9.0,
            'bbc.com': 9.0,
            'apnews.com': 9.0,
            'npr.org': 8.5,
            'cnn.com': 7.5,
            'nytimes.com': 8.0,
            'washingtonpost.com': 8.0,
            'theguardian.com': 8.0,
            'wsj.com': 8.5,
            'economist.com': 8.5
        }
        
        # Known low-credibility indicators
        self.suspicious_indicators = [
            'breaking', 'urgent', 'shocking', 'exclusive',
            'doctors hate', 'one weird trick', 'you won\'t believe',
            'fake news', 'hoax', 'conspiracy'
        ]
        
        logger.info("Source verifier initialized")
    
    def verify_source_credibility(self, source: Source) -> CredibilityAssessment:
        """
        Perform comprehensive source credibility assessment
        
        Args:
            source: Source object to assess
            
        Returns:
            CredibilityAssessment with detailed scoring
        """
        try:
            logger.info(f"Assessing credibility for source: {source.name}")
            
            # Initialize assessment
            assessment = CredibilityAssessment(
                source_id=source.id,
                credibility_score=0.0,
                credibility_level=CredibilityLevel.UNKNOWN,
                confidence=0.0
            )
            
            # Perform various verification checks
            scores = []
            
            # 1. Domain reputation check
            if source.url:
                domain_score = self._assess_domain_reputation(source.url)
                assessment.domain_reputation = domain_score
                scores.append(('domain', domain_score, 0.25))  # 25% weight
            
            # 2. Technical verification
            if source.url:
                tech_scores = self._verify_technical_aspects(source.url)
                assessment.ssl_valid = tech_scores.get('ssl_valid')
                assessment.domain_age_score = tech_scores.get('domain_age_score')
                if tech_scores.get('domain_age_score'):
                    scores.append(('technical', tech_scores['domain_age_score'], 0.15))
            
            # 3. Social media verification (if applicable)
            if source.platform:
                social_score = self._assess_social_media_credibility(source)
                assessment.social_media_presence = social_score
                scores.append(('social', social_score, 0.2))  # 20% weight
            
            # 4. Content quality assessment
            content_score = self._assess_content_quality(source)
            assessment.content_quality = content_score
            scores.append(('content', content_score, 0.2))  # 20% weight
            
            # 5. Fact-check record
            fact_check_score = self._check_fact_check_record(source)
            assessment.fact_check_record = fact_check_score
            if fact_check_score is not None:
                scores.append(('factcheck', fact_check_score, 0.2))  # 20% weight
            
            # Calculate weighted overall score
            if scores:
                total_weight = sum(weight for _, _, weight in scores)
                weighted_score = sum(score * weight for _, score, weight in scores)
                assessment.credibility_score = weighted_score / total_weight if total_weight > 0 else 0.0
            else:
                assessment.credibility_score = 5.0  # Neutral score if no data
            
            # Determine credibility level
            assessment.credibility_level = self._score_to_level(assessment.credibility_score)
            
            # Calculate confidence based on available data
            assessment.confidence = min(1.0, len(scores) / 5.0)  # 5 possible checks
            
            # Check for warning flags
            assessment.warning_flags = self._identify_warning_flags(source)
            
            # Add assessment notes
            assessment.assessment_notes = self._generate_assessment_notes(assessment, scores)
            
            logger.info(
                f"Credibility assessment completed: "
                f"Score {assessment.credibility_score:.1f}, "
                f"Level {assessment.credibility_level}, "
                f"Confidence {assessment.confidence:.2f}"
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in credibility assessment: {e}")
            raise VerificationError(f"Credibility assessment failed: {e}")
    
    def _assess_domain_reputation(self, url: str) -> float:
        """
        Assess domain reputation based on known databases and indicators
        
        Args:
            url: URL to assess
            
        Returns:
            Domain reputation score (0-10)
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check against trusted domains list
            if domain in self.trusted_domains:
                return self.trusted_domains[domain]
            
            # Check domain characteristics
            score = 5.0  # Start with neutral score
            
            # Domain length (shorter is often better for legitimate sites)
            if len(domain) < 15:
                score += 0.5
            elif len(domain) > 30:
                score -= 1.0
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'\d{3,}',  # Many numbers
                r'-{2,}',   # Multiple hyphens
                r'[^a-z0-9.-]',  # Non-standard characters
                r'^[0-9]+\.',    # Starts with numbers
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, domain):
                    score -= 0.5
            
            # Check TLD reputation
            high_reputation_tlds = {'.com', '.org', '.edu', '.gov', '.mil', '.net'}
            low_reputation_tlds = {'.tk', '.ml', '.ga', '.cf', '.buzz'}
            
            tld = '.' + domain.split('.')[-1]
            if tld in high_reputation_tlds:
                score += 0.3
            elif tld in low_reputation_tlds:
                score -= 2.0
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            logger.warning(f"Error assessing domain reputation: {e}")
            return 5.0  # Neutral score on error
    
    def _verify_technical_aspects(self, url: str) -> Dict[str, Any]:
        """
        Verify technical aspects of the source URL
        
        Args:
            url: URL to verify
            
        Returns:
            Dict containing technical verification results
        """
        results = {}
        
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # 1. SSL Certificate verification
            try:
                if parsed_url.scheme == 'https':
                    context = ssl.create_default_context()
                    with socket.create_connection((domain, 443), timeout=10) as sock:
                        with context.wrap_socket(sock, server_hostname=domain) as ssock:
                            cert = ssock.getpeercert()
                            
                            # Check if certificate is valid
                            not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                            results['ssl_valid'] = not_after > datetime.utcnow()
                            results['ssl_expiry'] = not_after
                else:
                    results['ssl_valid'] = False  # HTTP only
                    
            except Exception as e:
                logger.warning(f"SSL verification failed: {e}")
                results['ssl_valid'] = False
            
            # 2. Domain age check using WHOIS
            try:
                domain_info = whois.whois(domain)
                if domain_info and hasattr(domain_info, 'creation_date'):
                    creation_date = domain_info.creation_date
                    if isinstance(creation_date, list):
                        creation_date = creation_date[0]
                    
                    if creation_date:
                        domain_age = (datetime.now() - creation_date).days
                        
                        # Score based on age (older domains are generally more trustworthy)
                        if domain_age > 365 * 5:  # 5+ years
                            age_score = 9.0
                        elif domain_age > 365 * 2:  # 2+ years
                            age_score = 7.0
                        elif domain_age > 365:  # 1+ year
                            age_score = 6.0
                        elif domain_age > 90:  # 3+ months
                            age_score = 4.0
                        else:  # Very new domain
                            age_score = 2.0
                        
                        results['domain_age_days'] = domain_age
                        results['domain_age_score'] = age_score
                        
            except Exception as e:
                logger.warning(f"WHOIS lookup failed: {e}")
            
            # 3. DNS verification
            try:
                import socket
                socket.gethostbyname(domain)
                results['dns_resolves'] = True
            except Exception:
                results['dns_resolves'] = False
                
        except Exception as e:
            logger.warning(f"Technical verification failed: {e}")
        
        return results
    
    def _assess_social_media_credibility(self, source: Source) -> float:
        """
        Assess social media account credibility
        
        Args:
            source: Source with social media information
            
        Returns:
            Social media credibility score (0-10)
        """
        score = 5.0  # Start with neutral
        
        try:
            # Verification badge (if available)
            if source.has_verified_badge:
                score += 2.0
            
            # Account age
            if source.account_age_days:
                if source.account_age_days > 365 * 3:  # 3+ years
                    score += 1.5
                elif source.account_age_days > 365:  # 1+ year
                    score += 1.0
                elif source.account_age_days < 30:  # Very new
                    score -= 2.0
            
            # Follower count (be careful of fake followers)
            if source.follower_count:
                if source.follower_count > 100000:
                    score += 1.0
                elif source.follower_count > 10000:
                    score += 0.5
                elif source.follower_count < 100:
                    score -= 0.5
            
            # Platform-specific checks
            if source.platform == PlatformType.TWITTER:
                # Twitter verification is generally reliable
                if source.has_verified_badge:
                    score += 1.0
                    
            elif source.platform == PlatformType.FACEBOOK:
                # Facebook page verification
                if source.has_verified_badge:
                    score += 1.0
                    
            elif source.platform == PlatformType.YOUTUBE:
                # YouTube channel verification
                if source.has_verified_badge:
                    score += 0.5  # Less weight than other platforms
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            logger.warning(f"Error assessing social media credibility: {e}")
            return 5.0
    
    def _assess_content_quality(self, source: Source) -> float:
        """
        Assess content quality based on available information
        
        Args:
            source: Source to assess
            
        Returns:
            Content quality score (0-10)
        """
        score = 5.0  # Start neutral
        
        try:
            # Check source name for quality indicators
            source_name_lower = source.name.lower()
            
            # Professional news organization patterns
            news_indicators = [
                'news', 'times', 'post', 'herald', 'gazette', 'journal',
                'tribune', 'chronicle', 'observer', 'guardian', 'telegraph'
            ]
            
            for indicator in news_indicators:
                if indicator in source_name_lower:
                    score += 0.5
                    break
            
            # Check for suspicious indicators in name
            for indicator in self.suspicious_indicators:
                if indicator in source_name_lower:
                    score -= 1.0
            
            # URL structure quality
            if source.url:
                parsed_url = urlparse(source.url)
                
                # Professional URL structure
                if parsed_url.path and len(parsed_url.path.split('/')) >= 3:
                    score += 0.3
                
                # Avoid suspicious URL patterns
                url_lower = source.url.lower()
                suspicious_url_patterns = [
                    'bit.ly', 'tinyurl', 'goo.gl',  # URL shorteners
                    'blogspot', 'wordpress.com',    # Free hosting
                    'click', 'viral', 'buzz'       # Clickbait indicators
                ]
                
                for pattern in suspicious_url_patterns:
                    if pattern in url_lower:
                        score -= 0.5
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            logger.warning(f"Error assessing content quality: {e}")
            return 5.0
    
    def _check_fact_check_record(self, source: Source) -> Optional[float]:
        """
        Check source against fact-checking databases
        
        Args:
            source: Source to check
            
        Returns:
            Fact-check score (0-10) or None if no data
        """
        try:
            if not source.url:
                return None
                
            parsed_url = urlparse(source.url)
            domain = parsed_url.netloc.lower()
            
            # This is a simplified implementation
            # In production, you would integrate with real fact-checking APIs
            
            # Known fact-checking organizations
            fact_checkers = {
                'snopes.com': 9.0,
                'factcheck.org': 9.0,
                'politifact.com': 8.5,
                'fullfact.org': 8.5,
                'checkyourfact.com': 8.0
            }
            
            if domain in fact_checkers:
                return fact_checkers[domain]
            
            # Known problematic sources
            problematic_sources = {
                'infowars.com': 1.0,
                'naturalnews.com': 2.0,
                'breitbart.com': 3.0
            }
            
            if domain in problematic_sources:
                return problematic_sources[domain]
            
            # For other sources, we can't determine without API access
            return None
            
        except Exception as e:
            logger.warning(f"Error checking fact-check record: {e}")
            return None
    
    def _score_to_level(self, score: float) -> CredibilityLevel:
        """Convert numerical score to credibility level"""
        if score >= 8.5:
            return CredibilityLevel.VERY_HIGH
        elif score >= 7.0:
            return CredibilityLevel.HIGH
        elif score >= 5.0:
            return CredibilityLevel.MEDIUM
        elif score >= 3.0:
            return CredibilityLevel.LOW
        else:
            return CredibilityLevel.VERY_LOW
    
    def _identify_warning_flags(self, source: Source) -> List[str]:
        """Identify potential warning flags for the source"""
        flags = []
        
        try:
            # Check source name for warning indicators
            source_name_lower = source.name.lower()
            
            # Clickbait/sensational language
            for indicator in self.suspicious_indicators:
                if indicator in source_name_lower:
                    flags.append(f"Suspicious language: '{indicator}' in source name")
            
            # URL warnings
            if source.url:
                url_lower = source.url.lower()
                
                # No HTTPS
                if not url_lower.startswith('https://'):
                    flags.append("No HTTPS encryption")
                
                # URL shorteners
                shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']
                for shortener in shorteners:
                    if shortener in url_lower:
                        flags.append(f"URL shortener used: {shortener}")
                
                # Suspicious TLDs
                parsed_url = urlparse(source.url)
                domain = parsed_url.netloc.lower()
                suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
                
                for tld in suspicious_tlds:
                    if domain.endswith(tld):
                        flags.append(f"Suspicious TLD: {tld}")
            
            # Social media warnings
            if source.platform and source.account_age_days:
                if source.account_age_days < 30:
                    flags.append("Very new social media account")
            
            if source.has_verified_badge is False and source.platform:
                flags.append(f"Unverified {source.platform} account")
                
        except Exception as e:
            logger.warning(f"Error identifying warning flags: {e}")
        
        return flags
    
    def _generate_assessment_notes(
        self, 
        assessment: CredibilityAssessment,
        scores: List[Tuple[str, float, float]]
    ) -> str:
        """Generate human-readable assessment notes"""
        notes = []
        
        # Overall assessment
        notes.append(f"Overall credibility: {assessment.credibility_level.value}")
        notes.append(f"Confidence in assessment: {assessment.confidence:.0%}")
        
        # Factor breakdown
        if scores:
            notes.append("\nFactor breakdown:")
            for factor, score, weight in scores:
                notes.append(f"- {factor.title()}: {score:.1f}/10 (weight: {weight:.0%})")
        
        # Key findings
        if assessment.ssl_valid is True:
            notes.append("✓ SSL certificate valid")
        elif assessment.ssl_valid is False:
            notes.append("⚠ SSL certificate invalid or missing")
        
        if assessment.verification_status:
            notes.append("✓ Platform verification badge present")
        
        if assessment.domain_age_score and assessment.domain_age_score > 7.0:
            notes.append("✓ Established domain (good age)")
        elif assessment.domain_age_score and assessment.domain_age_score < 4.0:
            notes.append("⚠ Recently registered domain")
        
        # Warnings
        if assessment.warning_flags:
            notes.append("\nWarning flags:")
            for flag in assessment.warning_flags:
                notes.append(f"⚠ {flag}")
        
        return "\n".join(notes)
    
    def batch_verify_sources(self, sources: List[Source]) -> List[CredibilityAssessment]:
        """
        Perform batch credibility assessment on multiple sources
        
        Args:
            sources: List of sources to assess
            
        Returns:
            List of CredibilityAssessment objects
        """
        assessments = []
        
        for source in sources:
            try:
                assessment = self.verify_source_credibility(source)
                assessments.append(assessment)
                logger.info(f"Assessed source: {source.name}")
            except Exception as e:
                logger.error(f"Failed to assess source {source.name}: {e}")
                
                # Create minimal assessment for failed verification
                assessment = CredibilityAssessment(
                    source_id=source.id,
                    credibility_score=0.0,
                    credibility_level=CredibilityLevel.UNKNOWN,
                    confidence=0.0,
                    assessment_notes=f"Assessment failed: {e}"
                )
                assessments.append(assessment)
        
        return assessments
    
    def create_credibility_report(
        self,
        assessments: List[CredibilityAssessment],
        output_path: str
    ) -> bool:
        """
        Create comprehensive credibility report
        
        Args:
            assessments: List of assessments to include
            output_path: Path to save report
            
        Returns:
            bool: True if report created successfully
        """
        try:
            # Calculate summary statistics
            scores = [a.credibility_score for a in assessments]
            levels = [a.credibility_level for a in assessments]
            
            report = {
                'report_generated': datetime.utcnow().isoformat(),
                'tool': 'lemkin-osint-source-verifier',
                'version': '1.0',
                'summary': {
                    'total_sources': len(assessments),
                    'average_score': sum(scores) / len(scores) if scores else 0,
                    'score_distribution': {
                        'very_high': sum(1 for l in levels if l == CredibilityLevel.VERY_HIGH),
                        'high': sum(1 for l in levels if l == CredibilityLevel.HIGH),
                        'medium': sum(1 for l in levels if l == CredibilityLevel.MEDIUM),
                        'low': sum(1 for l in levels if l == CredibilityLevel.LOW),
                        'very_low': sum(1 for l in levels if l == CredibilityLevel.VERY_LOW),
                        'unknown': sum(1 for l in levels if l == CredibilityLevel.UNKNOWN)
                    }
                },
                'assessments': [assessment.dict() for assessment in assessments]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Credibility report created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating credibility report: {e}")
            return False
    
    def update_source_credibility(
        self, 
        source: Source, 
        assessment: CredibilityAssessment
    ) -> Source:
        """
        Update source object with credibility assessment results
        
        Args:
            source: Source to update
            assessment: Assessment results
            
        Returns:
            Updated Source object
        """
        source.credibility_score = assessment.credibility_score
        source.credibility_level = assessment.credibility_level
        source.last_verified = assessment.assessed_at
        
        # Update technical fields if available
        if assessment.ssl_valid is not None:
            source.ssl_certificate_valid = assessment.ssl_valid
        
        if assessment.domain_age_score:
            # Estimate domain age from score (reverse calculation)
            if assessment.domain_age_score >= 9.0:
                source.domain_age_days = 365 * 5  # 5+ years estimate
            elif assessment.domain_age_score >= 7.0:
                source.domain_age_days = 365 * 2  # 2+ years estimate
            elif assessment.domain_age_score >= 6.0:
                source.domain_age_days = 365  # 1+ year estimate
        
        return source
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("Source verifier closed")