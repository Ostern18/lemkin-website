"""
Legal citation parsing and validation module.

This module provides comprehensive legal citation parsing, validation, and
formatting for major citation styles including Bluebook, ALWD, and others.
Supports case citations, statute citations, regulation citations, and more.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from .core import (
    LegalCitation, CitationMatch, CitationType, CitationStyle, CitationFormat,
    JurisdictionType, ResearchConfig
)


# Citation pattern definitions
@dataclass
class CitationPattern:
    """Citation pattern definition with regex and metadata"""
    pattern: str
    citation_type: CitationType
    format_name: str
    extraction_groups: Dict[str, int]  # Maps field names to regex group numbers
    validation_rules: List[str] = None


class CitationPatterns:
    """Legal citation regex patterns for different citation types"""
    
    # Case citation patterns
    CASE_PATTERNS = [
        CitationPattern(
            pattern=r'(\d+)\s+([A-Za-z\.]+\.?)\s+(\d+)(?:\s*,\s*(\d+))?\s*\(([^)]+)\s+(\d{4})\)',
            citation_type=CitationType.CASE,
            format_name="standard_case",
            extraction_groups={
                'volume': 1, 'reporter': 2, 'page': 3, 'pin_cite': 4,
                'court': 5, 'year': 6
            }
        ),
        CitationPattern(
            pattern=r'(\d+)\s+(U\.?S\.?)\s+(\d+)(?:\s*,\s*(\d+))?\s*\((\d{4})\)',
            citation_type=CitationType.CASE,
            format_name="supreme_court",
            extraction_groups={
                'volume': 1, 'reporter': 2, 'page': 3, 'pin_cite': 4, 'year': 5
            }
        ),
        CitationPattern(
            pattern=r'(\d+)\s+(F\.?\s?\d*d?)\s+(\d+)(?:\s*,\s*(\d+))?\s*\(([^)]+)\s+(\d{4})\)',
            citation_type=CitationType.CASE,
            format_name="federal_reporter",
            extraction_groups={
                'volume': 1, 'reporter': 2, 'page': 3, 'pin_cite': 4,
                'court': 5, 'year': 6
            }
        ),
        CitationPattern(
            pattern=r'(\d+)\s+(F\.\s?Supp\.?\s?\d*)\s+(\d+)(?:\s*,\s*(\d+))?\s*\(([^)]+)\s+(\d{4})\)',
            citation_type=CitationType.CASE,
            format_name="federal_supplement",
            extraction_groups={
                'volume': 1, 'reporter': 2, 'page': 3, 'pin_cite': 4,
                'court': 5, 'year': 6
            }
        ),
        CitationPattern(
            pattern=r'([^,]+),\s*(\d+)\s+([A-Za-z\.]+\.?)\s+(\d+)(?:\s*,\s*(\d+))?\s*\(([^)]+)\s+(\d{4})\)',
            citation_type=CitationType.CASE,
            format_name="case_name_first",
            extraction_groups={
                'case_name': 1, 'volume': 2, 'reporter': 3, 'page': 4,
                'pin_cite': 5, 'court': 6, 'year': 7
            }
        )
    ]
    
    # Statute citation patterns
    STATUTE_PATTERNS = [
        CitationPattern(
            pattern=r'(\d+)\s+(U\.?S\.?C\.?)\s+§\s*(\d+(?:[a-z])?(?:\(\w+\))?)',
            citation_type=CitationType.STATUTE,
            format_name="usc",
            extraction_groups={'title': 1, 'code': 2, 'section': 3}
        ),
        CitationPattern(
            pattern=r'(\w+)\s+(Code)\s+§\s*(\d+(?:\.\d+)*(?:[a-z])?)',
            citation_type=CitationType.STATUTE,
            format_name="state_code",
            extraction_groups={'state': 1, 'code': 2, 'section': 3}
        ),
        CitationPattern(
            pattern=r'(\w+)\s+(Rev\.?\s?Stat\.?)\s+§\s*(\d+(?:\.\d+)*)',
            citation_type=CitationType.STATUTE,
            format_name="revised_statutes",
            extraction_groups={'state': 1, 'code': 2, 'section': 3}
        ),
        CitationPattern(
            pattern=r'Pub\.?\s?L\.?\s?No\.?\s?(\d+-\d+)(?:\s*,\s*(\d+)\s+Stat\.?\s+(\d+))?(?:\s*\((\d{4})\))?',
            citation_type=CitationType.STATUTE,
            format_name="public_law",
            extraction_groups={'pub_law': 1, 'stat_vol': 2, 'stat_page': 3, 'year': 4}
        )
    ]
    
    # Regulation patterns
    REGULATION_PATTERNS = [
        CitationPattern(
            pattern=r'(\d+)\s+C\.?F\.?R\.?\s+§\s*(\d+(?:\.\d+)*)',
            citation_type=CitationType.REGULATION,
            format_name="cfr",
            extraction_groups={'title': 1, 'section': 2}
        ),
        CitationPattern(
            pattern=r'(\d+)\s+Fed\.?\s?Reg\.?\s+(\d+)(?:\s*,\s*(\d+))?\s*\(([^)]+)\s+(\d{1,2}),\s+(\d{4})\)',
            citation_type=CitationType.REGULATION,
            format_name="federal_register",
            extraction_groups={
                'volume': 1, 'page': 2, 'pin_cite': 3, 
                'month': 4, 'day': 5, 'year': 6
            }
        )
    ]
    
    # Constitutional citations
    CONSTITUTIONAL_PATTERNS = [
        CitationPattern(
            pattern=r'U\.?S\.?\s+Const\.?\s+(?:art|amend)\.?\s+([IVX]+|[0-9]+)(?:\s*,\s*§\s*(\d+))?',
            citation_type=CitationType.CONSTITUTIONAL,
            format_name="us_constitution",
            extraction_groups={'article_amendment': 1, 'section': 2}
        ),
        CitationPattern(
            pattern=r'(\w+)\s+Const\.?\s+(?:art|amend)\.?\s+([IVX]+|[0-9]+)(?:\s*,\s*§\s*(\d+))?',
            citation_type=CitationType.CONSTITUTIONAL,
            format_name="state_constitution",
            extraction_groups={'state': 1, 'article_amendment': 2, 'section': 3}
        )
    ]
    
    # Secondary source patterns
    SECONDARY_PATTERNS = [
        CitationPattern(
            pattern=r'([^,]+),\s*(\d+)\s+([A-Za-z\.\s]+)\s+(\d+)(?:\s*,\s*(\d+))?\s*\((\d{4})\)',
            citation_type=CitationType.SECONDARY,
            format_name="law_review",
            extraction_groups={
                'author': 1, 'volume': 2, 'journal': 3, 'page': 4,
                'pin_cite': 5, 'year': 6
            }
        ),
        CitationPattern(
            pattern=r'([^,]+),\s*([^,]+)\s+§\s*(\d+(?:\.\d+)*)\s*\((\d{4})\)',
            citation_type=CitationType.SECONDARY,
            format_name="treatise",
            extraction_groups={'author': 1, 'title': 2, 'section': 3, 'year': 4}
        )
    ]
    
    @classmethod
    def get_all_patterns(cls) -> List[CitationPattern]:
        """Get all citation patterns"""
        return (cls.CASE_PATTERNS + cls.STATUTE_PATTERNS + 
                cls.REGULATION_PATTERNS + cls.CONSTITUTIONAL_PATTERNS +
                cls.SECONDARY_PATTERNS)


class CitationValidator:
    """Validates parsed legal citations"""
    
    def __init__(self):
        # Known reporter abbreviations
        self.case_reporters = {
            'U.S.', 'US', 'S. Ct.', 'S.Ct.', 'L. Ed.', 'L.Ed.',
            'F.', 'F.2d', 'F.3d', 'F.4th', 'F. Supp.', 'F.Supp.',
            'F. Supp. 2d', 'F.Supp.2d', 'F. Supp. 3d', 'F.Supp.3d',
            'A.', 'A.2d', 'A.3d', 'P.', 'P.2d', 'P.3d',
            'N.E.', 'N.E.2d', 'N.E.3d', 'S.E.', 'S.E.2d', 'S.E.3d',
            'S.W.', 'S.W.2d', 'S.W.3d', 'N.W.', 'N.W.2d', 'N.W.3d',
            'So.', 'So.2d', 'So.3d', 'Cal. Rptr.', 'Cal.Rptr.',
            'N.Y.S.', 'N.Y.S.2d', 'N.Y.S.3d'
        }
        
        # Federal courts
        self.federal_courts = {
            'U.S.', 'S. Ct.', '1st Cir.', '2nd Cir.', '3rd Cir.',
            '4th Cir.', '5th Cir.', '6th Cir.', '7th Cir.', '8th Cir.',
            '9th Cir.', '10th Cir.', '11th Cir.', 'D.C. Cir.', 'Fed. Cir.',
            'D. Del.', 'S.D.N.Y.', 'N.D. Cal.', 'C.D. Cal.', 'E.D. Pa.'
        }
        
        # Valid years range
        self.min_year = 1600
        self.max_year = datetime.now().year + 1
    
    def validate_case_citation(self, citation: LegalCitation) -> Tuple[bool, List[str]]:
        """Validate a case citation"""
        errors = []
        
        # Check required fields
        if not citation.volume:
            errors.append("Volume number is required")
        else:
            try:
                vol = int(citation.volume)
                if vol <= 0 or vol > 9999:
                    errors.append("Volume number is out of valid range")
            except ValueError:
                errors.append("Volume must be a number")
        
        if not citation.reporter:
            errors.append("Reporter abbreviation is required")
        elif citation.reporter not in self.case_reporters:
            errors.append(f"Unknown reporter abbreviation: {citation.reporter}")
        
        if not citation.page:
            errors.append("Page number is required")
        else:
            try:
                page = int(citation.page)
                if page <= 0:
                    errors.append("Page number must be positive")
            except ValueError:
                errors.append("Page must be a number")
        
        # Validate year
        if citation.year:
            if citation.year < self.min_year or citation.year > self.max_year:
                errors.append(f"Year {citation.year} is out of valid range")
        else:
            errors.append("Year is required for case citations")
        
        # Validate pin cite if present
        if citation.pin_cite:
            try:
                pin = int(citation.pin_cite)
                page = int(citation.page) if citation.page else 0
                if pin <= page:
                    errors.append("Pin cite should be greater than starting page")
            except ValueError:
                pass  # Pin cites can be non-numeric (e.g., "n.15")
        
        return len(errors) == 0, errors
    
    def validate_statute_citation(self, citation: LegalCitation) -> Tuple[bool, List[str]]:
        """Validate a statute citation"""
        errors = []
        
        # Basic validation for statute citations
        if not citation.volume and not citation.raw_citation:
            errors.append("Statute title or code is required")
        
        # Validate section format
        if hasattr(citation, 'section') and citation.metadata.get('section'):
            section = citation.metadata['section']
            if not re.match(r'\d+(?:[a-z])?(?:\(\w+\))?(?:\.\d+)*', section):
                errors.append(f"Invalid section format: {section}")
        
        return len(errors) == 0, errors
    
    def validate_regulation_citation(self, citation: LegalCitation) -> Tuple[bool, List[str]]:
        """Validate a regulation citation"""
        errors = []
        
        # CFR validation
        if citation.reporter and 'C.F.R.' in citation.reporter:
            if not citation.volume:
                errors.append("CFR title is required")
            else:
                try:
                    title = int(citation.volume)
                    if title <= 0 or title > 50:
                        errors.append("CFR title must be between 1 and 50")
                except ValueError:
                    errors.append("CFR title must be a number")
        
        return len(errors) == 0, errors


class CitationFormatter:
    """Formats citations according to different style guides"""
    
    def __init__(self, style: CitationStyle = CitationStyle.BLUEBOOK):
        self.style = style
        self.format_rules = self._load_format_rules()
    
    def _load_format_rules(self) -> Dict[str, Any]:
        """Load formatting rules for the selected style"""
        rules = {
            CitationStyle.BLUEBOOK: {
                'case_format': '{case_name}, {volume} {reporter} {page}{pin_cite} ({court} {year})',
                'statute_format': '{title} U.S.C. § {section} ({year})',
                'regulation_format': '{title} C.F.R. § {section} ({year})',
                'italics': ['case_name'],
                'abbreviations': {
                    'United States': 'U.S.',
                    'versus': 'v.',
                    'Section': '§',
                    'Corporation': 'Corp.'
                },
                'punctuation': {
                    'case_name_separator': ', ',
                    'pin_cite_prefix': ', ',
                    'court_parentheses': True
                }
            },
            CitationStyle.ALWD: {
                'case_format': '{case_name}, {volume} {reporter} {page}{pin_cite} ({court} {year})',
                'statute_format': '{title} U.S.C. § {section} ({year})',
                'regulation_format': '{title} C.F.R. § {section} ({year})',
                'italics': ['case_name'],
                'abbreviations': {
                    'United States': 'U.S.',
                    'versus': 'v.',
                    'Section': '§'
                }
            }
        }
        return rules.get(self.style, rules[CitationStyle.BLUEBOOK])
    
    def format_citation(self, citation: LegalCitation) -> str:
        """Format a citation according to the selected style"""
        if citation.citation_type == CitationType.CASE:
            return self._format_case_citation(citation)
        elif citation.citation_type == CitationType.STATUTE:
            return self._format_statute_citation(citation)
        elif citation.citation_type == CitationType.REGULATION:
            return self._format_regulation_citation(citation)
        else:
            return citation.raw_citation  # Fallback to original
    
    def _format_case_citation(self, citation: LegalCitation) -> str:
        """Format a case citation"""
        template = self.format_rules['case_format']
        
        # Prepare values
        values = {
            'case_name': citation.case_name or '',
            'volume': citation.volume or '',
            'reporter': citation.reporter or '',
            'page': citation.page or '',
            'pin_cite': f", {citation.pin_cite}" if citation.pin_cite else '',
            'court': citation.court or '',
            'year': str(citation.year) if citation.year else ''
        }
        
        # Apply abbreviations
        for field, value in values.items():
            if isinstance(value, str):
                for full, abbrev in self.format_rules['abbreviations'].items():
                    value = value.replace(full, abbrev)
                values[field] = value
        
        # Format citation
        formatted = template.format(**values)
        
        # Clean up extra spaces
        formatted = re.sub(r'\s+', ' ', formatted).strip()
        
        return formatted
    
    def _format_statute_citation(self, citation: LegalCitation) -> str:
        """Format a statute citation"""
        template = self.format_rules['statute_format']
        
        values = {
            'title': citation.volume or '',
            'section': citation.page or citation.metadata.get('section', ''),
            'year': str(citation.year) if citation.year else ''
        }
        
        return template.format(**values)
    
    def _format_regulation_citation(self, citation: LegalCitation) -> str:
        """Format a regulation citation"""
        template = self.format_rules['regulation_format']
        
        values = {
            'title': citation.volume or '',
            'section': citation.page or citation.metadata.get('section', ''),
            'year': str(citation.year) if citation.year else ''
        }
        
        return template.format(**values)


class CitationNormalizer:
    """Normalizes citation text for better parsing"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize citation text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize common abbreviations
        abbreviations = {
            r'\bUS\b': 'U.S.',
            r'\bUSC\b': 'U.S.C.',
            r'\bCFR\b': 'C.F.R.',
            r'\bFed\s+Reg\b': 'Fed. Reg.',
            r'\bSupp\b': 'Supp.',
            r'\bCir\b': 'Cir.',
            r'\bCt\b': 'Ct.',
            r'\bSect(?:ion)?\b': '§',
        }
        
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Standardize section symbols
        text = re.sub(r'(?:§|sect?\.?|section)\s*', '§ ', text, flags=re.IGNORECASE)
        
        # Normalize parentheses spacing
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        # Fix common OCR errors
        text = text.replace('0', 'O')  # Sometimes O is read as 0 in court names
        text = re.sub(r'(\d)\s*[Il]\s*(\d)', r'\1l\2', text)  # Fix "1 l 1" to "111"
        
        return text.strip()


class CitationProcessor:
    """
    Main citation processing engine.
    
    Provides comprehensive citation parsing, validation, and formatting
    for all major legal citation types and styles.
    """
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        
        # Initialize components
        self.patterns = CitationPatterns()
        self.validator = CitationValidator()
        self.formatter = CitationFormatter(config.default_citation_style)
        self.normalizer = CitationNormalizer()
        
        # Compiled regex patterns for performance
        self._compiled_patterns = {}
        self._compile_patterns()
        
        logger.info(f"Citation processor initialized with {self.config.default_citation_style.value} style")
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        for pattern in self.patterns.get_all_patterns():
            try:
                self._compiled_patterns[pattern.format_name] = re.compile(
                    pattern.pattern, re.IGNORECASE
                )
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern.format_name}: {e}")
    
    def parse(
        self,
        text: str,
        citation_style: Optional[CitationStyle] = None
    ) -> CitationMatch:
        """
        Parse and validate legal citations from text
        
        Args:
            text: Text containing legal citations
            citation_style: Target citation style for formatting
            
        Returns:
            CitationMatch with parsed citations and validation
        """
        logger.info(f"Parsing citations from text: {text[:100]}...")
        
        start_time = datetime.now()
        
        # Normalize the input text
        normalized_text = self.normalizer.normalize_text(text)
        
        # Extract citations
        citations_found = []
        extraction_methods = []
        
        # Try each pattern type
        for pattern in self.patterns.get_all_patterns():
            if pattern.format_name not in self._compiled_patterns:
                continue
            
            regex = self._compiled_patterns[pattern.format_name]
            matches = regex.finditer(normalized_text)
            
            for match in matches:
                citation = self._create_citation_from_match(match, pattern)
                if citation:
                    citations_found.append(citation)
                    extraction_methods.append(pattern.format_name)
        
        # Remove duplicates based on raw citation
        unique_citations = self._deduplicate_citations(citations_found)
        
        # Validate citations
        for citation in unique_citations:
            self._validate_citation(citation)
        
        # Format citations if requested
        if citation_style and citation_style != self.config.default_citation_style:
            formatter = CitationFormatter(citation_style)
            standardized = []
            for citation in unique_citations:
                formatted = formatter.format_citation(citation)
                standardized.append(formatted)
        else:
            standardized = [self.formatter.format_citation(c) for c in unique_citations]
        
        # Calculate confidence score
        confidence = self._calculate_parsing_confidence(
            unique_citations, normalized_text
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = CitationMatch(
            original_text=text,
            citations_found=unique_citations,
            parsing_confidence=confidence,
            extraction_method="regex_patterns",
            validation_passed=all(c.is_valid for c in unique_citations),
            standardized_citations=standardized,
            formatting_suggestions=self._generate_formatting_suggestions(unique_citations),
            metadata={
                'processing_time': processing_time,
                'extraction_methods': list(set(extraction_methods)),
                'normalized_text': normalized_text
            }
        )
        
        logger.info(f"Parsed {len(unique_citations)} citations with {confidence:.2f} confidence")
        return result
    
    def _create_citation_from_match(
        self, 
        match: re.Match, 
        pattern: CitationPattern
    ) -> Optional[LegalCitation]:
        """Create a LegalCitation from a regex match"""
        try:
            # Extract matched text
            raw_citation = match.group(0)
            
            # Initialize citation
            citation = LegalCitation(
                raw_citation=raw_citation,
                citation_type=pattern.citation_type,
                citation_style=self.config.default_citation_style,
                metadata={'pattern_used': pattern.format_name}
            )
            
            # Extract fields based on pattern groups
            groups = match.groups()
            for field_name, group_idx in pattern.extraction_groups.items():
                if group_idx <= len(groups) and groups[group_idx - 1]:
                    value = groups[group_idx - 1].strip()
                    
                    # Map to citation fields
                    if field_name == 'volume':
                        citation.volume = value
                    elif field_name == 'reporter':
                        citation.reporter = value
                    elif field_name == 'page':
                        citation.page = value
                    elif field_name == 'pin_cite':
                        citation.pin_cite = value
                    elif field_name == 'court':
                        citation.court = value
                        citation.jurisdiction = self._infer_jurisdiction(value)
                    elif field_name == 'year':
                        try:
                            citation.year = int(value)
                        except ValueError:
                            pass
                    elif field_name == 'case_name':
                        citation.case_name = value
                    else:
                        # Store in metadata
                        citation.metadata[field_name] = value
            
            return citation
            
        except Exception as e:
            logger.warning(f"Error creating citation from match: {e}")
            return None
    
    def _infer_jurisdiction(self, court: str) -> Optional[JurisdictionType]:
        """Infer jurisdiction from court name"""
        court_lower = court.lower()
        
        if any(term in court_lower for term in ['u.s.', 'supreme', 'scotus']):
            return JurisdictionType.FEDERAL
        elif any(term in court_lower for term in ['circuit', 'cir.', 'federal']):
            return JurisdictionType.FEDERAL
        elif any(term in court_lower for term in ['d.', 'district']):
            return JurisdictionType.FEDERAL
        else:
            return JurisdictionType.STATE
    
    def _deduplicate_citations(self, citations: List[LegalCitation]) -> List[LegalCitation]:
        """Remove duplicate citations"""
        seen = set()
        unique = []
        
        for citation in citations:
            # Create a key based on normalized citation text
            key = self._normalize_citation_key(citation.raw_citation)
            
            if key not in seen:
                seen.add(key)
                unique.append(citation)
        
        return unique
    
    def _normalize_citation_key(self, citation_text: str) -> str:
        """Create a normalized key for deduplication"""
        # Remove punctuation and extra spaces, convert to lowercase
        key = re.sub(r'[^\w\s]', '', citation_text.lower())
        key = re.sub(r'\s+', ' ', key).strip()
        return key
    
    def _validate_citation(self, citation: LegalCitation):
        """Validate a citation and update its validation status"""
        if citation.citation_type == CitationType.CASE:
            is_valid, errors = self.validator.validate_case_citation(citation)
        elif citation.citation_type == CitationType.STATUTE:
            is_valid, errors = self.validator.validate_statute_citation(citation)
        elif citation.citation_type == CitationType.REGULATION:
            is_valid, errors = self.validator.validate_regulation_citation(citation)
        else:
            is_valid, errors = True, []  # Default to valid for unknown types
        
        citation.is_valid = is_valid
        citation.validation_errors = errors
        
        if is_valid:
            # Generate standardized citation
            citation.parsed_citation = self.formatter.format_citation(citation)
    
    def _calculate_parsing_confidence(
        self, 
        citations: List[LegalCitation], 
        text: str
    ) -> float:
        """Calculate confidence score for citation parsing"""
        if not citations:
            return 0.0
        
        # Base confidence on number of valid citations
        valid_count = sum(1 for c in citations if c.is_valid)
        base_confidence = valid_count / len(citations)
        
        # Boost confidence if citations cover significant portion of text
        total_citation_length = sum(len(c.raw_citation) for c in citations)
        coverage = min(1.0, total_citation_length / len(text))
        
        # Penalty for validation errors
        error_count = sum(len(c.validation_errors) for c in citations)
        error_penalty = min(0.5, error_count * 0.1)
        
        confidence = (base_confidence * 0.7 + coverage * 0.3) - error_penalty
        return max(0.0, min(1.0, confidence))
    
    def _generate_formatting_suggestions(
        self, 
        citations: List[LegalCitation]
    ) -> List[str]:
        """Generate formatting suggestions for citations"""
        suggestions = []
        
        for citation in citations:
            if not citation.is_valid:
                suggestions.extend([
                    f"Citation '{citation.raw_citation}': {error}"
                    for error in citation.validation_errors
                ])
            
            # Style-specific suggestions
            if citation.citation_type == CitationType.CASE:
                if citation.case_name and not citation.case_name.endswith(' v. '):
                    suggestions.append(
                        f"Consider italicizing case name in '{citation.raw_citation}'"
                    )
                
                if citation.pin_cite and not citation.pin_cite.startswith(','):
                    suggestions.append(
                        f"Pin cite should be preceded by comma in '{citation.raw_citation}'"
                    )
        
        return list(set(suggestions))  # Remove duplicates
    
    def extract_citations_from_document(
        self, 
        document_path: str,
        citation_style: Optional[CitationStyle] = None
    ) -> List[CitationMatch]:
        """
        Extract citations from a legal document
        
        Args:
            document_path: Path to document file
            citation_style: Target citation style
            
        Returns:
            List of CitationMatch objects, one per document section/paragraph
        """
        # This would integrate with document parsing libraries
        # For now, return placeholder
        logger.info(f"Extracting citations from document: {document_path}")
        
        # Placeholder implementation
        return []
    
    def validate_citation_format(
        self, 
        citation_text: str,
        expected_style: CitationStyle
    ) -> Dict[str, Any]:
        """
        Validate if a citation follows a specific format style
        
        Args:
            citation_text: Citation text to validate
            expected_style: Expected citation style
            
        Returns:
            Dictionary with validation results
        """
        result = self.parse(citation_text, expected_style)
        
        if not result.citations_found:
            return {
                'valid': False,
                'style_compliant': False,
                'errors': ['No valid citations found'],
                'suggestions': ['Check citation format']
            }
        
        citation = result.citations_found[0]
        
        # Check style compliance
        expected_formatter = CitationFormatter(expected_style)
        expected_format = expected_formatter.format_citation(citation)
        
        style_compliant = citation_text.strip() == expected_format.strip()
        
        return {
            'valid': citation.is_valid,
            'style_compliant': style_compliant,
            'errors': citation.validation_errors,
            'suggestions': result.formatting_suggestions,
            'expected_format': expected_format,
            'confidence': result.parsing_confidence
        }
    
    def convert_citation_style(
        self, 
        citation_text: str,
        from_style: CitationStyle,
        to_style: CitationStyle
    ) -> str:
        """
        Convert citation from one style to another
        
        Args:
            citation_text: Original citation
            from_style: Source citation style  
            to_style: Target citation style
            
        Returns:
            Converted citation text
        """
        # Parse with original style
        original_formatter = CitationFormatter(from_style)
        self.formatter = original_formatter
        
        result = self.parse(citation_text)
        
        if not result.citations_found:
            logger.warning(f"Could not parse citation: {citation_text}")
            return citation_text
        
        # Format with new style
        new_formatter = CitationFormatter(to_style)
        converted = new_formatter.format_citation(result.citations_found[0])
        
        logger.info(f"Converted citation from {from_style.value} to {to_style.value}")
        return converted
    
    def get_citation_statistics(self, text: str) -> Dict[str, Any]:
        """Get statistics about citations in text"""
        result = self.parse(text)
        
        stats = {
            'total_citations': len(result.citations_found),
            'valid_citations': sum(1 for c in result.citations_found if c.is_valid),
            'citation_types': {},
            'years_range': {},
            'jurisdictions': {},
            'most_common_reporters': {}
        }
        
        # Analyze citation types
        for citation in result.citations_found:
            cite_type = citation.citation_type.value
            stats['citation_types'][cite_type] = stats['citation_types'].get(cite_type, 0) + 1
            
            # Year analysis
            if citation.year:
                stats['years_range'][str(citation.year)] = stats['years_range'].get(str(citation.year), 0) + 1
            
            # Jurisdiction analysis  
            if citation.jurisdiction:
                juris = citation.jurisdiction.value
                stats['jurisdictions'][juris] = stats['jurisdictions'].get(juris, 0) + 1
            
            # Reporter analysis
            if citation.reporter:
                reporter = citation.reporter
                stats['most_common_reporters'][reporter] = stats['most_common_reporters'].get(reporter, 0) + 1
        
        return stats


# Convenience function for direct module usage
def parse_legal_citations(
    text: str,
    citation_style: Optional[CitationStyle] = None,
    config: Optional[ResearchConfig] = None
) -> CitationMatch:
    """
    Convenience function to parse legal citations
    
    Args:
        text: Text containing legal citations
        citation_style: Target citation style for formatting
        config: Research configuration (uses default if not provided)
        
    Returns:
        CitationMatch with parsed citations and validation
    """
    if config is None:
        from .core import ResearchConfig
        config = ResearchConfig()
    
    processor = CitationProcessor(config)
    return processor.parse(text, citation_style)


# Export main classes and functions
__all__ = [
    'CitationProcessor',
    'CitationValidator',
    'CitationFormatter',
    'CitationNormalizer',
    'CitationPatterns',
    'CitationPattern',
    'parse_legal_citations'
]