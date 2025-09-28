"""
Temporal extraction module for detecting and parsing dates, times, and durations from text.
"""

import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import arrow
import parsedatetime as pdt
import spacy
from spacy.matcher import Matcher
import logging
from loguru import logger

from .core import TemporalEntity, TemporalEntityType, LanguageCode, TimelineConfig


class DateExtractor:
    """Extracts and parses date references from text"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.cal = pdt.Calendar()
        
        # Compile regex patterns for different date formats
        self.date_patterns = self._compile_date_patterns()
        self.fuzzy_date_patterns = self._compile_fuzzy_date_patterns()
        
    def _compile_date_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for date extraction"""
        patterns = {}
        
        # Standard date patterns
        patterns['iso_date'] = re.compile(
            r'\b\d{4}-\d{1,2}-\d{1,2}\b', re.IGNORECASE
        )
        patterns['us_date'] = re.compile(
            r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b', re.IGNORECASE
        )
        patterns['eu_date'] = re.compile(
            r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b', re.IGNORECASE
        )
        patterns['written_date'] = re.compile(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b',
            re.IGNORECASE
        )
        patterns['partial_date'] = re.compile(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
            re.IGNORECASE
        )
        patterns['year_only'] = re.compile(
            r'\b(?:19|20)\d{2}\b'
        )
        
        # Multilingual date patterns
        if LanguageCode.ES in self.config.supported_languages:
            patterns['spanish_date'] = re.compile(
                r'\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d{1,2},?\s+\d{2,4}\b',
                re.IGNORECASE
            )
        
        if LanguageCode.FR in self.config.supported_languages:
            patterns['french_date'] = re.compile(
                r'\b(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{1,2},?\s+\d{2,4}\b',
                re.IGNORECASE
            )
        
        if LanguageCode.DE in self.config.supported_languages:
            patterns['german_date'] = re.compile(
                r'\b(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{1,2},?\s+\d{2,4}\b',
                re.IGNORECASE
            )
        
        return patterns
    
    def _compile_fuzzy_date_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for fuzzy/relative dates"""
        patterns = {}
        
        patterns['relative_days'] = re.compile(
            r'\b(?:yesterday|today|tomorrow|the\s+day\s+before|the\s+next\s+day)\b',
            re.IGNORECASE
        )
        patterns['relative_weeks'] = re.compile(
            r'\b(?:last\s+week|this\s+week|next\s+week)\b',
            re.IGNORECASE
        )
        patterns['relative_months'] = re.compile(
            r'\b(?:last\s+month|this\s+month|next\s+month)\b',
            re.IGNORECASE
        )
        patterns['relative_years'] = re.compile(
            r'\b(?:last\s+year|this\s+year|next\s+year)\b',
            re.IGNORECASE
        )
        patterns['approximate'] = re.compile(
            r'\b(?:around|about|approximately|circa|roughly|near|close\s+to)\s+',
            re.IGNORECASE
        )
        patterns['seasons'] = re.compile(
            r'\b(?:spring|summer|fall|autumn|winter)\s+(?:of\s+)?(?:19|20)?\d{2}\b',
            re.IGNORECASE
        )
        patterns['decades'] = re.compile(
            r'\b(?:early|mid|late)\s+(?:19|20)\d{1}0s?\b',
            re.IGNORECASE
        )
        
        return patterns
    
    def extract_dates(self, text: str, document_id: str, 
                     language: LanguageCode) -> List[TemporalEntity]:
        """Extract date references from text"""
        entities = []
        
        # Extract with different pattern types
        entities.extend(self._extract_with_patterns(text, document_id, language))
        entities.extend(self._extract_with_parsedatetime(text, document_id, language))
        entities.extend(self._extract_fuzzy_dates(text, document_id, language))
        
        # Remove duplicates and overlaps
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_with_patterns(self, text: str, document_id: str,
                              language: LanguageCode) -> List[TemporalEntity]:
        """Extract dates using regex patterns"""
        entities = []
        
        for pattern_name, pattern in self.date_patterns.items():
            for match in pattern.finditer(text):
                try:
                    date_text = match.group().strip()
                    parsed_date = self._parse_date_string(date_text, language)
                    
                    if parsed_date:
                        entity = TemporalEntity(
                            text=date_text,
                            entity_type=TemporalEntityType.DATE,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=self._calculate_confidence(pattern_name, date_text),
                            language=language,
                            document_id=document_id,
                            parsed_date=parsed_date,
                            context=self._extract_context(text, match.start(), match.end()),
                            normalized_form=parsed_date.strftime('%Y-%m-%d'),
                            metadata={'pattern_type': pattern_name, 'extraction_method': 'regex'}
                        )
                        entities.append(entity)
                        
                except Exception as e:
                    logger.debug("Failed to parse date '{}': {}", date_text, e)
                    continue
        
        return entities
    
    def _extract_with_parsedatetime(self, text: str, document_id: str,
                                   language: LanguageCode) -> List[TemporalEntity]:
        """Extract dates using parsedatetime library"""
        entities = []
        
        try:
            # Use parsedatetime to find date/time expressions
            parse_result = self.cal.nlp(text)
            
            for item in parse_result:
                if item[1] == 1 or item[1] == 3:  # Date or datetime
                    try:
                        parsed_date = datetime(*item[0][:6])
                        
                        # Find the original text span (approximate)
                        date_str = item[4]
                        start_pos = text.find(date_str)
                        
                        if start_pos >= 0:
                            entity = TemporalEntity(
                                text=date_str,
                                entity_type=TemporalEntityType.DATE if item[1] == 1 else TemporalEntityType.DATETIME,
                                start_pos=start_pos,
                                end_pos=start_pos + len(date_str),
                                confidence=0.8,  # parsedatetime confidence
                                language=language,
                                document_id=document_id,
                                parsed_date=parsed_date,
                                context=self._extract_context(text, start_pos, start_pos + len(date_str)),
                                normalized_form=parsed_date.strftime('%Y-%m-%d'),
                                metadata={'extraction_method': 'parsedatetime'}
                            )
                            entities.append(entity)
                            
                    except Exception as e:
                        logger.debug("Failed to process parsedatetime result: {}", e)
                        continue
                        
        except Exception as e:
            logger.warning("Error using parsedatetime: {}", e)
        
        return entities
    
    def _extract_fuzzy_dates(self, text: str, document_id: str,
                            language: LanguageCode) -> List[TemporalEntity]:
        """Extract fuzzy and relative date references"""
        if not self.config.handle_fuzzy_dates:
            return []
        
        entities = []
        
        for pattern_name, pattern in self.fuzzy_date_patterns.items():
            for match in pattern.finditer(text):
                try:
                    date_text = match.group().strip()
                    parsed_date, uncertainty = self._parse_fuzzy_date(date_text, pattern_name)
                    
                    if parsed_date:
                        entity = TemporalEntity(
                            text=date_text,
                            entity_type=TemporalEntityType.FUZZY_TIME,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=0.6,  # Lower confidence for fuzzy dates
                            language=language,
                            document_id=document_id,
                            parsed_date=parsed_date,
                            is_fuzzy=True,
                            uncertainty_range=uncertainty,
                            context=self._extract_context(text, match.start(), match.end()),
                            normalized_form=self._normalize_fuzzy_date(date_text),
                            metadata={'pattern_type': pattern_name, 'extraction_method': 'fuzzy'}
                        )
                        entities.append(entity)
                        
                except Exception as e:
                    logger.debug("Failed to parse fuzzy date '{}': {}", date_text, e)
                    continue
        
        return entities
    
    def _parse_date_string(self, date_str: str, language: LanguageCode) -> Optional[datetime]:
        """Parse a date string using multiple methods"""
        try:
            # Try dateutil parser first (handles many formats)
            return date_parser.parse(date_str, fuzzy=True)
        except:
            pass
        
        try:
            # Try arrow parser
            return arrow.get(date_str).datetime
        except:
            pass
        
        try:
            # Try parsedatetime
            time_struct, parse_status = self.cal.parse(date_str)
            if parse_status > 0:
                return datetime(*time_struct[:6])
        except:
            pass
        
        return None
    
    def _parse_fuzzy_date(self, date_str: str, pattern_type: str) -> Tuple[Optional[datetime], Optional[Tuple[datetime, datetime]]]:
        """Parse fuzzy date and return date with uncertainty range"""
        now = datetime.now()
        
        if pattern_type == 'relative_days':
            if 'yesterday' in date_str.lower():
                date = now - timedelta(days=1)
                uncertainty = (date - timedelta(hours=2), date + timedelta(hours=2))
            elif 'today' in date_str.lower():
                date = now
                uncertainty = (date - timedelta(hours=1), date + timedelta(hours=1))
            elif 'tomorrow' in date_str.lower():
                date = now + timedelta(days=1)
                uncertainty = (date - timedelta(hours=2), date + timedelta(hours=2))
            else:
                return None, None
                
            return date, uncertainty
        
        elif pattern_type == 'seasons':
            # Extract year from season reference
            year_match = re.search(r'(?:19|20)?\d{2}', date_str)
            if year_match:
                year = int(year_match.group())
                if year < 100:
                    year += 2000 if year < 50 else 1900
                
                if 'spring' in date_str.lower():
                    date = datetime(year, 4, 1)  # Approximate spring start
                    uncertainty = (datetime(year, 3, 15), datetime(year, 5, 15))
                elif 'summer' in date_str.lower():
                    date = datetime(year, 7, 1)
                    uncertainty = (datetime(year, 6, 15), datetime(year, 8, 15))
                elif any(x in date_str.lower() for x in ['fall', 'autumn']):
                    date = datetime(year, 10, 1)
                    uncertainty = (datetime(year, 9, 15), datetime(year, 11, 15))
                elif 'winter' in date_str.lower():
                    date = datetime(year, 1, 1)
                    uncertainty = (datetime(year-1, 12, 15), datetime(year, 2, 15))
                else:
                    return None, None
                    
                return date, uncertainty
        
        elif pattern_type == 'decades':
            decade_match = re.search(r'(?:19|20)(\d{1})0s?', date_str)
            if decade_match:
                decade_start = int(decade_match.group())
                
                if 'early' in date_str.lower():
                    date = datetime(decade_start, 1, 1)
                    uncertainty = (date, datetime(decade_start + 3, 12, 31))
                elif 'mid' in date_str.lower():
                    date = datetime(decade_start + 5, 1, 1)
                    uncertainty = (datetime(decade_start + 3, 1, 1), datetime(decade_start + 7, 12, 31))
                elif 'late' in date_str.lower():
                    date = datetime(decade_start + 7, 1, 1)
                    uncertainty = (date, datetime(decade_start + 9, 12, 31))
                else:
                    date = datetime(decade_start + 5, 1, 1)  # Middle of decade
                    uncertainty = (datetime(decade_start, 1, 1), datetime(decade_start + 9, 12, 31))
                
                return date, uncertainty
        
        # Default parsing attempt
        try:
            parsed = date_parser.parse(date_str, fuzzy=True)
            uncertainty_days = 7  # Default 1 week uncertainty for fuzzy dates
            uncertainty = (
                parsed - timedelta(days=uncertainty_days),
                parsed + timedelta(days=uncertainty_days)
            )
            return parsed, uncertainty
        except:
            return None, None
    
    def _calculate_confidence(self, pattern_type: str, date_text: str) -> float:
        """Calculate confidence score for date extraction"""
        base_confidence = {
            'iso_date': 0.95,
            'written_date': 0.9,
            'us_date': 0.85,
            'eu_date': 0.85,
            'partial_date': 0.7,
            'year_only': 0.6,
            'spanish_date': 0.88,
            'french_date': 0.88,
            'german_date': 0.88
        }
        
        confidence = base_confidence.get(pattern_type, 0.7)
        
        # Adjust based on text characteristics
        if len(date_text) > 10:  # Longer, more specific dates
            confidence += 0.05
        if re.search(r'\d{4}', date_text):  # Contains 4-digit year
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _normalize_fuzzy_date(self, date_text: str) -> str:
        """Normalize fuzzy date to standard form"""
        date_text = date_text.lower().strip()
        
        normalizations = {
            'yesterday': 'previous day',
            'today': 'current day', 
            'tomorrow': 'next day',
            'last week': 'previous week',
            'this week': 'current week',
            'next week': 'following week',
            'last month': 'previous month',
            'this month': 'current month',
            'next month': 'following month',
            'last year': 'previous year',
            'this year': 'current year',
            'next year': 'following year'
        }
        
        for original, normalized in normalizations.items():
            if original in date_text:
                return normalized
        
        return date_text
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, 
                        window: int = 50) -> str:
        """Extract surrounding context for temporal entity"""
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)
        return text[context_start:context_end].strip()
    
    def _deduplicate_entities(self, entities: List[TemporalEntity]) -> List[TemporalEntity]:
        """Remove duplicate and overlapping temporal entities"""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x.start_pos)
        deduplicated = [sorted_entities[0]]
        
        for current in sorted_entities[1:]:
            last_added = deduplicated[-1]
            
            # Check for overlap
            if (current.start_pos < last_added.end_pos and 
                current.end_pos > last_added.start_pos):
                # Overlapping entities - keep the one with higher confidence
                if current.confidence > last_added.confidence:
                    deduplicated[-1] = current
            else:
                deduplicated.append(current)
        
        return deduplicated


class TimeExtractor:
    """Extracts and parses time references from text"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.time_patterns = self._compile_time_patterns()
        
    def _compile_time_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for time extraction"""
        patterns = {}
        
        # Standard time patterns
        patterns['24_hour'] = re.compile(
            r'\b([01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\b', re.IGNORECASE
        )
        patterns['12_hour_am_pm'] = re.compile(
            r'\b([01]?[0-9]):[0-5][0-9](?::[0-5][0-9])?\s*[ap]\.?m\.?\b', re.IGNORECASE
        )
        patterns['hour_only_am_pm'] = re.compile(
            r'\b([01]?[0-9])\s*[ap]\.?m\.?\b', re.IGNORECASE
        )
        patterns['written_time'] = re.compile(
            r'\b(?:at\s+)?(?:noon|midnight|midday)\b', re.IGNORECASE
        )
        patterns['relative_time'] = re.compile(
            r'\b(?:in\s+the\s+)?(?:morning|afternoon|evening|night)\b', re.IGNORECASE
        )
        
        return patterns
    
    def extract_times(self, text: str, document_id: str,
                     language: LanguageCode) -> List[TemporalEntity]:
        """Extract time references from text"""
        entities = []
        
        for pattern_name, pattern in self.time_patterns.items():
            for match in pattern.finditer(text):
                try:
                    time_text = match.group().strip()
                    parsed_time = self._parse_time_string(time_text, pattern_name)
                    
                    if parsed_time:
                        entity = TemporalEntity(
                            text=time_text,
                            entity_type=TemporalEntityType.TIME,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=self._calculate_time_confidence(pattern_name, time_text),
                            language=language,
                            document_id=document_id,
                            parsed_date=parsed_time,
                            context=self._extract_context(text, match.start(), match.end()),
                            normalized_form=parsed_time.strftime('%H:%M:%S'),
                            metadata={'pattern_type': pattern_name, 'extraction_method': 'time_regex'}
                        )
                        entities.append(entity)
                        
                except Exception as e:
                    logger.debug("Failed to parse time '{}': {}", time_text, e)
                    continue
        
        return entities
    
    def _parse_time_string(self, time_str: str, pattern_type: str) -> Optional[datetime]:
        """Parse time string and return datetime with today's date"""
        today = datetime.now().date()
        
        try:
            if pattern_type == 'written_time':
                if 'noon' in time_str.lower() or 'midday' in time_str.lower():
                    return datetime.combine(today, datetime.strptime('12:00:00', '%H:%M:%S').time())
                elif 'midnight' in time_str.lower():
                    return datetime.combine(today, datetime.strptime('00:00:00', '%H:%M:%S').time())
            
            elif pattern_type == 'relative_time':
                if 'morning' in time_str.lower():
                    return datetime.combine(today, datetime.strptime('08:00:00', '%H:%M:%S').time())
                elif 'afternoon' in time_str.lower():
                    return datetime.combine(today, datetime.strptime('14:00:00', '%H:%M:%S').time())
                elif 'evening' in time_str.lower():
                    return datetime.combine(today, datetime.strptime('19:00:00', '%H:%M:%S').time())
                elif 'night' in time_str.lower():
                    return datetime.combine(today, datetime.strptime('22:00:00', '%H:%M:%S').time())
            
            else:
                # Try standard parsing
                parsed = date_parser.parse(time_str)
                return datetime.combine(today, parsed.time())
                
        except Exception as e:
            logger.debug("Failed to parse time string '{}': {}", time_str, e)
        
        return None
    
    def _calculate_time_confidence(self, pattern_type: str, time_text: str) -> float:
        """Calculate confidence score for time extraction"""
        base_confidence = {
            '24_hour': 0.95,
            '12_hour_am_pm': 0.9,
            'hour_only_am_pm': 0.8,
            'written_time': 0.85,
            'relative_time': 0.6
        }
        
        return base_confidence.get(pattern_type, 0.7)
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int,
                        window: int = 30) -> str:
        """Extract surrounding context for time entity"""
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)
        return text[context_start:context_end].strip()


class DurationExtractor:
    """Extracts and parses duration references from text"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.duration_patterns = self._compile_duration_patterns()
        
    def _compile_duration_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for duration extraction"""
        patterns = {}
        
        patterns['explicit_duration'] = re.compile(
            r'\b(?:for\s+|during\s+|lasted\s+|took\s+)?(\d+)\s*(second|minute|hour|day|week|month|year)s?\b',
            re.IGNORECASE
        )
        patterns['range_duration'] = re.compile(
            r'\bfrom\s+.+?\s+to\s+.+?\b', re.IGNORECASE
        )
        patterns['approximate_duration'] = re.compile(
            r'\b(?:about|approximately|around|roughly)\s+(\d+)\s*(second|minute|hour|day|week|month|year)s?\b',
            re.IGNORECASE
        )
        
        return patterns
    
    def extract_durations(self, text: str, document_id: str,
                         language: LanguageCode) -> List[TemporalEntity]:
        """Extract duration references from text"""
        entities = []
        
        for pattern_name, pattern in self.duration_patterns.items():
            for match in pattern.finditer(text):
                try:
                    duration_text = match.group().strip()
                    duration_seconds = self._parse_duration_string(duration_text, pattern_name)
                    
                    if duration_seconds:
                        entity = TemporalEntity(
                            text=duration_text,
                            entity_type=TemporalEntityType.DURATION,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=self._calculate_duration_confidence(pattern_name),
                            language=language,
                            document_id=document_id,
                            duration_seconds=duration_seconds,
                            is_fuzzy='approximate' in pattern_name,
                            context=self._extract_context(text, match.start(), match.end()),
                            normalized_form=self._format_duration(duration_seconds),
                            metadata={'pattern_type': pattern_name, 'extraction_method': 'duration_regex'}
                        )
                        entities.append(entity)
                        
                except Exception as e:
                    logger.debug("Failed to parse duration '{}': {}", duration_text, e)
                    continue
        
        return entities
    
    def _parse_duration_string(self, duration_str: str, pattern_type: str) -> Optional[int]:
        """Parse duration string and return duration in seconds"""
        try:
            if pattern_type in ['explicit_duration', 'approximate_duration']:
                # Extract number and unit
                match = re.search(r'(\d+)\s*(second|minute|hour|day|week|month|year)s?', 
                                duration_str, re.IGNORECASE)
                
                if match:
                    number = int(match.group(1))
                    unit = match.group(2).lower()
                    
                    multipliers = {
                        'second': 1,
                        'minute': 60,
                        'hour': 3600,
                        'day': 86400,
                        'week': 604800,
                        'month': 2629746,  # Average month in seconds
                        'year': 31556952   # Average year in seconds
                    }
                    
                    return number * multipliers.get(unit, 1)
            
            elif pattern_type == 'range_duration':
                # Try to parse start and end times
                # This is more complex and would require integration with date/time extraction
                pass
                
        except Exception as e:
            logger.debug("Failed to parse duration '{}': {}", duration_str, e)
        
        return None
    
    def _calculate_duration_confidence(self, pattern_type: str) -> float:
        """Calculate confidence score for duration extraction"""
        base_confidence = {
            'explicit_duration': 0.9,
            'approximate_duration': 0.7,
            'range_duration': 0.8
        }
        
        return base_confidence.get(pattern_type, 0.7)
    
    def _format_duration(self, duration_seconds: int) -> str:
        """Format duration in seconds to human-readable string"""
        if duration_seconds < 60:
            return f"{duration_seconds} seconds"
        elif duration_seconds < 3600:
            minutes = duration_seconds // 60
            return f"{minutes} minutes"
        elif duration_seconds < 86400:
            hours = duration_seconds // 3600
            return f"{hours} hours"
        elif duration_seconds < 2629746:  # Less than a month
            days = duration_seconds // 86400
            return f"{days} days"
        elif duration_seconds < 31556952:  # Less than a year
            months = duration_seconds // 2629746
            return f"{months} months"
        else:
            years = duration_seconds // 31556952
            return f"{years} years"
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int,
                        window: int = 40) -> str:
        """Extract surrounding context for duration entity"""
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)
        return text[context_start:context_end].strip()


class TemporalExtractor:
    """Main temporal extraction coordinator"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.date_extractor = DateExtractor(config)
        self.time_extractor = TimeExtractor(config)
        self.duration_extractor = DurationExtractor(config)
        
        # Initialize spaCy for advanced NLP features
        self.nlp = None
        self._initialize_spacy()
        
    def _initialize_spacy(self) -> None:
        """Initialize spaCy models for supported languages"""
        try:
            # Load primary language model
            model_mapping = {
                LanguageCode.EN: 'en_core_web_sm',
                LanguageCode.ES: 'es_core_news_sm',
                LanguageCode.FR: 'fr_core_news_sm',
                LanguageCode.DE: 'de_core_news_sm',
                # Add more mappings as needed
            }
            
            model_name = model_mapping.get(self.config.primary_language, 'en_core_web_sm')
            try:
                self.nlp = spacy.load(model_name)
                logger.info("Loaded spaCy model: {}", model_name)
            except OSError:
                logger.warning("spaCy model {} not found, using basic extraction", model_name)
                
        except Exception as e:
            logger.warning("Error initializing spaCy: {}", e)
    
    def extract_temporal_references(self, text: str, document_id: str,
                                   language: Optional[LanguageCode] = None) -> List[TemporalEntity]:
        """
        Extract all temporal references from text
        
        Args:
            text: Input text to analyze
            document_id: Document identifier
            language: Language of the text (auto-detected if not provided)
            
        Returns:
            List of extracted temporal entities
        """
        if not text or not text.strip():
            return []
        
        if not document_id:
            document_id = str(uuid.uuid4())
        
        # Detect language if not provided
        if not language:
            language = self._detect_language(text)
        
        logger.debug("Extracting temporal references from {} chars of text in {}",
                    len(text), language.value)
        
        all_entities = []
        
        try:
            # Extract dates
            if self.config.extract_dates:
                date_entities = self.date_extractor.extract_dates(text, document_id, language)
                all_entities.extend(date_entities)
                logger.debug("Extracted {} date entities", len(date_entities))
            
            # Extract times
            if self.config.extract_times:
                time_entities = self.time_extractor.extract_times(text, document_id, language)
                all_entities.extend(time_entities)
                logger.debug("Extracted {} time entities", len(time_entities))
            
            # Extract durations
            if self.config.extract_durations:
                duration_entities = self.duration_extractor.extract_durations(text, document_id, language)
                all_entities.extend(duration_entities)
                logger.debug("Extracted {} duration entities", len(duration_entities))
            
            # Use spaCy for additional extraction if available
            if self.nlp:
                spacy_entities = self._extract_with_spacy(text, document_id, language)
                all_entities.extend(spacy_entities)
                logger.debug("Extracted {} spaCy entities", len(spacy_entities))
            
            # Filter by confidence and deduplicate
            filtered_entities = [
                entity for entity in all_entities
                if entity.confidence >= self.config.min_confidence
            ]
            
            final_entities = self._global_deduplication(filtered_entities)
            
            logger.info("Extracted {} temporal entities (filtered from {})",
                       len(final_entities), len(all_entities))
            
            return final_entities
            
        except Exception as e:
            logger.error("Error extracting temporal references: {}", e)
            return []
    
    def _detect_language(self, text: str) -> LanguageCode:
        """Detect language of input text"""
        if not self.config.auto_detect_language:
            return self.config.primary_language
        
        try:
            # Simple language detection based on common temporal keywords
            language_indicators = {
                LanguageCode.EN: ['january', 'february', 'today', 'yesterday', 'tomorrow', 'morning', 'afternoon'],
                LanguageCode.ES: ['enero', 'febrero', 'hoy', 'ayer', 'mañana', 'por la mañana'],
                LanguageCode.FR: ['janvier', 'février', 'aujourd\'hui', 'hier', 'demain', 'matin'],
                LanguageCode.DE: ['januar', 'februar', 'heute', 'gestern', 'morgen', 'vormittag']
            }
            
            text_lower = text.lower()
            language_scores = {}
            
            for lang, indicators in language_indicators.items():
                if lang in self.config.supported_languages:
                    score = sum(1 for indicator in indicators if indicator in text_lower)
                    language_scores[lang] = score
            
            if language_scores:
                detected_language = max(language_scores.items(), key=lambda x: x[1])[0]
                if language_scores[detected_language] > 0:
                    return detected_language
            
        except Exception as e:
            logger.debug("Language detection failed: {}", e)
        
        return self.config.primary_language
    
    def _extract_with_spacy(self, text: str, document_id: str,
                           language: LanguageCode) -> List[TemporalEntity]:
        """Extract temporal entities using spaCy NER"""
        if not self.nlp:
            return []
        
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME', 'EVENT', 'ORDINAL']:
                    try:
                        # Try to parse the entity text
                        parsed_date = None
                        entity_type = TemporalEntityType.FUZZY_TIME
                        
                        if ent.label_ == 'DATE':
                            parsed_date = date_parser.parse(ent.text, fuzzy=True)
                            entity_type = TemporalEntityType.DATE
                        elif ent.label_ == 'TIME':
                            parsed_date = date_parser.parse(ent.text, fuzzy=True)
                            entity_type = TemporalEntityType.TIME
                        
                        if parsed_date or ent.label_ in ['EVENT', 'ORDINAL']:
                            entity = TemporalEntity(
                                text=ent.text,
                                entity_type=entity_type,
                                start_pos=ent.start_char,
                                end_pos=ent.end_char,
                                confidence=0.75,  # spaCy confidence
                                language=language,
                                document_id=document_id,
                                parsed_date=parsed_date,
                                context=str(ent.sent) if ent.sent else None,
                                normalized_form=ent.text.lower(),
                                metadata={'spacy_label': ent.label_, 'extraction_method': 'spacy'}
                            )
                            entities.append(entity)
                            
                    except Exception as e:
                        logger.debug("Failed to process spaCy entity '{}': {}", ent.text, e)
                        continue
                        
        except Exception as e:
            logger.warning("Error using spaCy for temporal extraction: {}", e)
        
        return entities
    
    def _global_deduplication(self, entities: List[TemporalEntity]) -> List[TemporalEntity]:
        """Perform global deduplication of all temporal entities"""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x.start_pos)
        deduplicated = []
        
        for current in sorted_entities:
            # Check for overlap with existing entities
            overlaps = False
            
            for i, existing in enumerate(deduplicated):
                if (current.start_pos < existing.end_pos and 
                    current.end_pos > existing.start_pos):
                    # Overlapping entities
                    if current.confidence > existing.confidence:
                        # Replace with higher confidence entity
                        deduplicated[i] = current
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(current)
        
        return deduplicated


# Convenience function for direct usage
def extract_temporal_references(text: str, config: Optional[TimelineConfig] = None,
                               document_id: Optional[str] = None,
                               language: Optional[LanguageCode] = None) -> List[TemporalEntity]:
    """
    Convenience function to extract temporal references from text
    
    Args:
        text: Input text to analyze
        config: Timeline configuration (uses default if not provided)
        document_id: Document identifier
        language: Text language
        
    Returns:
        List of extracted temporal entities
    """
    if config is None:
        from .core import create_default_timeline_config
        config = create_default_timeline_config()
    
    extractor = TemporalExtractor(config)
    return extractor.extract_temporal_references(text, document_id or str(uuid.uuid4()), language)