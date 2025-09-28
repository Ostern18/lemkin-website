"""
Language-specific processing and cross-language entity matching.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import unicodedata
from langdetect import detect, detect_langs, LangDetectException
import stanza
from loguru import logger

from .core import Entity, EntityType, LanguageCode, NERConfig


class MultilingualProcessor:
    """
    Handles language detection, normalization, and cross-language processing
    """
    
    def __init__(self, config: NERConfig):
        """
        Initialize multilingual processor
        
        Args:
            config: NER configuration
        """
        self.config = config
        self.stanza_pipelines = {}
        self.language_patterns = {}
        self.normalization_rules = {}
        self.transliteration_tables = {}
        
        # Initialize language processing components
        self._initialize_stanza_pipelines()
        self._load_language_patterns()
        self._load_normalization_rules()
        self._load_transliteration_tables()
        
        logger.info("MultilingualProcessor initialized for languages: {}", 
                   [lang.value for lang in config.supported_languages])
    
    def _initialize_stanza_pipelines(self) -> None:
        """Initialize Stanza pipelines for supported languages"""
        # Stanza language code mappings
        stanza_lang_map = {
            LanguageCode.EN: "en",
            LanguageCode.ES: "es", 
            LanguageCode.FR: "fr",
            LanguageCode.DE: "de",
            LanguageCode.IT: "it",
            LanguageCode.PT: "pt",
            LanguageCode.RU: "ru",
            LanguageCode.ZH: "zh-hans",
            LanguageCode.AR: "ar",
            LanguageCode.JA: "ja"
        }
        
        for language in self.config.supported_languages:
            if language in stanza_lang_map:
                try:
                    stanza_lang = stanza_lang_map[language]
                    
                    # Download model if not available
                    try:
                        pipeline = stanza.Pipeline(
                            stanza_lang, 
                            processors='tokenize,mwt,pos,lemma,ner',
                            verbose=False,
                            use_gpu=False  # Set to True if GPU available
                        )
                        self.stanza_pipelines[language] = pipeline
                        logger.info("Loaded Stanza pipeline for {}", language.value)
                    except Exception as e:
                        logger.warning("Failed to load Stanza pipeline for {}: {}", language.value, e)
                        # Try downloading the model
                        try:
                            stanza.download(stanza_lang, verbose=False)
                            pipeline = stanza.Pipeline(
                                stanza_lang, 
                                processors='tokenize,mwt,pos,lemma,ner',
                                verbose=False,
                                use_gpu=False
                            )
                            self.stanza_pipelines[language] = pipeline
                            logger.info("Downloaded and loaded Stanza pipeline for {}", language.value)
                        except Exception as e2:
                            logger.error("Failed to download Stanza model for {}: {}", language.value, e2)
                            
                except Exception as e:
                    logger.error("Error initializing Stanza for {}: {}", language.value, e)
    
    def _load_language_patterns(self) -> None:
        """Load language-specific text patterns"""
        self.language_patterns = {
            LanguageCode.EN: {
                "name_prefixes": r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Hon\.?)\b',
                "name_suffixes": r'\b(?:Jr\.?|Sr\.?|II|III|IV|V|Esq\.?|Ph\.?D\.?|M\.?D\.?)\b',
                "organization_indicators": r'\b(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Co\.?|Company|Corporation)\b',
                "location_indicators": r'\b(?:City|County|State|Province|Country|District)\b',
                "legal_titles": r'\b(?:Judge|Justice|Attorney|Lawyer|Counsel|Prosecutor|Defendant)\b'
            },
            LanguageCode.ES: {
                "name_prefixes": r'\b(?:Sr\.?|Sra\.?|Srta\.?|Dr\.?|Dra\.?|Prof\.?|Hon\.?)\b',
                "name_suffixes": r'\b(?:Jr\.?|II|III|IV|V)\b',
                "organization_indicators": r'\b(?:S\.?A\.?|S\.?L\.?|Ltda\.?|Cía\.?|Empresa|Corporación)\b',
                "location_indicators": r'\b(?:Ciudad|Condado|Estado|Provincia|País|Distrito)\b',
                "legal_titles": r'\b(?:Juez|Magistrado|Abogado|Fiscal|Demandado|Acusado)\b'
            },
            LanguageCode.FR: {
                "name_prefixes": r'\b(?:M\.?|Mme\.?|Mlle\.?|Dr\.?|Prof\.?|Hon\.?)\b',
                "name_suffixes": r'\b(?:Jr\.?|II|III|IV|V)\b',
                "organization_indicators": r'\b(?:S\.?A\.?|S\.?A\.?R\.?L\.?|Cie\.?|Société|Corporation)\b',
                "location_indicators": r'\b(?:Ville|Département|État|Province|Pays|District)\b',
                "legal_titles": r'\b(?:Juge|Magistrat|Avocat|Procureur|Défendeur|Accusé)\b'
            },
            LanguageCode.DE: {
                "name_prefixes": r'\b(?:Herr|Frau|Dr\.?|Prof\.?|Hon\.?)\b',
                "name_suffixes": r'\b(?:Jr\.?|II|III|IV|V)\b',
                "organization_indicators": r'\b(?:GmbH|AG|KG|OHG|Unternehmen|Gesellschaft)\b',
                "location_indicators": r'\b(?:Stadt|Landkreis|Staat|Land|Bezirk)\b',
                "legal_titles": r'\b(?:Richter|Anwalt|Staatsanwalt|Angeklagte|Beklagte)\b'
            },
            LanguageCode.RU: {
                "name_prefixes": r'\b(?:г-н|г-жа|д-р|проф\.?)\b',
                "organization_indicators": r'\b(?:ООО|ЗАО|ОАО|ИП|компания|корпорация)\b',
                "location_indicators": r'\b(?:город|область|край|республика|район)\b',
                "legal_titles": r'\b(?:судья|адвокат|прокурор|обвиняемый|ответчик)\b'
            },
            LanguageCode.ZH: {
                "name_prefixes": r'(?:先生|女士|博士|教授)',
                "organization_indicators": r'(?:公司|企业|集团|有限公司|股份有限公司)',
                "location_indicators": r'(?:市|省|县|区|州|国)',
                "legal_titles": r'(?:法官|律师|检察官|被告|原告)'
            },
            LanguageCode.AR: {
                "name_prefixes": r'(?:السيد|السيدة|الدكتور|الأستاذ)',
                "organization_indicators": r'(?:شركة|مؤسسة|مجموعة)',
                "location_indicators": r'(?:مدينة|محافظة|منطقة|دولة)',
                "legal_titles": r'(?:قاضي|محامي|مدعي عام|متهم|مدعى عليه)'
            }
        }
    
    def _load_normalization_rules(self) -> None:
        """Load text normalization rules for each language"""
        self.normalization_rules = {
            LanguageCode.EN: {
                "contractions": {
                    "won't": "will not",
                    "can't": "cannot", 
                    "n't": " not",
                    "'re": " are",
                    "'ve": " have",
                    "'ll": " will",
                    "'d": " would"
                },
                "abbreviations": {
                    "St.": "Street",
                    "Ave.": "Avenue", 
                    "Blvd.": "Boulevard",
                    "Dr.": "Drive",
                    "Corp.": "Corporation",
                    "Inc.": "Incorporated",
                    "Ltd.": "Limited"
                }
            },
            LanguageCode.ES: {
                "contractions": {
                    "del": "de el",
                    "al": "a el"
                },
                "abbreviations": {
                    "Sr.": "Señor",
                    "Sra.": "Señora",
                    "Dr.": "Doctor",
                    "Dra.": "Doctora"
                }
            },
            LanguageCode.FR: {
                "contractions": {
                    "du": "de le",
                    "au": "à le", 
                    "des": "de les",
                    "aux": "à les"
                },
                "abbreviations": {
                    "M.": "Monsieur",
                    "Mme.": "Madame",
                    "Dr.": "Docteur"
                }
            }
        }
    
    def _load_transliteration_tables(self) -> None:
        """Load transliteration tables for different scripts"""
        # Simplified transliteration mappings
        self.transliteration_tables = {
            "cyrillic_to_latin": {
                'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
                'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k',
                'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
                'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
                'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '',
                'э': 'e', 'ю': 'yu', 'я': 'ya'
            },
            "arabic_to_latin": {
                'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h',
                'خ': 'kh', 'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's',
                'ش': 'sh', 'ص': 's', 'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': "'",
                'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
                'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y'
            }
        }
    
    def detect_language(self, text: str, min_confidence: float = 0.7) -> LanguageCode:
        """
        Detect language of input text
        
        Args:
            text: Input text
            min_confidence: Minimum confidence threshold
            
        Returns:
            Detected language code
        """
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for reliable language detection")
            return self.config.primary_language
        
        try:
            # Try to detect language with confidence scores
            lang_probs = detect_langs(text)
            
            for lang_prob in lang_probs:
                if lang_prob.prob >= min_confidence:
                    # Map langdetect codes to our LanguageCode enum
                    detected_lang = self._map_langdetect_code(lang_prob.lang)
                    if detected_lang and detected_lang in self.config.supported_languages:
                        logger.debug("Detected language: {} (confidence: {:.3f})", 
                                   detected_lang.value, lang_prob.prob)
                        return detected_lang
            
            # Fallback to simple detection
            detected_code = detect(text)
            detected_lang = self._map_langdetect_code(detected_code)
            
            if detected_lang and detected_lang in self.config.supported_languages:
                logger.debug("Detected language (fallback): {}", detected_lang.value)
                return detected_lang
            
        except LangDetectException as e:
            logger.warning("Language detection failed: {}", e)
        
        logger.info("Using primary language as fallback: {}", self.config.primary_language.value)
        return self.config.primary_language
    
    def normalize_entity(self, entity: Entity) -> Entity:
        """
        Normalize entity text based on language-specific rules
        
        Args:
            entity: Entity to normalize
            
        Returns:
            Entity with normalized text
        """
        normalized_text = self.normalize_text(entity.text, entity.language)
        
        # Create a copy with normalized text
        normalized_entity = Entity(
            entity_id=entity.entity_id,
            text=entity.text,  # Keep original text
            entity_type=entity.entity_type,
            start_pos=entity.start_pos,
            end_pos=entity.end_pos,
            confidence=entity.confidence,
            language=entity.language,
            document_id=entity.document_id,
            context=entity.context,
            normalized_form=normalized_text,  # Set normalized form
            aliases=entity.aliases,
            metadata=entity.metadata,
            timestamp=entity.timestamp
        )
        
        # Add transliteration if text uses non-Latin script
        if self._needs_transliteration(normalized_text):
            transliterated = self.transliterate_text(normalized_text, entity.language)
            if transliterated and transliterated != normalized_text:
                normalized_entity.aliases.append(transliterated)
        
        # Add language-specific variants
        variants = self._generate_name_variants(normalized_text, entity.language, entity.entity_type)
        normalized_entity.aliases.extend(variants)
        
        return normalized_entity
    
    def normalize_text(self, text: str, language: LanguageCode) -> str:
        """
        Normalize text using language-specific rules
        
        Args:
            text: Text to normalize
            language: Language of the text
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        normalized = text
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKD', normalized)
        
        # Language-specific normalization
        if language in self.normalization_rules:
            rules = self.normalization_rules[language]
            
            # Apply contractions
            for contraction, expansion in rules.get("contractions", {}).items():
                normalized = normalized.replace(contraction, expansion)
            
            # Apply abbreviations
            for abbrev, expansion in rules.get("abbreviations", {}).items():
                normalized = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, normalized, flags=re.IGNORECASE)
        
        # General normalization
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Fix capitalization for names (if it's all caps or all lower)
        if normalized.isupper() or normalized.islower():
            normalized = normalized.title()
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def transliterate_text(self, text: str, source_language: LanguageCode) -> Optional[str]:
        """
        Transliterate text from non-Latin scripts to Latin
        
        Args:
            text: Text to transliterate
            source_language: Source language
            
        Returns:
            Transliterated text or None if not applicable
        """
        if not text:
            return None
        
        try:
            # Russian/Cyrillic transliteration
            if source_language == LanguageCode.RU:
                return self._apply_transliteration(text, "cyrillic_to_latin")
            
            # Arabic transliteration
            elif source_language == LanguageCode.AR:
                return self._apply_transliteration(text, "arabic_to_latin")
            
            # Chinese pinyin (simplified approach)
            elif source_language == LanguageCode.ZH:
                return self._chinese_to_pinyin(text)
            
            # For other languages, return None (no transliteration needed or available)
            return None
            
        except Exception as e:
            logger.warning("Transliteration failed for {}: {}", source_language.value, e)
            return None
    
    def cross_language_match(self, entity1: Entity, entity2: Entity) -> float:
        """
        Calculate cross-language similarity between entities
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if entity1.language == entity2.language:
            return 0.0  # Same language, not cross-language
        
        # Compare normalized forms
        norm1 = entity1.normalized_form or entity1.text
        norm2 = entity2.normalized_form or entity2.text
        
        # Direct normalized comparison
        if norm1.lower() == norm2.lower():
            return 0.9
        
        # Transliteration comparison
        transliterated_scores = []
        
        # Try transliterating entity1 to Latin and compare with entity2
        trans1 = self.transliterate_text(entity1.text, entity1.language)
        if trans1:
            from fuzzywuzzy import fuzz
            score = fuzz.ratio(trans1.lower(), norm2.lower()) / 100.0
            transliterated_scores.append(score)
        
        # Try transliterating entity2 to Latin and compare with entity1
        trans2 = self.transliterate_text(entity2.text, entity2.language)
        if trans2:
            from fuzzywuzzy import fuzz
            score = fuzz.ratio(norm1.lower(), trans2.lower()) / 100.0
            transliterated_scores.append(score)
        
        # Check aliases for cross-language matches
        alias_scores = []
        all_aliases1 = [entity1.text, entity1.normalized_form or ""] + entity1.aliases
        all_aliases2 = [entity2.text, entity2.normalized_form or ""] + entity2.aliases
        
        for alias1 in all_aliases1:
            if not alias1:
                continue
            for alias2 in all_aliases2:
                if not alias2:
                    continue
                if alias1.lower() == alias2.lower():
                    alias_scores.append(1.0)
                else:
                    from fuzzywuzzy import fuzz
                    score = fuzz.ratio(alias1.lower(), alias2.lower()) / 100.0
                    if score >= 0.8:  # High threshold for cross-language alias matching
                        alias_scores.append(score)
        
        # Combine scores
        all_scores = transliterated_scores + alias_scores
        if all_scores:
            return max(all_scores)
        
        return 0.0
    
    def process_with_stanza(self, text: str, language: LanguageCode) -> Optional[Any]:
        """
        Process text with Stanza pipeline
        
        Args:
            text: Text to process
            language: Language of the text
            
        Returns:
            Stanza document object or None
        """
        if language not in self.stanza_pipelines:
            logger.warning("No Stanza pipeline available for {}", language.value)
            return None
        
        try:
            pipeline = self.stanza_pipelines[language]
            doc = pipeline(text)
            return doc
        except Exception as e:
            logger.error("Stanza processing failed for {}: {}", language.value, e)
            return None
    
    def extract_linguistic_features(self, entity: Entity) -> Dict[str, Any]:
        """
        Extract linguistic features from entity using Stanza
        
        Args:
            entity: Entity to analyze
            
        Returns:
            Dictionary of linguistic features
        """
        features = {}
        
        # Process with Stanza if available
        stanza_doc = self.process_with_stanza(entity.text, entity.language)
        if stanza_doc:
            # Extract features from Stanza analysis
            tokens = []
            lemmas = []
            pos_tags = []
            
            for sentence in stanza_doc.sentences:
                for token in sentence.tokens:
                    for word in token.words:
                        tokens.append(word.text)
                        lemmas.append(word.lemma)
                        pos_tags.append(word.upos)
            
            features.update({
                "tokens": tokens,
                "lemmas": lemmas,
                "pos_tags": pos_tags,
                "token_count": len(tokens)
            })
        
        # Add language-specific pattern features
        if entity.language in self.language_patterns:
            patterns = self.language_patterns[entity.language]
            
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, entity.text, re.IGNORECASE):
                    features[f"has_{pattern_name}"] = True
        
        return features
    
    def _map_langdetect_code(self, lang_code: str) -> Optional[LanguageCode]:
        """Map langdetect language codes to our LanguageCode enum"""
        mapping = {
            'en': LanguageCode.EN,
            'es': LanguageCode.ES,
            'fr': LanguageCode.FR,
            'de': LanguageCode.DE,
            'it': LanguageCode.IT,
            'pt': LanguageCode.PT,
            'ru': LanguageCode.RU,
            'zh-cn': LanguageCode.ZH,
            'zh': LanguageCode.ZH,
            'ar': LanguageCode.AR,
            'ja': LanguageCode.JA
        }
        return mapping.get(lang_code)
    
    def _needs_transliteration(self, text: str) -> bool:
        """Check if text contains non-Latin scripts"""
        if not text:
            return False
        
        # Check for Cyrillic characters
        if re.search(r'[а-яё]', text, re.IGNORECASE):
            return True
        
        # Check for Arabic characters
        if re.search(r'[\u0600-\u06FF]', text):
            return True
        
        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return True
        
        return False
    
    def _apply_transliteration(self, text: str, table_name: str) -> str:
        """Apply transliteration using specified table"""
        if table_name not in self.transliteration_tables:
            return text
        
        table = self.transliteration_tables[table_name]
        result = ""
        
        for char in text.lower():
            if char in table:
                result += table[char]
            else:
                result += char
        
        return result
    
    def _chinese_to_pinyin(self, text: str) -> str:
        """Convert Chinese characters to Pinyin (simplified approach)"""
        try:
            # This would require a proper Pinyin library like pypinyin
            # For now, return text as-is
            logger.warning("Chinese to Pinyin conversion not fully implemented")
            return text
        except Exception:
            return text
    
    def _generate_name_variants(self, text: str, language: LanguageCode, 
                              entity_type: EntityType) -> List[str]:
        """Generate common name variants based on language and entity type"""
        variants = []
        
        if entity_type != EntityType.PERSON:
            return variants
        
        # Split name into parts
        name_parts = text.split()
        if len(name_parts) < 2:
            return variants
        
        # Generate common variants
        if language == LanguageCode.EN:
            # First Last -> F. Last, Last F., Last, F, First L.
            first = name_parts[0]
            last = name_parts[-1]
            
            if len(first) > 0:
                variants.append(f"{first[0]}. {last}")
                variants.append(f"{last}, {first[0]}.")
                variants.append(f"{first} {last[0]}.")
            
            variants.append(last)  # Last name only
            
            # Middle name handling
            if len(name_parts) > 2:
                middle = " ".join(name_parts[1:-1])
                variants.append(f"{first} {middle[0]}. {last}")
        
        elif language == LanguageCode.ES:
            # Spanish name conventions (often two surnames)
            if len(name_parts) >= 3:
                first = name_parts[0]
                paternal = name_parts[-2] 
                maternal = name_parts[-1]
                variants.append(f"{first} {paternal}")
                variants.append(f"{paternal} {maternal}")
                variants.append(paternal)
        
        elif language == LanguageCode.RU:
            # Russian name conventions (First Middle Last)
            if len(name_parts) == 3:
                first, middle, last = name_parts
                variants.append(f"{first} {last}")
                variants.append(f"{last} {first}")
                variants.append(f"{first[0]}. {middle[0]}. {last}")
        
        # Remove duplicates and empty variants
        variants = list(set(v for v in variants if v and v != text))
        
        return variants
    
    def get_language_info(self, language: LanguageCode) -> Dict[str, Any]:
        """Get information about language support and capabilities"""
        return {
            "language_code": language.value,
            "stanza_available": language in self.stanza_pipelines,
            "patterns_available": language in self.language_patterns,
            "normalization_available": language in self.normalization_rules,
            "transliteration_available": self._has_transliteration_support(language),
            "script_type": self._get_script_type(language)
        }
    
    def _has_transliteration_support(self, language: LanguageCode) -> bool:
        """Check if transliteration is supported for language"""
        return language in [LanguageCode.RU, LanguageCode.AR, LanguageCode.ZH]
    
    def _get_script_type(self, language: LanguageCode) -> str:
        """Get the script type for a language"""
        script_mapping = {
            LanguageCode.EN: "Latin",
            LanguageCode.ES: "Latin", 
            LanguageCode.FR: "Latin",
            LanguageCode.DE: "Latin",
            LanguageCode.IT: "Latin",
            LanguageCode.PT: "Latin",
            LanguageCode.RU: "Cyrillic",
            LanguageCode.AR: "Arabic",
            LanguageCode.ZH: "Han",
            LanguageCode.JA: "Mixed"
        }
        return script_mapping.get(language, "Unknown")