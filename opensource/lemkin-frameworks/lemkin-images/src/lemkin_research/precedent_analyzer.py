"""
Legal precedent analysis module using semantic embeddings and legal reasoning.

This module provides comprehensive precedent analysis including case similarity
analysis, binding authority evaluation, and Shepardizing-equivalent functionality
for validating case law currency and treatment.
"""

import json
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available, using sklearn for similarity search")
    FAISS_AVAILABLE = False

from .core import (
    CaseOpinion, SimilarCase, Precedent, PrecedentMatch, SearchQuery,
    RelevanceScore, JurisdictionType, ResearchConfig
)


class LegalEmbeddingModel:
    """Legal-specific text embedding model wrapper"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self.embedding_dim = None
        
        # Legal-specific model names (would use actual legal BERT models)
        self.legal_models = {
            "legal-bert": "nlpaueb/legal-bert-base-uncased",
            "legal-bert-small": "nlpaueb/legal-bert-small-uncased", 
            "sentence-legal": "sentence-transformers/legal-bert-base-uncased",
            "default": "sentence-transformers/all-MiniLM-L6-v2"
        }
    
    def load_model(self):
        """Load the embedding model"""
        if self._model is not None:
            return
        
        try:
            if self.model_name in self.legal_models:
                model_path = self.legal_models[self.model_name]
            else:
                model_path = self.model_name
            
            # Try to load as sentence transformer first
            try:
                self._model = SentenceTransformer(model_path)
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded SentenceTransformer model: {model_path}")
            except:
                # Fallback to transformers
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._model = AutoModel.from_pretrained(model_path)
                self.embedding_dim = self._model.config.hidden_size
                logger.info(f"Loaded Transformer model: {model_path}")
                
        except Exception as e:
            logger.warning(f"Failed to load model {self.model_name}: {e}")
            # Fallback to default
            self._model = SentenceTransformer(self.legal_models["default"])
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embeddings"""
        if self._model is None:
            self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            if isinstance(self._model, SentenceTransformer):
                embeddings = self._model.encode(texts, convert_to_numpy=True)
            else:
                # Use transformers model
                embeddings = self._encode_with_transformers(texts)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.embedding_dim))
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode using transformers library"""
        embeddings = []
        
        for text in texts:
            inputs = self._tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embeddings.append(embedding.numpy())
        
        return np.array(embeddings)


class CaseVectorDatabase:
    """Vector database for fast case similarity search"""
    
    def __init__(self, embedding_model: LegalEmbeddingModel, use_faiss: bool = True):
        self.embedding_model = embedding_model
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        # Storage
        self.case_embeddings = []
        self.case_metadata = []
        self.case_index = None
        
        # FAISS index
        if self.use_faiss:
            self.faiss_index = None
    
    def add_cases(self, cases: List[CaseOpinion]):
        """Add cases to the vector database"""
        if not cases:
            return
        
        logger.info(f"Adding {len(cases)} cases to vector database")
        
        # Prepare texts for embedding
        texts = []
        for case in cases:
            # Combine relevant text fields for embedding
            case_text = self._prepare_case_text(case)
            texts.append(case_text)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Store embeddings and metadata
        start_idx = len(self.case_embeddings)
        self.case_embeddings.extend(embeddings)
        
        for i, case in enumerate(cases):
            metadata = {
                'case_id': case.id,
                'case_name': case.case_name,
                'citation': case.citation,
                'court': case.court,
                'date_decided': case.date_decided,
                'jurisdiction': case.jurisdiction,
                'index': start_idx + i,
                'case_object': case
            }
            self.case_metadata.append(metadata)
        
        # Update indices
        self._update_indices()
        
        logger.info(f"Vector database now contains {len(self.case_embeddings)} cases")
    
    def _prepare_case_text(self, case: CaseOpinion) -> str:
        """Prepare case text for embedding"""
        text_parts = []
        
        # Case name (important for similarity)
        if case.case_name:
            text_parts.append(f"Case: {case.case_name}")
        
        # Holdings and key legal principles
        if case.holdings:
            text_parts.append(f"Holdings: {' '.join(case.holdings)}")
        
        # Legal issues
        if case.legal_issues:
            text_parts.append(f"Legal Issues: {' '.join(case.legal_issues)}")
        
        # Key facts
        if case.key_facts:
            text_parts.append(f"Key Facts: {' '.join(case.key_facts)}")
        
        # Summary if available
        if case.summary:
            text_parts.append(f"Summary: {case.summary}")
        
        # Subject areas for topical similarity
        if case.subject_areas:
            text_parts.append(f"Subject Areas: {' '.join(case.subject_areas)}")
        
        # Full text excerpt (if available and not too long)
        if case.full_text and len(case.full_text) > 100:
            # Take first 1000 chars of full text
            text_parts.append(case.full_text[:1000])
        
        return " ".join(text_parts)
    
    def _update_indices(self):
        """Update search indices"""
        if not self.case_embeddings:
            return
        
        embeddings_array = np.array(self.case_embeddings)
        
        if self.use_faiss:
            # Update FAISS index
            dimension = embeddings_array.shape[1]
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array.astype('float32'))
            self.faiss_index.add(embeddings_array.astype('float32'))
    
    def search_similar(
        self, 
        query_case: Union[str, CaseOpinion], 
        k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar cases"""
        if not self.case_embeddings:
            logger.warning("No cases in vector database")
            return []
        
        # Prepare query embedding
        if isinstance(query_case, str):
            query_text = query_case
        else:
            query_text = self._prepare_case_text(query_case)
        
        query_embedding = self.embedding_model.encode([query_text])
        
        if self.use_faiss and self.faiss_index is not None:
            return self._search_with_faiss(query_embedding[0], k, threshold)
        else:
            return self._search_with_sklearn(query_embedding[0], k, threshold)
    
    def _search_with_faiss(
        self, 
        query_embedding: np.ndarray, 
        k: int, 
        threshold: float
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search using FAISS index"""
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= threshold and idx < len(self.case_metadata):
                results.append((self.case_metadata[idx], float(similarity)))
        
        return results
    
    def _search_with_sklearn(
        self, 
        query_embedding: np.ndarray, 
        k: int, 
        threshold: float
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search using sklearn cosine similarity"""
        embeddings_array = np.array(self.case_embeddings)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], embeddings_array)[0]
        
        # Get top k similar cases above threshold
        similar_indices = np.where(similarities >= threshold)[0]
        if len(similar_indices) == 0:
            return []
        
        # Sort by similarity
        sorted_indices = similar_indices[np.argsort(similarities[similar_indices])[::-1]]
        
        results = []
        for idx in sorted_indices[:k]:
            if idx < len(self.case_metadata):
                results.append((self.case_metadata[idx], similarities[idx]))
        
        return results
    
    def get_case_count(self) -> int:
        """Get number of cases in database"""
        return len(self.case_embeddings)
    
    def save_to_disk(self, path: Path):
        """Save vector database to disk"""
        save_data = {
            'embeddings': self.case_embeddings,
            'metadata': self.case_metadata,
            'model_name': self.embedding_model.model_name
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save FAISS index separately if available
        if self.use_faiss and self.faiss_index is not None:
            faiss_path = path.with_suffix('.faiss')
            faiss.write_index(self.faiss_index, str(faiss_path))
    
    def load_from_disk(self, path: Path):
        """Load vector database from disk"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.case_embeddings = save_data['embeddings']
        self.case_metadata = save_data['metadata']
        
        # Load FAISS index if available
        if self.use_faiss:
            faiss_path = path.with_suffix('.faiss')
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))


class JurisdictionalAnalyzer:
    """Analyzer for jurisdictional authority and binding precedent"""
    
    # Hierarchy weights (higher = more authoritative)
    JURISDICTIONAL_HIERARCHY = {
        'US Supreme Court': 1.0,
        'Federal Circuit Courts': 0.9,
        'Federal District Courts': 0.7,
        'State Supreme Courts': 0.8,
        'State Appellate Courts': 0.6,
        'State Trial Courts': 0.4,
        'Administrative Courts': 0.3,
        'International Courts': 0.2
    }
    
    # Court name patterns for classification
    COURT_PATTERNS = {
        'US Supreme Court': [r'supreme court.*united states', r'scotus', r'u\.?s\.? supreme'],
        'Federal Circuit Courts': [r'circuit.*appeal', r'court.*appeal.*circuit', r'\d+(th|st|nd|rd)\s+circuit'],
        'Federal District Courts': [r'district.*court', r'eastern.*district', r'western.*district'],
        'State Supreme Courts': [r'supreme court.*[state]', r'court.*appeal.*[state]'],
        'State Appellate Courts': [r'appellate.*court', r'court.*appeal'],
        'State Trial Courts': [r'trial.*court', r'district.*court.*[state]', r'county.*court'],
        'Administrative Courts': [r'administrative', r'tax.*court', r'immigration'],
        'International Courts': [r'international', r'icj', r'world.*court']
    }
    
    def __init__(self):
        self.court_cache = {}
    
    def classify_court(self, court_name: str) -> str:
        """Classify court type and authority level"""
        if court_name in self.court_cache:
            return self.court_cache[court_name]
        
        court_lower = court_name.lower()
        
        for court_type, patterns in self.COURT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, court_lower):
                    self.court_cache[court_name] = court_type
                    return court_type
        
        # Default classification
        self.court_cache[court_name] = 'State Trial Courts'
        return 'State Trial Courts'
    
    def calculate_jurisdictional_weight(
        self, 
        case: CaseOpinion,
        target_jurisdiction: Optional[JurisdictionType] = None
    ) -> float:
        """Calculate jurisdictional weight for precedential value"""
        court_type = self.classify_court(case.court)
        base_weight = self.JURISDICTIONAL_HIERARCHY.get(court_type, 0.3)
        
        # Adjust for jurisdiction match
        if target_jurisdiction and case.jurisdiction:
            if case.jurisdiction == target_jurisdiction:
                base_weight *= 1.2  # Boost for same jurisdiction
            elif case.jurisdiction == JurisdictionType.FEDERAL:
                base_weight *= 1.1  # Federal cases have broader authority
        
        # Temporal decay (older cases have less weight)
        if case.date_decided:
            years_old = (datetime.now().date() - case.date_decided).days / 365.25
            temporal_factor = max(0.5, 1.0 - (years_old * 0.02))  # 2% decay per year, min 50%
            base_weight *= temporal_factor
        
        return min(base_weight, 1.0)
    
    def is_binding_precedent(
        self,
        precedent_case: CaseOpinion,
        current_jurisdiction: JurisdictionType,
        current_court: str
    ) -> Tuple[bool, str]:
        """Determine if a case is binding precedent"""
        precedent_court_type = self.classify_court(precedent_case.court)
        current_court_type = self.classify_court(current_court)
        
        # Supreme Court cases are binding on all lower courts
        if precedent_court_type == 'US Supreme Court':
            return True, "binding_supreme_court"
        
        # Federal circuit decisions binding on district courts in same circuit
        if (precedent_court_type == 'Federal Circuit Courts' and 
            current_court_type == 'Federal District Courts' and
            current_jurisdiction == JurisdictionType.FEDERAL):
            # Would need circuit matching logic here
            return True, "binding_circuit"
        
        # State supreme court binding on lower state courts
        if (precedent_court_type == 'State Supreme Courts' and
            current_jurisdiction == JurisdictionType.STATE and
            precedent_case.jurisdiction == current_jurisdiction):
            return True, "binding_state_supreme"
        
        # Otherwise persuasive
        return False, "persuasive"


class PrecedentAnalyzer:
    """
    Main precedent analysis engine.
    
    Provides case similarity analysis, precedent ranking, and 
    Shepardizing-equivalent functionality.
    """
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        
        # Initialize components
        self.embedding_model = LegalEmbeddingModel(config.embedding_model)
        self.vector_db = CaseVectorDatabase(self.embedding_model)
        self.jurisdictional_analyzer = JurisdictionalAnalyzer()
        
        # NLP components
        self._nlp = None
        self._legal_tfidf = None
        
        # Cache for expensive operations
        self._similarity_cache = {}
        
        logger.info("Precedent analyzer initialized")
    
    @property
    def nlp(self):
        """Lazy-loaded spaCy model"""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, downloading...")
                spacy.cli.download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    def add_cases_to_index(self, cases: List[CaseOpinion]):
        """Add cases to the search index"""
        self.vector_db.add_cases(cases)
        logger.info(f"Added {len(cases)} cases to precedent index")
    
    def find_similar_precedents(
        self,
        reference_case: Union[str, CaseOpinion],
        max_results: int = 10,
        similarity_threshold: float = None
    ) -> List[PrecedentMatch]:
        """
        Find similar legal precedents for a reference case
        
        Args:
            reference_case: Case name/citation or CaseOpinion object
            max_results: Maximum number of precedents to return
            similarity_threshold: Minimum similarity score (uses config default)
            
        Returns:
            List of PrecedentMatch objects with similarity analysis
        """
        if similarity_threshold is None:
            similarity_threshold = self.config.precedent_threshold
        
        logger.info(f"Finding precedents for: {reference_case}")
        
        # Search for similar cases
        similar_results = self.vector_db.search_similar(
            reference_case, 
            k=max_results * 2,  # Get more results to filter and rank
            threshold=similarity_threshold
        )
        
        if not similar_results:
            logger.info("No similar precedents found")
            return []
        
        # Convert to PrecedentMatch objects with detailed analysis
        precedent_matches = []
        
        for metadata, similarity_score in similar_results[:max_results]:
            case = metadata['case_object']
            
            # Create precedent object
            precedent = self._create_precedent(case, reference_case)
            
            # Analyze the match
            match_analysis = self._analyze_precedent_match(
                reference_case, case, similarity_score
            )
            
            precedent_match = PrecedentMatch(
                query_case=str(reference_case),
                matched_precedent=precedent,
                match_confidence=similarity_score,
                analysis_method="semantic_similarity",
                supporting_evidence=match_analysis['supporting_evidence'],
                distinguishing_factors=match_analysis['distinguishing_factors'],
                recommendation=match_analysis['recommendation']
            )
            
            precedent_matches.append(precedent_match)
        
        # Sort by combined relevance score
        precedent_matches.sort(key=lambda x: x.matched_precedent.precedential_value, reverse=True)
        
        logger.info(f"Found {len(precedent_matches)} precedent matches")
        return precedent_matches
    
    def _create_precedent(
        self, 
        case: CaseOpinion, 
        reference_case: Union[str, CaseOpinion]
    ) -> Precedent:
        """Create Precedent object with authority analysis"""
        # Calculate jurisdictional weight
        jurisdictional_weight = self.jurisdictional_analyzer.calculate_jurisdictional_weight(case)
        
        # Calculate temporal relevance (how recent/current the case is)
        temporal_relevance = self._calculate_temporal_relevance(case)
        
        # Calculate subject matter relevance
        subject_matter_relevance = self._calculate_subject_matter_relevance(
            case, reference_case
        )
        
        # Overall precedential value
        precedential_value = (
            jurisdictional_weight * 0.4 +
            temporal_relevance * 0.3 +
            subject_matter_relevance * 0.3
        )
        
        # Determine binding strength
        binding_strength = self._determine_binding_strength(case)
        
        return Precedent(
            case_opinion=case,
            binding_strength=binding_strength,
            precedential_value=precedential_value,
            jurisdictional_weight=jurisdictional_weight,
            temporal_relevance=temporal_relevance,
            subject_matter_relevance=subject_matter_relevance,
            distinguishable_factors=self._identify_distinguishing_factors(case, reference_case),
            supporting_rationale=self._extract_supporting_rationale(case)
        )
    
    def _analyze_precedent_match(
        self,
        reference_case: Union[str, CaseOpinion],
        matched_case: CaseOpinion,
        similarity_score: float
    ) -> Dict[str, Any]:
        """Analyze the quality and relevance of a precedent match"""
        
        supporting_evidence = []
        distinguishing_factors = []
        
        # Analyze factual similarity
        if isinstance(reference_case, CaseOpinion) and matched_case.key_facts:
            ref_facts = reference_case.key_facts if reference_case.key_facts else []
            factual_overlap = self._calculate_factual_overlap(ref_facts, matched_case.key_facts)
            if factual_overlap > 0.3:
                supporting_evidence.append(f"Similar factual patterns (overlap: {factual_overlap:.2f})")
        
        # Analyze legal issues
        if isinstance(reference_case, CaseOpinion) and matched_case.legal_issues:
            ref_issues = reference_case.legal_issues if reference_case.legal_issues else []
            legal_overlap = self._calculate_legal_issue_overlap(ref_issues, matched_case.legal_issues)
            if legal_overlap > 0.4:
                supporting_evidence.append(f"Similar legal issues (overlap: {legal_overlap:.2f})")
        
        # Check for subject area alignment
        if isinstance(reference_case, CaseOpinion) and matched_case.subject_areas:
            ref_subjects = reference_case.subject_areas if reference_case.subject_areas else []
            subject_overlap = len(set(ref_subjects) & set(matched_case.subject_areas))
            if subject_overlap > 0:
                supporting_evidence.append(f"{subject_overlap} shared subject area(s)")
        
        # Check for potential distinguishing factors
        if matched_case.date_decided and isinstance(reference_case, CaseOpinion):
            if reference_case.date_decided:
                time_diff = abs((matched_case.date_decided - reference_case.date_decided).days)
                if time_diff > 3650:  # 10 years
                    distinguishing_factors.append(f"Significant time gap ({time_diff // 365} years)")
        
        # Jurisdictional differences
        if (isinstance(reference_case, CaseOpinion) and 
            matched_case.jurisdiction != reference_case.jurisdiction):
            distinguishing_factors.append(f"Different jurisdictions: {matched_case.jurisdiction} vs {reference_case.jurisdiction}")
        
        # Determine recommendation
        if similarity_score >= 0.8 and len(supporting_evidence) >= 2:
            recommendation = "highly_relevant"
        elif similarity_score >= 0.6 and len(supporting_evidence) >= 1:
            recommendation = "relevant"
        elif len(distinguishing_factors) > len(supporting_evidence):
            recommendation = "review_carefully"
        else:
            recommendation = "review_manually"
        
        return {
            'supporting_evidence': supporting_evidence,
            'distinguishing_factors': distinguishing_factors,
            'recommendation': recommendation
        }
    
    def _calculate_temporal_relevance(self, case: CaseOpinion) -> float:
        """Calculate how temporally relevant a case is"""
        if not case.date_decided:
            return 0.5  # Unknown date gets medium relevance
        
        years_old = (datetime.now().date() - case.date_decided).days / 365.25
        
        # Legal cases maintain relevance longer than other domains
        if years_old <= 5:
            return 1.0
        elif years_old <= 15:
            return 0.9
        elif years_old <= 30:
            return 0.7
        else:
            return 0.5
    
    def _calculate_subject_matter_relevance(
        self, 
        case: CaseOpinion, 
        reference_case: Union[str, CaseOpinion]
    ) -> float:
        """Calculate subject matter relevance between cases"""
        if not isinstance(reference_case, CaseOpinion):
            return 0.5  # Cannot compare subjects for string queries
        
        if not case.subject_areas or not reference_case.subject_areas:
            return 0.5  # Unknown subjects
        
        # Calculate Jaccard similarity of subject areas
        case_subjects = set(case.subject_areas)
        ref_subjects = set(reference_case.subject_areas)
        
        intersection = len(case_subjects & ref_subjects)
        union = len(case_subjects | ref_subjects)
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_binding_strength(self, case: CaseOpinion) -> str:
        """Determine the binding strength of a precedent"""
        court_type = self.jurisdictional_analyzer.classify_court(case.court)
        
        if court_type == 'US Supreme Court':
            return "binding_supreme"
        elif court_type in ['Federal Circuit Courts', 'State Supreme Courts']:
            return "binding_appellate"
        elif court_type in ['Federal District Courts', 'State Appellate Courts']:
            return "persuasive_strong"
        else:
            return "persuasive"
    
    def _identify_distinguishing_factors(
        self, 
        case: CaseOpinion, 
        reference_case: Union[str, CaseOpinion]
    ) -> List[str]:
        """Identify factors that distinguish this case from the reference"""
        factors = []
        
        if isinstance(reference_case, CaseOpinion):
            # Jurisdictional differences
            if case.jurisdiction != reference_case.jurisdiction:
                factors.append(f"Different jurisdiction: {case.jurisdiction}")
            
            # Court level differences
            case_court_type = self.jurisdictional_analyzer.classify_court(case.court)
            ref_court_type = self.jurisdictional_analyzer.classify_court(reference_case.court)
            if case_court_type != ref_court_type:
                factors.append(f"Different court level: {case_court_type}")
            
            # Time period differences
            if case.date_decided and reference_case.date_decided:
                time_diff = abs((case.date_decided - reference_case.date_decided).days)
                if time_diff > 1825:  # 5 years
                    factors.append(f"Decided {time_diff // 365} years apart")
        
        # Case status issues
        if case.overruled:
            factors.append("Case has been overruled")
        if case.reversed:
            factors.append("Case has been reversed")
        
        return factors
    
    def _extract_supporting_rationale(self, case: CaseOpinion) -> List[str]:
        """Extract key supporting rationale from the case"""
        rationale = []
        
        if case.holdings:
            rationale.extend([f"Holding: {holding}" for holding in case.holdings[:3]])
        
        if case.legal_issues:
            rationale.extend([f"Legal Issue: {issue}" for issue in case.legal_issues[:2]])
        
        # Extract key sentences from summary if available
        if case.summary and len(case.summary) > 100:
            summary_sentences = case.summary.split('.')[:3]
            rationale.extend([s.strip() for s in summary_sentences if len(s.strip()) > 20])
        
        return rationale
    
    def _calculate_factual_overlap(self, facts1: List[str], facts2: List[str]) -> float:
        """Calculate factual overlap between two sets of facts"""
        if not facts1 or not facts2:
            return 0.0
        
        # Use TF-IDF for more sophisticated similarity
        all_facts = facts1 + facts2
        
        if self._legal_tfidf is None:
            self._legal_tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        
        try:
            tfidf_matrix = self._legal_tfidf.fit_transform(all_facts)
            
            # Calculate similarity between fact sets
            facts1_vectors = tfidf_matrix[:len(facts1)]
            facts2_vectors = tfidf_matrix[len(facts1):]
            
            similarity_matrix = cosine_similarity(facts1_vectors, facts2_vectors)
            
            # Return maximum similarity
            return float(np.max(similarity_matrix))
        except:
            # Fallback to simple word overlap
            words1 = set(' '.join(facts1).lower().split())
            words2 = set(' '.join(facts2).lower().split())
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
    
    def _calculate_legal_issue_overlap(self, issues1: List[str], issues2: List[str]) -> float:
        """Calculate legal issue overlap between two sets of issues"""
        return self._calculate_factual_overlap(issues1, issues2)  # Same logic
    
    def shepardize_case(self, case_citation: str) -> Dict[str, Any]:
        """
        Perform Shepardizing-equivalent analysis on a case
        
        Args:
            case_citation: Citation of case to analyze
            
        Returns:
            Dictionary with case treatment information
        """
        logger.info(f"Shepardizing case: {case_citation}")
        
        # Search for the case in our database
        search_results = self.vector_db.search_similar(case_citation, k=1, threshold=0.8)
        
        if not search_results:
            logger.warning(f"Case not found in database: {case_citation}")
            return {
                'case_citation': case_citation,
                'found': False,
                'treatment': 'unknown',
                'citing_cases': [],
                'treatment_summary': 'Case not found in database'
            }
        
        case_metadata, _ = search_results[0]
        case = case_metadata['case_object']
        
        # Analyze treatment
        treatment_analysis = {
            'case_citation': case_citation,
            'case_name': case.case_name,
            'found': True,
            'current_status': self._determine_case_status(case),
            'citing_cases': case.citing_cases[:20] if case.citing_cases else [],
            'cited_cases': case.cited_cases[:10] if case.cited_cases else [],
            'overruled': case.overruled,
            'reversed': case.reversed,
            'treatment_summary': self._generate_treatment_summary(case)
        }
        
        # Find cases that cite this case for more analysis
        citing_analysis = self._analyze_citing_cases(case)
        treatment_analysis.update(citing_analysis)
        
        return treatment_analysis
    
    def _determine_case_status(self, case: CaseOpinion) -> str:
        """Determine the current status of a case"""
        if case.overruled:
            return "overruled"
        elif case.reversed:
            return "reversed"
        else:
            return "good_law"
    
    def _generate_treatment_summary(self, case: CaseOpinion) -> str:
        """Generate a summary of case treatment"""
        if case.overruled:
            return f"This case has been overruled and is no longer good law."
        elif case.reversed:
            return f"This case has been reversed on appeal."
        elif case.citing_cases and len(case.citing_cases) > 10:
            return f"This case has been frequently cited ({len(case.citing_cases)} citations) and appears to be good law."
        else:
            return f"This case appears to be good law with {len(case.citing_cases) if case.citing_cases else 0} citing cases."
    
    def _analyze_citing_cases(self, case: CaseOpinion) -> Dict[str, Any]:
        """Analyze cases that cite the given case"""
        analysis = {
            'positive_treatment': 0,
            'negative_treatment': 0,
            'neutral_treatment': 0,
            'recent_citations': [],
            'treatment_trend': 'stable'
        }
        
        # This would require more sophisticated analysis of citing cases
        # For now, provide basic metrics
        if case.citing_cases:
            analysis['total_citing_cases'] = len(case.citing_cases)
            
            # Estimate based on citation count and case age
            if len(case.citing_cases) > 50:
                analysis['positive_treatment'] = len(case.citing_cases) * 0.7
                analysis['treatment_trend'] = 'positive'
            elif len(case.citing_cases) > 10:
                analysis['positive_treatment'] = len(case.citing_cases) * 0.6
                analysis['treatment_trend'] = 'stable'
            else:
                analysis['neutral_treatment'] = len(case.citing_cases)
                analysis['treatment_trend'] = 'limited'
        
        return analysis
    
    def rank_precedents_by_relevance(
        self,
        precedents: List[Precedent],
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[Precedent]:
        """
        Rank precedents by relevance score
        
        Args:
            precedents: List of precedents to rank
            query_context: Additional context for ranking (jurisdiction, etc.)
            
        Returns:
            Ranked list of precedents
        """
        def relevance_score(precedent: Precedent) -> float:
            score = precedent.precedential_value
            
            # Boost based on query context
            if query_context:
                target_jurisdiction = query_context.get('jurisdiction')
                if target_jurisdiction and precedent.case_opinion.jurisdiction == target_jurisdiction:
                    score *= 1.2
                
                # Boost recent cases if specified
                if query_context.get('prefer_recent', False):
                    score *= precedent.temporal_relevance
            
            return score
        
        return sorted(precedents, key=relevance_score, reverse=True)
    
    def save_index(self, path: Path):
        """Save the vector database index"""
        self.vector_db.save_to_disk(path)
        logger.info(f"Precedent index saved to {path}")
    
    def load_index(self, path: Path):
        """Load vector database index from disk"""
        self.vector_db.load_from_disk(path)
        logger.info(f"Precedent index loaded from {path}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the precedent index"""
        return {
            'total_cases': self.vector_db.get_case_count(),
            'embedding_model': self.embedding_model.model_name,
            'embedding_dimension': self.embedding_model.embedding_dim,
            'faiss_enabled': self.vector_db.use_faiss
        }


# Convenience function for direct module usage
def find_similar_precedents(
    reference_case: Union[str, CaseOpinion],
    cases_database: List[CaseOpinion],
    max_results: int = 10,
    config: Optional[ResearchConfig] = None
) -> List[PrecedentMatch]:
    """
    Convenience function to find similar precedents
    
    Args:
        reference_case: Case to find precedents for
        cases_database: List of cases to search through
        max_results: Maximum number of results
        config: Research configuration
        
    Returns:
        List of PrecedentMatch objects
    """
    if config is None:
        from .core import ResearchConfig
        config = ResearchConfig()
    
    analyzer = PrecedentAnalyzer(config)
    analyzer.add_cases_to_index(cases_database)
    
    return analyzer.find_similar_precedents(reference_case, max_results)


# Export main classes and functions
__all__ = [
    'PrecedentAnalyzer',
    'LegalEmbeddingModel',
    'CaseVectorDatabase',
    'JurisdictionalAnalyzer',
    'find_similar_precedents'
]