"""
Cross-document entity resolution and linking.
"""

import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
from fuzzywuzzy import fuzz, process
import re
from loguru import logger

from .core import Entity, EntityType, EntityGraph, EntityLinkResult, LanguageCode, NERConfig


class EntityLinker:
    """
    Links and resolves entities across documents
    """
    
    def __init__(self, config: NERConfig):
        """
        Initialize entity linker
        
        Args:
            config: NER configuration
        """
        self.config = config
        self.entity_cache = {}
        self.similarity_cache = {}
        self.vectorizer = None
        self.entity_vectors = {}
        
        logger.info("EntityLinker initialized")
    
    def link_entities(self, source_entity: Entity, candidate_entities: List[Entity]) -> EntityLinkResult:
        """
        Link a source entity to candidate entities
        
        Args:
            source_entity: Entity to find links for
            candidate_entities: Potential matching entities
            
        Returns:
            EntityLinkResult with linked entities and confidence scores
        """
        if not candidate_entities:
            return EntityLinkResult(
                source_entity=source_entity,
                target_entities=[],
                similarity_scores={},
                link_confidence=0.0
            )
        
        logger.debug("Linking entity '{}' against {} candidates", 
                    source_entity.text, len(candidate_entities))
        
        try:
            # Calculate similarity scores
            similarity_scores = {}
            for candidate in candidate_entities:
                # Skip if same entity
                if candidate.entity_id == source_entity.entity_id:
                    continue
                
                # Skip if different types (unless configured otherwise)
                if (source_entity.entity_type != candidate.entity_type and
                    not self._can_link_entity_types(source_entity.entity_type, candidate.entity_type)):
                    continue
                
                score = self._calculate_similarity(source_entity, candidate)
                similarity_scores[candidate.entity_id] = score
            
            # Filter by threshold
            filtered_candidates = [
                entity for entity in candidate_entities
                if similarity_scores.get(entity.entity_id, 0.0) >= self.config.similarity_threshold
            ]
            
            # Sort by similarity score
            filtered_candidates.sort(
                key=lambda e: similarity_scores.get(e.entity_id, 0.0), 
                reverse=True
            )
            
            # Calculate overall link confidence
            max_score = max(similarity_scores.values()) if similarity_scores else 0.0
            link_confidence = max_score
            
            # Determine link type
            link_type = self._determine_link_type(source_entity, filtered_candidates, similarity_scores)
            
            result = EntityLinkResult(
                source_entity=source_entity,
                target_entities=filtered_candidates,
                similarity_scores=similarity_scores,
                link_confidence=link_confidence,
                link_type=link_type,
                explanation=self._generate_link_explanation(source_entity, filtered_candidates, similarity_scores)
            )
            
            logger.debug("Linked entity '{}' to {} targets with confidence {:.3f}", 
                        source_entity.text, len(filtered_candidates), link_confidence)
            
            return result
            
        except Exception as e:
            logger.error("Error linking entities: {}", e)
            return EntityLinkResult(
                source_entity=source_entity,
                target_entities=[],
                similarity_scores={},
                link_confidence=0.0,
                explanation=f"Error during linking: {str(e)}"
            )
    
    def create_entity_graph(self, entities: List[Entity]) -> EntityGraph:
        """
        Create entity graph with relationships
        
        Args:
            entities: List of entities to analyze
            
        Returns:
            EntityGraph with entities and relationships
        """
        logger.info("Creating entity graph from {} entities", len(entities))
        
        entity_graph = EntityGraph()
        
        # Group entities by type for more efficient processing
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.entity_type].append(entity)
            entity_graph.add_entity(entity)
        
        # Find relationships within each entity type
        relationships_found = 0
        for entity_type, type_entities in entities_by_type.items():
            logger.debug("Processing {} {} entities", len(type_entities), entity_type.value)
            
            # Link entities within the same type
            for i, source_entity in enumerate(type_entities):
                candidates = type_entities[i+1:]  # Avoid duplicate comparisons
                
                if len(candidates) > self.config.max_link_distance:
                    # Limit candidates for performance
                    candidates = candidates[:self.config.max_link_distance]
                
                link_result = self.link_entities(source_entity, candidates)
                
                # Add relationships for linked entities
                for target_entity in link_result.target_entities:
                    confidence = link_result.similarity_scores.get(target_entity.entity_id, 0.0)
                    
                    entity_graph.add_relationship(
                        source_id=source_entity.entity_id,
                        target_id=target_entity.entity_id,
                        relationship_type=link_result.link_type,
                        confidence=confidence,
                        metadata={
                            "similarity_method": "multi_feature",
                            "entity_types": [source_entity.entity_type.value, target_entity.entity_type.value]
                        }
                    )
                    relationships_found += 1
        
        # Find cross-type relationships (e.g., PERSON -> ORGANIZATION)
        cross_type_relationships = self._find_cross_type_relationships(entities_by_type)
        for relationship in cross_type_relationships:
            entity_graph.add_relationship(**relationship)
            relationships_found += 1
        
        # Add graph metadata
        entity_graph.metadata.update({
            "total_entities": len(entities),
            "entity_type_counts": {et.value: len(ents) for et, ents in entities_by_type.items()},
            "total_relationships": relationships_found,
            "similarity_threshold": self.config.similarity_threshold,
            "processing_timestamp": entity_graph.created_at.isoformat()
        })
        
        logger.info("Entity graph created: {} entities, {} relationships", 
                   len(entity_graph.entities), len(entity_graph.relationships))
        
        return entity_graph
    
    def resolve_coreferences(self, entities: List[Entity]) -> List[List[Entity]]:
        """
        Group entities that refer to the same real-world entity
        
        Args:
            entities: List of entities to resolve
            
        Returns:
            List of entity clusters (co-reference chains)
        """
        logger.info("Resolving coreferences for {} entities", len(entities))
        
        if not entities:
            return []
        
        # Build similarity matrix
        n = len(entities)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = self._calculate_similarity(entities[i], entities[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Matrix is symmetric
        
        # Use hierarchical clustering to group similar entities
        clusters = self._cluster_entities(entities, similarity_matrix)
        
        # Filter clusters by minimum similarity threshold
        filtered_clusters = []
        for cluster in clusters:
            if len(cluster) > 1:  # Only keep clusters with multiple entities
                # Check if cluster meets similarity threshold
                cluster_similarities = []
                for i in range(len(cluster)):
                    for j in range(i+1, len(cluster)):
                        idx_i = entities.index(cluster[i])
                        idx_j = entities.index(cluster[j])
                        cluster_similarities.append(similarity_matrix[idx_i][idx_j])
                
                if cluster_similarities and np.mean(cluster_similarities) >= self.config.similarity_threshold:
                    filtered_clusters.append(cluster)
        
        logger.info("Found {} co-reference clusters", len(filtered_clusters))
        return filtered_clusters
    
    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate entities, keeping the highest confidence version
        
        Args:
            entities: List of entities to deduplicate
            
        Returns:
            Deduplicated list of entities
        """
        logger.info("Deduplicating {} entities", len(entities))
        
        if not entities:
            return []
        
        # Find co-reference clusters
        clusters = self.resolve_coreferences(entities)
        
        # Keep the highest confidence entity from each cluster
        deduplicated = []
        clustered_entity_ids = set()
        
        for cluster in clusters:
            # Sort by confidence and keep the best one
            best_entity = max(cluster, key=lambda e: e.confidence)
            deduplicated.append(best_entity)
            
            # Mark all entities in cluster as processed
            for entity in cluster:
                clustered_entity_ids.add(entity.entity_id)
            
            logger.debug("Cluster of {} entities -> kept '{}'", 
                        len(cluster), best_entity.text)
        
        # Add entities that weren't clustered
        for entity in entities:
            if entity.entity_id not in clustered_entity_ids:
                deduplicated.append(entity)
        
        logger.info("Deduplication complete: {} -> {} entities", 
                   len(entities), len(deduplicated))
        
        return deduplicated
    
    def _calculate_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between two entities"""
        # Check cache first
        cache_key = f"{entity1.entity_id}_{entity2.entity_id}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Multiple similarity metrics
        similarities = []
        
        # 1. Exact text match
        if entity1.text.lower() == entity2.text.lower():
            similarities.append(1.0)
        else:
            # 2. Fuzzy string matching
            fuzzy_score = fuzz.ratio(entity1.text.lower(), entity2.text.lower()) / 100.0
            similarities.append(fuzzy_score)
            
            # 3. Token-based similarity
            token_score = fuzz.token_sort_ratio(entity1.text.lower(), entity2.text.lower()) / 100.0
            similarities.append(token_score)
        
        # 4. Phonetic similarity (for names)
        if entity1.entity_type == EntityType.PERSON and entity2.entity_type == EntityType.PERSON:
            phonetic_score = self._calculate_phonetic_similarity(entity1.text, entity2.text)
            similarities.append(phonetic_score)
        
        # 5. Context similarity
        if entity1.context and entity2.context:
            context_score = self._calculate_context_similarity(entity1.context, entity2.context)
            similarities.append(context_score * 0.5)  # Lower weight for context
        
        # 6. Alias matching
        alias_score = self._calculate_alias_similarity(entity1, entity2)
        if alias_score > 0:
            similarities.append(alias_score)
        
        # 7. Cross-language similarity
        if entity1.language != entity2.language:
            cross_lang_score = self._calculate_cross_language_similarity(entity1, entity2)
            similarities.append(cross_lang_score * 0.8)  # Slightly lower weight
        
        # 8. Position-based similarity (if from same document)
        if entity1.document_id == entity2.document_id:
            position_score = self._calculate_position_similarity(entity1, entity2)
            similarities.append(position_score * 0.3)  # Lower weight for position
        
        # Calculate weighted average
        if similarities:
            final_score = np.mean(similarities)
        else:
            final_score = 0.0
        
        # Cache result
        self.similarity_cache[cache_key] = final_score
        self.similarity_cache[f"{entity2.entity_id}_{entity1.entity_id}"] = final_score
        
        return final_score
    
    def _calculate_phonetic_similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity using Soundex and Metaphone"""
        try:
            # Soundex similarity
            soundex1 = jellyfish.soundex(text1)
            soundex2 = jellyfish.soundex(text2)
            soundex_match = 1.0 if soundex1 == soundex2 else 0.0
            
            # Metaphone similarity
            metaphone1 = jellyfish.metaphone(text1)
            metaphone2 = jellyfish.metaphone(text2)
            metaphone_match = 1.0 if metaphone1 == metaphone2 else 0.0
            
            # NYSIIS similarity
            nysiis1 = jellyfish.nysiis(text1)
            nysiis2 = jellyfish.nysiis(text2)
            nysiis_match = 1.0 if nysiis1 == nysiis2 else 0.0
            
            return np.mean([soundex_match, metaphone_match, nysiis_match])
        except:
            return 0.0
    
    def _calculate_context_similarity(self, context1: str, context2: str) -> float:
        """Calculate similarity between entity contexts using TF-IDF"""
        try:
            if not hasattr(self, '_context_vectorizer'):
                self._context_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2)
                )
            
            # Fit and transform both contexts
            contexts = [context1, context2]
            tfidf_matrix = self._context_vectorizer.fit_transform(contexts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix[0][1]
        except:
            return 0.0
    
    def _calculate_alias_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Check if entities match through aliases"""
        # Check if entity texts are in each other's aliases
        text1_lower = entity1.text.lower()
        text2_lower = entity2.text.lower()
        
        aliases1 = [alias.lower() for alias in entity1.aliases]
        aliases2 = [alias.lower() for alias in entity2.aliases]
        
        # Direct alias match
        if text1_lower in aliases2 or text2_lower in aliases1:
            return 1.0
        
        # Check alias-to-alias matches
        for alias1 in aliases1:
            for alias2 in aliases2:
                if fuzz.ratio(alias1, alias2) >= 90:  # High threshold for alias matching
                    return 0.9
        
        return 0.0
    
    def _calculate_cross_language_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between entities in different languages"""
        # This would require translation or transliteration
        # For now, use basic character-level similarity for names
        
        if (entity1.entity_type == EntityType.PERSON and entity2.entity_type == EntityType.PERSON) or \
           (entity1.entity_type == EntityType.LOCATION and entity2.entity_type == EntityType.LOCATION):
            
            # Use Levenshtein distance for cross-language name similarity
            try:
                distance = jellyfish.levenshtein_distance(entity1.text.lower(), entity2.text.lower())
                max_len = max(len(entity1.text), len(entity2.text))
                if max_len == 0:
                    return 0.0
                similarity = 1.0 - (distance / max_len)
                return max(0.0, similarity)
            except:
                return 0.0
        
        return 0.0
    
    def _calculate_position_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity based on positions in the same document"""
        if entity1.document_id != entity2.document_id:
            return 0.0
        
        # Entities that appear close together are more likely to be related
        distance = abs(entity1.start_pos - entity2.start_pos)
        
        # Normalize by context window
        max_distance = self.config.context_window * 10
        if distance >= max_distance:
            return 0.0
        
        return 1.0 - (distance / max_distance)
    
    def _can_link_entity_types(self, type1: EntityType, type2: EntityType) -> bool:
        """Check if two entity types can be linked"""
        # Allow linking between related types
        related_types = {
            (EntityType.PERSON, EntityType.ORGANIZATION),
            (EntityType.ORGANIZATION, EntityType.LOCATION),
            (EntityType.COURT, EntityType.LOCATION),
            (EntityType.LEGAL_ENTITY, EntityType.ORGANIZATION),
            (EntityType.CASE_NAME, EntityType.COURT)
        }
        
        return (type1, type2) in related_types or (type2, type1) in related_types
    
    def _determine_link_type(self, source_entity: Entity, target_entities: List[Entity], 
                           similarity_scores: Dict[str, float]) -> str:
        """Determine the type of relationship between entities"""
        if not target_entities:
            return "NONE"
        
        max_score = max(similarity_scores.values())
        
        if max_score >= 0.95:
            return "IDENTICAL"
        elif max_score >= 0.8:
            return "SIMILAR"
        elif max_score >= 0.6:
            return "RELATED"
        else:
            return "WEAK_LINK"
    
    def _generate_link_explanation(self, source_entity: Entity, target_entities: List[Entity],
                                 similarity_scores: Dict[str, float]) -> str:
        """Generate human-readable explanation for entity links"""
        if not target_entities:
            return f"No entities similar to '{source_entity.text}' found above threshold {self.config.similarity_threshold}"
        
        explanations = []
        for target in target_entities[:3]:  # Limit to top 3
            score = similarity_scores.get(target.entity_id, 0.0)
            explanations.append(f"'{target.text}' (score: {score:.3f})")
        
        return f"'{source_entity.text}' linked to: " + ", ".join(explanations)
    
    def _cluster_entities(self, entities: List[Entity], similarity_matrix: np.ndarray) -> List[List[Entity]]:
        """Cluster entities using similarity matrix"""
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Convert similarity to distance
            distance_matrix = 1.0 - similarity_matrix
            
            # Use agglomerative clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.0 - self.config.similarity_threshold,
                metric='precomputed',
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group entities by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(entities[i])
            
            return list(clusters.values())
            
        except ImportError:
            logger.warning("scikit-learn clustering not available, using simple threshold clustering")
            return self._simple_threshold_clustering(entities, similarity_matrix)
    
    def _simple_threshold_clustering(self, entities: List[Entity], 
                                   similarity_matrix: np.ndarray) -> List[List[Entity]]:
        """Simple threshold-based clustering fallback"""
        clusters = []
        visited = set()
        
        for i, entity in enumerate(entities):
            if i in visited:
                continue
            
            # Start new cluster
            cluster = [entity]
            visited.add(i)
            
            # Find similar entities
            for j in range(i+1, len(entities)):
                if j not in visited and similarity_matrix[i][j] >= self.config.similarity_threshold:
                    cluster.append(entities[j])
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _find_cross_type_relationships(self, entities_by_type: Dict[EntityType, List[Entity]]) -> List[Dict[str, Any]]:
        """Find relationships between different entity types"""
        relationships = []
        
        # Define relationship patterns
        relationship_patterns = [
            (EntityType.PERSON, EntityType.ORGANIZATION, "WORKS_FOR"),
            (EntityType.PERSON, EntityType.LOCATION, "LOCATED_IN"),
            (EntityType.ORGANIZATION, EntityType.LOCATION, "BASED_IN"),
            (EntityType.COURT, EntityType.LOCATION, "LOCATED_IN"),
            (EntityType.CASE_NAME, EntityType.COURT, "HEARD_BY"),
            (EntityType.LEGAL_ENTITY, EntityType.ORGANIZATION, "IS_A")
        ]
        
        for source_type, target_type, rel_type in relationship_patterns:
            if source_type not in entities_by_type or target_type not in entities_by_type:
                continue
            
            source_entities = entities_by_type[source_type]
            target_entities = entities_by_type[target_type]
            
            # Find relationships based on context proximity
            for source in source_entities:
                for target in target_entities:
                    # Check if entities appear in similar contexts or same document
                    if self._entities_contextually_related(source, target):
                        relationships.append({
                            "source_id": source.entity_id,
                            "target_id": target.entity_id,
                            "relationship_type": rel_type,
                            "confidence": 0.6,  # Lower confidence for inferred relationships
                            "metadata": {
                                "inferred": True,
                                "source_type": source_type.value,
                                "target_type": target_type.value
                            }
                        })
        
        return relationships
    
    def _entities_contextually_related(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if entities are contextually related"""
        # Same document
        if entity1.document_id == entity2.document_id:
            # Check proximity
            distance = abs(entity1.start_pos - entity2.start_pos)
            if distance <= self.config.context_window * 5:  # Within extended context
                return True
        
        # Similar contexts
        if entity1.context and entity2.context:
            context_sim = self._calculate_context_similarity(entity1.context, entity2.context)
            if context_sim >= 0.3:  # Lower threshold for cross-type relationships
                return True
        
        return False