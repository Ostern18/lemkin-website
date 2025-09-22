"""
Event sequencing module for chronological ordering and uncertainty handling.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
import logging
from loguru import logger

from .core import (
    Event, Timeline, TemporalEntity, TimelineEventType, 
    TimelineConfig, LanguageCode
)


@dataclass
class EventRelationship:
    """Represents a relationship between two events"""
    source_event_id: str
    target_event_id: str
    relationship_type: str  # 'before', 'after', 'concurrent', 'contains', 'overlaps'
    confidence: float
    temporal_distance: Optional[timedelta] = None
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class UncertaintyHandler:
    """Handles temporal uncertainty in event sequencing"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.uncertainty_window = timedelta(hours=config.uncertainty_window_hours)
    
    def calculate_temporal_uncertainty(self, event: Event) -> Tuple[datetime, datetime]:
        """
        Calculate temporal uncertainty range for an event
        
        Args:
            event: Event to analyze
            
        Returns:
            Tuple of (earliest_possible_time, latest_possible_time)
        """
        if event.uncertainty_range:
            return event.uncertainty_range
        
        if event.is_fuzzy:
            # Larger uncertainty for fuzzy events
            uncertainty = self.uncertainty_window * 2
        else:
            uncertainty = self.uncertainty_window
        
        earliest = event.start_time - uncertainty
        latest = event.start_time + uncertainty
        
        # Adjust for event duration if present
        if event.end_time:
            latest = max(latest, event.end_time + uncertainty)
        
        return earliest, latest
    
    def events_might_overlap(self, event1: Event, event2: Event) -> bool:
        """
        Check if two events might temporally overlap given uncertainty
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            True if events might overlap considering uncertainty
        """
        range1 = self.calculate_temporal_uncertainty(event1)
        range2 = self.calculate_temporal_uncertainty(event2)
        
        # Check for overlap between uncertainty ranges
        return (range1[0] <= range2[1] and range2[0] <= range1[1])
    
    def calculate_confidence_overlap(self, event1: Event, event2: Event) -> float:
        """
        Calculate confidence that two events overlap
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Confidence score (0.0 to 1.0) that events overlap
        """
        if not self.events_might_overlap(event1, event2):
            return 0.0
        
        range1 = self.calculate_temporal_uncertainty(event1)
        range2 = self.calculate_temporal_uncertainty(event2)
        
        # Calculate overlap duration vs total span
        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        
        if overlap_duration <= 0:
            return 0.0
        
        # Calculate total span
        total_start = min(range1[0], range2[0])
        total_end = max(range1[1], range2[1])
        total_duration = (total_end - total_start).total_seconds()
        
        if total_duration == 0:
            return 1.0
        
        # Confidence based on overlap ratio and event confidences
        overlap_ratio = overlap_duration / total_duration
        combined_confidence = (event1.confidence + event2.confidence) / 2
        
        return overlap_ratio * combined_confidence
    
    def resolve_uncertain_ordering(self, events: List[Event]) -> List[Event]:
        """
        Resolve ordering of events with temporal uncertainty
        
        Args:
            events: List of events to order
            
        Returns:
            List of events in resolved chronological order
        """
        if len(events) <= 1:
            return events
        
        # Create graph for topological sorting with uncertainty
        graph = nx.DiGraph()
        
        # Add all events as nodes
        for event in events:
            graph.add_node(event.event_id, event=event)
        
        # Add edges based on temporal relationships with confidence
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                relationship = self._determine_temporal_relationship(event1, event2)
                if relationship:
                    if relationship['type'] == 'before':
                        graph.add_edge(event1.event_id, event2.event_id, 
                                     confidence=relationship['confidence'])
                    elif relationship['type'] == 'after':
                        graph.add_edge(event2.event_id, event1.event_id,
                                     confidence=relationship['confidence'])
        
        # Perform topological sort with confidence weighting
        try:
            ordered_ids = list(nx.topological_sort(graph))
            return [graph.nodes[event_id]['event'] for event_id in ordered_ids]
        except nx.NetworkXError:
            # Cycle detected - use confidence-based resolution
            logger.warning("Temporal cycle detected, using confidence-based ordering")
            return self._resolve_temporal_cycles(events, graph)
    
    def _determine_temporal_relationship(self, event1: Event, event2: Event) -> Optional[Dict[str, Any]]:
        """Determine temporal relationship between two events"""
        range1 = self.calculate_temporal_uncertainty(event1)
        range2 = self.calculate_temporal_uncertainty(event2)
        
        # Clear before/after relationship
        if range1[1] < range2[0]:
            confidence = min(event1.confidence, event2.confidence)
            return {'type': 'before', 'confidence': confidence}
        elif range2[1] < range1[0]:
            confidence = min(event1.confidence, event2.confidence)
            return {'type': 'after', 'confidence': confidence}
        
        # Potentially overlapping - use most likely centers
        center1 = range1[0] + (range1[1] - range1[0]) / 2
        center2 = range2[0] + (range2[1] - range2[0]) / 2
        
        if center1 < center2:
            # Likely before but with uncertainty
            overlap_conf = self.calculate_confidence_overlap(event1, event2)
            confidence = min(event1.confidence, event2.confidence) * (1 - overlap_conf)
            if confidence > 0.3:  # Threshold for uncertain ordering
                return {'type': 'before', 'confidence': confidence}
        elif center2 < center1:
            overlap_conf = self.calculate_confidence_overlap(event1, event2)
            confidence = min(event1.confidence, event2.confidence) * (1 - overlap_conf)
            if confidence > 0.3:
                return {'type': 'after', 'confidence': confidence}
        
        return None  # Too uncertain to determine order
    
    def _resolve_temporal_cycles(self, events: List[Event], graph: nx.DiGraph) -> List[Event]:
        """Resolve temporal cycles using confidence-based ordering"""
        # Sort by confidence-weighted temporal position
        def sort_key(event):
            uncertainty_range = self.calculate_temporal_uncertainty(event)
            center_time = uncertainty_range[0] + (uncertainty_range[1] - uncertainty_range[0]) / 2
            # Weight by confidence
            return (center_time.timestamp(), -event.confidence)
        
        return sorted(events, key=sort_key)


class ChronologicalSorter:
    """Sorts events in chronological order"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.uncertainty_handler = UncertaintyHandler(config)
    
    def sort_events_chronologically(self, events: List[Event]) -> List[Event]:
        """
        Sort events in chronological order handling uncertainty
        
        Args:
            events: List of events to sort
            
        Returns:
            Chronologically sorted list of events
        """
        if not events:
            return []
        
        logger.debug("Sorting {} events chronologically", len(events))
        
        # Handle uncertainty if enabled
        if self.config.enable_uncertainty_handling:
            sorted_events = self.uncertainty_handler.resolve_uncertain_ordering(events)
        else:
            # Simple sort by start time
            sorted_events = sorted(events, key=lambda x: x.start_time)
        
        logger.debug("Chronological sorting completed")
        return sorted_events
    
    def group_concurrent_events(self, events: List[Event]) -> Dict[str, List[Event]]:
        """
        Group events that occur concurrently
        
        Args:
            events: List of chronologically sorted events
            
        Returns:
            Dictionary mapping group IDs to lists of concurrent events
        """
        if not events:
            return {}
        
        concurrent_groups = {}
        current_group_id = None
        current_group = []
        
        for i, event in enumerate(events):
            if i == 0:
                # First event starts first group
                current_group_id = str(uuid.uuid4())
                current_group = [event]
            else:
                # Check if this event is concurrent with any in current group
                is_concurrent = False
                
                for group_event in current_group:
                    if self._are_concurrent(event, group_event):
                        is_concurrent = True
                        break
                
                if is_concurrent:
                    # Add to current group
                    current_group.append(event)
                else:
                    # Start new group
                    if current_group:
                        concurrent_groups[current_group_id] = current_group
                    
                    current_group_id = str(uuid.uuid4())
                    current_group = [event]
        
        # Add final group
        if current_group:
            concurrent_groups[current_group_id] = current_group
        
        return concurrent_groups
    
    def _are_concurrent(self, event1: Event, event2: Event) -> bool:
        """Check if two events are concurrent"""
        if not self.config.allow_overlapping_events:
            return False
        
        # Use uncertainty handler for more sophisticated concurrency detection
        if self.config.enable_uncertainty_handling:
            overlap_confidence = self.uncertainty_handler.calculate_confidence_overlap(event1, event2)
            return overlap_confidence > 0.5  # Threshold for considering concurrent
        
        # Simple overlap check
        if not event1.end_time or not event2.end_time:
            return False
        
        return event1.overlaps_with(event2)


class RelationshipInferrer:
    """Infers relationships between events based on temporal and contextual clues"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        
        # Causal relationship indicators
        self.causal_indicators = {
            'cause_words': [
                'because', 'due to', 'caused by', 'resulted in', 'led to',
                'triggered', 'prompted', 'brought about', 'gave rise to',
                'consequently', 'therefore', 'thus', 'hence', 'as a result'
            ],
            'sequence_words': [
                'then', 'next', 'after', 'following', 'subsequently', 'later',
                'before', 'prior to', 'earlier', 'previously', 'first', 'finally'
            ],
            'concurrent_words': [
                'while', 'during', 'meanwhile', 'simultaneously', 'at the same time',
                'concurrently', 'in parallel', 'alongside'
            ]
        }
    
    def infer_event_relationships(self, events: List[Event]) -> List[EventRelationship]:
        """
        Infer relationships between events
        
        Args:
            events: List of events to analyze
            
        Returns:
            List of inferred relationships
        """
        if not self.config.infer_event_relationships or len(events) < 2:
            return []
        
        logger.debug("Inferring relationships between {} events", len(events))
        
        relationships = []
        
        # Analyze each pair of events
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                relationship = self._analyze_event_pair(event1, event2)
                if relationship:
                    relationships.append(relationship)
        
        logger.debug("Inferred {} relationships", len(relationships))
        return relationships
    
    def _analyze_event_pair(self, event1: Event, event2: Event) -> Optional[EventRelationship]:
        """Analyze a pair of events for relationships"""
        # Temporal relationship analysis
        temporal_rel = self._analyze_temporal_relationship(event1, event2)
        
        # Contextual relationship analysis
        contextual_rel = self._analyze_contextual_relationship(event1, event2)
        
        # Combine analyses
        if temporal_rel or contextual_rel:
            relationship_type = self._determine_relationship_type(temporal_rel, contextual_rel)
            confidence = self._calculate_relationship_confidence(temporal_rel, contextual_rel)
            
            if confidence > 0.4:  # Minimum confidence threshold
                evidence = []
                if temporal_rel:
                    evidence.append(f"temporal: {temporal_rel}")
                if contextual_rel:
                    evidence.extend(contextual_rel.get('evidence', []))
                
                return EventRelationship(
                    source_event_id=event1.event_id,
                    target_event_id=event2.event_id,
                    relationship_type=relationship_type,
                    confidence=confidence,
                    evidence=evidence
                )
        
        return None
    
    def _analyze_temporal_relationship(self, event1: Event, event2: Event) -> Optional[str]:
        """Analyze temporal relationship between events"""
        # Simple temporal ordering
        if event1.end_time and event2.start_time:
            if event1.end_time <= event2.start_time:
                return 'before'
            elif event2.end_time and event2.end_time <= event1.start_time:
                return 'after'
            elif event1.overlaps_with(event2):
                return 'concurrent'
        elif event1.start_time < event2.start_time:
            return 'before'
        elif event2.start_time < event1.start_time:
            return 'after'
        
        return None
    
    def _analyze_contextual_relationship(self, event1: Event, event2: Event) -> Optional[Dict[str, Any]]:
        """Analyze contextual clues for relationships"""
        if not event1.source_text or not event2.source_text:
            return None
        
        # Combine text sources for analysis
        combined_text = f"{event1.source_text} {event2.source_text}"
        combined_text += " ".join(event1.context_sentences + event2.context_sentences)
        text_lower = combined_text.lower()
        
        evidence = []
        relationship_indicators = []
        
        # Check for causal indicators
        for word in self.causal_indicators['cause_words']:
            if word in text_lower:
                relationship_indicators.append('causal')
                evidence.append(f"causal indicator: '{word}'")
        
        # Check for sequence indicators
        for word in self.causal_indicators['sequence_words']:
            if word in text_lower:
                relationship_indicators.append('sequential')
                evidence.append(f"sequence indicator: '{word}'")
        
        # Check for concurrent indicators
        for word in self.causal_indicators['concurrent_words']:
            if word in text_lower:
                relationship_indicators.append('concurrent')
                evidence.append(f"concurrency indicator: '{word}'")
        
        if relationship_indicators:
            return {
                'indicators': relationship_indicators,
                'evidence': evidence,
                'confidence': min(1.0, len(relationship_indicators) * 0.3)
            }
        
        return None
    
    def _determine_relationship_type(self, temporal_rel: Optional[str], 
                                   contextual_rel: Optional[Dict[str, Any]]) -> str:
        """Determine the overall relationship type"""
        if contextual_rel and 'indicators' in contextual_rel:
            indicators = contextual_rel['indicators']
            
            if 'causal' in indicators:
                return 'causes'
            elif 'concurrent' in indicators:
                return 'concurrent'
            elif 'sequential' in indicators:
                return 'before' if temporal_rel == 'before' else 'after'
        
        # Fall back to temporal relationship
        return temporal_rel or 'related'
    
    def _calculate_relationship_confidence(self, temporal_rel: Optional[str],
                                         contextual_rel: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in relationship inference"""
        confidence = 0.0
        
        if temporal_rel:
            confidence += 0.5  # Base temporal confidence
        
        if contextual_rel:
            confidence += contextual_rel.get('confidence', 0.0)
        
        return min(1.0, confidence)


class EventSequencer:
    """Main event sequencing coordinator"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.chronological_sorter = ChronologicalSorter(config)
        self.relationship_inferrer = RelationshipInferrer(config)
        self.uncertainty_handler = UncertaintyHandler(config)
    
    def sequence_events(self, events: List[Event]) -> Timeline:
        """
        Create a chronologically sequenced timeline from events
        
        Args:
            events: List of events to sequence
            
        Returns:
            Timeline with chronologically ordered events
        """
        if not events:
            logger.warning("No events provided for sequencing")
            return Timeline(title="Empty Timeline")
        
        logger.info("Sequencing {} events into timeline", len(events))
        
        try:
            # Limit number of events if necessary
            if len(events) > self.config.max_events_per_timeline:
                logger.warning("Too many events ({}), limiting to {}",
                             len(events), self.config.max_events_per_timeline)
                # Sort by confidence and take top events
                events = sorted(events, key=lambda x: x.confidence, reverse=True)
                events = events[:self.config.max_events_per_timeline]
            
            # Sort events chronologically
            sorted_events = self.chronological_sorter.sort_events_chronologically(events)
            
            # Create timeline
            timeline = Timeline(
                title="Generated Timeline",
                events=sorted_events,
                construction_method="automatic_sequencing"
            )
            
            # Infer and add event relationships
            if self.config.infer_event_relationships:
                relationships = self.relationship_inferrer.infer_event_relationships(sorted_events)
                self._apply_relationships_to_events(sorted_events, relationships)
                
                # Store relationships in timeline metadata
                timeline.processing_metadata['relationships'] = [
                    {
                        'source': rel.source_event_id,
                        'target': rel.target_event_id,
                        'type': rel.relationship_type,
                        'confidence': rel.confidence,
                        'evidence': rel.evidence
                    }
                    for rel in relationships
                ]
            
            # Group concurrent events if enabled
            if self.config.allow_overlapping_events:
                concurrent_groups = self.chronological_sorter.group_concurrent_events(sorted_events)
                timeline.processing_metadata['concurrent_groups'] = {
                    group_id: [event.event_id for event in events]
                    for group_id, events in concurrent_groups.items()
                }
            
            # Update timeline metadata
            timeline.processing_metadata.update({
                'original_event_count': len(events),
                'final_event_count': len(sorted_events),
                'uncertainty_handling': self.config.enable_uncertainty_handling,
                'relationship_inference': self.config.infer_event_relationships,
                'sequencing_method': 'chronological_with_uncertainty' if self.config.enable_uncertainty_handling else 'simple_chronological'
            })
            
            logger.info("Timeline sequencing completed: {} events ordered", len(sorted_events))
            return timeline
            
        except Exception as e:
            logger.error("Error sequencing events: {}", e)
            # Return basic timeline with unsorted events
            return Timeline(
                title="Error Timeline",
                events=events,
                construction_method="error_fallback",
                processing_metadata={'error': str(e)}
            )
    
    def _apply_relationships_to_events(self, events: List[Event], 
                                     relationships: List[EventRelationship]) -> None:
        """Apply inferred relationships to event objects"""
        # Create lookup for events by ID
        event_lookup = {event.event_id: event for event in events}
        
        # Apply relationships
        for rel in relationships:
            source_event = event_lookup.get(rel.source_event_id)
            target_event = event_lookup.get(rel.target_event_id)
            
            if source_event and target_event:
                if rel.relationship_type == 'before':
                    source_event.after_events.append(rel.target_event_id)
                    target_event.before_events.append(rel.source_event_id)
                elif rel.relationship_type == 'after':
                    source_event.before_events.append(rel.target_event_id)
                    target_event.after_events.append(rel.source_event_id)
                elif rel.relationship_type == 'concurrent':
                    source_event.concurrent_events.append(rel.target_event_id)
                    target_event.concurrent_events.append(rel.source_event_id)
                elif rel.relationship_type == 'causes':
                    # Add causal relationship metadata
                    source_event.metadata.setdefault('causes', []).append({
                        'event_id': rel.target_event_id,
                        'confidence': rel.confidence,
                        'evidence': rel.evidence
                    })
                    target_event.metadata.setdefault('caused_by', []).append({
                        'event_id': rel.source_event_id,
                        'confidence': rel.confidence,
                        'evidence': rel.evidence
                    })
    
    def analyze_temporal_patterns(self, timeline: Timeline) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the timeline
        
        Args:
            timeline: Timeline to analyze
            
        Returns:
            Dictionary containing pattern analysis results
        """
        if not timeline.events:
            return {}
        
        logger.debug("Analyzing temporal patterns in timeline with {} events", len(timeline.events))
        
        analysis = {}
        
        # Time span analysis
        start_time = min(event.start_time for event in timeline.events)
        end_time = max(
            event.end_time or event.start_time for event in timeline.events
        )
        total_span = end_time - start_time
        
        analysis['temporal_span'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_days': total_span.days,
            'total_duration_seconds': total_span.total_seconds()
        }
        
        # Event density analysis
        if total_span.total_seconds() > 0:
            events_per_day = len(timeline.events) / (total_span.total_seconds() / 86400)
            analysis['event_density'] = {
                'events_per_day': events_per_day,
                'density_category': self._categorize_density(events_per_day)
            }
        
        # Uncertainty analysis
        uncertain_events = [e for e in timeline.events if e.is_fuzzy]
        analysis['uncertainty'] = {
            'uncertain_event_count': len(uncertain_events),
            'uncertainty_percentage': len(uncertain_events) / len(timeline.events) * 100,
            'avg_confidence': sum(e.confidence for e in timeline.events) / len(timeline.events)
        }
        
        # Concurrent event analysis
        concurrent_count = 0
        for event in timeline.events:
            concurrent_count += len(event.concurrent_events)
        
        analysis['concurrency'] = {
            'concurrent_relationships': concurrent_count // 2,  # Each relationship counted twice
            'events_with_concurrency': len([e for e in timeline.events if e.concurrent_events])
        }
        
        return analysis
    
    def _categorize_density(self, events_per_day: float) -> str:
        """Categorize event density"""
        if events_per_day < 0.1:
            return 'very_sparse'
        elif events_per_day < 1.0:
            return 'sparse'
        elif events_per_day < 5.0:
            return 'moderate'
        elif events_per_day < 20.0:
            return 'dense'
        else:
            return 'very_dense'


# Convenience function for direct usage
def sequence_events(events: List[Event], config: Optional[TimelineConfig] = None) -> Timeline:
    """
    Convenience function to sequence events into a timeline
    
    Args:
        events: List of events to sequence
        config: Timeline configuration (uses default if not provided)
        
    Returns:
        Chronologically sequenced timeline
    """
    if config is None:
        from .core import create_default_timeline_config
        config = create_default_timeline_config()
    
    sequencer = EventSequencer(config)
    return sequencer.sequence_events(events)