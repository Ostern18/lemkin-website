"""
Temporal validation module for consistency checking across timeline sources.
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
    Timeline, Event, TemporalEntity, Inconsistency, ValidationResult,
    InconsistencyType, TimelineEventType, TimelineConfig
)


@dataclass
class ConsistencyRule:
    """Represents a temporal consistency rule"""
    rule_id: str
    rule_type: str  # 'chronological', 'causal', 'duration', 'logical'
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    validation_function: callable
    applicable_event_types: List[TimelineEventType] = None
    
    def __post_init__(self):
        if self.applicable_event_types is None:
            self.applicable_event_types = list(TimelineEventType)


class TemporalLogicEngine:
    """Handles temporal logic and reasoning for consistency checking"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        
        # Temporal logic operators
        self.temporal_operators = {
            'before': self._check_before,
            'after': self._check_after,
            'during': self._check_during,
            'overlaps': self._check_overlaps,
            'meets': self._check_meets,
            'starts': self._check_starts,
            'finishes': self._check_finishes,
            'contains': self._check_contains
        }
    
    def evaluate_temporal_constraint(self, event1: Event, event2: Event,
                                   constraint: str) -> Tuple[bool, float, str]:
        """
        Evaluate temporal constraint between two events
        
        Args:
            event1: First event
            event2: Second event
            constraint: Temporal constraint to check
            
        Returns:
            Tuple of (is_satisfied, confidence, explanation)
        """
        if constraint not in self.temporal_operators:
            return False, 0.0, f"Unknown temporal constraint: {constraint}"
        
        try:
            return self.temporal_operators[constraint](event1, event2)
        except Exception as e:
            logger.error("Error evaluating temporal constraint '{}': {}", constraint, e)
            return False, 0.0, f"Error evaluating constraint: {str(e)}"
    
    def _check_before(self, event1: Event, event2: Event) -> Tuple[bool, float, str]:
        """Check if event1 occurs before event2"""
        if event1.end_time and event2.start_time:
            is_before = event1.end_time <= event2.start_time
            
            if is_before:
                gap = event2.start_time - event1.end_time
                confidence = min(1.0, (event1.confidence + event2.confidence) / 2)
                return True, confidence, f"Events separated by {gap}"
            else:
                overlap = event1.end_time - event2.start_time
                return False, 0.8, f"Events overlap by {overlap}"
        
        elif event1.start_time < event2.start_time:
            # Only start times available
            gap = event2.start_time - event1.start_time
            confidence = min(0.7, (event1.confidence + event2.confidence) / 2)
            return True, confidence, f"Start times separated by {gap}"
        
        else:
            return False, 0.9, "Event2 starts before or at the same time as Event1"
    
    def _check_after(self, event1: Event, event2: Event) -> Tuple[bool, float, str]:
        """Check if event1 occurs after event2"""
        return self._check_before(event2, event1)
    
    def _check_during(self, event1: Event, event2: Event) -> Tuple[bool, float, str]:
        """Check if event1 occurs during event2"""
        if not event2.end_time:
            return False, 0.5, "Event2 has no end time - cannot check containment"
        
        if (event1.start_time >= event2.start_time and 
            (event1.end_time or event1.start_time) <= event2.end_time):
            confidence = min(1.0, (event1.confidence + event2.confidence) / 2)
            return True, confidence, "Event1 occurs entirely within Event2"
        else:
            return False, 0.8, "Event1 extends outside Event2's time bounds"
    
    def _check_overlaps(self, event1: Event, event2: Event) -> Tuple[bool, float, str]:
        """Check if events overlap"""
        if not event1.end_time or not event2.end_time:
            # Can't determine overlap without end times
            same_start = event1.start_time == event2.start_time
            return same_start, 0.3 if same_start else 0.0, "Insufficient time information for overlap check"
        
        overlaps = event1.overlaps_with(event2)
        if overlaps:
            overlap_start = max(event1.start_time, event2.start_time)
            overlap_end = min(event1.end_time, event2.end_time)
            overlap_duration = overlap_end - overlap_start
            confidence = min(1.0, (event1.confidence + event2.confidence) / 2)
            return True, confidence, f"Events overlap for {overlap_duration}"
        else:
            return False, 0.9, "Events do not overlap"
    
    def _check_meets(self, event1: Event, event2: Event) -> Tuple[bool, float, str]:
        """Check if event1 meets event2 (end of event1 = start of event2)"""
        if not event1.end_time:
            return False, 0.5, "Event1 has no end time"
        
        meets = abs((event1.end_time - event2.start_time).total_seconds()) < 60  # 1 minute tolerance
        confidence = min(1.0, (event1.confidence + event2.confidence) / 2) if meets else 0.8
        
        if meets:
            return True, confidence, "Events meet (end of first = start of second)"
        else:
            gap = event2.start_time - event1.end_time
            return False, confidence, f"Gap of {gap} between events"
    
    def _check_starts(self, event1: Event, event2: Event) -> Tuple[bool, float, str]:
        """Check if events start at the same time"""
        time_diff = abs((event1.start_time - event2.start_time).total_seconds())
        starts_same = time_diff < 300  # 5 minute tolerance
        
        confidence = min(1.0, (event1.confidence + event2.confidence) / 2) if starts_same else 0.9
        
        if starts_same:
            return True, confidence, f"Events start within {time_diff} seconds"
        else:
            return False, confidence, f"Events start {time_diff} seconds apart"
    
    def _check_finishes(self, event1: Event, event2: Event) -> Tuple[bool, float, str]:
        """Check if events finish at the same time"""
        if not event1.end_time or not event2.end_time:
            return False, 0.5, "Both events need end times for finish comparison"
        
        time_diff = abs((event1.end_time - event2.end_time).total_seconds())
        finishes_same = time_diff < 300  # 5 minute tolerance
        
        confidence = min(1.0, (event1.confidence + event2.confidence) / 2) if finishes_same else 0.9
        
        if finishes_same:
            return True, confidence, f"Events finish within {time_diff} seconds"
        else:
            return False, confidence, f"Events finish {time_diff} seconds apart"
    
    def _check_contains(self, event1: Event, event2: Event) -> Tuple[bool, float, str]:
        """Check if event1 contains event2"""
        return self._check_during(event2, event1)
    
    def analyze_temporal_graph(self, events: List[Event]) -> Dict[str, Any]:
        """
        Analyze temporal relationships as a graph
        
        Args:
            events: List of events to analyze
            
        Returns:
            Graph analysis results
        """
        if len(events) < 2:
            return {'nodes': len(events), 'edges': 0, 'cycles': [], 'components': 1}
        
        # Build temporal graph
        graph = nx.DiGraph()
        
        # Add events as nodes
        for event in events:
            graph.add_node(event.event_id, event=event)
        
        # Add temporal relationships as edges
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                is_before, confidence, _ = self._check_before(event1, event2)
                if is_before and confidence > 0.5:
                    graph.add_edge(event1.event_id, event2.event_id, 
                                 type='before', confidence=confidence)
                else:
                    is_after, confidence, _ = self._check_after(event1, event2)
                    if is_after and confidence > 0.5:
                        graph.add_edge(event2.event_id, event1.event_id,
                                     type='before', confidence=confidence)
        
        # Analyze graph properties
        try:
            cycles = list(nx.simple_cycles(graph))
            strongly_connected = list(nx.strongly_connected_components(graph))
            
            return {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'cycles': cycles,
                'strongly_connected_components': len(strongly_connected),
                'is_dag': nx.is_directed_acyclic_graph(graph),
                'density': nx.density(graph)
            }
        except Exception as e:
            logger.warning("Error analyzing temporal graph: {}", e)
            return {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'error': str(e)
            }


class ConsistencyChecker:
    """Checks temporal consistency using various rules and heuristics"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.logic_engine = TemporalLogicEngine(config)
        self.consistency_rules = self._initialize_consistency_rules()
    
    def _initialize_consistency_rules(self) -> List[ConsistencyRule]:
        """Initialize built-in consistency rules"""
        rules = []
        
        # Chronological ordering rules
        rules.append(ConsistencyRule(
            rule_id="chronological_order",
            rule_type="chronological",
            description="Events should be in chronological order",
            severity="high",
            validation_function=self._validate_chronological_order
        ))
        
        # Duration consistency rules
        rules.append(ConsistencyRule(
            rule_id="duration_consistency",
            rule_type="duration",
            description="Event durations should be reasonable",
            severity="medium",
            validation_function=self._validate_duration_consistency
        ))
        
        # Causality rules
        rules.append(ConsistencyRule(
            rule_id="causality_order",
            rule_type="causal",
            description="Cause events should precede effect events",
            severity="critical",
            validation_function=self._validate_causality_order
        ))
        
        # Overlap rules
        rules.append(ConsistencyRule(
            rule_id="exclusive_events",
            rule_type="logical",
            description="Mutually exclusive events should not overlap",
            severity="high",
            validation_function=self._validate_exclusive_events
        ))
        
        # Date range rules
        rules.append(ConsistencyRule(
            rule_id="date_range_validity",
            rule_type="logical",
            description="Event dates should be within reasonable ranges",
            severity="medium",
            validation_function=self._validate_date_ranges
        ))
        
        # Participant consistency rules
        rules.append(ConsistencyRule(
            rule_id="participant_consistency",
            rule_type="logical",
            description="Participants should be consistently referenced",
            severity="low",
            validation_function=self._validate_participant_consistency
        ))
        
        return rules
    
    def check_timeline_consistency(self, timeline: Timeline) -> List[Inconsistency]:
        """
        Check timeline for temporal inconsistencies
        
        Args:
            timeline: Timeline to validate
            
        Returns:
            List of detected inconsistencies
        """
        if not timeline.events:
            return []
        
        logger.debug("Checking consistency for timeline with {} events", len(timeline.events))
        
        inconsistencies = []
        
        # Apply each consistency rule
        for rule in self.consistency_rules:
            try:
                rule_inconsistencies = rule.validation_function(timeline)
                inconsistencies.extend(rule_inconsistencies)
            except Exception as e:
                logger.warning("Error applying consistency rule {}: {}", rule.rule_id, e)
                # Create error inconsistency
                error_inconsistency = Inconsistency(
                    inconsistency_type=InconsistencyType.MISSING_TEMPORAL_CONTEXT,
                    severity="medium",
                    description=f"Error applying consistency rule '{rule.rule_id}': {str(e)}",
                    affected_events=[],
                    detection_method=f"rule_error_{rule.rule_id}",
                    confidence=0.8,
                    context_information={'rule_id': rule.rule_id, 'error': str(e)}
                )
                inconsistencies.append(error_inconsistency)
        
        logger.debug("Found {} inconsistencies", len(inconsistencies))
        return inconsistencies
    
    def _validate_chronological_order(self, timeline: Timeline) -> List[Inconsistency]:
        """Validate chronological ordering of events"""
        inconsistencies = []
        
        if not self.config.strict_chronology:
            return inconsistencies
        
        for i in range(len(timeline.events) - 1):
            event1 = timeline.events[i]
            event2 = timeline.events[i + 1]
            
            # Check if events are in wrong order
            if event1.start_time > event2.start_time:
                inconsistency = Inconsistency(
                    inconsistency_type=InconsistencyType.CHRONOLOGICAL_VIOLATION,
                    severity="high",
                    description=f"Event '{event1.title}' occurs after '{event2.title}' but appears earlier in timeline",
                    affected_events=[event1.event_id, event2.event_id],
                    detection_method="chronological_order_check",
                    confidence=0.9,
                    suggested_resolution=f"Reorder events: move '{event1.title}' after '{event2.title}'",
                    context_information={
                        'event1_start': event1.start_time.isoformat(),
                        'event2_start': event2.start_time.isoformat(),
                        'time_difference': (event1.start_time - event2.start_time).total_seconds()
                    }
                )
                inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _validate_duration_consistency(self, timeline: Timeline) -> List[Inconsistency]:
        """Validate event duration consistency"""
        inconsistencies = []
        
        for event in timeline.events:
            if not event.end_time:
                continue
            
            duration = event.end_time - event.start_time
            
            # Check for negative durations
            if duration.total_seconds() < 0:
                inconsistency = Inconsistency(
                    inconsistency_type=InconsistencyType.DURATION_MISMATCH,
                    severity="critical",
                    description=f"Event '{event.title}' has negative duration (end before start)",
                    affected_events=[event.event_id],
                    detection_method="negative_duration_check",
                    confidence=1.0,
                    suggested_resolution="Correct start and end times",
                    context_information={
                        'start_time': event.start_time.isoformat(),
                        'end_time': event.end_time.isoformat(),
                        'duration_seconds': duration.total_seconds()
                    }
                )
                inconsistencies.append(inconsistency)
            
            # Check for unreasonably long durations
            elif duration.total_seconds() > 365 * 24 * 3600:  # More than 1 year
                inconsistency = Inconsistency(
                    inconsistency_type=InconsistencyType.DURATION_MISMATCH,
                    severity="medium",
                    description=f"Event '{event.title}' has unusually long duration ({duration.days} days)",
                    affected_events=[event.event_id],
                    detection_method="long_duration_check",
                    confidence=0.7,
                    suggested_resolution="Verify event duration is correct",
                    context_information={
                        'duration_days': duration.days,
                        'duration_hours': duration.total_seconds() / 3600
                    }
                )
                inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _validate_causality_order(self, timeline: Timeline) -> List[Inconsistency]:
        """Validate causal ordering of events"""
        inconsistencies = []
        
        if not self.config.validate_causality:
            return inconsistencies
        
        # Look for causal relationships in event metadata
        for event in timeline.events:
            causes = event.metadata.get('causes', [])
            caused_by = event.metadata.get('caused_by', [])
            
            # Check cause relationships
            for cause_info in causes:
                caused_event_id = cause_info.get('event_id')
                caused_event = next((e for e in timeline.events if e.event_id == caused_event_id), None)
                
                if caused_event and event.start_time > caused_event.start_time:
                    inconsistency = Inconsistency(
                        inconsistency_type=InconsistencyType.CAUSALITY_VIOLATION,
                        severity="critical",
                        description=f"Cause '{event.title}' occurs after effect '{caused_event.title}'",
                        affected_events=[event.event_id, caused_event.event_id],
                        detection_method="causality_order_check",
                        confidence=cause_info.get('confidence', 0.8),
                        suggested_resolution="Verify causal relationship and timing",
                        context_information={
                            'cause_time': event.start_time.isoformat(),
                            'effect_time': caused_event.start_time.isoformat(),
                            'evidence': cause_info.get('evidence', [])
                        }
                    )
                    inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _validate_exclusive_events(self, timeline: Timeline) -> List[Inconsistency]:
        """Validate that mutually exclusive events don't overlap"""
        inconsistencies = []
        
        # Define mutually exclusive event patterns
        exclusive_patterns = [
            ['birth', 'death'],
            ['start', 'end'],
            ['arrival', 'departure']
        ]
        
        for pattern in exclusive_patterns:
            pattern_events = []
            
            # Find events matching this pattern
            for event in timeline.events:
                for keyword in pattern:
                    if keyword.lower() in event.title.lower():
                        pattern_events.append((event, keyword))
                        break
            
            # Check for overlapping exclusive events
            for i, (event1, type1) in enumerate(pattern_events):
                for event2, type2 in pattern_events[i+1:]:
                    if (type1 != type2 and event1.end_time and event2.end_time and
                        event1.overlaps_with(event2)):
                        
                        inconsistency = Inconsistency(
                            inconsistency_type=InconsistencyType.OVERLAPPING_EXCLUSIVE_EVENTS,
                            severity="high",
                            description=f"Mutually exclusive events overlap: '{event1.title}' and '{event2.title}'",
                            affected_events=[event1.event_id, event2.event_id],
                            detection_method="exclusive_events_check",
                            confidence=0.8,
                            suggested_resolution="Verify event timing or relationship",
                            context_information={
                                'pattern': pattern,
                                'event1_type': type1,
                                'event2_type': type2
                            }
                        )
                        inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _validate_date_ranges(self, timeline: Timeline) -> List[Inconsistency]:
        """Validate that event dates are within reasonable ranges"""
        inconsistencies = []
        
        now = datetime.now(timezone.utc)
        
        for event in timeline.events:
            # Check for future dates that seem unlikely
            if event.start_time > now + timedelta(days=365 * 10):  # More than 10 years in future
                inconsistency = Inconsistency(
                    inconsistency_type=InconsistencyType.DATE_CONFLICT,
                    severity="medium",
                    description=f"Event '{event.title}' is scheduled far in the future ({event.start_time.year})",
                    affected_events=[event.event_id],
                    detection_method="future_date_check",
                    confidence=0.6,
                    suggested_resolution="Verify date is correct",
                    context_information={
                        'event_date': event.start_time.isoformat(),
                        'years_in_future': (event.start_time - now).days // 365
                    }
                )
                inconsistencies.append(inconsistency)
            
            # Check for very old dates that might be errors
            elif event.start_time < datetime(1900, 1, 1, tzinfo=timezone.utc):
                inconsistency = Inconsistency(
                    inconsistency_type=InconsistencyType.DATE_CONFLICT,
                    severity="medium",
                    description=f"Event '{event.title}' has very old date ({event.start_time.year})",
                    affected_events=[event.event_id],
                    detection_method="old_date_check",
                    confidence=0.6,
                    suggested_resolution="Verify historical date is correct",
                    context_information={
                        'event_date': event.start_time.isoformat(),
                        'event_year': event.start_time.year
                    }
                )
                inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _validate_participant_consistency(self, timeline: Timeline) -> List[Inconsistency]:
        """Validate participant name consistency across events"""
        inconsistencies = []
        
        # Collect all participant names
        participant_variants = defaultdict(set)
        
        for event in timeline.events:
            for participant in event.participants:
                # Simple normalization
                normalized = participant.lower().strip()
                participant_variants[normalized].add(participant)
        
        # Look for potential name variants
        for normalized_name, variants in participant_variants.items():
            if len(variants) > 1:
                # Check if variants are similar (might be inconsistent naming)
                variant_list = list(variants)
                
                for i, name1 in enumerate(variant_list):
                    for name2 in variant_list[i+1:]:
                        # Simple similarity check
                        if self._names_might_be_same_person(name1, name2):
                            affected_events = [
                                event.event_id for event in timeline.events
                                if name1 in event.participants or name2 in event.participants
                            ]
                            
                            inconsistency = Inconsistency(
                                inconsistency_type=InconsistencyType.AMBIGUOUS_REFERENCE,
                                severity="low",
                                description=f"Possible name variants for same person: '{name1}' and '{name2}'",
                                affected_events=affected_events,
                                detection_method="participant_consistency_check",
                                confidence=0.5,
                                suggested_resolution="Standardize participant names",
                                context_information={
                                    'name_variants': [name1, name2],
                                    'normalized_name': normalized_name
                                }
                            )
                            inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _names_might_be_same_person(self, name1: str, name2: str) -> bool:
        """Simple heuristic to check if two names might refer to the same person"""
        # Split names into parts
        parts1 = set(name1.lower().split())
        parts2 = set(name2.lower().split())
        
        # Check for significant overlap
        overlap = parts1.intersection(parts2)
        
        # If they share most name parts, they might be the same person
        return len(overlap) >= min(len(parts1), len(parts2)) * 0.7


class TemporalValidator:
    """Main temporal validation coordinator"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.consistency_checker = ConsistencyChecker(config)
        self.logic_engine = TemporalLogicEngine(config)
    
    def detect_temporal_inconsistencies(self, timeline: Timeline) -> List[Inconsistency]:
        """
        Detect temporal inconsistencies in timeline
        
        Args:
            timeline: Timeline to validate
            
        Returns:
            List of detected inconsistencies
        """
        if not self.config.enable_validation:
            logger.info("Temporal validation is disabled")
            return []
        
        logger.info("Detecting temporal inconsistencies in timeline: {}", timeline.timeline_id)
        
        try:
            # Run consistency checks
            inconsistencies = self.consistency_checker.check_timeline_consistency(timeline)
            
            # Add temporal graph analysis
            graph_analysis = self.logic_engine.analyze_temporal_graph(timeline.events)
            
            # Check for temporal cycles (impossible sequences)
            if graph_analysis.get('cycles'):
                for cycle in graph_analysis['cycles']:
                    cycle_events = [eid for eid in cycle]
                    inconsistency = Inconsistency(
                        inconsistency_type=InconsistencyType.IMPOSSIBLE_SEQUENCE,
                        severity="critical",
                        description=f"Impossible temporal cycle detected involving {len(cycle_events)} events",
                        affected_events=cycle_events,
                        detection_method="temporal_graph_analysis",
                        confidence=0.95,
                        suggested_resolution="Review and correct event ordering",
                        context_information={'cycle': cycle_events, 'cycle_length': len(cycle_events)}
                    )
                    inconsistencies.append(inconsistency)
            
            # Categorize inconsistencies by severity
            critical_count = len([i for i in inconsistencies if i.severity == 'critical'])
            high_count = len([i for i in inconsistencies if i.severity == 'high'])
            medium_count = len([i for i in inconsistencies if i.severity == 'medium'])
            low_count = len([i for i in inconsistencies if i.severity == 'low'])
            
            logger.info(
                "Inconsistency detection completed: {} total ({} critical, {} high, {} medium, {} low)",
                len(inconsistencies), critical_count, high_count, medium_count, low_count
            )
            
            return inconsistencies
            
        except Exception as e:
            logger.error("Error detecting temporal inconsistencies: {}", e)
            # Return error as inconsistency
            error_inconsistency = Inconsistency(
                inconsistency_type=InconsistencyType.MISSING_TEMPORAL_CONTEXT,
                severity="high",
                description=f"Error during validation process: {str(e)}",
                affected_events=[],
                detection_method="validation_error",
                confidence=1.0,
                context_information={'error': str(e)}
            )
            return [error_inconsistency]
    
    def validate_timeline(self, timeline: Timeline) -> ValidationResult:
        """
        Perform comprehensive timeline validation
        
        Args:
            timeline: Timeline to validate
            
        Returns:
            Comprehensive validation result
        """
        logger.info("Performing comprehensive timeline validation")
        
        # Detect inconsistencies
        inconsistencies = self.detect_temporal_inconsistencies(timeline)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(timeline, inconsistencies)
        
        # Categorize issues by severity
        critical_issues = len([i for i in inconsistencies if i.severity == 'critical'])
        high_priority_issues = len([i for i in inconsistencies if i.severity == 'high'])
        medium_priority_issues = len([i for i in inconsistencies if i.severity == 'medium'])
        low_priority_issues = len([i for i in inconsistencies if i.severity == 'low'])
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(inconsistencies)
        
        # Assess overall confidence
        confidence_assessment = self._assess_confidence(timeline, inconsistencies)
        
        # Calculate quality metrics
        temporal_accuracy = self._calculate_temporal_accuracy(timeline, inconsistencies)
        completeness = self._calculate_completeness(timeline)
        logical_consistency = self._calculate_logical_consistency(timeline, inconsistencies)
        
        # Create validation result
        validation_result = ValidationResult(
            timeline_id=timeline.timeline_id,
            is_consistent=critical_issues == 0 and consistency_score >= self.config.consistency_threshold,
            consistency_score=consistency_score,
            inconsistencies=inconsistencies,
            critical_issues=critical_issues,
            high_priority_issues=high_priority_issues,
            medium_priority_issues=medium_priority_issues,
            low_priority_issues=low_priority_issues,
            validation_method="comprehensive_temporal_validation",
            validation_coverage=1.0,  # Full validation
            improvement_suggestions=suggestions,
            confidence_assessment=confidence_assessment,
            temporal_accuracy=temporal_accuracy,
            completeness=completeness,
            logical_consistency=logical_consistency
        )
        
        logger.info(
            "Timeline validation completed: {} consistent, score {:.3f}",
            "PASS" if validation_result.is_consistent else "FAIL",
            consistency_score
        )
        
        return validation_result
    
    def _calculate_consistency_score(self, timeline: Timeline, 
                                   inconsistencies: List[Inconsistency]) -> float:
        """Calculate overall consistency score"""
        if not timeline.events:
            return 1.0
        
        if not inconsistencies:
            return 1.0
        
        # Weight inconsistencies by severity
        severity_weights = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.1}
        
        total_penalty = 0.0
        for inconsistency in inconsistencies:
            weight = severity_weights.get(inconsistency.severity, 0.5)
            penalty = weight * (1.0 - inconsistency.confidence)
            total_penalty += penalty
        
        # Normalize by number of events
        if timeline.events:
            penalty_per_event = total_penalty / len(timeline.events)
            consistency_score = max(0.0, 1.0 - penalty_per_event)
        else:
            consistency_score = 1.0
        
        return consistency_score
    
    def _generate_improvement_suggestions(self, inconsistencies: List[Inconsistency]) -> List[str]:
        """Generate suggestions for improving timeline"""
        suggestions = []
        
        # Group inconsistencies by type
        inconsistency_groups = defaultdict(list)
        for inconsistency in inconsistencies:
            inconsistency_groups[inconsistency.inconsistency_type].append(inconsistency)
        
        # Generate type-specific suggestions
        if InconsistencyType.CHRONOLOGICAL_VIOLATION in inconsistency_groups:
            suggestions.append("Review event ordering to ensure chronological consistency")
        
        if InconsistencyType.DURATION_MISMATCH in inconsistency_groups:
            suggestions.append("Verify event start and end times for duration accuracy")
        
        if InconsistencyType.CAUSALITY_VIOLATION in inconsistency_groups:
            suggestions.append("Check causal relationships to ensure causes precede effects")
        
        if InconsistencyType.DATE_CONFLICT in inconsistency_groups:
            suggestions.append("Validate unusual dates that may be data entry errors")
        
        if InconsistencyType.AMBIGUOUS_REFERENCE in inconsistency_groups:
            suggestions.append("Standardize participant and location names for consistency")
        
        # Add general suggestions
        if inconsistencies:
            high_confidence_issues = [i for i in inconsistencies if i.confidence > 0.8]
            if high_confidence_issues:
                suggestions.append(f"Address {len(high_confidence_issues)} high-confidence issues first")
        
        return suggestions
    
    def _assess_confidence(self, timeline: Timeline, inconsistencies: List[Inconsistency]) -> str:
        """Assess overall confidence in timeline"""
        if not inconsistencies:
            return "High confidence - no inconsistencies detected"
        
        critical_issues = [i for i in inconsistencies if i.severity == 'critical']
        high_issues = [i for i in inconsistencies if i.severity == 'high']
        
        if critical_issues:
            return f"Low confidence - {len(critical_issues)} critical issues found"
        elif len(high_issues) > len(timeline.events) * 0.1:  # More than 10% of events have high issues
            return f"Medium confidence - {len(high_issues)} high-priority issues found"
        elif len(inconsistencies) > len(timeline.events) * 0.2:  # More than 20% of events have issues
            return "Medium confidence - multiple consistency concerns"
        else:
            return "High confidence - minor issues only"
    
    def _calculate_temporal_accuracy(self, timeline: Timeline, 
                                   inconsistencies: List[Inconsistency]) -> float:
        """Calculate temporal accuracy score"""
        if not timeline.events:
            return 1.0
        
        temporal_issues = [
            i for i in inconsistencies
            if i.inconsistency_type in [
                InconsistencyType.CHRONOLOGICAL_VIOLATION,
                InconsistencyType.DURATION_MISMATCH,
                InconsistencyType.DATE_CONFLICT
            ]
        ]
        
        if not temporal_issues:
            return 1.0
        
        # Calculate penalty based on temporal issues
        penalty = len(temporal_issues) / len(timeline.events)
        return max(0.0, 1.0 - penalty)
    
    def _calculate_completeness(self, timeline: Timeline) -> float:
        """Calculate timeline completeness score"""
        if not timeline.events:
            return 0.0
        
        # Factors contributing to completeness
        completeness_factors = []
        
        # Event description completeness
        events_with_descriptions = len([e for e in timeline.events if e.description])
        description_completeness = events_with_descriptions / len(timeline.events)
        completeness_factors.append(description_completeness * 0.3)
        
        # Time information completeness
        events_with_end_times = len([e for e in timeline.events if e.end_time])
        time_completeness = events_with_end_times / len(timeline.events)
        completeness_factors.append(time_completeness * 0.2)
        
        # Participant information completeness
        events_with_participants = len([e for e in timeline.events if e.participants])
        participant_completeness = events_with_participants / len(timeline.events)
        completeness_factors.append(participant_completeness * 0.2)
        
        # Location information completeness
        events_with_locations = len([e for e in timeline.events if e.locations])
        location_completeness = events_with_locations / len(timeline.events)
        completeness_factors.append(location_completeness * 0.1)
        
        # Context information completeness
        events_with_context = len([e for e in timeline.events if e.context_sentences])
        context_completeness = events_with_context / len(timeline.events)
        completeness_factors.append(context_completeness * 0.2)
        
        return sum(completeness_factors)
    
    def _calculate_logical_consistency(self, timeline: Timeline,
                                     inconsistencies: List[Inconsistency]) -> float:
        """Calculate logical consistency score"""
        if not timeline.events:
            return 1.0
        
        logical_issues = [
            i for i in inconsistencies
            if i.inconsistency_type in [
                InconsistencyType.CAUSALITY_VIOLATION,
                InconsistencyType.IMPOSSIBLE_SEQUENCE,
                InconsistencyType.OVERLAPPING_EXCLUSIVE_EVENTS
            ]
        ]
        
        if not logical_issues:
            return 1.0
        
        # Weight logical issues more heavily
        penalty = sum(
            0.8 if i.severity in ['critical', 'high'] else 0.4
            for i in logical_issues
        ) / len(timeline.events)
        
        return max(0.0, 1.0 - penalty)


# Convenience function for direct usage
def detect_temporal_inconsistencies(timeline: Timeline,
                                   config: Optional[TimelineConfig] = None) -> List[Inconsistency]:
    """
    Convenience function to detect temporal inconsistencies in timeline
    
    Args:
        timeline: Timeline to validate
        config: Timeline configuration (uses default if not provided)
        
    Returns:
        List of detected inconsistencies
    """
    if config is None:
        from .core import create_default_timeline_config
        config = create_default_timeline_config()
    
    validator = TemporalValidator(config)
    return validator.detect_temporal_inconsistencies(timeline)