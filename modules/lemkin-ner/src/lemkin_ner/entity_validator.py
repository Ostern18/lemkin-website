"""
Human validation workflows and quality assurance for entity extraction.
"""

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import re
from loguru import logger

from .core import Entity, EntityType, ValidationResult, LanguageCode, NERConfig


class ValidationStatus(Enum):
    """Status of validation process"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    CORRECTED = "corrected"


class QualityMetric(Enum):
    """Quality metrics for entity validation"""
    CONFIDENCE_SCORE = "confidence_score"
    CONTEXT_RELEVANCE = "context_relevance"
    TYPE_ACCURACY = "type_accuracy"
    BOUNDARY_ACCURACY = "boundary_accuracy"
    CONSISTENCY = "consistency"


class EntityValidator:
    """
    Validates extracted entities through automated checks and human review
    """
    
    def __init__(self, config: NERConfig):
        """
        Initialize entity validator
        
        Args:
            config: NER configuration
        """
        self.config = config
        self.validation_rules = {}
        self.quality_thresholds = {}
        self.validation_history = {}
        self.reviewer_stats = {}
        
        # Initialize validation components
        self._load_validation_rules()
        self._set_quality_thresholds()
        
        logger.info("EntityValidator initialized")
    
    def _load_validation_rules(self) -> None:
        """Load validation rules for different entity types"""
        self.validation_rules = {
            EntityType.PERSON: {
                "min_length": 2,
                "max_length": 100,
                "required_patterns": [r'[A-Za-z]'],
                "forbidden_patterns": [r'^\d+$', r'^[^\w\s]+$'],
                "case_rules": ["title_case_preferred"],
                "context_requirements": ["not_in_url", "not_all_caps"]
            },
            EntityType.ORGANIZATION: {
                "min_length": 2,
                "max_length": 200,
                "required_patterns": [r'[A-Za-z]'],
                "forbidden_patterns": [r'^\d+$'],
                "valid_suffixes": ["Inc", "Corp", "LLC", "Ltd", "Company", "Corporation", "Foundation"],
                "context_requirements": ["not_in_url"]
            },
            EntityType.LOCATION: {
                "min_length": 2,
                "max_length": 100,
                "required_patterns": [r'[A-Za-z]'],
                "forbidden_patterns": [r'^\d+$'],
                "context_requirements": ["not_in_email", "not_in_url"]
            },
            EntityType.DATE: {
                "min_length": 4,
                "max_length": 50,
                "required_patterns": [r'\d'],
                "valid_formats": [
                    r'\d{1,2}/\d{1,2}/\d{2,4}',
                    r'\d{1,2}-\d{1,2}-\d{2,4}',
                    r'\w+ \d{1,2}, \d{4}',
                    r'\d{1,2} \w+ \d{4}'
                ]
            },
            EntityType.LEGAL_ENTITY: {
                "min_length": 3,
                "max_length": 300,
                "required_patterns": [r'[A-Za-z]'],
                "legal_indicators": [
                    "court", "judge", "attorney", "lawyer", "statute", "case", "v.", "vs."
                ]
            }
        }
    
    def _set_quality_thresholds(self) -> None:
        """Set quality thresholds for different metrics"""
        self.quality_thresholds = {
            QualityMetric.CONFIDENCE_SCORE: {
                "excellent": 0.9,
                "good": 0.7,
                "acceptable": self.config.validation_threshold,
                "poor": 0.3
            },
            QualityMetric.CONTEXT_RELEVANCE: {
                "excellent": 0.8,
                "good": 0.6,
                "acceptable": 0.4,
                "poor": 0.2
            },
            QualityMetric.TYPE_ACCURACY: {
                "excellent": 0.95,
                "good": 0.85,
                "acceptable": 0.7,
                "poor": 0.5
            },
            QualityMetric.BOUNDARY_ACCURACY: {
                "excellent": 0.95,
                "good": 0.85,
                "acceptable": 0.7,
                "poor": 0.5
            }
        }
    
    def validate_entity(self, entity: Entity, context_entities: Optional[List[Entity]] = None) -> ValidationResult:
        """
        Validate a single entity
        
        Args:
            entity: Entity to validate
            context_entities: Other entities in the same document for consistency checking
            
        Returns:
            ValidationResult with validation outcome
        """
        logger.debug("Validating entity: '{}' (type: {})", entity.text, entity.entity_type.value)
        
        issues = []
        suggestions = []
        is_valid = True
        validation_confidence = 1.0
        
        try:
            # Rule-based validation
            rule_issues, rule_suggestions = self._apply_validation_rules(entity)
            issues.extend(rule_issues)
            suggestions.extend(rule_suggestions)
            
            # Quality metric evaluation
            quality_scores = self._evaluate_quality_metrics(entity)
            
            # Context validation
            if entity.context:
                context_issues, context_suggestions = self._validate_context(entity)
                issues.extend(context_issues)
                suggestions.extend(context_suggestions)
            
            # Consistency validation with other entities
            if context_entities:
                consistency_issues, consistency_suggestions = self._validate_consistency(entity, context_entities)
                issues.extend(consistency_issues)
                suggestions.extend(consistency_suggestions)
            
            # Language-specific validation
            lang_issues, lang_suggestions = self._validate_language_specific(entity)
            issues.extend(lang_issues)
            suggestions.extend(lang_suggestions)
            
            # Determine overall validation result
            if issues:
                critical_issues = [issue for issue in issues if issue.startswith("CRITICAL:")]
                if critical_issues:
                    is_valid = False
                    validation_confidence = 0.0
                else:
                    is_valid = True
                    validation_confidence = max(0.1, 1.0 - (len(issues) * 0.2))
            
            # Adjust confidence based on quality scores
            if quality_scores:
                avg_quality = sum(quality_scores.values()) / len(quality_scores)
                validation_confidence = min(validation_confidence, avg_quality)
            
            # Create validation result
            result = ValidationResult(
                entity=entity,
                is_valid=is_valid,
                validation_confidence=validation_confidence,
                issues=issues,
                suggestions=suggestions,
                metadata={
                    "quality_scores": quality_scores,
                    "validation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "validator_version": "1.0"
                }
            )
            
            logger.debug("Validation complete for '{}': {} (confidence: {:.3f})", 
                        entity.text, "VALID" if is_valid else "INVALID", validation_confidence)
            
            return result
            
        except Exception as e:
            logger.error("Error validating entity '{}': {}", entity.text, e)
            return ValidationResult(
                entity=entity,
                is_valid=False,
                validation_confidence=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Manual review required due to validation error"]
            )
    
    def validate_batch(self, entities: List[Entity]) -> List[ValidationResult]:
        """
        Validate multiple entities in batch
        
        Args:
            entities: List of entities to validate
            
        Returns:
            List of validation results
        """
        logger.info("Validating batch of {} entities", len(entities))
        
        # Group entities by document for context validation
        entities_by_doc = {}
        for entity in entities:
            doc_id = entity.document_id
            if doc_id not in entities_by_doc:
                entities_by_doc[doc_id] = []
            entities_by_doc[doc_id].append(entity)
        
        # Validate each entity with document context
        results = []
        for doc_id, doc_entities in entities_by_doc.items():
            logger.debug("Validating {} entities from document: {}", len(doc_entities), doc_id)
            
            for entity in doc_entities:
                # Get other entities as context (excluding current entity)
                context_entities = [e for e in doc_entities if e.entity_id != entity.entity_id]
                result = self.validate_entity(entity, context_entities)
                results.append(result)
        
        # Generate batch statistics
        valid_count = sum(1 for r in results if r.is_valid)
        avg_confidence = sum(r.validation_confidence for r in results) / len(results) if results else 0.0
        
        logger.info("Batch validation complete: {}/{} valid (avg confidence: {:.3f})", 
                   valid_count, len(results), avg_confidence)
        
        return results
    
    def create_human_review_tasks(self, validation_results: List[ValidationResult], 
                                 output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Create human review tasks for entities that need manual validation
        
        Args:
            validation_results: Results from automated validation
            output_dir: Directory to save review tasks
            
        Returns:
            Summary of created review tasks
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating human review tasks for {} validation results", len(validation_results))
        
        # Filter entities that need human review
        review_needed = []
        for result in validation_results:
            needs_review = (
                not result.is_valid or
                result.validation_confidence < self.config.validation_threshold or
                len(result.issues) > 2 or
                self.config.require_human_review
            )
            
            if needs_review:
                review_needed.append(result)
        
        if not review_needed:
            logger.info("No entities require human review")
            return {
                "total_results": len(validation_results),
                "review_needed": 0,
                "tasks_created": 0,
                "output_directory": str(output_dir)
            }
        
        # Create review tasks grouped by priority
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for result in review_needed:
            if not result.is_valid or result.validation_confidence < 0.3:
                high_priority.append(result)
            elif result.validation_confidence < 0.6:
                medium_priority.append(result)
            else:
                low_priority.append(result)
        
        # Save review tasks to files
        tasks_created = 0
        
        # High priority tasks
        if high_priority:
            high_priority_file = output_dir / "high_priority_review.json"
            self._save_review_tasks(high_priority, high_priority_file, "high")
            tasks_created += 1
        
        # Medium priority tasks
        if medium_priority:
            medium_priority_file = output_dir / "medium_priority_review.json"
            self._save_review_tasks(medium_priority, medium_priority_file, "medium")
            tasks_created += 1
        
        # Low priority tasks
        if low_priority:
            low_priority_file = output_dir / "low_priority_review.json"
            self._save_review_tasks(low_priority, low_priority_file, "low")
            tasks_created += 1
        
        # Create CSV export for easier review
        csv_file = output_dir / "entity_review_tasks.csv"
        self._export_review_tasks_csv(review_needed, csv_file)
        
        # Create review instructions
        instructions_file = output_dir / "review_instructions.md"
        self._create_review_instructions(instructions_file)
        
        summary = {
            "total_results": len(validation_results),
            "review_needed": len(review_needed),
            "high_priority": len(high_priority),
            "medium_priority": len(medium_priority),
            "low_priority": len(low_priority),
            "tasks_created": tasks_created,
            "output_directory": str(output_dir),
            "files_created": [
                str(f) for f in output_dir.glob("*") 
                if f.is_file() and f.stat().st_mtime > (datetime.now().timestamp() - 60)
            ]
        }
        
        logger.info("Created {} review task files for {} entities requiring review", 
                   tasks_created, len(review_needed))
        
        return summary
    
    def process_human_feedback(self, feedback_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Process human feedback on entity validation
        
        Args:
            feedback_file: Path to file containing human feedback
            
        Returns:
            Summary of processed feedback
        """
        feedback_file = Path(feedback_file)
        if not feedback_file.exists():
            raise FileNotFoundError(f"Feedback file not found: {feedback_file}")
        
        logger.info("Processing human feedback from: {}", feedback_file)
        
        try:
            # Load feedback data
            if feedback_file.suffix.lower() == '.json':
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
            elif feedback_file.suffix.lower() == '.csv':
                feedback_data = self._load_csv_feedback(feedback_file)
            else:
                raise ValueError(f"Unsupported feedback file format: {feedback_file.suffix}")
            
            # Process each feedback entry
            processed = 0
            corrections = 0
            approvals = 0
            rejections = 0
            
            for entry in feedback_data:
                try:
                    entity_id = entry.get('entity_id')
                    reviewer = entry.get('reviewer', 'unknown')
                    decision = entry.get('decision', '').lower()
                    comments = entry.get('comments', '')
                    corrected_text = entry.get('corrected_text', '')
                    corrected_type = entry.get('corrected_type', '')
                    
                    if not entity_id:
                        logger.warning("Skipping feedback entry without entity_id")
                        continue
                    
                    # Store feedback in validation history
                    feedback_record = {
                        "entity_id": entity_id,
                        "reviewer": reviewer,
                        "decision": decision,
                        "comments": comments,
                        "corrected_text": corrected_text,
                        "corrected_type": corrected_type,
                        "review_timestamp": datetime.now(timezone.utc).isoformat(),
                        "feedback_file": str(feedback_file)
                    }
                    
                    self.validation_history[entity_id] = feedback_record
                    
                    # Update reviewer statistics
                    if reviewer not in self.reviewer_stats:
                        self.reviewer_stats[reviewer] = {
                            "total_reviewed": 0,
                            "approvals": 0,
                            "rejections": 0,
                            "corrections": 0
                        }
                    
                    self.reviewer_stats[reviewer]["total_reviewed"] += 1
                    
                    if decision == 'approved':
                        approvals += 1
                        self.reviewer_stats[reviewer]["approvals"] += 1
                    elif decision == 'rejected':
                        rejections += 1
                        self.reviewer_stats[reviewer]["rejections"] += 1
                    elif decision == 'corrected':
                        corrections += 1
                        self.reviewer_stats[reviewer]["corrections"] += 1
                    
                    processed += 1
                    
                except Exception as e:
                    logger.error("Error processing feedback entry: {}", e)
                    continue
            
            summary = {
                "feedback_file": str(feedback_file),
                "total_entries": len(feedback_data),
                "processed": processed,
                "approvals": approvals,
                "rejections": rejections,
                "corrections": corrections,
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("Processed {} feedback entries: {} approved, {} rejected, {} corrected", 
                       processed, approvals, rejections, corrections)
            
            return summary
            
        except Exception as e:
            logger.error("Error processing human feedback: {}", e)
            raise
    
    def generate_quality_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate quality assessment report
        
        Args:
            validation_results: List of validation results to analyze
            
        Returns:
            Quality report dictionary
        """
        logger.info("Generating quality report for {} validation results", len(validation_results))
        
        if not validation_results:
            return {"error": "No validation results provided"}
        
        # Overall statistics
        total_entities = len(validation_results)
        valid_entities = sum(1 for r in validation_results if r.is_valid)
        avg_confidence = sum(r.validation_confidence for r in validation_results) / total_entities
        
        # Quality metrics by entity type
        type_stats = {}
        for result in validation_results:
            entity_type = result.entity.entity_type.value
            if entity_type not in type_stats:
                type_stats[entity_type] = {
                    "total": 0,
                    "valid": 0,
                    "confidence_scores": []
                }
            
            type_stats[entity_type]["total"] += 1
            if result.is_valid:
                type_stats[entity_type]["valid"] += 1
            type_stats[entity_type]["confidence_scores"].append(result.validation_confidence)
        
        # Calculate type-specific metrics
        for entity_type, stats in type_stats.items():
            stats["validity_rate"] = stats["valid"] / stats["total"] if stats["total"] > 0 else 0
            stats["avg_confidence"] = sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
            stats.pop("confidence_scores")  # Remove raw scores from final report
        
        # Common issues analysis
        issue_counts = {}
        for result in validation_results:
            for issue in result.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort issues by frequency
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Quality distribution
        quality_distribution = {
            "excellent": 0,
            "good": 0,
            "acceptable": 0,
            "poor": 0
        }
        
        for result in validation_results:
            confidence = result.validation_confidence
            if confidence >= 0.9:
                quality_distribution["excellent"] += 1
            elif confidence >= 0.7:
                quality_distribution["good"] += 1
            elif confidence >= self.config.validation_threshold:
                quality_distribution["acceptable"] += 1
            else:
                quality_distribution["poor"] += 1
        
        # Human review statistics (if available)
        review_stats = self._get_review_statistics()
        
        report = {
            "summary": {
                "total_entities": total_entities,
                "valid_entities": valid_entities,
                "validity_rate": valid_entities / total_entities,
                "average_confidence": avg_confidence,
                "report_timestamp": datetime.now(timezone.utc).isoformat()
            },
            "by_entity_type": type_stats,
            "quality_distribution": quality_distribution,
            "top_issues": [{"issue": issue, "count": count} for issue, count in top_issues],
            "human_review_stats": review_stats,
            "recommendations": self._generate_quality_recommendations(validation_results)
        }
        
        logger.info("Quality report generated: {:.1f}% validity rate, {:.3f} avg confidence", 
                   report["summary"]["validity_rate"] * 100, avg_confidence)
        
        return report
    
    def _apply_validation_rules(self, entity: Entity) -> Tuple[List[str], List[str]]:
        """Apply entity type-specific validation rules"""
        issues = []
        suggestions = []
        
        entity_type = entity.entity_type
        if entity_type not in self.validation_rules:
            return issues, suggestions
        
        rules = self.validation_rules[entity_type]
        text = entity.text.strip()
        
        # Length validation
        if len(text) < rules.get("min_length", 1):
            issues.append(f"CRITICAL: Text too short ({len(text)} < {rules['min_length']})")
        
        if len(text) > rules.get("max_length", 1000):
            issues.append(f"Text too long ({len(text)} > {rules['max_length']})")
            suggestions.append("Consider splitting or truncating the entity")
        
        # Pattern validation
        for pattern in rules.get("required_patterns", []):
            if not re.search(pattern, text):
                issues.append(f"Missing required pattern: {pattern}")
        
        for pattern in rules.get("forbidden_patterns", []):
            if re.search(pattern, text):
                issues.append(f"Contains forbidden pattern: {pattern}")
        
        # Case validation
        if "title_case_preferred" in rules.get("case_rules", []):
            if text.islower() or text.isupper():
                suggestions.append("Consider using title case")
        
        # Type-specific validations
        if entity_type == EntityType.DATE:
            if not self._is_valid_date_format(text):
                issues.append("Invalid date format")
                suggestions.append("Use standard date format (MM/DD/YYYY or DD-MM-YYYY)")
        
        elif entity_type == EntityType.LEGAL_ENTITY:
            legal_indicators = rules.get("legal_indicators", [])
            if not any(indicator.lower() in text.lower() for indicator in legal_indicators):
                suggestions.append("Consider verifying this is a legal entity")
        
        return issues, suggestions
    
    def _evaluate_quality_metrics(self, entity: Entity) -> Dict[str, float]:
        """Evaluate quality metrics for entity"""
        scores = {}
        
        # Confidence score (normalized to 0-1)
        scores[QualityMetric.CONFIDENCE_SCORE.value] = entity.confidence
        
        # Context relevance (if context available)
        if entity.context:
            scores[QualityMetric.CONTEXT_RELEVANCE.value] = self._calculate_context_relevance(entity)
        
        # Type accuracy (heuristic based on text patterns)
        scores[QualityMetric.TYPE_ACCURACY.value] = self._estimate_type_accuracy(entity)
        
        # Boundary accuracy (heuristic based on text structure)
        scores[QualityMetric.BOUNDARY_ACCURACY.value] = self._estimate_boundary_accuracy(entity)
        
        return scores
    
    def _validate_context(self, entity: Entity) -> Tuple[List[str], List[str]]:
        """Validate entity based on its context"""
        issues = []
        suggestions = []
        
        if not entity.context:
            return issues, suggestions
        
        context = entity.context.lower()
        entity_text = entity.text.lower()
        
        # Check if entity appears in problematic contexts
        if entity.entity_type == EntityType.PERSON:
            if "@" in context and entity_text in context:
                issues.append("Person name appears in email address")
            
            if "http://" in context or "https://" in context:
                issues.append("Person name appears in URL")
        
        # Check for context consistency
        if entity.entity_type == EntityType.ORGANIZATION:
            org_indicators = ["company", "corporation", "inc", "ltd", "llc"]
            if not any(indicator in context for indicator in org_indicators):
                suggestions.append("Consider verifying organization context")
        
        return issues, suggestions
    
    def _validate_consistency(self, entity: Entity, context_entities: List[Entity]) -> Tuple[List[str], List[str]]:
        """Validate consistency with other entities in the same document"""
        issues = []
        suggestions = []
        
        # Find similar entities
        similar_entities = []
        for other in context_entities:
            if (other.entity_type == entity.entity_type and 
                self._are_entities_similar(entity, other)):
                similar_entities.append(other)
        
        # Check for inconsistent representations of the same entity
        if similar_entities:
            confidences = [e.confidence for e in similar_entities] + [entity.confidence]
            if max(confidences) - min(confidences) > 0.3:
                issues.append("Inconsistent confidence scores for similar entities")
        
        # Check for contradictory entity types for the same text
        same_text_entities = [
            e for e in context_entities 
            if e.text.lower() == entity.text.lower() and e.entity_type != entity.entity_type
        ]
        
        if same_text_entities:
            other_types = [e.entity_type.value for e in same_text_entities]
            issues.append(f"Same text classified as different types: {', '.join(other_types)}")
            suggestions.append("Review entity type classification")
        
        return issues, suggestions
    
    def _validate_language_specific(self, entity: Entity) -> Tuple[List[str], List[str]]:
        """Validate entity based on language-specific rules"""
        issues = []
        suggestions = []
        
        # Check if text contains characters appropriate for the detected language
        text = entity.text
        language = entity.language
        
        # Basic script validation
        if language == LanguageCode.EN:
            if re.search(r'[^\x00-\x7F]', text) and entity.entity_type == EntityType.PERSON:
                suggestions.append("Non-ASCII characters in English person name - verify correctness")
        
        elif language == LanguageCode.RU:
            if not re.search(r'[а-яё]', text, re.IGNORECASE) and len(text) > 3:
                issues.append("Russian language detected but no Cyrillic characters found")
        
        elif language == LanguageCode.AR:
            if not re.search(r'[\u0600-\u06FF]', text):
                issues.append("Arabic language detected but no Arabic script found")
        
        elif language == LanguageCode.ZH:
            if not re.search(r'[\u4e00-\u9fff]', text):
                issues.append("Chinese language detected but no Chinese characters found")
        
        return issues, suggestions
    
    def _save_review_tasks(self, validation_results: List[ValidationResult], 
                          output_file: Path, priority: str) -> None:
        """Save review tasks to JSON file"""
        tasks = []
        
        for result in validation_results:
            task = {
                "entity_id": result.entity.entity_id,
                "text": result.entity.text,
                "entity_type": result.entity.entity_type.value,
                "document_id": result.entity.document_id,
                "language": result.entity.language.value,
                "confidence": result.entity.confidence,
                "context": result.entity.context,
                "validation_confidence": result.validation_confidence,
                "is_valid": result.is_valid,
                "issues": result.issues,
                "suggestions": result.suggestions,
                "priority": priority,
                "review_fields": {
                    "decision": "",  # approved/rejected/corrected
                    "corrected_text": "",
                    "corrected_type": "",
                    "comments": "",
                    "reviewer": ""
                }
            }
            tasks.append(task)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "priority": priority,
                "total_tasks": len(tasks),
                "created_timestamp": datetime.now(timezone.utc).isoformat(),
                "tasks": tasks
            }, f, indent=2, ensure_ascii=False)
        
        logger.debug("Saved {} {} priority review tasks to: {}", len(tasks), priority, output_file)
    
    def _export_review_tasks_csv(self, validation_results: List[ValidationResult], output_file: Path) -> None:
        """Export review tasks to CSV file"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'entity_id', 'text', 'entity_type', 'document_id', 'language',
                'confidence', 'validation_confidence', 'is_valid', 'issues',
                'suggestions', 'context', 'decision', 'corrected_text', 
                'corrected_type', 'comments', 'reviewer'
            ])
            
            # Write data rows
            for result in validation_results:
                writer.writerow([
                    result.entity.entity_id,
                    result.entity.text,
                    result.entity.entity_type.value,
                    result.entity.document_id,
                    result.entity.language.value,
                    result.entity.confidence,
                    result.validation_confidence,
                    result.is_valid,
                    '; '.join(result.issues),
                    '; '.join(result.suggestions),
                    result.entity.context or '',
                    '',  # decision - to be filled by reviewer
                    '',  # corrected_text
                    '',  # corrected_type
                    '',  # comments
                    ''   # reviewer
                ])
        
        logger.debug("Exported {} review tasks to CSV: {}", len(validation_results), output_file)
    
    def _create_review_instructions(self, output_file: Path) -> None:
        """Create markdown file with review instructions"""
        instructions = """# Entity Review Instructions

## Overview
This document contains instructions for reviewing automatically extracted entities from legal documents.

## Review Process

### 1. Review Files
- `high_priority_review.json`: Entities requiring immediate attention
- `medium_priority_review.json`: Entities with moderate confidence issues
- `low_priority_review.json`: Entities for quality assurance review
- `entity_review_tasks.csv`: All tasks in spreadsheet format

### 2. Decision Categories
- **approved**: Entity is correct as extracted
- **rejected**: Entity should not have been extracted
- **corrected**: Entity is partially correct but needs modification

### 3. Review Fields
Fill out the following fields for each entity:

- `decision`: One of "approved", "rejected", or "corrected"
- `corrected_text`: If corrected, provide the correct entity text
- `corrected_type`: If corrected, provide the correct entity type
- `comments`: Explanation for decision
- `reviewer`: Your name/ID

### 4. Entity Types
- **PERSON**: Individual names
- **ORGANIZATION**: Companies, institutions, groups
- **LOCATION**: Places, addresses, geographic entities
- **DATE**: Dates and time expressions
- **TIME**: Time-specific expressions
- **EVENT**: Named events, incidents
- **LEGAL_ENTITY**: Legal terms, roles, documents
- **COURT**: Court names and judicial bodies
- **STATUTE**: Laws, regulations, statutes
- **CASE_NAME**: Legal case names
- **CONTRACT**: Contract types and names

### 5. Common Issues to Watch For
- Incorrect entity boundaries (too much/little text included)
- Wrong entity type classification
- False positives (non-entities extracted as entities)
- Missing entities that should have been found
- Language/script inconsistencies

### 6. Quality Guidelines
- Verify entity makes sense in context
- Check for consistent treatment of similar entities
- Ensure proper names are correctly capitalized
- Validate dates and legal references format

## Submitting Reviews
Save completed review files and submit according to project guidelines.
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logger.debug("Created review instructions: {}", output_file)
    
    def _load_csv_feedback(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Load feedback data from CSV file"""
        feedback_data = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Skip rows without decisions
                if not row.get('decision'):
                    continue
                
                feedback_data.append({
                    'entity_id': row.get('entity_id'),
                    'decision': row.get('decision'),
                    'corrected_text': row.get('corrected_text'),
                    'corrected_type': row.get('corrected_type'),
                    'comments': row.get('comments'),
                    'reviewer': row.get('reviewer')
                })
        
        return feedback_data
    
    def _get_review_statistics(self) -> Dict[str, Any]:
        """Get statistics from human review data"""
        if not self.validation_history:
            return {"message": "No human review data available"}
        
        total_reviewed = len(self.validation_history)
        decisions = [entry['decision'] for entry in self.validation_history.values()]
        
        stats = {
            "total_reviewed": total_reviewed,
            "approved": decisions.count('approved'),
            "rejected": decisions.count('rejected'),
            "corrected": decisions.count('corrected'),
            "reviewers": list(self.reviewer_stats.keys()),
            "reviewer_stats": self.reviewer_stats
        }
        
        return stats
    
    def _generate_quality_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        total_results = len(validation_results)
        if total_results == 0:
            return recommendations
        
        # Calculate statistics for recommendations
        invalid_rate = sum(1 for r in validation_results if not r.is_valid) / total_results
        low_confidence_rate = sum(1 for r in validation_results if r.validation_confidence < 0.6) / total_results
        
        # Generate recommendations based on issues
        if invalid_rate > 0.1:
            recommendations.append(f"High invalid entity rate ({invalid_rate:.1%}). Consider adjusting extraction thresholds.")
        
        if low_confidence_rate > 0.2:
            recommendations.append(f"Many entities have low confidence ({low_confidence_rate:.1%}). Review model performance.")
        
        # Issue-specific recommendations
        issue_counts = {}
        for result in validation_results:
            for issue in result.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        if issue_counts:
            most_common_issue = max(issue_counts.items(), key=lambda x: x[1])
            if most_common_issue[1] > total_results * 0.1:
                recommendations.append(f"Address frequent issue: {most_common_issue[0]} (affects {most_common_issue[1]} entities)")
        
        # Type-specific recommendations
        type_performance = {}
        for result in validation_results:
            entity_type = result.entity.entity_type.value
            if entity_type not in type_performance:
                type_performance[entity_type] = {"total": 0, "valid": 0}
            
            type_performance[entity_type]["total"] += 1
            if result.is_valid:
                type_performance[entity_type]["valid"] += 1
        
        for entity_type, perf in type_performance.items():
            if perf["total"] >= 5:  # Only for types with sufficient samples
                validity_rate = perf["valid"] / perf["total"]
                if validity_rate < 0.7:
                    recommendations.append(f"Poor performance for {entity_type} entities ({validity_rate:.1%} valid). Review extraction rules.")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _is_valid_date_format(self, text: str) -> bool:
        """Check if text matches common date formats"""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\w+\s+\d{1,2},\s+\d{4}',
            r'\d{1,2}\s+\w+\s+\d{4}',
            r'\w+\s+\d{4}',
            r'\d{4}'
        ]
        
        return any(re.match(pattern, text.strip(), re.IGNORECASE) for pattern in date_patterns)
    
    def _calculate_context_relevance(self, entity: Entity) -> float:
        """Calculate how relevant the entity is to its context"""
        if not entity.context:
            return 0.5  # Neutral score if no context
        
        context = entity.context.lower()
        entity_text = entity.text.lower()
        
        # Basic relevance heuristics
        relevance_score = 0.5  # Base score
        
        # Check if entity appears naturally in context
        if entity_text in context:
            # Check surrounding words for relevance
            context_words = context.split()
            if entity.entity_type == EntityType.PERSON:
                person_indicators = ["said", "stated", "testified", "argued", "claimed"]
                if any(indicator in context_words for indicator in person_indicators):
                    relevance_score += 0.3
            
            elif entity.entity_type == EntityType.ORGANIZATION:
                org_indicators = ["company", "corporation", "firm", "organization"]
                if any(indicator in context_words for indicator in org_indicators):
                    relevance_score += 0.3
            
            elif entity.entity_type == EntityType.LOCATION:
                loc_indicators = ["located", "in", "at", "from", "city", "state"]
                if any(indicator in context_words for indicator in loc_indicators):
                    relevance_score += 0.3
        
        return min(1.0, relevance_score)
    
    def _estimate_type_accuracy(self, entity: Entity) -> float:
        """Estimate how accurate the entity type classification is"""
        text = entity.text.lower()
        entity_type = entity.entity_type
        
        # Basic heuristics for type accuracy
        if entity_type == EntityType.PERSON:
            # Check for name-like patterns
            if len(text.split()) >= 2 and text.istitle():
                return 0.8
            elif any(title in text for title in ["mr.", "mrs.", "ms.", "dr."]):
                return 0.9
            else:
                return 0.6
        
        elif entity_type == EntityType.ORGANIZATION:
            org_suffixes = ["inc", "corp", "llc", "ltd", "company", "corporation"]
            if any(suffix in text for suffix in org_suffixes):
                return 0.9
            else:
                return 0.7
        
        elif entity_type == EntityType.DATE:
            if self._is_valid_date_format(entity.text):
                return 0.9
            else:
                return 0.4
        
        # Default moderate confidence
        return 0.7
    
    def _estimate_boundary_accuracy(self, entity: Entity) -> float:
        """Estimate how accurate the entity boundaries are"""
        text = entity.text.strip()
        
        # Check for common boundary issues
        boundary_score = 0.8  # Base score
        
        # Check for leading/trailing punctuation or articles
        if text.startswith(('the ', 'a ', 'an ', '.', ',', ';')):
            boundary_score -= 0.2
        
        if text.endswith(('.', ',', ';', ':', '!', '?')):
            boundary_score -= 0.1
        
        # Check for incomplete words
        if text.startswith((' ', '\t')) or text.endswith((' ', '\t')):
            boundary_score -= 0.3
        
        # Check for reasonable length
        if len(text) < 2:
            boundary_score -= 0.4
        elif len(text) > 100:
            boundary_score -= 0.2
        
        return max(0.1, boundary_score)
    
    def _are_entities_similar(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities are similar (likely the same entity)"""
        text1 = entity1.text.lower().strip()
        text2 = entity2.text.lower().strip()
        
        # Exact match
        if text1 == text2:
            return True
        
        # Fuzzy match for similar entities
        from fuzzywuzzy import fuzz
        if fuzz.ratio(text1, text2) >= 85:
            return True
        
        # Check aliases
        all_aliases1 = [text1] + [alias.lower() for alias in entity1.aliases]
        all_aliases2 = [text2] + [alias.lower() for alias in entity2.aliases]
        
        for alias1 in all_aliases1:
            for alias2 in all_aliases2:
                if alias1 == alias2 or fuzz.ratio(alias1, alias2) >= 90:
                    return True
        
        return False