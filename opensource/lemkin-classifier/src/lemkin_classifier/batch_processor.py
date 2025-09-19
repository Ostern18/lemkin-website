"""
Batch processing system for high-volume legal document classification.

This module provides efficient batch processing capabilities for classifying
large numbers of documents with progress tracking, error handling, and
performance optimization.
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Iterator
from dataclasses import dataclass, asdict
from enum import Enum
import threading

import pandas as pd
from pydantic import BaseModel, Field, validator
from tqdm import tqdm

from .core import DocumentClassifier, DocumentContent, ClassificationResult, ClassificationConfig
from .confidence_scorer import ConfidenceScorer, ConfidenceAssessment
from .legal_taxonomy import DocumentType, LegalDomain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(str, Enum):
    """Status of batch processing"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ProcessingMode(str, Enum):
    """Processing execution mode"""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    ASYNC = "async"


class DocumentBatch(BaseModel):
    """Batch of documents for processing"""
    
    batch_id: str = Field(description="Unique batch identifier")
    documents: List[Union[str, Path, DocumentContent]] = Field(description="List of documents to process")
    batch_name: Optional[str] = Field(default=None, description="Optional batch name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")
    
    # Processing configuration
    priority: int = Field(default=0, description="Processing priority (higher = more priority)")
    max_retries: int = Field(default=3, description="Maximum retry attempts for failed documents")
    timeout_seconds: int = Field(default=300, description="Timeout per document in seconds")
    
    # Filtering and validation
    file_types: Optional[List[str]] = Field(default=None, description="Allowed file types")
    min_size: Optional[int] = Field(default=None, description="Minimum file size in bytes")
    max_size: Optional[int] = Field(default=None, description="Maximum file size in bytes")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('documents', pre=True)
    def validate_documents(cls, v):
        if not v:
            raise ValueError("Document list cannot be empty")
        return v
    
    def __len__(self) -> int:
        return len(self.documents)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }


class ProcessingConfig(BaseModel):
    """Configuration for batch processing"""
    
    # Performance settings
    max_workers: int = Field(default=4, ge=1, description="Maximum number of worker threads/processes")
    batch_size: int = Field(default=32, ge=1, description="Size of processing batches")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.THREADED, description="Processing execution mode")
    
    # Resource management
    memory_limit_gb: Optional[float] = Field(default=None, description="Memory limit in GB")
    gpu_enabled: bool = Field(default=True, description="Enable GPU acceleration if available")
    cache_size: int = Field(default=1000, description="Number of results to cache")
    
    # Error handling
    fail_fast: bool = Field(default=False, description="Stop processing on first error")
    continue_on_error: bool = Field(default=True, description="Continue processing other documents on error")
    error_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Max error rate before stopping")
    
    # Output configuration
    output_format: str = Field(default="json", description="Output format: json, csv, parquet")
    include_confidence: bool = Field(default=True, description="Include confidence assessments")
    include_raw_predictions: bool = Field(default=False, description="Include raw model predictions")
    
    # Progress tracking
    enable_progress_bar: bool = Field(default=True, description="Show progress bar")
    log_interval: int = Field(default=100, description="Log progress every N documents")
    checkpoint_interval: int = Field(default=500, description="Save checkpoint every N documents")
    
    # Quality control
    enable_quality_checks: bool = Field(default=True, description="Enable quality validation")
    min_confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence for results")


class BatchMetrics(BaseModel):
    """Metrics and statistics for batch processing"""
    
    # Timing metrics
    start_time: datetime = Field(description="Processing start time")
    end_time: Optional[datetime] = Field(default=None, description="Processing end time")
    total_duration: Optional[float] = Field(default=None, description="Total processing time in seconds")
    average_time_per_document: Optional[float] = Field(default=None, description="Average processing time per document")
    
    # Volume metrics
    total_documents: int = Field(description="Total number of documents processed")
    successful_documents: int = Field(default=0, description="Successfully processed documents")
    failed_documents: int = Field(default=0, description="Failed documents")
    skipped_documents: int = Field(default=0, description="Skipped documents")
    
    # Quality metrics
    average_confidence: Optional[float] = Field(default=None, description="Average confidence score")
    confidence_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of confidence levels")
    review_required_count: int = Field(default=0, description="Documents requiring human review")
    
    # Performance metrics
    documents_per_second: Optional[float] = Field(default=None, description="Processing throughput")
    peak_memory_usage: Optional[float] = Field(default=None, description="Peak memory usage in GB")
    cpu_utilization: Optional[float] = Field(default=None, description="Average CPU utilization")
    
    # Error statistics
    error_rate: float = Field(default=0.0, description="Error rate (0-1)")
    error_types: Dict[str, int] = Field(default_factory=dict, description="Counts of different error types")
    retry_count: int = Field(default=0, description="Total number of retries")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchProcessingResult(BaseModel):
    """Complete result of batch processing"""
    
    batch_id: str = Field(description="Batch identifier")
    status: ProcessingStatus = Field(description="Processing status")
    
    # Results
    results: List[ClassificationResult] = Field(description="Classification results")
    failed_documents: List[Dict[str, Any]] = Field(default_factory=list, description="Failed document information")
    
    # Metrics and statistics
    metrics: BatchMetrics = Field(description="Processing metrics")
    
    # Configuration used
    processing_config: ProcessingConfig = Field(description="Configuration used for processing")
    classification_config: ClassificationConfig = Field(description="Classification model configuration")
    
    # Output information
    output_files: List[str] = Field(default_factory=list, description="Generated output files")
    checkpoint_files: List[str] = Field(default_factory=list, description="Checkpoint files created")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@dataclass
class ProcessingTask:
    """Individual processing task"""
    document_id: str
    document: Union[str, Path, DocumentContent]
    batch_id: str
    retry_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[ClassificationResult] = None
    error: Optional[str] = None


class BatchProcessor:
    """High-performance batch processor for legal document classification"""
    
    def __init__(
        self,
        classifier: DocumentClassifier,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        config: Optional[ProcessingConfig] = None
    ):
        """
        Initialize the batch processor
        
        Args:
            classifier: Document classifier instance
            confidence_scorer: Optional confidence scorer
            config: Processing configuration
        """
        self.classifier = classifier
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.config = config or ProcessingConfig()
        
        # Processing state
        self.current_batch: Optional[DocumentBatch] = None
        self.processing_status = ProcessingStatus.PENDING
        self.results_cache = {}
        self.error_counts = {}
        
        # Threading and synchronization
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()
        
        logger.info(f"BatchProcessor initialized with {self.config.max_workers} workers")
    
    def process_batch(
        self,
        batch: DocumentBatch,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> BatchProcessingResult:
        """
        Process a batch of documents
        
        Args:
            batch: Document batch to process
            output_dir: Optional output directory for results
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchProcessingResult with complete processing information
        """
        self.current_batch = batch
        self.processing_status = ProcessingStatus.RUNNING
        
        # Initialize metrics
        metrics = BatchMetrics(
            start_time=datetime.now(timezone.utc),
            total_documents=len(batch.documents)
        )
        
        results = []
        failed_documents = []
        
        try:
            # Create processing tasks
            tasks = [
                ProcessingTask(
                    document_id=f"{batch.batch_id}_{i}",
                    document=doc,
                    batch_id=batch.batch_id
                )
                for i, doc in enumerate(batch.documents)
            ]
            
            # Process based on selected mode
            if self.config.processing_mode == ProcessingMode.SEQUENTIAL:
                results, failed_documents = self._process_sequential(tasks, progress_callback)
            elif self.config.processing_mode == ProcessingMode.THREADED:
                results, failed_documents = self._process_threaded(tasks, progress_callback)
            elif self.config.processing_mode == ProcessingMode.MULTIPROCESS:
                results, failed_documents = self._process_multiprocess(tasks, progress_callback)
            elif self.config.processing_mode == ProcessingMode.ASYNC:
                results, failed_documents = asyncio.run(self._process_async(tasks, progress_callback))
            
            self.processing_status = ProcessingStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.processing_status = ProcessingStatus.FAILED
            failed_documents.append({
                'error': str(e),
                'error_type': type(e).__name__,
                'batch_level': True
            })
        
        # Finalize metrics
        metrics.end_time = datetime.now(timezone.utc)
        metrics.total_duration = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.successful_documents = len(results)
        metrics.failed_documents = len(failed_documents)
        metrics.error_rate = metrics.failed_documents / metrics.total_documents if metrics.total_documents > 0 else 0
        
        if metrics.total_duration > 0:
            metrics.documents_per_second = metrics.successful_documents / metrics.total_duration
            metrics.average_time_per_document = metrics.total_duration / metrics.total_documents
        
        # Calculate confidence statistics
        if results:
            confidences = [r.classification.confidence_score for r in results]
            metrics.average_confidence = sum(confidences) / len(confidences)
            metrics.confidence_distribution = self._calculate_confidence_distribution(results)
            metrics.review_required_count = sum(1 for r in results if r.requires_review)
        
        # Create final result
        batch_result = BatchProcessingResult(
            batch_id=batch.batch_id,
            status=self.processing_status,
            results=results,
            failed_documents=failed_documents,
            metrics=metrics,
            processing_config=self.config,
            classification_config=self.classifier.config
        )
        
        # Save results if output directory specified
        if output_dir:
            batch_result.output_files = self._save_results(batch_result, Path(output_dir))
        
        logger.info(
            f"Batch processing completed: {metrics.successful_documents}/{metrics.total_documents} successful, "
            f"{metrics.error_rate:.2%} error rate"
        )
        
        return batch_result
    
    def _process_sequential(
        self,
        tasks: List[ProcessingTask],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[List[ClassificationResult], List[Dict[str, Any]]]:
        """Process tasks sequentially"""
        results = []
        failed_documents = []
        
        progress_bar = None
        if self.config.enable_progress_bar:
            progress_bar = tqdm(total=len(tasks), desc="Processing documents")
        
        for i, task in enumerate(tasks):
            if self._stop_event.is_set():
                break
            
            while self._pause_event.is_set():
                time.sleep(0.1)
            
            try:
                task.start_time = datetime.now(timezone.utc)
                result = self._process_single_document(task.document)
                task.end_time = datetime.now(timezone.utc)
                task.result = result
                results.append(result)
                
                if progress_callback:
                    progress = (i + 1) / len(tasks)
                    progress_callback(progress, f"Processed {i + 1}/{len(tasks)} documents")
                
            except Exception as e:
                error_info = {
                    'document_id': task.document_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                failed_documents.append(error_info)
                logger.error(f"Failed to process document {task.document_id}: {e}")
                
                if not self.config.continue_on_error:
                    break
            
            if progress_bar:
                progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
        
        return results, failed_documents
    
    def _process_threaded(
        self,
        tasks: List[ProcessingTask],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[List[ClassificationResult], List[Dict[str, Any]]]:
        """Process tasks using thread pool"""
        results = []
        failed_documents = []
        completed_count = 0
        
        progress_bar = None
        if self.config.enable_progress_bar:
            progress_bar = tqdm(total=len(tasks), desc="Processing documents")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_document, task.document): task
                for task in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                if self._stop_event.is_set():
                    break
                
                task = future_to_task[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    task.result = result
                    results.append(result)
                    
                except Exception as e:
                    error_info = {
                        'document_id': task.document_id,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    failed_documents.append(error_info)
                    logger.error(f"Failed to process document {task.document_id}: {e}")
                
                if progress_bar:
                    progress_bar.update(1)
                
                if progress_callback:
                    progress = completed_count / len(tasks)
                    progress_callback(progress, f"Processed {completed_count}/{len(tasks)} documents")
                
                # Check error threshold
                error_rate = len(failed_documents) / completed_count
                if error_rate > self.config.error_threshold and not self.config.continue_on_error:
                    logger.error(f"Error rate {error_rate:.2%} exceeds threshold {self.config.error_threshold:.2%}")
                    break
        
        if progress_bar:
            progress_bar.close()
        
        return results, failed_documents
    
    def _process_multiprocess(
        self,
        tasks: List[ProcessingTask],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[List[ClassificationResult], List[Dict[str, Any]]]:
        """Process tasks using process pool"""
        # Note: This is a simplified implementation
        # In practice, you'd need to handle model serialization/loading in each process
        logger.warning("Multiprocess mode not fully implemented - falling back to threaded mode")
        return self._process_threaded(tasks, progress_callback)
    
    async def _process_async(
        self,
        tasks: List[ProcessingTask],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[List[ClassificationResult], List[Dict[str, Any]]]:
        """Process tasks asynchronously"""
        results = []
        failed_documents = []
        
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_task(task: ProcessingTask):
            async with semaphore:
                try:
                    # Run synchronous classification in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, self._process_single_document, task.document
                    )
                    return task, result, None
                except Exception as e:
                    return task, None, e
        
        # Create and run all tasks
        task_coroutines = [process_task(task) for task in tasks]
        
        progress_bar = None
        if self.config.enable_progress_bar:
            progress_bar = tqdm(total=len(tasks), desc="Processing documents")
        
        completed_count = 0
        for coro in asyncio.as_completed(task_coroutines):
            task, result, error = await coro
            completed_count += 1
            
            if error:
                error_info = {
                    'document_id': task.document_id,
                    'error': str(error),
                    'error_type': type(error).__name__,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                failed_documents.append(error_info)
            else:
                results.append(result)
            
            if progress_bar:
                progress_bar.update(1)
            
            if progress_callback:
                progress = completed_count / len(tasks)
                progress_callback(progress, f"Processed {completed_count}/{len(tasks)} documents")
        
        if progress_bar:
            progress_bar.close()
        
        return results, failed_documents
    
    def _process_single_document(self, document: Union[str, Path, DocumentContent]) -> ClassificationResult:
        """Process a single document"""
        if isinstance(document, DocumentContent):
            return self.classifier.classify_document(document)
        else:
            return self.classifier.classify_file(document)
    
    def _calculate_confidence_distribution(self, results: List[ClassificationResult]) -> Dict[str, int]:
        """Calculate distribution of confidence levels"""
        distribution = {
            'very_low': 0,
            'low': 0,
            'medium': 0,
            'high': 0,
            'very_high': 0
        }
        
        for result in results:
            confidence = result.classification.confidence_score
            if confidence >= 0.9:
                distribution['very_high'] += 1
            elif confidence >= 0.8:
                distribution['high'] += 1
            elif confidence >= 0.6:
                distribution['medium'] += 1
            elif confidence >= 0.4:
                distribution['low'] += 1
            else:
                distribution['very_low'] += 1
        
        return distribution
    
    def _save_results(self, batch_result: BatchProcessingResult, output_dir: Path) -> List[str]:
        """Save batch results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []
        
        base_name = f"batch_{batch_result.batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save main results
        if self.config.output_format.lower() == "json":
            results_file = output_dir / f"{base_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump([result.dict() for result in batch_result.results], f, indent=2, default=str)
            output_files.append(str(results_file))
            
        elif self.config.output_format.lower() == "csv":
            results_file = output_dir / f"{base_name}_results.csv"
            df = pd.DataFrame([self._flatten_result(result) for result in batch_result.results])
            df.to_csv(results_file, index=False)
            output_files.append(str(results_file))
        
        # Save metrics
        metrics_file = output_dir / f"{base_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(batch_result.metrics.dict(), f, indent=2, default=str)
        output_files.append(str(metrics_file))
        
        # Save failed documents info
        if batch_result.failed_documents:
            failures_file = output_dir / f"{base_name}_failures.json"
            with open(failures_file, 'w') as f:
                json.dump(batch_result.failed_documents, f, indent=2, default=str)
            output_files.append(str(failures_file))
        
        return output_files
    
    def _flatten_result(self, result: ClassificationResult) -> Dict[str, Any]:
        """Flatten classification result for CSV export"""
        return {
            'file_path': result.document_content.file_path,
            'document_type': result.classification.document_type.value,
            'legal_domain': result.classification.legal_domain.value,
            'confidence_score': result.classification.confidence_score,
            'urgency_level': result.urgency_level,
            'sensitivity_level': result.sensitivity_level,
            'requires_review': result.requires_review,
            'review_reasons': '; '.join(result.review_reasons),
            'processing_time': result.processing_time,
            'text_length': result.document_content.length,
            'word_count': result.document_content.word_count,
        }
    
    def create_batch_from_directory(
        self,
        directory: Path,
        batch_name: Optional[str] = None,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True
    ) -> DocumentBatch:
        """
        Create a document batch from a directory
        
        Args:
            directory: Directory containing documents
            batch_name: Optional name for the batch
            file_patterns: File patterns to match (e.g., ['*.pdf', '*.docx'])
            recursive: Whether to search recursively
            
        Returns:
            DocumentBatch ready for processing
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find matching files
        documents = []
        patterns = file_patterns or ['*.pdf', '*.docx', '*.txt']
        
        for pattern in patterns:
            if recursive:
                documents.extend(directory.rglob(pattern))
            else:
                documents.extend(directory.glob(pattern))
        
        # Remove duplicates and sort
        documents = sorted(set(documents))
        
        if not documents:
            raise ValueError(f"No matching documents found in {directory}")
        
        batch_id = f"dir_{directory.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return DocumentBatch(
            batch_id=batch_id,
            documents=documents,
            batch_name=batch_name or f"Directory batch: {directory.name}",
            metadata={
                'source_directory': str(directory),
                'file_patterns': patterns,
                'recursive': recursive,
                'total_files': len(documents)
            }
        )
    
    def stop_processing(self) -> None:
        """Stop current processing"""
        self._stop_event.set()
        self.processing_status = ProcessingStatus.CANCELLED
        logger.info("Processing stopped by user")
    
    def pause_processing(self) -> None:
        """Pause current processing"""
        self._pause_event.set()
        self.processing_status = ProcessingStatus.PAUSED
        logger.info("Processing paused")
    
    def resume_processing(self) -> None:
        """Resume paused processing"""
        self._pause_event.clear()
        self.processing_status = ProcessingStatus.RUNNING
        logger.info("Processing resumed")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            'status': self.processing_status.value,
            'current_batch': self.current_batch.batch_id if self.current_batch else None,
            'cache_size': len(self.results_cache),
            'error_counts': self.error_counts,
            'is_stopped': self._stop_event.is_set(),
            'is_paused': self._pause_event.is_set(),
        }