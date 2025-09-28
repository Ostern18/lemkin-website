"""
Evidence Cataloger for Lemkin Report Generator Suite.

This module provides the EvidenceCataloger class for creating comprehensive
evidence inventories with chain of custody tracking, authenticity verification,
and detailed cataloging suitable for legal proceedings.
"""

from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
from collections import defaultdict

from loguru import logger

from .core import (
    BaseReportModel, ReportConfig, EvidenceCatalog, CaseData, PersonInfo,
    Evidence, EvidenceType, EvidenceAuthenticity, ConfidentialityLevel,
    ReportType, ReportStatus
)


class ChainOfCustodyEntry(BaseReportModel):
    """Individual entry in chain of custody log"""
    custodian_name: str = Field(..., min_length=1)
    custodian_role: str = Field(..., min_length=1)
    transfer_date: datetime = Field(..., description="Date and time of transfer")
    transfer_reason: str = Field(..., min_length=1)
    transfer_method: str = Field(..., description="How evidence was transferred")
    receiving_party: Optional[str] = None
    location: str = Field(..., description="Storage location")
    condition_notes: Optional[str] = None
    witness: Optional[str] = None
    signature: Optional[str] = None
    
    # Integrity verification
    hash_verified: bool = Field(default=False)
    seal_intact: bool = Field(default=True)
    tamper_evident: bool = Field(default=False)


class EvidenceAnalysis(BaseReportModel):
    """Analysis results for individual evidence item"""
    evidence_id: str = Field(..., min_length=1)
    analysis_type: str = Field(..., min_length=1)
    analyst: PersonInfo
    analysis_date: date = Field(default_factory=date.today)
    
    # Results
    findings: str = Field(..., min_length=1)
    conclusions: List[str] = Field(default_factory=list)
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    
    # Technical details
    methodology: str = Field(..., min_length=1)
    equipment_used: List[str] = Field(default_factory=list)
    standards_followed: List[str] = Field(default_factory=list)
    
    # Quality assurance
    peer_reviewed: bool = Field(default=False)
    reviewer: Optional[PersonInfo] = None
    review_date: Optional[date] = None
    
    # Supporting materials
    reports_generated: List[str] = Field(default_factory=list)
    images_created: List[str] = Field(default_factory=list)
    data_files: List[str] = Field(default_factory=list)


class EvidenceSearchIndex:
    """Search index for evidence catalog"""
    
    def __init__(self):
        self.by_type: Dict[str, List[str]] = defaultdict(list)
        self.by_source: Dict[str, List[str]] = defaultdict(list)
        self.by_date: Dict[str, List[str]] = defaultdict(list)
        self.by_custodian: Dict[str, List[str]] = defaultdict(list)
        self.by_authenticity: Dict[str, List[str]] = defaultdict(list)
        self.by_relevance: Dict[float, List[str]] = defaultdict(list)
        self.by_tags: Dict[str, List[str]] = defaultdict(list)
        self.full_text: Dict[str, str] = {}
    
    def index_evidence(self, evidence: Evidence):
        """Add evidence item to search indices"""
        eid = evidence.evidence_id
        
        # Index by type
        self.by_type[evidence.evidence_type.value].append(eid)
        
        # Index by source
        self.by_source[evidence.source].append(eid)
        
        # Index by collection date
        if evidence.date_collected:
            date_key = evidence.date_collected.strftime("%Y-%m")
            self.by_date[date_key].append(eid)
        
        # Index by custodian
        self.by_custodian[evidence.custodian].append(eid)
        
        # Index by authenticity status
        self.by_authenticity[evidence.authenticity_status.value].append(eid)
        
        # Index by relevance score
        relevance_tier = round(evidence.evidential_weight * 10) / 10
        self.by_relevance[relevance_tier].append(eid)
        
        # Index by tags
        for tag in evidence.tags:
            self.by_tags[tag.lower()].append(eid)
        
        # Full text indexing
        searchable_text = " ".join([
            evidence.title,
            evidence.description,
            evidence.source,
            evidence.relevance_to_case,
            " ".join(evidence.tags),
            " ".join(evidence.categories)
        ]).lower()
        self.full_text[eid] = searchable_text
    
    def search(self, query: str) -> List[str]:
        """Search evidence by text query"""
        query_lower = query.lower()
        results = []
        
        for eid, text in self.full_text.items():
            if query_lower in text:
                results.append(eid)
        
        return results
    
    def filter_by_criteria(self, criteria: Dict[str, Any]) -> List[str]:
        """Filter evidence by multiple criteria"""
        result_sets = []
        
        if 'type' in criteria:
            result_sets.append(set(self.by_type.get(criteria['type'], [])))
        
        if 'source' in criteria:
            result_sets.append(set(self.by_source.get(criteria['source'], [])))
        
        if 'custodian' in criteria:
            result_sets.append(set(self.by_custodian.get(criteria['custodian'], [])))
        
        if 'authenticity' in criteria:
            result_sets.append(set(self.by_authenticity.get(criteria['authenticity'], [])))
        
        if 'min_relevance' in criteria:
            min_rel = criteria['min_relevance']
            relevant_items = set()
            for rel_score, items in self.by_relevance.items():
                if rel_score >= min_rel:
                    relevant_items.update(items)
            result_sets.append(relevant_items)
        
        if 'tags' in criteria:
            tag_items = set()
            for tag in criteria['tags']:
                tag_items.update(self.by_tags.get(tag.lower(), []))
            result_sets.append(tag_items)
        
        # Return intersection of all criteria
        if result_sets:
            return list(set.intersection(*result_sets))
        else:
            return list(self.full_text.keys())


class EvidenceQualityChecker:
    """Quality assurance for evidence cataloging"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logger.bind(component="evidence_quality_checker")
    
    def check_evidence_quality(self, evidence: Evidence) -> Dict[str, Any]:
        """Perform quality checks on evidence item"""
        quality_report = {
            "score": 0.0,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "completeness": 0.0,
            "chain_of_custody_intact": True,
            "authentication_status": "pending"
        }
        
        quality_factors = []
        
        # Check required fields
        if not evidence.evidence_id:
            quality_report["issues"].append("Evidence ID is required")
        else:
            quality_factors.append(1.0)
        
        if not evidence.title or len(evidence.title.strip()) < 5:
            quality_report["issues"].append("Title must be at least 5 characters")
            quality_factors.append(0.0)
        else:
            quality_factors.append(1.0)
        
        if not evidence.description or len(evidence.description.strip()) < 20:
            quality_report["warnings"].append("Description should be more detailed")
            quality_factors.append(0.7)
        else:
            quality_factors.append(1.0)
        
        # Check chain of custody
        if not evidence.chain_of_custody:
            quality_report["warnings"].append("No chain of custody entries")
            quality_report["chain_of_custody_intact"] = False
            quality_factors.append(0.5)
        else:
            # Verify chain integrity
            custody_score = self._verify_chain_of_custody(evidence.chain_of_custody)
            if custody_score < 0.8:
                quality_report["issues"].append("Chain of custody has gaps or inconsistencies")
                quality_report["chain_of_custody_intact"] = False
            quality_factors.append(custody_score)
        
        # Check file information
        if evidence.evidence_type in [EvidenceType.DOCUMENT, EvidenceType.PHOTOGRAPH, 
                                      EvidenceType.VIDEO, EvidenceType.AUDIO, EvidenceType.DIGITAL]:
            if not evidence.file_path:
                quality_report["warnings"].append("File path not specified for digital evidence")
                quality_factors.append(0.6)
            elif not evidence.file_hash:
                quality_report["recommendations"].append("Calculate file hash for integrity verification")
                quality_factors.append(0.8)
            else:
                quality_factors.append(1.0)
        
        # Check authentication status
        if evidence.authenticity_status == EvidenceAuthenticity.PENDING_VERIFICATION:
            quality_report["recommendations"].append("Complete authenticity verification")
            quality_factors.append(0.7)
        elif evidence.authenticity_status == EvidenceAuthenticity.AUTHENTIC:
            if not evidence.authentication_method:
                quality_report["warnings"].append("Authentication method not documented")
                quality_factors.append(0.8)
            else:
                quality_factors.append(1.0)
        elif evidence.authenticity_status == EvidenceAuthenticity.SUSPICIOUS:
            quality_report["issues"].append("Evidence marked as suspicious - requires review")
            quality_factors.append(0.3)
        
        # Check relevance documentation
        if not evidence.relevance_to_case:
            quality_report["issues"].append("Relevance to case must be documented")
            quality_factors.append(0.0)
        elif len(evidence.relevance_to_case.strip()) < 20:
            quality_report["warnings"].append("Relevance description should be more detailed")
            quality_factors.append(0.7)
        else:
            quality_factors.append(1.0)
        
        # Check privilege and confidentiality
        if evidence.privilege_claims and not evidence.confidentiality_level:
            quality_report["warnings"].append("Confidentiality level should be set for privileged material")
        
        # Calculate overall scores
        if quality_factors:
            quality_report["score"] = sum(quality_factors) / len(quality_factors)
            quality_report["completeness"] = quality_report["score"]
        
        # Set authentication status
        if evidence.authenticity_status != EvidenceAuthenticity.PENDING_VERIFICATION:
            quality_report["authentication_status"] = evidence.authenticity_status.value
        
        return quality_report
    
    def _verify_chain_of_custody(self, chain_entries: List[Dict[str, Any]]) -> float:
        """Verify integrity of chain of custody"""
        if not chain_entries:
            return 0.0
        
        integrity_score = 1.0
        issues = []
        
        # Sort entries by date
        try:
            sorted_entries = sorted(chain_entries, 
                                  key=lambda x: x.get('transfer_date', datetime.min))
        except (TypeError, KeyError):
            return 0.5  # Invalid date format
        
        # Check for gaps in chain
        for i in range(1, len(sorted_entries)):
            prev_entry = sorted_entries[i-1]
            curr_entry = sorted_entries[i]
            
            # Check if receiving party matches next custodian
            prev_receiver = prev_entry.get('receiving_party')
            curr_custodian = curr_entry.get('custodian_name')
            
            if prev_receiver and curr_custodian and prev_receiver != curr_custodian:
                integrity_score *= 0.8  # Reduce score for chain gaps
        
        # Check for required fields
        required_fields = ['custodian_name', 'transfer_date', 'location']
        for entry in chain_entries:
            missing_fields = [f for f in required_fields if not entry.get(f)]
            if missing_fields:
                integrity_score *= 0.9  # Reduce score for missing fields
        
        return max(integrity_score, 0.0)
    
    def check_catalog_completeness(self, catalog: EvidenceCatalog) -> Dict[str, Any]:
        """Check completeness of evidence catalog"""
        completeness_report = {
            "overall_score": 0.0,
            "item_count": len(catalog.case_data.evidence_list),
            "authenticated_count": 0,
            "pending_authentication": 0,
            "missing_information": [],
            "quality_issues": [],
            "recommendations": []
        }
        
        if not catalog.case_data.evidence_list:
            completeness_report["missing_information"].append("No evidence items cataloged")
            return completeness_report
        
        # Analyze each evidence item
        quality_scores = []
        auth_counts = defaultdict(int)
        
        for evidence in catalog.case_data.evidence_list:
            quality_check = self.check_evidence_quality(evidence)
            quality_scores.append(quality_check["score"])
            auth_counts[evidence.authenticity_status.value] += 1
        
        completeness_report["overall_score"] = sum(quality_scores) / len(quality_scores)
        completeness_report["authenticated_count"] = auth_counts.get("authentic", 0)
        completeness_report["pending_authentication"] = auth_counts.get("pending_verification", 0)
        
        # Check catalog-level completeness
        if not catalog.inventory_complete:
            completeness_report["recommendations"].append("Mark inventory as complete after verification")
        
        if not catalog.authentication_summary:
            completeness_report["missing_information"].append("Authentication summary not provided")
        
        if catalog.pending_authentications:
            completeness_report["recommendations"].append(
                f"Complete authentication for {len(catalog.pending_authentications)} pending items"
            )
        
        return completeness_report


class EvidenceReportGenerator:
    """Generates various evidence-related reports"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logger.bind(component="evidence_report_generator")
    
    def generate_chain_of_custody_report(self, evidence: Evidence) -> str:
        """Generate detailed chain of custody report for evidence item"""
        report_lines = [
            f"CHAIN OF CUSTODY REPORT",
            f"Evidence ID: {evidence.evidence_id}",
            f"Evidence Title: {evidence.title}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "CHAIN OF CUSTODY ENTRIES:",
            ""
        ]
        
        if not evidence.chain_of_custody:
            report_lines.append("No chain of custody entries recorded.")
            return "\n".join(report_lines)
        
        # Sort entries by date
        sorted_entries = sorted(evidence.chain_of_custody, 
                               key=lambda x: x.get('transfer_date', datetime.min))
        
        for i, entry in enumerate(sorted_entries, 1):
            report_lines.extend([
                f"Entry #{i}:",
                f"  Custodian: {entry.get('custodian_name', 'Unknown')}",
                f"  Role: {entry.get('custodian_role', 'Unknown')}",
                f"  Date: {entry.get('transfer_date', 'Unknown')}",
                f"  Reason: {entry.get('transfer_reason', 'Not specified')}",
                f"  Method: {entry.get('transfer_method', 'Not specified')}",
                f"  Location: {entry.get('location', 'Not specified')}",
                ""
            ])
            
            if entry.get('receiving_party'):
                report_lines.append(f"  Transferred to: {entry['receiving_party']}")
            
            if entry.get('condition_notes'):
                report_lines.append(f"  Condition: {entry['condition_notes']}")
            
            if entry.get('witness'):
                report_lines.append(f"  Witness: {entry['witness']}")
            
            # Add integrity checks
            integrity_notes = []
            if entry.get('hash_verified'):
                integrity_notes.append("Hash verified")
            if not entry.get('seal_intact', True):
                integrity_notes.append("SEAL BROKEN")
            if entry.get('tamper_evident'):
                integrity_notes.append("TAMPERING DETECTED")
            
            if integrity_notes:
                report_lines.append(f"  Integrity: {', '.join(integrity_notes)}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def generate_authenticity_summary(self, evidence_list: List[Evidence]) -> str:
        """Generate summary of evidence authenticity status"""
        auth_counts = defaultdict(int)
        auth_methods = defaultdict(int)
        
        for evidence in evidence_list:
            auth_counts[evidence.authenticity_status.value] += 1
            if evidence.authentication_method:
                auth_methods[evidence.authentication_method] += 1
        
        summary_lines = [
            "EVIDENCE AUTHENTICITY SUMMARY",
            f"Total Items: {len(evidence_list)}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "AUTHENTICITY STATUS:",
        ]
        
        for status, count in auth_counts.items():
            percentage = (count / len(evidence_list)) * 100
            summary_lines.append(f"  {status.title()}: {count} ({percentage:.1f}%)")
        
        if auth_methods:
            summary_lines.extend([
                "",
                "AUTHENTICATION METHODS USED:",
            ])
            for method, count in auth_methods.items():
                summary_lines.append(f"  {method}: {count} items")
        
        return "\n".join(summary_lines)
    
    def generate_evidence_timeline(self, evidence_list: List[Evidence]) -> str:
        """Generate chronological timeline of evidence collection"""
        dated_evidence = [e for e in evidence_list if e.date_collected]
        if not dated_evidence:
            return "No evidence collection dates available for timeline."
        
        sorted_evidence = sorted(dated_evidence, key=lambda e: e.date_collected)
        
        timeline_lines = [
            "EVIDENCE COLLECTION TIMELINE",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        current_date = None
        for evidence in sorted_evidence:
            evidence_date = evidence.date_collected
            
            if evidence_date != current_date:
                current_date = evidence_date
                timeline_lines.append(f"{current_date.strftime('%Y-%m-%d')}:")
            
            timeline_lines.append(f"  • {evidence.title} ({evidence.evidence_type.value})")
            if evidence.source:
                timeline_lines.append(f"    Source: {evidence.source}")
        
        return "\n".join(timeline_lines)


class EvidenceCataloger:
    """
    Comprehensive evidence cataloging and inventory management.
    
    Provides systematic evidence organization, chain of custody tracking,
    authenticity verification, and detailed reporting suitable for legal
    proceedings and case management.
    """
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.quality_checker = EvidenceQualityChecker(config)
        self.report_generator = EvidenceReportGenerator(config)
        self.logger = logger.bind(component="evidence_cataloger")
    
    def catalog(
        self,
        evidence_list: List[Evidence],
        case_data: CaseData,
        custodian: PersonInfo,
        custom_organization: Optional[Dict[str, Any]] = None
    ) -> EvidenceCatalog:
        """
        Create comprehensive evidence catalog with organization and tracking
        
        Args:
            evidence_list: List of evidence items to catalog
            case_data: Associated case information
            custodian: Person responsible for evidence custody
            custom_organization: Custom organization scheme (optional)
            
        Returns:
            EvidenceCatalog with organized evidence inventory
        """
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting evidence cataloging for {len(evidence_list)} items")
        
        # Initialize catalog
        catalog = EvidenceCatalog(
            case_data=case_data,
            title=f"Evidence Catalog: {case_data.case_info.case_name}",
            custodian=custodian,
            catalog_date=date.today(),
            inventory_complete=False
        )
        
        try:
            # Create search index
            search_index = EvidenceSearchIndex()
            for evidence in evidence_list:
                search_index.index_evidence(evidence)
            
            # Organize evidence by category
            catalog.evidence_by_category = self._organize_by_category(evidence_list)
            
            # Organize evidence by relevance
            catalog.evidence_by_relevance = self._organize_by_relevance(evidence_list)
            
            # Create evidence timeline
            catalog.evidence_timeline = self._create_evidence_timeline(evidence_list)
            
            # Generate custody log
            catalog.custody_log = self._generate_custody_log(evidence_list)
            
            # Create authentication summary
            catalog.authentication_summary = self._create_authentication_summary(evidence_list)
            
            # Identify pending authentications
            catalog.pending_authentications = self._identify_pending_authentications(evidence_list)
            
            # Generate analysis summaries
            catalog.forensic_analysis_summary = self._generate_forensic_summary(evidence_list)
            catalog.expert_opinions_summary = self._generate_expert_opinions_summary(evidence_list)
            
            # Create admissibility analysis
            catalog.admissibility_analysis = self._analyze_admissibility(evidence_list)
            
            # Perform quality control
            completeness_check = self.quality_checker.check_catalog_completeness(catalog)
            catalog.discrepancies_noted = completeness_check.get("quality_issues", [])
            
            # Mark as complete if quality threshold met
            if completeness_check["overall_score"] >= 0.8:
                catalog.inventory_complete = True
                catalog.verification_date = date.today()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Evidence cataloging completed in {duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Evidence cataloging failed: {str(e)}")
            catalog.discrepancies_noted.append(f"Cataloging error: {str(e)}")
            raise
        
        return catalog
    
    def _organize_by_category(self, evidence_list: List[Evidence]) -> Dict[str, List[str]]:
        """Organize evidence by type and category"""
        organization = defaultdict(list)
        
        # Organize by evidence type
        for evidence in evidence_list:
            organization[evidence.evidence_type.value].append(evidence.evidence_id)
        
        # Organize by custom categories
        for evidence in evidence_list:
            for category in evidence.categories:
                organization[category].append(evidence.evidence_id)
        
        # Organize by confidentiality level
        for evidence in evidence_list:
            conf_key = f"confidential_{evidence.confidentiality_level.value}"
            organization[conf_key].append(evidence.evidence_id)
        
        return dict(organization)
    
    def _organize_by_relevance(self, evidence_list: List[Evidence]) -> Dict[str, List[str]]:
        """Organize evidence by relevance tiers"""
        organization = {
            "high_relevance": [],
            "medium_relevance": [],
            "low_relevance": [],
            "supporting": [],
            "contradictory": []
        }
        
        for evidence in evidence_list:
            # Organize by evidential weight
            if evidence.evidential_weight >= 0.8:
                organization["high_relevance"].append(evidence.evidence_id)
            elif evidence.evidential_weight >= 0.6:
                organization["medium_relevance"].append(evidence.evidence_id)
            else:
                organization["low_relevance"].append(evidence.evidence_id)
            
            # Organize by relationship type
            if evidence.supporting_evidence_ids:
                organization["supporting"].append(evidence.evidence_id)
            if evidence.contradictory_evidence_ids:
                organization["contradictory"].append(evidence.evidence_id)
        
        return organization
    
    def _create_evidence_timeline(self, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """Create chronological timeline of evidence"""
        timeline_entries = []
        
        # Add collection events
        for evidence in evidence_list:
            if evidence.date_collected:
                timeline_entries.append({
                    "date": evidence.date_collected,
                    "event_type": "collection",
                    "evidence_id": evidence.evidence_id,
                    "description": f"Collected: {evidence.title}",
                    "custodian": evidence.custodian,
                    "source": evidence.source
                })
        
        # Add authentication events
        for evidence in evidence_list:
            if evidence.authentication_date:
                timeline_entries.append({
                    "date": evidence.authentication_date,
                    "event_type": "authentication", 
                    "evidence_id": evidence.evidence_id,
                    "description": f"Authenticated: {evidence.title}",
                    "method": evidence.authentication_method,
                    "party": evidence.authenticating_party,
                    "status": evidence.authenticity_status.value
                })
        
        # Sort by date
        timeline_entries.sort(key=lambda x: x["date"])
        
        return timeline_entries
    
    def _generate_custody_log(self, evidence_list: List[Evidence]) -> List[Dict[str, Any]]:
        """Generate comprehensive custody log"""
        custody_log = []
        
        for evidence in evidence_list:
            if evidence.chain_of_custody:
                for entry in evidence.chain_of_custody:
                    log_entry = {
                        "evidence_id": evidence.evidence_id,
                        "evidence_title": evidence.title,
                        **entry
                    }
                    custody_log.append(log_entry)
        
        # Sort by transfer date
        custody_log.sort(key=lambda x: x.get("transfer_date", datetime.min))
        
        return custody_log
    
    def _create_authentication_summary(self, evidence_list: List[Evidence]) -> Dict[str, int]:
        """Create summary of authentication status"""
        summary = defaultdict(int)
        
        for evidence in evidence_list:
            summary[evidence.authenticity_status.value] += 1
        
        return dict(summary)
    
    def _identify_pending_authentications(self, evidence_list: List[Evidence]) -> List[str]:
        """Identify evidence items pending authentication"""
        pending = []
        
        for evidence in evidence_list:
            if evidence.authenticity_status == EvidenceAuthenticity.PENDING_VERIFICATION:
                pending.append(evidence.evidence_id)
        
        return pending
    
    def _generate_forensic_summary(self, evidence_list: List[Evidence]) -> str:
        """Generate summary of forensic analysis results"""
        forensic_items = [e for e in evidence_list if e.forensic_analysis]
        
        if not forensic_items:
            return "No forensic analysis results available."
        
        summary_parts = [
            f"Forensic analysis completed on {len(forensic_items)} evidence items.",
            ""
        ]
        
        # Categorize by analysis type
        analysis_types = defaultdict(int)
        for evidence in forensic_items:
            if evidence.forensic_analysis:
                analysis_type = evidence.forensic_analysis.get("analysis_type", "unknown")
                analysis_types[analysis_type] += 1
        
        if analysis_types:
            summary_parts.append("Analysis Types:")
            for analysis_type, count in analysis_types.items():
                summary_parts.append(f"  • {analysis_type}: {count} items")
        
        return "\n".join(summary_parts)
    
    def _generate_expert_opinions_summary(self, evidence_list: List[Evidence]) -> str:
        """Generate summary of expert opinions"""
        expert_items = [e for e in evidence_list if e.expert_opinions]
        
        if not expert_items:
            return "No expert opinions recorded."
        
        summary_parts = [
            f"Expert opinions provided for {len(expert_items)} evidence items.",
            ""
        ]
        
        # Count expert types
        expert_types = defaultdict(int)
        for evidence in expert_items:
            for opinion in evidence.expert_opinions:
                expert_type = opinion.get("expert_type", "unknown")
                expert_types[expert_type] += 1
        
        if expert_types:
            summary_parts.append("Expert Types:")
            for expert_type, count in expert_types.items():
                summary_parts.append(f"  • {expert_type}: {count} opinions")
        
        return "\n".join(summary_parts)
    
    def _analyze_admissibility(self, evidence_list: List[Evidence]) -> Dict[str, str]:
        """Analyze admissibility status of evidence"""
        analysis = {}
        
        for evidence in evidence_list:
            admissibility_factors = []
            
            # Authentication factor
            if evidence.authenticity_status == EvidenceAuthenticity.AUTHENTIC:
                admissibility_factors.append("authenticated")
            elif evidence.authenticity_status in [EvidenceAuthenticity.SUSPICIOUS, EvidenceAuthenticity.MANIPULATED]:
                admissibility_factors.append("authentication_issues")
            
            # Privilege factors
            if evidence.privilege_claims:
                admissibility_factors.append("privilege_claimed")
            
            # Chain of custody
            if not evidence.chain_of_custody:
                admissibility_factors.append("custody_gap")
            
            # Best evidence rule for documents
            if evidence.evidence_type == EvidenceType.DOCUMENT:
                if "original" in evidence.title.lower() or "original" in evidence.description.lower():
                    admissibility_factors.append("original_document")
                else:
                    admissibility_factors.append("copy_document")
            
            # Determine overall admissibility
            if "authentication_issues" in admissibility_factors:
                admissibility = "questionable - authentication concerns"
            elif "custody_gap" in admissibility_factors:
                admissibility = "questionable - chain of custody issues"
            elif "privilege_claimed" in admissibility_factors:
                admissibility = "subject to privilege determination"
            elif "authenticated" in admissibility_factors:
                admissibility = "likely admissible"
            else:
                admissibility = "requires foundation"
            
            analysis[evidence.evidence_id] = admissibility
        
        return analysis
    
    def search_evidence(
        self,
        evidence_list: List[Evidence],
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        """Search evidence catalog with text query and filters"""
        # Create search index
        search_index = EvidenceSearchIndex()
        for evidence in evidence_list:
            search_index.index_evidence(evidence)
        
        # Get matching evidence IDs
        if query:
            matching_ids = set(search_index.search(query))
        else:
            matching_ids = set(evidence.evidence_id for evidence in evidence_list)
        
        # Apply filters
        if filters:
            filtered_ids = set(search_index.filter_by_criteria(filters))
            matching_ids = matching_ids.intersection(filtered_ids)
        
        # Return matching evidence objects
        evidence_dict = {e.evidence_id: e for e in evidence_list}
        return [evidence_dict[eid] for eid in matching_ids if eid in evidence_dict]
    
    def generate_chain_of_custody_report(self, evidence: Evidence) -> str:
        """Generate detailed chain of custody report for specific evidence"""
        return self.report_generator.generate_chain_of_custody_report(evidence)
    
    def generate_authenticity_summary(self, evidence_list: List[Evidence]) -> str:
        """Generate authenticity status summary report"""
        return self.report_generator.generate_authenticity_summary(evidence_list)
    
    def generate_evidence_timeline_report(self, evidence_list: List[Evidence]) -> str:
        """Generate chronological evidence timeline report"""
        return self.report_generator.generate_evidence_timeline(evidence_list)
    
    def validate_evidence_item(self, evidence: Evidence) -> Dict[str, Any]:
        """Validate individual evidence item"""
        return self.quality_checker.check_evidence_quality(evidence)
    
    def update_chain_of_custody(
        self,
        evidence: Evidence,
        new_custodian: str,
        new_custodian_role: str,
        transfer_reason: str,
        location: str,
        transfer_method: str = "physical_handoff",
        witness: Optional[str] = None
    ) -> Evidence:
        """Add new entry to evidence chain of custody"""
        custody_entry = {
            "custodian_name": new_custodian,
            "custodian_role": new_custodian_role,
            "transfer_date": datetime.utcnow(),
            "transfer_reason": transfer_reason,
            "transfer_method": transfer_method,
            "receiving_party": new_custodian,
            "location": location,
            "witness": witness,
            "hash_verified": bool(evidence.file_hash),
            "seal_intact": True,
            "tamper_evident": False
        }
        
        evidence.chain_of_custody.append(custody_entry)
        evidence.custodian = new_custodian
        evidence.updated_at = datetime.utcnow()
        
        self.logger.info(f"Updated chain of custody for evidence {evidence.evidence_id}")
        
        return evidence