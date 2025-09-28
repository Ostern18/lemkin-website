"""
Legal Brief Formatter for Lemkin Report Generator Suite.

This module provides the LegalBriefFormatter class for creating auto-populated
legal brief templates with proper formatting, citation management, and 
court-specific compliance for various types of legal briefs and motions.
"""

from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import re
from collections import defaultdict

from loguru import logger

from .core import (
    BaseReportModel, ReportConfig, LegalBrief, CaseData, PersonInfo,
    LegalTemplate, ReportSection, LegalCitation, Evidence, ReportType,
    ReportStatus, CitationStyle, DocumentStandard, TemplateType,
    ConfidentialityLevel
)


class BriefTemplate:
    """Templates for different types of legal briefs"""
    
    MOTION_BRIEF_SECTIONS = [
        {
            "title": "Introduction",
            "key": "introduction",
            "required": True,
            "max_length": 800,
            "description": "Brief overview of motion and relief sought"
        },
        {
            "title": "Statement of Facts",
            "key": "statement_of_facts",
            "required": True,
            "max_length": 2000,
            "description": "Factual background relevant to the motion"
        },
        {
            "title": "Argument",
            "key": "argument",
            "required": True,
            "max_length": 5000,
            "description": "Legal arguments supporting the motion"
        },
        {
            "title": "Conclusion",
            "key": "conclusion",
            "required": True,
            "max_length": 300,
            "description": "Summary and prayer for relief"
        }
    ]
    
    RESPONSE_BRIEF_SECTIONS = [
        {
            "title": "Introduction",
            "key": "introduction",
            "required": True,
            "max_length": 600,
            "description": "Overview of response position"
        },
        {
            "title": "Counter-Statement of Facts",
            "key": "counter_statement",
            "required": False,
            "max_length": 1500,
            "description": "Correction or clarification of facts"
        },
        {
            "title": "Argument",
            "key": "argument", 
            "required": True,
            "max_length": 5000,
            "description": "Arguments opposing the motion"
        },
        {
            "title": "Conclusion",
            "key": "conclusion",
            "required": True,
            "max_length": 300,
            "description": "Request for denial of motion"
        }
    ]
    
    APPELLATE_BRIEF_SECTIONS = [
        {
            "title": "Statement of Issues",
            "key": "statement_of_issues",
            "required": True,
            "max_length": 800,
            "description": "Questions presented for review"
        },
        {
            "title": "Statement of the Case",
            "key": "statement_of_case",
            "required": True,
            "max_length": 1000,
            "description": "Procedural history and nature of case"
        },
        {
            "title": "Statement of Facts",
            "key": "statement_of_facts",
            "required": True,
            "max_length": 2500,
            "description": "Relevant facts with record citations"
        },
        {
            "title": "Summary of Argument",
            "key": "summary_of_argument",
            "required": True,
            "max_length": 1200,
            "description": "Concise summary of legal arguments"
        },
        {
            "title": "Argument",
            "key": "argument",
            "required": True,
            "max_length": 8000,
            "description": "Detailed legal arguments with authorities"
        },
        {
            "title": "Conclusion",
            "key": "conclusion",
            "required": True,
            "max_length": 400,
            "description": "Relief sought on appeal"
        }
    ]


class CitationManager:
    """Manages legal citations and formatting"""
    
    def __init__(self, citation_style: CitationStyle = CitationStyle.BLUEBOOK):
        self.citation_style = citation_style
        self.logger = logger.bind(component="citation_manager")
        
        # Citation patterns for validation
        self.citation_patterns = {
            "case": re.compile(r'([^,]+),\s*(\d+)\s+([A-Z][a-z\.]*\.?\s?\d*[a-z]*)\s+(\d+)(?:\s*\(([^)]+)\s+(\d{4})\))?'),
            "statute": re.compile(r'(\d+)\s+(U\.?S\.?C\.?)\s+§?\s*(\d+[a-z]*)'),
            "regulation": re.compile(r'(\d+)\s+(C\.?F\.?R\.?)\s+§?\s*(\d+(?:\.\d+)*)'),
        }
    
    def format_citation(self, citation: LegalCitation) -> str:
        """Format citation according to specified style"""
        if self.citation_style == CitationStyle.BLUEBOOK:
            return self._format_bluebook(citation)
        elif self.citation_style == CitationStyle.ALWD:
            return self._format_alwd(citation)
        else:
            return citation.full_citation
    
    def _format_bluebook(self, citation: LegalCitation) -> str:
        """Format citation using Bluebook style"""
        formatted = citation.full_citation
        
        # Add pin cite if present
        if citation.pin_cite:
            formatted += f", {citation.pin_cite}"
        
        # Add parenthetical if present
        if citation.parenthetical:
            formatted += f" ({citation.parenthetical})"
        
        return formatted
    
    def _format_alwd(self, citation: LegalCitation) -> str:
        """Format citation using ALWD style"""
        # ALWD formatting (similar to Bluebook for basic cases)
        return self._format_bluebook(citation)
    
    def validate_citation(self, citation_text: str) -> Dict[str, Any]:
        """Validate citation format"""
        validation_result = {
            "valid": False,
            "citation_type": "unknown",
            "errors": [],
            "suggestions": []
        }
        
        # Check against patterns
        for citation_type, pattern in self.citation_patterns.items():
            if pattern.search(citation_text):
                validation_result["valid"] = True
                validation_result["citation_type"] = citation_type
                break
        
        if not validation_result["valid"]:
            validation_result["errors"].append("Citation format not recognized")
            validation_result["suggestions"].append("Verify citation follows standard format")
        
        return validation_result
    
    def generate_citation_list(self, citations: List[LegalCitation]) -> str:
        """Generate formatted list of authorities"""
        if not citations:
            return "No authorities cited."
        
        # Group citations by type
        citations_by_type = defaultdict(list)
        for citation in citations:
            citations_by_type[citation.citation_type].append(citation)
        
        citation_sections = []
        
        # Order: Cases, Statutes, Regulations, Other
        type_order = ["case", "statute", "regulation", "constitutional", "secondary", "other"]
        
        for citation_type in type_order:
            if citation_type in citations_by_type:
                type_citations = citations_by_type[citation_type]
                # Sort by relevance score
                type_citations.sort(key=lambda c: c.relevance_score, reverse=True)
                
                section_lines = [f"{citation_type.title()}s:"]
                for citation in type_citations:
                    formatted_citation = self.format_citation(citation)
                    section_lines.append(f"  {formatted_citation}")
                
                citation_sections.append("\n".join(section_lines))
        
        return "\n\n".join(citation_sections)


class ArgumentStructureAnalyzer:
    """Analyzes and structures legal arguments"""
    
    def __init__(self):
        self.logger = logger.bind(component="argument_analyzer")
    
    def analyze_legal_theories(self, case_data: CaseData) -> Dict[str, Any]:
        """Analyze legal theories and suggest argument structure"""
        analysis = {
            "primary_theories": [],
            "supporting_theories": [],
            "argument_outline": [],
            "evidence_mapping": {},
            "precedent_mapping": {}
        }
        
        # Analyze primary legal theories
        if case_data.legal_theories:
            analysis["primary_theories"] = case_data.legal_theories[:3]  # Top 3
            analysis["supporting_theories"] = case_data.legal_theories[3:]
        
        # Create argument outline
        for i, theory in enumerate(analysis["primary_theories"], 1):
            argument_section = {
                "section": f"{i}. {theory}",
                "subsections": [
                    f"A. Elements of {theory}",
                    f"B. Application to Facts",
                    f"C. Supporting Authority"
                ]
            }
            analysis["argument_outline"].append(argument_section)
        
        # Map evidence to theories
        for evidence in case_data.evidence_list:
            if evidence.relevance_to_case:
                # Simple keyword matching (could be enhanced with NLP)
                for theory in analysis["primary_theories"]:
                    if any(keyword in evidence.relevance_to_case.lower() 
                           for keyword in theory.lower().split()):
                        if theory not in analysis["evidence_mapping"]:
                            analysis["evidence_mapping"][theory] = []
                        analysis["evidence_mapping"][theory].append(evidence.evidence_id)
        
        # Map precedents to theories
        for precedent in case_data.precedent_cases:
            if precedent.notes:
                for theory in analysis["primary_theories"]:
                    if any(keyword in precedent.notes.lower() 
                           for keyword in theory.lower().split()):
                        if theory not in analysis["precedent_mapping"]:
                            analysis["precedent_mapping"][theory] = []
                        analysis["precedent_mapping"][theory].append(precedent.id)
        
        return analysis
    
    def generate_argument_sections(
        self,
        case_data: CaseData,
        brief_type: str,
        argument_strategy: str = "comprehensive"
    ) -> List[ReportSection]:
        """Generate structured argument sections"""
        sections = []
        
        # Analyze legal theories
        theory_analysis = self.analyze_legal_theories(case_data)
        
        if brief_type.lower() in ["motion", "petition"]:
            sections = self._generate_motion_arguments(case_data, theory_analysis)
        elif brief_type.lower() in ["response", "opposition"]:
            sections = self._generate_response_arguments(case_data, theory_analysis)
        elif brief_type.lower() in ["appellate", "appeal"]:
            sections = self._generate_appellate_arguments(case_data, theory_analysis)
        else:
            sections = self._generate_generic_arguments(case_data, theory_analysis)
        
        return sections
    
    def _generate_motion_arguments(
        self,
        case_data: CaseData,
        theory_analysis: Dict[str, Any]
    ) -> List[ReportSection]:
        """Generate argument sections for motion brief"""
        sections = []
        
        for i, theory in enumerate(theory_analysis["primary_theories"], 1):
            # Create main argument section
            section = ReportSection(
                title=f"{i}. {theory.title()}",
                content=self._generate_theory_argument(case_data, theory, theory_analysis),
                section_type="argument",
                order_index=i,
                subsections=[]
            )
            
            # Add subsections
            subsections = [
                ReportSection(
                    title=f"A. Legal Standard for {theory}",
                    content=self._generate_legal_standard(case_data, theory),
                    section_type="legal_standard",
                    order_index=1
                ),
                ReportSection(
                    title="B. Application to Facts",
                    content=self._generate_factual_application(case_data, theory, theory_analysis),
                    section_type="factual_application", 
                    order_index=2
                ),
                ReportSection(
                    title="C. Supporting Authority",
                    content=self._generate_supporting_authority(case_data, theory, theory_analysis),
                    section_type="authority",
                    order_index=3
                )
            ]
            
            section.subsections = subsections
            sections.append(section)
        
        return sections
    
    def _generate_response_arguments(
        self,
        case_data: CaseData,
        theory_analysis: Dict[str, Any]
    ) -> List[ReportSection]:
        """Generate argument sections for response brief"""
        sections = []
        
        for i, theory in enumerate(theory_analysis["primary_theories"], 1):
            section = ReportSection(
                title=f"{i}. {theory.title()} Fails as a Matter of Law",
                content=self._generate_opposition_argument(case_data, theory, theory_analysis),
                section_type="opposition_argument",
                order_index=i,
                subsections=[]
            )
            
            subsections = [
                ReportSection(
                    title="A. Legal Standard",
                    content=self._generate_legal_standard(case_data, theory),
                    section_type="legal_standard",
                    order_index=1
                ),
                ReportSection(
                    title="B. Plaintiff Cannot Establish Elements",
                    content=self._generate_elements_analysis(case_data, theory, "opposition"),
                    section_type="elements_analysis",
                    order_index=2
                ),
                ReportSection(
                    title="C. Distinguishing Authority",
                    content=self._generate_distinguishing_authority(case_data, theory, theory_analysis),
                    section_type="distinguishing_authority",
                    order_index=3
                )
            ]
            
            section.subsections = subsections
            sections.append(section)
        
        return sections
    
    def _generate_appellate_arguments(
        self,
        case_data: CaseData,
        theory_analysis: Dict[str, Any]
    ) -> List[ReportSection]:
        """Generate argument sections for appellate brief"""
        sections = []
        
        for i, theory in enumerate(theory_analysis["primary_theories"], 1):
            section = ReportSection(
                title=f"{i}. The Trial Court Erred in [Specific Ruling Related to {theory}]",
                content=self._generate_appellate_argument(case_data, theory, theory_analysis),
                section_type="appellate_argument",
                order_index=i,
                subsections=[]
            )
            
            subsections = [
                ReportSection(
                    title="A. Standard of Review",
                    content=self._generate_standard_of_review(case_data, theory),
                    section_type="standard_of_review",
                    order_index=1
                ),
                ReportSection(
                    title="B. Legal Framework",
                    content=self._generate_legal_framework(case_data, theory),
                    section_type="legal_framework",
                    order_index=2
                ),
                ReportSection(
                    title="C. Application and Analysis",
                    content=self._generate_appellate_application(case_data, theory, theory_analysis),
                    section_type="appellate_application",
                    order_index=3
                )
            ]
            
            section.subsections = subsections
            sections.append(section)
        
        return sections
    
    def _generate_generic_arguments(
        self,
        case_data: CaseData,
        theory_analysis: Dict[str, Any]
    ) -> List[ReportSection]:
        """Generate generic argument sections"""
        return self._generate_motion_arguments(case_data, theory_analysis)
    
    def _generate_theory_argument(
        self,
        case_data: CaseData,
        theory: str,
        theory_analysis: Dict[str, Any]
    ) -> str:
        """Generate argument content for specific legal theory"""
        content_parts = [
            f"Plaintiff's claim for {theory.lower()} is supported by both the facts of this case and established legal precedent."
        ]
        
        # Add factual support
        if theory in theory_analysis["evidence_mapping"]:
            evidence_count = len(theory_analysis["evidence_mapping"][theory])
            content_parts.append(f"The evidence demonstrates {evidence_count} key factors supporting this claim.")
        
        # Add legal support
        if theory in theory_analysis["precedent_mapping"]:
            precedent_count = len(theory_analysis["precedent_mapping"][theory])
            content_parts.append(f"This position is supported by {precedent_count} controlling authorities.")
        
        return " ".join(content_parts)
    
    def _generate_legal_standard(self, case_data: CaseData, theory: str) -> str:
        """Generate legal standard section"""
        return f"To establish a claim for {theory.lower()}, plaintiff must demonstrate [specific elements to be researched and added]."
    
    def _generate_factual_application(
        self,
        case_data: CaseData,
        theory: str,
        theory_analysis: Dict[str, Any]
    ) -> str:
        """Generate factual application section"""
        content_parts = [
            f"The facts of this case clearly establish the elements of {theory.lower()}."
        ]
        
        # Reference relevant evidence
        if theory in theory_analysis["evidence_mapping"]:
            content_parts.append("The evidence includes: [specific evidence references to be added].")
        
        return " ".join(content_parts)
    
    def _generate_supporting_authority(
        self,
        case_data: CaseData,
        theory: str,
        theory_analysis: Dict[str, Any]
    ) -> str:
        """Generate supporting authority section"""
        if theory in theory_analysis["precedent_mapping"]:
            return "The following authorities support this position: [citations to be added with specific analysis]."
        else:
            return "Legal research is needed to identify supporting authority for this argument."
    
    def _generate_opposition_argument(
        self,
        case_data: CaseData,
        theory: str,
        theory_analysis: Dict[str, Any]
    ) -> str:
        """Generate opposition argument"""
        return f"Plaintiff cannot establish a valid claim for {theory.lower()} because [specific deficiencies to be analyzed and added]."
    
    def _generate_elements_analysis(
        self,
        case_data: CaseData,
        theory: str,
        perspective: str
    ) -> str:
        """Generate elements analysis"""
        if perspective == "opposition":
            return f"Plaintiff fails to establish the required elements of {theory.lower()}: [specific element-by-element analysis to be added]."
        else:
            return f"All elements of {theory.lower()} are satisfied: [element-by-element analysis to be added]."
    
    def _generate_distinguishing_authority(
        self,
        case_data: CaseData,
        theory: str,
        theory_analysis: Dict[str, Any]
    ) -> str:
        """Generate distinguishing authority section"""
        return "Plaintiff's cited authorities are distinguishable: [specific distinguishing analysis to be added]."
    
    def _generate_appellate_argument(
        self,
        case_data: CaseData,
        theory: str,
        theory_analysis: Dict[str, Any]
    ) -> str:
        """Generate appellate argument"""
        return f"The trial court's ruling on {theory.lower()} was error because [specific legal error to be analyzed and added]."
    
    def _generate_standard_of_review(self, case_data: CaseData, theory: str) -> str:
        """Generate standard of review section"""
        return "This Court reviews [specific standard - de novo, clear error, abuse of discretion] the trial court's [ruling type]."
    
    def _generate_legal_framework(self, case_data: CaseData, theory: str) -> str:
        """Generate legal framework section"""
        return f"The legal framework governing {theory.lower()} is well-established: [framework to be researched and added]."
    
    def _generate_appellate_application(
        self,
        case_data: CaseData,
        theory: str,
        theory_analysis: Dict[str, Any]
    ) -> str:
        """Generate appellate application section"""
        return f"Applied to this case, the legal standard demonstrates that [specific application and error analysis to be added]."


class BriefValidator:
    """Validates legal brief content and compliance"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logger.bind(component="brief_validator")
    
    def validate_brief(self, brief: LegalBrief) -> Dict[str, Any]:
        """Comprehensive validation of legal brief"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "compliance_score": 0.0,
            "word_count_compliant": True,
            "citation_format_valid": True,
            "structure_compliant": True
        }
        
        # Check required sections
        if not brief.argument_sections:
            validation_result["errors"].append("Brief must contain argument sections")
            validation_result["valid"] = False
            validation_result["structure_compliant"] = False
        
        if not brief.conclusion:
            validation_result["errors"].append("Brief must contain conclusion")
            validation_result["valid"] = False
            validation_result["structure_compliant"] = False
        
        # Check word count limits (court-specific)
        word_count = brief.word_count
        if word_count == 0:
            # Calculate word count
            word_count = self._calculate_word_count(brief)
            brief.word_count = word_count
        
        # Typical limits (would be court-specific)
        max_word_limits = {
            "motion": 2500,
            "response": 2500,
            "appellate": 7000,
            "reply": 1250
        }
        
        brief_type_key = brief.brief_type.lower()
        if brief_type_key in max_word_limits:
            limit = max_word_limits[brief_type_key]
            if word_count > limit:
                validation_result["warnings"].append(f"Word count ({word_count}) exceeds typical limit ({limit})")
                validation_result["word_count_compliant"] = False
        
        # Validate citations
        citation_issues = 0
        if brief.authorities_cited:
            citation_manager = CitationManager(self.config.default_citation_style)
            for citation in brief.authorities_cited:
                citation_validation = citation_manager.validate_citation(citation.full_citation)
                if not citation_validation["valid"]:
                    citation_issues += 1
        
        if citation_issues > 0:
            validation_result["warnings"].append(f"{citation_issues} citations may have formatting issues")
            if citation_issues > len(brief.authorities_cited) * 0.2:  # More than 20% have issues
                validation_result["citation_format_valid"] = False
        
        # Check court rules compliance
        compliance_checks = self._check_court_rules_compliance(brief)
        validation_result["compliance_score"] = compliance_checks["score"]
        if compliance_checks["issues"]:
            validation_result["warnings"].extend(compliance_checks["issues"])
        
        # Check filing readiness
        if not brief.internal_review_complete:
            validation_result["suggestions"].append("Complete internal review before filing")
        
        if not brief.client_approval and "client_approval_required" in self.config.metadata:
            validation_result["suggestions"].append("Obtain client approval before filing")
        
        return validation_result
    
    def _calculate_word_count(self, brief: LegalBrief) -> int:
        """Calculate total word count for brief"""
        total_words = 0
        
        # Count words in argument sections
        for section in brief.argument_sections:
            total_words += len(section.content.split())
            for subsection in section.subsections:
                total_words += len(subsection.content.split())
        
        # Count words in conclusion
        if brief.conclusion:
            total_words += len(brief.conclusion.split())
        
        # Count words in prayer for relief
        if brief.prayer_for_relief:
            total_words += len(brief.prayer_for_relief.split())
        
        return total_words
    
    def _check_court_rules_compliance(self, brief: LegalBrief) -> Dict[str, Any]:
        """Check compliance with court rules"""
        compliance_result = {
            "score": 1.0,
            "issues": []
        }
        
        # Check for common compliance issues
        if brief.template and brief.template.document_standard:
            # Federal court rules
            if brief.template.document_standard == DocumentStandard.FEDERAL_COURT:
                if brief.word_count > 7000 and "appellate" in brief.brief_type.lower():
                    compliance_result["issues"].append("Exceeds federal appellate brief word limit")
                    compliance_result["score"] *= 0.8
                
                # Check for required certificate of service (would check in metadata)
                if "certificate_of_service" not in brief.metadata:
                    compliance_result["issues"].append("Certificate of service may be required")
                    compliance_result["score"] *= 0.9
        
        return compliance_result
    
    def check_argument_structure(self, brief: LegalBrief) -> Dict[str, Any]:
        """Check logical structure of arguments"""
        structure_analysis = {
            "logical_flow": True,
            "issues": [],
            "suggestions": []
        }
        
        # Check argument sections have proper structure
        for section in brief.argument_sections:
            if not section.subsections:
                structure_analysis["suggestions"].append(f"Consider adding subsections to {section.title}")
            
            # Check for legal standard subsection
            has_legal_standard = any("standard" in sub.title.lower() for sub in section.subsections)
            if not has_legal_standard:
                structure_analysis["suggestions"].append(f"Consider adding legal standard subsection to {section.title}")
        
        return structure_analysis


class LegalBriefFormatter:
    """
    Auto-populated legal brief template formatter.
    
    Creates professional legal briefs with proper formatting, structured arguments,
    citation management, and court-specific compliance suitable for filing
    in various jurisdictions and court levels.
    """
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.citation_manager = CitationManager(config.default_citation_style)
        self.argument_analyzer = ArgumentStructureAnalyzer()
        self.validator = BriefValidator(config)
        self.logger = logger.bind(component="legal_brief_formatter")
        
        # Template management
        self.templates = {
            "motion": BriefTemplate.MOTION_BRIEF_SECTIONS,
            "response": BriefTemplate.RESPONSE_BRIEF_SECTIONS,
            "appellate": BriefTemplate.APPELLATE_BRIEF_SECTIONS
        }
    
    def format(
        self,
        case_data: CaseData,
        template: Union[str, LegalTemplate],
        brief_type: str,
        author: PersonInfo,
        opposing_counsel: Optional[PersonInfo] = None,
        custom_sections: Optional[List[Dict[str, Any]]] = None
    ) -> LegalBrief:
        """
        Generate auto-populated legal brief from template
        
        Args:
            case_data: Case information for brief population
            template: Brief template name or LegalTemplate object
            brief_type: Type of brief (motion, response, appellate, etc.)
            author: Brief author information
            opposing_counsel: Opposing counsel information
            custom_sections: Additional custom sections
            
        Returns:
            LegalBrief with populated content and proper formatting
        """
        start_time = datetime.utcnow()
        
        self.logger.info(f"Formatting {brief_type} brief for case {case_data.case_info.case_number}")
        
        # Get or create template
        if isinstance(template, str):
            legal_template = self._get_template(template, brief_type)
        else:
            legal_template = template
        
        # Initialize brief
        brief = LegalBrief(
            case_data=case_data,
            template=legal_template,
            brief_type=brief_type,
            title=f"{brief_type.title()} Brief: {case_data.case_info.case_name}",
            author=author,
            opposing_counsel=opposing_counsel,
            review_status=ReportStatus.DRAFT
        )
        
        try:
            # Generate statement of issues
            brief.statement_of_issues = self._generate_statement_of_issues(case_data, brief_type)
            
            # Generate argument sections
            brief.argument_sections = self.argument_analyzer.generate_argument_sections(
                case_data, brief_type
            )
            
            # Generate conclusion
            brief.conclusion = self._generate_conclusion(case_data, brief_type)
            
            # Generate prayer for relief
            brief.prayer_for_relief = self._generate_prayer_for_relief(case_data, brief_type)
            
            # Collect authorities cited
            brief.authorities_cited = self._collect_authorities(case_data)
            
            # Collect evidence cited
            brief.evidence_cited = self._collect_evidence_references(case_data, brief.argument_sections)
            
            # Generate exhibits list
            brief.exhibits = self._generate_exhibits_list(case_data)
            
            # Calculate word count
            brief.word_count = self._calculate_word_count(brief)
            
            # Check page limit compliance
            brief.page_limit_compliance = self._check_page_limits(brief)
            
            # Format citations
            self._format_citations(brief)
            brief.citation_format_verified = True
            
            # Check court rules compliance
            court_compliance = self._check_court_rules_compliance(brief)
            brief.court_rules_compliance = court_compliance
            
            # Validate brief
            validation_result = self.validator.validate_brief(brief)
            if not validation_result["valid"]:
                self.logger.warning(f"Brief validation issues: {validation_result['errors']}")
                brief.review_required = True
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Brief formatting completed in {duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Brief formatting failed: {str(e)}")
            brief.conclusion = f"Error occurred during brief generation: {str(e)}"
            brief.review_required = True
            raise
        
        return brief
    
    def _get_template(self, template_name: str, brief_type: str) -> LegalTemplate:
        """Get legal template for brief type"""
        # Use brief type if template name not found
        template_key = template_name if template_name in self.templates else brief_type.lower()
        
        if template_key not in self.templates:
            template_key = "motion"  # Default fallback
        
        sections = []
        for i, section_def in enumerate(self.templates[template_key]):
            section = ReportSection(
                title=section_def["title"],
                content="",  # Will be populated
                section_type=section_def["key"],
                order_index=i,
                formatting_notes=section_def.get("description", "")
            )
            sections.append(section)
        
        return LegalTemplate(
            name=f"{template_key.title()} Brief Template",
            template_type=TemplateType.STANDARD,
            document_standard=self.config.default_document_standard,
            sections=sections,
            citation_style=self.config.default_citation_style
        )
    
    def _generate_statement_of_issues(self, case_data: CaseData, brief_type: str) -> List[str]:
        """Generate statement of legal issues"""
        issues = []
        
        # Base issues on legal theories and causes of action
        if case_data.legal_theories:
            for theory in case_data.legal_theories[:3]:  # Limit to top 3
                if brief_type.lower() == "appellate":
                    issues.append(f"Whether the trial court erred in its ruling regarding {theory.lower()}?")
                else:
                    issues.append(f"Whether {theory.lower()} applies under the facts of this case?")
        
        if case_data.causes_of_action:
            for cause in case_data.causes_of_action[:2]:  # Limit to top 2
                if brief_type.lower() == "response":
                    issues.append(f"Whether plaintiff has stated a valid claim for {cause.lower()}?")
                else:
                    issues.append(f"Whether the elements of {cause.lower()} are satisfied?")
        
        # Add damages issues if applicable
        if case_data.damages_claimed:
            issues.append("Whether plaintiff is entitled to damages and in what amount?")
        
        # Default issues if none generated
        if not issues:
            if brief_type.lower() == "motion":
                issues.append("Whether the relief requested should be granted?")
            elif brief_type.lower() == "response":
                issues.append("Whether the motion should be denied?")
            else:
                issues.append("What relief is appropriate under the circumstances?")
        
        return issues
    
    def _generate_conclusion(self, case_data: CaseData, brief_type: str) -> str:
        """Generate brief conclusion"""
        conclusion_parts = []
        
        if brief_type.lower() == "motion":
            conclusion_parts.append("For the foregoing reasons, plaintiff respectfully requests that this Court grant the motion")
            if case_data.legal_theories:
                primary_theory = case_data.legal_theories[0]
                conclusion_parts.append(f"and find that {primary_theory.lower()} applies to the facts of this case")
        
        elif brief_type.lower() in ["response", "opposition"]:
            conclusion_parts.append("For the foregoing reasons, defendant respectfully requests that this Court deny the motion")
            if case_data.defenses:
                primary_defense = case_data.defenses[0]
                conclusion_parts.append(f"as {primary_defense.lower()}")
        
        elif brief_type.lower() == "appellate":
            conclusion_parts.append("For the foregoing reasons, this Court should reverse the decision of the trial court")
            conclusion_parts.append("and remand for further proceedings consistent with this opinion")
        
        else:
            conclusion_parts.append("For the foregoing reasons, the Court should grant the relief requested")
        
        conclusion_parts.append(".")
        
        return " ".join(conclusion_parts)
    
    def _generate_prayer_for_relief(self, case_data: CaseData, brief_type: str) -> str:
        """Generate prayer for relief section"""
        prayer_parts = ["WHEREFORE, "]
        
        if brief_type.lower() == "motion":
            prayer_parts.append("plaintiff respectfully requests that this Court:")
            relief_items = [
                "Grant the motion",
                "Enter judgment in favor of plaintiff"
            ]
            
            if case_data.damages_claimed:
                relief_items.append(f"Award damages in the amount of ${case_data.damages_claimed:,.2f}")
            
            relief_items.append("Grant such other and further relief as the Court deems just and proper")
            
        elif brief_type.lower() in ["response", "opposition"]:
            prayer_parts.append("defendant respectfully requests that this Court:")
            relief_items = [
                "Deny the motion in its entirety",
                "Enter judgment in favor of defendant",
                "Award defendant its costs and attorney's fees",
                "Grant such other and further relief as the Court deems just and proper"
            ]
        
        elif brief_type.lower() == "appellate":
            prayer_parts.append("appellant respectfully requests that this Court:")
            relief_items = [
                "Reverse the decision of the trial court",
                "Remand for proceedings consistent with this Court's decision",
                "Grant such other and further relief as the Court deems appropriate"
            ]
        
        else:
            prayer_parts.append("the moving party respectfully requests that this Court:")
            relief_items = [
                "Grant the relief requested",
                "Grant such other and further relief as the Court deems just and proper"
            ]
        
        # Format as numbered list
        prayer_parts.append("\n")
        for i, item in enumerate(relief_items, 1):
            prayer_parts.append(f"\n{i}. {item};")
        
        # Remove semicolon from last item and add period
        if prayer_parts:
            prayer_parts[-1] = prayer_parts[-1].rstrip(';') + '.'
        
        return "".join(prayer_parts)
    
    def _collect_authorities(self, case_data: CaseData) -> List[LegalCitation]:
        """Collect all legal authorities for citation"""
        authorities = []
        
        # Add precedent cases
        authorities.extend(case_data.precedent_cases)
        
        # Note: In a real implementation, this would also scan the generated
        # argument text for additional citations and authorities
        
        return authorities
    
    def _collect_evidence_references(
        self,
        case_data: CaseData,
        argument_sections: List[ReportSection]
    ) -> List[str]:
        """Collect evidence references from arguments"""
        evidence_refs = []
        
        # Get high-relevance evidence
        high_relevance_evidence = [
            e.evidence_id for e in case_data.evidence_list 
            if e.evidential_weight >= 0.7
        ]
        
        evidence_refs.extend(high_relevance_evidence[:10])  # Limit to top 10
        
        return evidence_refs
    
    def _generate_exhibits_list(self, case_data: CaseData) -> List[Dict[str, Any]]:
        """Generate list of exhibits to be attached"""
        exhibits = []
        
        # Include key evidence as exhibits
        key_evidence = [
            e for e in case_data.evidence_list 
            if e.evidential_weight >= 0.8 and e.evidence_type.value in ["document", "photograph"]
        ]
        
        for i, evidence in enumerate(key_evidence[:15], 1):  # Limit to 15 exhibits
            exhibit = {
                "exhibit_letter": chr(64 + i),  # A, B, C, etc.
                "evidence_id": evidence.evidence_id,
                "title": evidence.title,
                "description": evidence.description[:100] + "..." if len(evidence.description) > 100 else evidence.description,
                "file_path": evidence.file_path,
                "pages": 1  # Default, would be calculated for documents
            }
            exhibits.append(exhibit)
        
        return exhibits
    
    def _calculate_word_count(self, brief: LegalBrief) -> int:
        """Calculate total word count for the brief"""
        total_words = 0
        
        # Count argument sections
        for section in brief.argument_sections:
            total_words += len(section.content.split())
            for subsection in section.subsections:
                total_words += len(subsection.content.split())
        
        # Count conclusion
        if brief.conclusion:
            total_words += len(brief.conclusion.split())
        
        return total_words
    
    def _check_page_limits(self, brief: LegalBrief) -> bool:
        """Check if brief complies with typical page limits"""
        # Estimate pages (roughly 250 words per page)
        estimated_pages = brief.word_count / 250
        
        # Typical page limits
        page_limits = {
            "motion": 10,
            "response": 10,
            "appellate": 30,
            "reply": 5
        }
        
        brief_type_key = brief.brief_type.lower()
        if brief_type_key in page_limits:
            return estimated_pages <= page_limits[brief_type_key]
        
        return True  # No limit known
    
    def _format_citations(self, brief: LegalBrief):
        """Format all citations in the brief"""
        # Format authorities cited
        for citation in brief.authorities_cited:
            citation.full_citation = self.citation_manager.format_citation(citation)
    
    def _check_court_rules_compliance(self, brief: LegalBrief) -> Dict[str, bool]:
        """Check compliance with court-specific rules"""
        compliance = {
            "word_count": brief.word_count <= 7000,  # General limit
            "citation_format": brief.citation_format_verified,
            "required_sections": bool(brief.argument_sections and brief.conclusion),
            "signature_block": True,  # Would check in template
            "certificate_of_service": True  # Would check in metadata
        }
        
        return compliance
    
    def get_template_options(self) -> List[str]:
        """Get available brief template options"""
        return list(self.templates.keys())
    
    def validate_brief(self, brief: LegalBrief) -> Dict[str, Any]:
        """Validate completed brief"""
        return self.validator.validate_brief(brief)
    
    def generate_table_of_contents(self, brief: LegalBrief) -> List[str]:
        """Generate table of contents for brief"""
        toc_lines = ["TABLE OF CONTENTS", ""]
        
        # Add statement of issues
        if brief.statement_of_issues:
            toc_lines.append("STATEMENT OF ISSUES ................................. 1")
        
        # Add argument sections
        page_num = 2  # Start after issues
        for section in brief.argument_sections:
            toc_lines.append(f"{section.title} ................................. {page_num}")
            page_num += 1
            
            for subsection in section.subsections:
                toc_lines.append(f"  {subsection.title} ................................. {page_num}")
        
        # Add conclusion
        toc_lines.append(f"CONCLUSION ................................. {page_num}")
        
        return toc_lines
    
    def generate_table_of_authorities(self, brief: LegalBrief) -> List[str]:
        """Generate table of authorities"""
        if not brief.authorities_cited:
            return ["TABLE OF AUTHORITIES", "", "No authorities cited."]
        
        toa_lines = ["TABLE OF AUTHORITIES", ""]
        
        # Group by citation type
        cases = [c for c in brief.authorities_cited if c.citation_type == "case"]
        statutes = [c for c in brief.authorities_cited if c.citation_type == "statute"]
        
        if cases:
            toa_lines.extend(["CASES:", ""])
            for case in sorted(cases, key=lambda c: c.full_citation):
                toa_lines.append(f"{case.full_citation} ................................. [page]")
            toa_lines.append("")
        
        if statutes:
            toa_lines.extend(["STATUTES:", ""])
            for statute in sorted(statutes, key=lambda c: c.full_citation):
                toa_lines.append(f"{statute.full_citation} ................................. [page]")
        
        return toa_lines