"""
Fact Sheet Generator for Lemkin Report Generator Suite.

This module provides the FactSheetGenerator class for creating standardized
legal fact sheets with consistent formatting, comprehensive case summaries,
and professional presentation suitable for legal proceedings.
"""

from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

from loguru import logger

from .core import (
    BaseReportModel, ReportConfig, FactSheet, CaseData, PersonInfo,
    ReportSection, LegalCitation, Evidence, ReportStatus, 
    ConfidentialityLevel, ReportType
)


class FactSheetTemplate:
    """Template structure for fact sheet generation"""
    
    DEFAULT_SECTIONS = [
        {
            "title": "Executive Summary",
            "key": "executive_summary",
            "required": True,
            "max_length": 1000,
            "description": "Concise overview of the case and key issues"
        },
        {
            "title": "Case Information",
            "key": "case_information", 
            "required": True,
            "max_length": 500,
            "description": "Basic case details and parties"
        },
        {
            "title": "Factual Background",
            "key": "factual_background",
            "required": True,
            "max_length": 2000,
            "description": "Chronological presentation of relevant facts"
        },
        {
            "title": "Legal Issues",
            "key": "legal_issues",
            "required": True,
            "max_length": 1000,
            "description": "Primary legal questions presented"
        },
        {
            "title": "Key Evidence",
            "key": "key_evidence",
            "required": False,
            "max_length": 1500,
            "description": "Summary of most relevant evidence"
        },
        {
            "title": "Witness Summary",
            "key": "witness_summary",
            "required": False,
            "max_length": 1000,
            "description": "Overview of witness testimony and credibility"
        },
        {
            "title": "Preliminary Legal Analysis",
            "key": "preliminary_analysis",
            "required": False,
            "max_length": 1500,
            "description": "Initial assessment of legal strengths and weaknesses"
        },
        {
            "title": "Recommendations",
            "key": "recommendations",
            "required": False,
            "max_length": 800,
            "description": "Strategic recommendations and next steps"
        }
    ]


class FactSheetFormatter:
    """Handles formatting and styling of fact sheet content"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logger.bind(component="fact_sheet_formatter")
    
    def format_case_header(self, case_data: CaseData) -> str:
        """Format the case header section"""
        case_info = case_data.case_info
        
        header_lines = [
            f"CASE: {case_info.case_name}",
            f"CASE NUMBER: {case_info.case_number}",
            f"COURT: {case_info.court}",
            f"JURISDICTION: {case_info.jurisdiction}"
        ]
        
        if case_info.judge:
            header_lines.append(f"JUDGE: {case_info.judge}")
        
        if case_info.practice_area:
            header_lines.append(f"PRACTICE AREA: {case_info.practice_area}")
        
        if case_info.filing_date:
            header_lines.append(f"FILED: {case_info.filing_date.strftime('%B %d, %Y')}")
        
        return "\n".join(header_lines)
    
    def format_parties_section(self, case_data: CaseData) -> str:
        """Format the parties information section"""
        sections = []
        
        # Group parties by role
        parties_by_role = {}
        all_parties = case_data.case_info.key_parties + case_data.attorneys
        
        for party in all_parties:
            role = party.role.title()
            if role not in parties_by_role:
                parties_by_role[role] = []
            parties_by_role[role].append(party)
        
        for role, parties in parties_by_role.items():
            role_section = [f"{role.upper()}S:"]
            for party in parties:
                party_line = f"  • {party.full_name}"
                if party.organization:
                    party_line += f" ({party.organization})"
                if party.contact_email:
                    party_line += f" - {party.contact_email}"
                role_section.append(party_line)
            sections.append("\n".join(role_section))
        
        return "\n\n".join(sections)
    
    def format_timeline(self, case_data: CaseData) -> str:
        """Format key dates and chronology"""
        timeline_items = []
        
        # Add key dates
        if case_data.key_dates:
            sorted_dates = sorted(case_data.key_dates.items(), key=lambda x: x[1])
            for event, event_date in sorted_dates:
                timeline_items.append(f"• {event_date.strftime('%B %d, %Y')}: {event}")
        
        # Add chronology items
        if case_data.chronology:
            for item in case_data.chronology:
                if 'date' in item and 'event' in item:
                    event_date = item['date']
                    if isinstance(event_date, str):
                        timeline_items.append(f"• {event_date}: {item['event']}")
                    else:
                        timeline_items.append(f"• {event_date.strftime('%B %d, %Y')}: {item['event']}")
        
        return "\n".join(timeline_items) if timeline_items else "No key dates identified."
    
    def format_legal_theories(self, case_data: CaseData) -> str:
        """Format legal theories and causes of action"""
        sections = []
        
        if case_data.legal_theories:
            sections.append("LEGAL THEORIES:")
            for theory in case_data.legal_theories:
                sections.append(f"  • {theory}")
        
        if case_data.causes_of_action:
            sections.append("\nCAUSES OF ACTION:")
            for cause in case_data.causes_of_action:
                sections.append(f"  • {cause}")
        
        if case_data.defenses:
            sections.append("\nDEFENSES:")
            for defense in case_data.defenses:
                sections.append(f"  • {defense}")
        
        return "\n".join(sections) if sections else "Legal theories to be determined."
    
    def format_evidence_summary(self, evidence_list: List[Evidence]) -> str:
        """Format summary of key evidence"""
        if not evidence_list:
            return "Evidence inventory pending."
        
        # Group evidence by type
        evidence_by_type = {}
        for evidence in evidence_list:
            evidence_type = evidence.evidence_type.value.title()
            if evidence_type not in evidence_by_type:
                evidence_by_type[evidence_type] = []
            evidence_by_type[evidence_type].append(evidence)
        
        sections = []
        for evidence_type, items in evidence_by_type.items():
            sections.append(f"{evidence_type.upper()} EVIDENCE:")
            for item in items[:5]:  # Limit to top 5 per type
                relevance = f" (Weight: {item.evidential_weight:.1f})" if item.evidential_weight > 0 else ""
                sections.append(f"  • {item.title}{relevance}")
                if item.description and len(item.description) <= 100:
                    sections.append(f"    {item.description}")
            
            if len(items) > 5:
                sections.append(f"    ... and {len(items) - 5} additional items")
        
        return "\n".join(sections)
    
    def format_citations(self, citations: List[LegalCitation]) -> str:
        """Format legal citations consistently"""
        if not citations:
            return "Legal research in progress."
        
        formatted_citations = []
        for citation in sorted(citations, key=lambda c: c.relevance_score, reverse=True):
            formatted_line = f"• {citation.short_citation}"
            if citation.parenthetical:
                formatted_line += f" ({citation.parenthetical})"
            if citation.relevance_score > 0:
                formatted_line += f" [Relevance: {citation.relevance_score:.1f}]"
            formatted_citations.append(formatted_line)
        
        return "\n".join(formatted_citations)


class FactSheetValidator:
    """Validates fact sheet content for completeness and quality"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logger.bind(component="fact_sheet_validator")
    
    def validate_content(self, fact_sheet: FactSheet) -> Dict[str, Any]:
        """Validate fact sheet content and structure"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "completeness_score": 0.0,
            "quality_indicators": {}
        }
        
        # Check required fields
        if not fact_sheet.executive_summary or len(fact_sheet.executive_summary.strip()) < 100:
            validation_result["errors"].append("Executive summary must be at least 100 characters")
            validation_result["valid"] = False
        
        if not fact_sheet.factual_background or len(fact_sheet.factual_background.strip()) < 200:
            validation_result["errors"].append("Factual background must be at least 200 characters")
            validation_result["valid"] = False
        
        if not fact_sheet.legal_issues:
            validation_result["warnings"].append("No legal issues identified")
        
        # Check content quality
        exec_summary_word_count = len(fact_sheet.executive_summary.split())
        if exec_summary_word_count > 200:
            validation_result["warnings"].append(f"Executive summary is lengthy ({exec_summary_word_count} words)")
        
        # Check for key information presence
        completeness_factors = []
        
        if fact_sheet.case_data.case_info.case_number:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.0)
            validation_result["errors"].append("Case number is required")
            validation_result["valid"] = False
        
        if fact_sheet.case_data.case_info.key_parties:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.5)
            validation_result["warnings"].append("No parties identified")
        
        if fact_sheet.case_data.evidence_list:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.3)
            validation_result["suggestions"].append("Consider adding evidence summary")
        
        if fact_sheet.key_evidence:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.5)
            validation_result["suggestions"].append("Identify key evidence items")
        
        if fact_sheet.recommendations:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.7)
            validation_result["suggestions"].append("Add strategic recommendations")
        
        # Calculate completeness score
        validation_result["completeness_score"] = sum(completeness_factors) / len(completeness_factors)
        
        # Quality indicators
        validation_result["quality_indicators"] = {
            "word_count": len(fact_sheet.executive_summary.split()) + len(fact_sheet.factual_background.split()),
            "sections_completed": sum([
                1 if fact_sheet.executive_summary else 0,
                1 if fact_sheet.factual_background else 0,
                1 if fact_sheet.legal_issues else 0,
                1 if fact_sheet.key_evidence else 0,
                1 if fact_sheet.witness_summary else 0,
                1 if fact_sheet.preliminary_analysis else 0,
                1 if fact_sheet.recommendations else 0
            ]),
            "has_case_details": bool(fact_sheet.case_data.case_info.case_number),
            "has_parties": bool(fact_sheet.case_data.case_info.key_parties),
            "has_evidence": bool(fact_sheet.case_data.evidence_list),
            "has_timeline": bool(fact_sheet.case_data.key_dates or fact_sheet.case_data.chronology)
        }
        
        return validation_result
    
    def suggest_improvements(self, fact_sheet: FactSheet) -> List[str]:
        """Suggest specific improvements for the fact sheet"""
        suggestions = []
        
        # Analyze content depth
        if len(fact_sheet.factual_background.split()) < 300:
            suggestions.append("Expand factual background with more detail")
        
        if not fact_sheet.case_data.precedent_cases:
            suggestions.append("Research and include relevant precedent cases")
        
        if not fact_sheet.case_data.key_dates:
            suggestions.append("Create timeline of key events")
        
        if len(fact_sheet.legal_issues) < 2:
            suggestions.append("Consider identifying additional legal issues")
        
        # Check for specific legal analysis elements
        if "damages" not in fact_sheet.factual_background.lower() and fact_sheet.case_data.damages_claimed:
            suggestions.append("Include damages analysis in factual background")
        
        if not fact_sheet.witness_summary and fact_sheet.case_data.witnesses:
            suggestions.append("Add witness credibility analysis")
        
        return suggestions


class FactSheetGenerator:
    """
    Generator for standardized legal fact sheets.
    
    Creates comprehensive case summaries with consistent formatting,
    professional presentation, and all necessary legal elements for
    use in legal proceedings, client communications, and case management.
    """
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.formatter = FactSheetFormatter(config)
        self.validator = FactSheetValidator(config)
        self.logger = logger.bind(component="fact_sheet_generator")
        
        # Template management
        self.templates = {
            "standard": FactSheetTemplate.DEFAULT_SECTIONS,
            "litigation": self._get_litigation_template(),
            "transactional": self._get_transactional_template(),
            "appellate": self._get_appellate_template()
        }
    
    def generate(
        self,
        case_data: CaseData,
        template: Optional[str] = None,
        author: Optional[PersonInfo] = None,
        custom_sections: Optional[List[Dict[str, Any]]] = None
    ) -> FactSheet:
        """
        Generate a comprehensive fact sheet for a case
        
        Args:
            case_data: Complete case information
            template: Template name to use (default: "standard")
            author: Author information for the fact sheet
            custom_sections: Additional custom sections to include
            
        Returns:
            FactSheet with standardized case summary
        """
        start_time = datetime.utcnow()
        template_name = template or "standard"
        
        self.logger.info(f"Generating fact sheet for case {case_data.case_info.case_number} using {template_name} template")
        
        # Get default author if not provided
        if not author:
            author = case_data.attorneys[0] if case_data.attorneys else PersonInfo(
                full_name="Unknown Author",
                role="attorney",
                organization=self.config.firm_name
            )
        
        # Initialize fact sheet
        fact_sheet = FactSheet(
            case_data=case_data,
            title=f"Case Fact Sheet: {case_data.case_info.case_name}",
            author=author,
            preparation_date=date.today(),
            executive_summary="",
            factual_background="",
            review_status=ReportStatus.DRAFT,
            confidentiality_level=self.config.default_confidentiality
        )
        
        try:
            # Generate executive summary
            fact_sheet.executive_summary = self._generate_executive_summary(case_data)
            
            # Generate factual background
            fact_sheet.factual_background = self._generate_factual_background(case_data)
            
            # Generate legal issues
            fact_sheet.legal_issues = self._generate_legal_issues(case_data)
            
            # Generate key evidence summary
            fact_sheet.key_evidence = self._generate_key_evidence_ids(case_data)
            
            # Generate witness summary
            fact_sheet.witness_summary = self._generate_witness_summary(case_data)
            
            # Generate preliminary analysis
            fact_sheet.preliminary_analysis = self._generate_preliminary_analysis(case_data)
            
            # Generate recommendations
            fact_sheet.recommendations = self._generate_recommendations(case_data)
            
            # Apply formatting
            self._apply_formatting(fact_sheet)
            
            # Validate content
            validation_result = self.validator.validate_content(fact_sheet)
            if not validation_result["valid"]:
                self.logger.warning(f"Fact sheet validation failed: {validation_result['errors']}")
                fact_sheet.review_required = True
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Fact sheet generation completed in {duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Fact sheet generation failed: {str(e)}")
            fact_sheet.executive_summary = "Error occurred during fact sheet generation"
            fact_sheet.factual_background = f"Generation error: {str(e)}"
            fact_sheet.review_required = True
            raise
        
        return fact_sheet
    
    def _generate_executive_summary(self, case_data: CaseData) -> str:
        """Generate concise executive summary"""
        case_info = case_data.case_info
        
        summary_parts = [
            f"This fact sheet provides an overview of {case_info.case_name}, a {case_info.case_type} matter in {case_info.court}."
        ]
        
        # Add practice area context
        if case_info.practice_area:
            summary_parts.append(f"This case involves {case_info.practice_area.lower()} law.")
        
        # Add key legal theories
        if case_data.legal_theories:
            theories_text = ", ".join(case_data.legal_theories[:3])  # Limit to first 3
            summary_parts.append(f"The primary legal theories include {theories_text}.")
        
        # Add damages information
        if case_data.damages_claimed:
            summary_parts.append(f"Damages claimed total ${case_data.damages_claimed:,.2f}.")
        
        # Add procedural status
        if case_info.filing_date:
            summary_parts.append(f"The case was filed on {case_info.filing_date.strftime('%B %d, %Y')}.")
        
        if case_info.trial_date:
            summary_parts.append(f"Trial is scheduled for {case_info.trial_date.strftime('%B %d, %Y')}.")
        
        # Add strategic assessment
        if case_data.strengths and case_data.weaknesses:
            summary_parts.append("The case presents both significant opportunities and notable challenges that require careful strategic consideration.")
        
        return " ".join(summary_parts)
    
    def _generate_factual_background(self, case_data: CaseData) -> str:
        """Generate comprehensive factual background"""
        background_sections = []
        
        # Add statement of facts if available
        if case_data.statement_of_facts:
            background_sections.append("BACKGROUND:")
            background_sections.append(case_data.statement_of_facts)
        
        # Add chronology
        if case_data.chronology or case_data.key_dates:
            background_sections.append("\nKEY EVENTS:")
            timeline = self.formatter.format_timeline(case_data)
            background_sections.append(timeline)
        
        # Add parties information
        if case_data.case_info.key_parties:
            background_sections.append("\nPARTIES:")
            parties_info = self.formatter.format_parties_section(case_data)
            background_sections.append(parties_info)
        
        # Add disputed vs undisputed facts
        if case_data.disputed_facts or case_data.undisputed_facts:
            if case_data.undisputed_facts:
                background_sections.append("\nUNDISPUTED FACTS:")
                for fact in case_data.undisputed_facts:
                    background_sections.append(f"• {fact}")
            
            if case_data.disputed_facts:
                background_sections.append("\nDISPUTED FACTS:")
                for fact in case_data.disputed_facts:
                    background_sections.append(f"• {fact}")
        
        # Add discovery information
        if case_data.document_requests or case_data.interrogatories or case_data.depositions:
            background_sections.append("\nDISCOVERY STATUS:")
            discovery_items = []
            if case_data.document_requests:
                discovery_items.append(f"Document requests: {len(case_data.document_requests)} served")
            if case_data.interrogatories:
                discovery_items.append(f"Interrogatories: {len(case_data.interrogatories)} served")
            if case_data.depositions:
                discovery_items.append(f"Depositions: {len(case_data.depositions)} scheduled/completed")
            background_sections.extend([f"• {item}" for item in discovery_items])
        
        return "\n".join(background_sections) if background_sections else "Factual background to be developed."
    
    def _generate_legal_issues(self, case_data: CaseData) -> List[str]:
        """Generate list of primary legal issues"""
        issues = []
        
        # Add causes of action as legal issues
        if case_data.causes_of_action:
            for cause in case_data.causes_of_action:
                issues.append(f"Whether {cause.lower()} can be established")
        
        # Add legal theories as issues
        if case_data.legal_theories:
            for theory in case_data.legal_theories:
                if theory not in [c.lower() for c in case_data.causes_of_action]:
                    issues.append(f"Application of {theory.lower()} doctrine")
        
        # Add damages issues
        if case_data.damages_claimed:
            issues.append("Calculation and recovery of damages")
        
        # Add procedural issues based on case type
        if case_data.case_info.case_type.lower() == "civil":
            issues.append("Subject matter jurisdiction")
            issues.append("Personal jurisdiction over defendants")
        
        # Add evidence-based issues
        if case_data.evidence_list:
            authenticity_issues = [e for e in case_data.evidence_list 
                                 if e.authenticity_status.value in ["suspicious", "pending_verification"]]
            if authenticity_issues:
                issues.append("Admissibility and authenticity of key evidence")
        
        return issues[:8]  # Limit to top 8 issues
    
    def _generate_key_evidence_ids(self, case_data: CaseData) -> List[str]:
        """Generate list of key evidence item IDs"""
        if not case_data.evidence_list:
            return []
        
        # Sort evidence by relevance/weight
        sorted_evidence = sorted(
            case_data.evidence_list,
            key=lambda e: e.evidential_weight if e.evidential_weight > 0 else 0.5,
            reverse=True
        )
        
        # Return IDs of top evidence items
        return [e.evidence_id for e in sorted_evidence[:10]]
    
    def _generate_witness_summary(self, case_data: CaseData) -> str:
        """Generate summary of witness information"""
        if not case_data.witnesses:
            return "Witness list under development."
        
        witness_sections = []
        
        # Group witnesses by type/role
        fact_witnesses = [w for w in case_data.witnesses if "fact" in w.role.lower() or "witness" in w.role.lower()]
        expert_witnesses = [w for w in case_data.witnesses if "expert" in w.role.lower()]
        
        if fact_witnesses:
            witness_sections.append("FACT WITNESSES:")
            for witness in fact_witnesses:
                witness_line = f"• {witness.full_name}"
                if witness.organization:
                    witness_line += f" ({witness.organization})"
                witness_sections.append(witness_line)
        
        if expert_witnesses:
            witness_sections.append("\nEXPERT WITNESSES:")
            for expert in expert_witnesses:
                expert_line = f"• {expert.full_name}"
                if expert.qualifications:
                    expert_line += f" - {', '.join(expert.qualifications[:2])}"
                witness_sections.append(expert_line)
        
        return "\n".join(witness_sections)
    
    def _generate_preliminary_analysis(self, case_data: CaseData) -> str:
        """Generate preliminary legal analysis"""
        analysis_sections = []
        
        # Add strengths and weaknesses analysis
        if case_data.strengths or case_data.weaknesses:
            analysis_sections.append("CASE ASSESSMENT:")
            
            if case_data.strengths:
                analysis_sections.append("\nStrengths:")
                for strength in case_data.strengths:
                    analysis_sections.append(f"• {strength}")
            
            if case_data.weaknesses:
                analysis_sections.append("\nWeaknesses:")
                for weakness in case_data.weaknesses:
                    analysis_sections.append(f"• {weakness}")
        
        # Add risk analysis
        if case_data.risks:
            analysis_sections.append("\nRISK FACTORS:")
            for risk in case_data.risks:
                analysis_sections.append(f"• {risk}")
        
        # Add precedent analysis
        if case_data.precedent_cases:
            analysis_sections.append("\nRELEVANT PRECEDENT:")
            citations = self.formatter.format_citations(case_data.precedent_cases[:5])
            analysis_sections.append(citations)
        
        # Add legal theories analysis
        if case_data.legal_theories:
            analysis_sections.append("\nLEGAL THEORIES:")
            theories_analysis = self.formatter.format_legal_theories(case_data)
            analysis_sections.append(theories_analysis)
        
        return "\n".join(analysis_sections) if analysis_sections else "Legal analysis pending further research."
    
    def _generate_recommendations(self, case_data: CaseData) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Discovery recommendations
        if not case_data.document_requests and not case_data.interrogatories:
            recommendations.append("Initiate discovery process with document requests and interrogatories")
        
        # Evidence recommendations
        if case_data.evidence_list:
            pending_auth = [e for e in case_data.evidence_list if e.authenticity_status.value == "pending_verification"]
            if pending_auth:
                recommendations.append(f"Complete authentication of {len(pending_auth)} evidence items")
        
        # Expert witness recommendations
        expert_witnesses = [w for w in case_data.witnesses if "expert" in w.role.lower()]
        if not expert_witnesses and case_data.case_info.practice_area in ["personal injury", "medical malpractice", "intellectual property"]:
            recommendations.append("Consider retaining expert witnesses for technical testimony")
        
        # Settlement considerations
        if case_data.settlement_offers:
            recommendations.append("Evaluate settlement options in light of case strengths and risks")
        
        # Timeline recommendations
        if case_data.case_info.trial_date:
            days_to_trial = (case_data.case_info.trial_date - date.today()).days
            if days_to_trial < 90:
                recommendations.append("Prioritize trial preparation given approaching trial date")
        
        # General strategic recommendations
        if case_data.weaknesses:
            recommendations.append("Address identified case weaknesses through additional evidence or legal research")
        
        if case_data.opportunities:
            recommendations.append("Capitalize on strategic opportunities identified in case analysis")
        
        # Default recommendations if none generated
        if not recommendations:
            recommendations = [
                "Continue factual development and evidence gathering",
                "Research applicable legal precedents",
                "Assess settlement possibilities",
                "Prepare for upcoming procedural deadlines"
            ]
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def _apply_formatting(self, fact_sheet: FactSheet):
        """Apply consistent formatting to fact sheet content"""
        fact_sheet.formatting_applied = True
        fact_sheet.updated_at = datetime.utcnow()
    
    def _get_litigation_template(self) -> List[Dict[str, Any]]:
        """Get template optimized for litigation cases"""
        return FactSheetTemplate.DEFAULT_SECTIONS + [
            {
                "title": "Discovery Status",
                "key": "discovery_status",
                "required": False,
                "max_length": 800,
                "description": "Status of document production, depositions, and other discovery"
            },
            {
                "title": "Motion Practice",
                "key": "motion_practice", 
                "required": False,
                "max_length": 600,
                "description": "Summary of filed motions and anticipated motion practice"
            }
        ]
    
    def _get_transactional_template(self) -> List[Dict[str, Any]]:
        """Get template optimized for transactional matters"""
        return [
            {
                "title": "Transaction Overview",
                "key": "transaction_overview",
                "required": True,
                "max_length": 1000,
                "description": "Overview of the business transaction"
            },
            {
                "title": "Parties and Structure",
                "key": "parties_structure",
                "required": True,
                "max_length": 800,
                "description": "Transaction parties and proposed structure"
            },
            {
                "title": "Key Terms",
                "key": "key_terms",
                "required": True,
                "max_length": 1200,
                "description": "Material terms and conditions"
            },
            {
                "title": "Legal Issues and Risks",
                "key": "legal_issues_risks",
                "required": True,
                "max_length": 1500,
                "description": "Legal issues and risk assessment"
            },
            {
                "title": "Due Diligence",
                "key": "due_diligence",
                "required": False,
                "max_length": 1000,
                "description": "Due diligence findings and recommendations"
            },
            {
                "title": "Next Steps",
                "key": "next_steps",
                "required": True,
                "max_length": 600,
                "description": "Recommended next steps and timeline"
            }
        ]
    
    def _get_appellate_template(self) -> List[Dict[str, Any]]:
        """Get template optimized for appellate matters"""
        return [
            {
                "title": "Procedural Posture",
                "key": "procedural_posture",
                "required": True,
                "max_length": 800,
                "description": "Lower court proceedings and appeal basis"
            },
            {
                "title": "Issues on Appeal",
                "key": "issues_on_appeal",
                "required": True,
                "max_length": 1000,
                "description": "Questions of law presented on appeal"
            },
            {
                "title": "Standard of Review",
                "key": "standard_of_review",
                "required": True,
                "max_length": 600,
                "description": "Applicable standards of review"
            },
            {
                "title": "Argument Summary",
                "key": "argument_summary",
                "required": True,
                "max_length": 1500,
                "description": "Summary of appellate arguments"
            },
            {
                "title": "Record References",
                "key": "record_references", 
                "required": False,
                "max_length": 800,
                "description": "Key record citations and evidence"
            },
            {
                "title": "Briefing Schedule",
                "key": "briefing_schedule",
                "required": False,
                "max_length": 400,
                "description": "Appellate briefing deadlines and oral argument"
            }
        ]
    
    def get_template_options(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())
    
    def validate_fact_sheet(self, fact_sheet: FactSheet) -> Dict[str, Any]:
        """Validate a completed fact sheet"""
        return self.validator.validate_content(fact_sheet)
    
    def suggest_improvements(self, fact_sheet: FactSheet) -> List[str]:
        """Suggest improvements for a fact sheet"""
        return self.validator.suggest_improvements(fact_sheet)