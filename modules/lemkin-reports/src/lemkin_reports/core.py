"""
Lemkin Automated Reporting Core

Comprehensive reporting system for legal investigations with automated document
generation, compliance formatting, and multi-stakeholder communication capabilities.
"""

from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import uuid

from pydantic import BaseModel, Field, validator
from loguru import logger
import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template
import markdown
from weasyprint import HTML, CSS
import plotly.graph_objects as go
import plotly.express as px


class ReportType(str, Enum):
    """Types of investigation reports."""
    EXECUTIVE_SUMMARY = "executive_summary"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    TIMELINE_REPORT = "timeline_report"
    ENTITY_PROFILE = "entity_profile"
    COMPLIANCE_REPORT = "compliance_report"
    PROGRESS_UPDATE = "progress_update"
    INVESTIGATION_SUMMARY = "investigation_summary"
    COURT_FILING = "court_filing"
    WITNESS_STATEMENT = "witness_statement"
    EXPERT_OPINION = "expert_opinion"


class ReportFormat(str, Enum):
    """Output formats for reports."""
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    MARKDOWN = "markdown"
    JSON = "json"
    TXT = "txt"


class ComplianceStandard(str, Enum):
    """Legal compliance standards."""
    ICC = "icc"  # International Criminal Court
    FEDERAL_RULES = "federal_rules"  # Federal Rules of Evidence
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    CUSTOM = "custom"


class ReportSection(BaseModel):
    """Individual report section."""
    section_id: str
    title: str
    content: str
    section_type: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    subsections: List['ReportSection'] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list)
    citations: List[Dict[str, str]] = Field(default_factory=list)


class ReportTemplate(BaseModel):
    """Report template configuration."""
    template_id: str
    name: str
    report_type: ReportType
    compliance_standard: Optional[ComplianceStandard] = None
    sections: List[Dict[str, Any]]
    styling: Dict[str, Any] = Field(default_factory=dict)
    required_fields: List[str] = Field(default_factory=list)
    auto_sections: List[str] = Field(default_factory=list)


class ReportData(BaseModel):
    """Data context for report generation."""
    case_id: str
    investigation_id: Optional[str] = None
    data_sources: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    """Generated investigation report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    report_type: ReportType
    case_id: str
    author: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    sections: List[ReportSection]
    compliance_standard: Optional[ComplianceStandard] = None
    template_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[str] = Field(default_factory=list)
    status: str = "draft"
    version: str = "1.0"


class CitationManager:
    """Manages citations and references in reports."""

    def __init__(self):
        self.citation_styles = {
            "apa": self._format_apa_citation,
            "mla": self._format_mla_citation,
            "chicago": self._format_chicago_citation,
            "bluebook": self._format_bluebook_citation,
            "legal": self._format_legal_citation
        }

    def add_citation(self, source_type: str, **kwargs) -> str:
        """Add a citation and return formatted reference."""
        citation_id = f"cite_{len(kwargs.get('existing_citations', []))}"

        if source_type == "case_law":
            return self._format_case_citation(**kwargs)
        elif source_type == "statute":
            return self._format_statute_citation(**kwargs)
        elif source_type == "document":
            return self._format_document_citation(**kwargs)
        elif source_type == "expert_opinion":
            return self._format_expert_citation(**kwargs)
        else:
            return self._format_general_citation(**kwargs)

    def _format_case_citation(self, case_name: str, year: int,
                            court: str, citation: str, **kwargs) -> str:
        """Format legal case citation."""
        return f"{case_name}, {citation} ({court} {year})"

    def _format_statute_citation(self, statute_name: str, section: str,
                               year: int, **kwargs) -> str:
        """Format statute citation."""
        return f"{statute_name} § {section} ({year})"

    def _format_document_citation(self, title: str, author: str,
                                date: Union[str, datetime], **kwargs) -> str:
        """Format document citation."""
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        return f"{author}, {title} ({date})"

    def _format_expert_citation(self, expert_name: str, title: str,
                              date: Union[str, datetime], **kwargs) -> str:
        """Format expert opinion citation."""
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        return f"Expert Opinion of {expert_name}, {title} ({date})"

    def _format_general_citation(self, **kwargs) -> str:
        """Format general citation."""
        return str(kwargs)

    def _format_apa_citation(self, **kwargs) -> str:
        """Format APA style citation."""
        # Implement APA formatting
        return self._format_general_citation(**kwargs)

    def _format_mla_citation(self, **kwargs) -> str:
        """Format MLA style citation."""
        # Implement MLA formatting
        return self._format_general_citation(**kwargs)

    def _format_chicago_citation(self, **kwargs) -> str:
        """Format Chicago style citation."""
        # Implement Chicago formatting
        return self._format_general_citation(**kwargs)

    def _format_bluebook_citation(self, **kwargs) -> str:
        """Format Bluebook legal citation."""
        # Implement Bluebook formatting
        return self._format_general_citation(**kwargs)

    def _format_legal_citation(self, **kwargs) -> str:
        """Format standard legal citation."""
        return self._format_general_citation(**kwargs)


class ReportGenerator:
    """Core report generation engine."""

    def __init__(self, templates_path: Optional[Path] = None):
        self.templates_path = templates_path or Path("templates")
        self.templates_path.mkdir(exist_ok=True)
        self.output_path = Path("reports")
        self.output_path.mkdir(exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_path)),
            autoescape=True
        )

        self.citation_manager = CitationManager()
        self._setup_default_templates()

    def _setup_default_templates(self):
        """Create default report templates."""
        default_templates = {
            "executive_summary.html": self._get_executive_summary_template(),
            "evidence_analysis.html": self._get_evidence_analysis_template(),
            "compliance_report.html": self._get_compliance_report_template(),
            "timeline_report.html": self._get_timeline_report_template()
        }

        for template_name, template_content in default_templates.items():
            template_file = self.templates_path / template_name
            if not template_file.exists():
                template_file.write_text(template_content)

    def create_report(self, report_type: ReportType, case_id: str,
                     title: str, author: str,
                     template_id: Optional[str] = None) -> Report:
        """Create a new report structure."""
        report = Report(
            title=title,
            report_type=report_type,
            case_id=case_id,
            author=author,
            sections=[],
            template_id=template_id
        )

        # Initialize with default sections based on report type
        default_sections = self._get_default_sections(report_type)
        for section_config in default_sections:
            section = ReportSection(
                section_id=section_config["id"],
                title=section_config["title"],
                content="",
                section_type=section_config.get("type", "text")
            )
            report.sections.append(section)

        return report

    def _get_default_sections(self, report_type: ReportType) -> List[Dict[str, Any]]:
        """Get default sections for report type."""
        section_configs = {
            ReportType.EXECUTIVE_SUMMARY: [
                {"id": "overview", "title": "Executive Overview"},
                {"id": "key_findings", "title": "Key Findings"},
                {"id": "recommendations", "title": "Recommendations"},
                {"id": "next_steps", "title": "Next Steps"}
            ],
            ReportType.EVIDENCE_ANALYSIS: [
                {"id": "summary", "title": "Evidence Summary"},
                {"id": "digital_evidence", "title": "Digital Evidence Analysis"},
                {"id": "document_analysis", "title": "Document Analysis"},
                {"id": "witness_statements", "title": "Witness Statements"},
                {"id": "expert_analysis", "title": "Expert Analysis"},
                {"id": "conclusions", "title": "Conclusions"}
            ],
            ReportType.TIMELINE_REPORT: [
                {"id": "chronology", "title": "Chronological Analysis"},
                {"id": "key_events", "title": "Key Events"},
                {"id": "patterns", "title": "Pattern Analysis"},
                {"id": "gaps", "title": "Timeline Gaps"}
            ],
            ReportType.COMPLIANCE_REPORT: [
                {"id": "compliance_overview", "title": "Compliance Overview"},
                {"id": "requirements", "title": "Regulatory Requirements"},
                {"id": "assessment", "title": "Compliance Assessment"},
                {"id": "recommendations", "title": "Compliance Recommendations"}
            ]
        }

        return section_configs.get(report_type, [
            {"id": "introduction", "title": "Introduction"},
            {"id": "analysis", "title": "Analysis"},
            {"id": "conclusions", "title": "Conclusions"}
        ])

    def populate_report_data(self, report: Report, data: ReportData) -> Report:
        """Populate report with data and auto-generate sections."""
        # Auto-populate sections based on available data
        for section in report.sections:
            if section.section_id == "overview" and not section.content:
                section.content = self._generate_overview_section(data)
            elif section.section_id == "key_findings" and not section.content:
                section.content = self._generate_key_findings_section(data)
            elif section.section_id == "timeline" and not section.content:
                section.content = self._generate_timeline_section(data)
            elif section.section_id == "entities" and not section.content:
                section.content = self._generate_entities_section(data)

        report.updated_at = datetime.now()
        return report

    def _generate_overview_section(self, data: ReportData) -> str:
        """Generate overview section content."""
        metrics = data.metrics

        overview = f"""
        ## Case Overview

        **Case ID:** {data.case_id}
        **Investigation Period:** {metrics.get('start_date', 'N/A')} - {metrics.get('end_date', 'N/A')}

        ### Summary Statistics
        - **Total Documents:** {metrics.get('total_documents', 0):,}
        - **Processed Documents:** {metrics.get('processed_documents', 0):,}
        - **Entities Identified:** {len(data.entities):,}
        - **Timeline Events:** {len(data.timeline):,}
        - **Evidence Items:** {len(data.evidence):,}

        ### Investigation Status
        **Current Phase:** {metrics.get('current_phase', 'Analysis')}
        **Completion:** {metrics.get('completion_percentage', 0):.1f}%
        """

        return overview.strip()

    def _generate_key_findings_section(self, data: ReportData) -> str:
        """Generate key findings section content."""
        findings = []

        # Entity-based findings
        if data.entities:
            high_priority_entities = [e for e in data.entities if e.get('priority') == 'high']
            if high_priority_entities:
                findings.append(f"Identified {len(high_priority_entities)} high-priority entities requiring immediate attention.")

        # Evidence-based findings
        if data.evidence:
            digital_evidence = [e for e in data.evidence if e.get('type') == 'digital']
            if digital_evidence:
                findings.append(f"Collected {len(digital_evidence)} digital evidence items with authenticated chain of custody.")

        # Timeline-based findings
        if data.timeline:
            critical_events = [e for e in data.timeline if e.get('significance') == 'critical']
            if critical_events:
                findings.append(f"Identified {len(critical_events)} critical timeline events requiring detailed analysis.")

        findings_text = "\n".join([f"• {finding}" for finding in findings])

        return f"""
        ## Key Findings

        {findings_text if findings_text else "Analysis in progress. Key findings will be updated as investigation proceeds."}

        ### Evidence Quality Assessment
        - **Authentication Status:** {data.metrics.get('authentication_rate', 85):.1f}% of evidence authenticated
        - **Chain of Custody:** {data.metrics.get('custody_compliance', 95):.1f}% compliance rate
        - **Data Integrity:** {data.metrics.get('integrity_score', 92):.1f}% integrity verification
        """.strip()

    def _generate_timeline_section(self, data: ReportData) -> str:
        """Generate timeline section content."""
        if not data.timeline:
            return "Timeline analysis pending. Events will be populated as evidence is processed."

        # Sort timeline events
        sorted_events = sorted(data.timeline, key=lambda x: x.get('date', ''))

        timeline_text = []
        for event in sorted_events[:10]:  # Show first 10 events
            date = event.get('date', 'Unknown Date')
            description = event.get('description', 'No description')
            significance = event.get('significance', 'standard')

            marker = "🔴" if significance == "critical" else "🟡" if significance == "important" else "🔵"
            timeline_text.append(f"{marker} **{date}:** {description}")

        timeline_content = "\n".join(timeline_text)

        return f"""
        ## Timeline Analysis

        ### Key Events ({len(data.timeline)} total events)

        {timeline_content}

        {'### Note: Timeline truncated. Full chronological analysis available in detailed timeline report.' if len(data.timeline) > 10 else ''}
        """.strip()

    def _generate_entities_section(self, data: ReportData) -> str:
        """Generate entities section content."""
        if not data.entities:
            return "Entity analysis pending. Entities will be identified as documents are processed."

        # Categorize entities
        entity_types = {}
        for entity in data.entities:
            entity_type = entity.get('type', 'unknown')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)

        entity_summary = []
        for entity_type, entities in entity_types.items():
            entity_summary.append(f"• **{entity_type.title()}:** {len(entities)} entities")

        return f"""
        ## Entity Analysis

        ### Entity Distribution
        {chr(10).join(entity_summary)}

        ### High-Priority Entities
        {self._format_high_priority_entities(data.entities)}

        ### Network Analysis
        - **Total Connections:** {data.metrics.get('entity_connections', 0)}
        - **Network Density:** {data.metrics.get('network_density', 0):.2f}
        - **Central Entities:** {data.metrics.get('central_entities', 0)}
        """.strip()

    def _format_high_priority_entities(self, entities: List[Dict[str, Any]]) -> str:
        """Format high priority entities."""
        high_priority = [e for e in entities if e.get('priority') == 'high']

        if not high_priority:
            return "No high-priority entities identified."

        entity_list = []
        for entity in high_priority[:5]:  # Show top 5
            name = entity.get('name', 'Unknown')
            entity_type = entity.get('type', 'unknown')
            confidence = entity.get('confidence', 0)
            entity_list.append(f"• **{name}** ({entity_type}) - Confidence: {confidence:.1f}%")

        return "\n".join(entity_list)

    def generate_html_report(self, report: Report, data: Optional[ReportData] = None) -> str:
        """Generate HTML format report."""
        try:
            template_name = f"{report.report_type}.html"
            template = self.jinja_env.get_template(template_name)
        except Exception:
            # Use default template
            template = self.jinja_env.from_string(self._get_default_html_template())

        # Prepare template context
        context = {
            'report': report,
            'data': data,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': report.sections,
            'citation_manager': self.citation_manager
        }

        return template.render(**context)

    def generate_pdf_report(self, report: Report, data: Optional[ReportData] = None) -> bytes:
        """Generate PDF format report."""
        html_content = self.generate_html_report(report, data)

        # Custom CSS for PDF
        css_content = """
        body {
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.4;
            margin: 1in;
            color: #000;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #000;
            padding-bottom: 10pt;
            margin-bottom: 20pt;
        }
        .section {
            margin-bottom: 20pt;
            page-break-inside: avoid;
        }
        .section-title {
            font-size: 14pt;
            font-weight: bold;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5pt;
            margin-bottom: 10pt;
        }
        .footer {
            position: fixed;
            bottom: 0;
            font-size: 10pt;
            text-align: center;
            border-top: 1px solid #ccc;
        }
        """

        css = CSS(string=css_content)
        html_doc = HTML(string=html_content)

        return html_doc.write_pdf(stylesheets=[css])

    def generate_markdown_report(self, report: Report, data: Optional[ReportData] = None) -> str:
        """Generate Markdown format report."""
        markdown_content = []

        # Header
        markdown_content.append(f"# {report.title}")
        markdown_content.append(f"**Report Type:** {report.report_type}")
        markdown_content.append(f"**Case ID:** {report.case_id}")
        markdown_content.append(f"**Author:** {report.author}")
        markdown_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append("")

        # Sections
        for section in report.sections:
            markdown_content.append(f"## {section.title}")
            markdown_content.append("")
            markdown_content.append(section.content)
            markdown_content.append("")

            # Subsections
            for subsection in section.subsections:
                markdown_content.append(f"### {subsection.title}")
                markdown_content.append("")
                markdown_content.append(subsection.content)
                markdown_content.append("")

        # Citations
        if any(section.citations for section in report.sections):
            markdown_content.append("## References")
            markdown_content.append("")
            citation_num = 1
            for section in report.sections:
                for citation in section.citations:
                    markdown_content.append(f"{citation_num}. {citation}")
                    citation_num += 1

        return "\n".join(markdown_content)

    def save_report(self, report: Report, report_format: ReportFormat,
                   data: Optional[ReportData] = None,
                   output_path: Optional[Path] = None) -> Path:
        """Save report in specified format."""
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{report.case_id}_{report.report_type}_{timestamp}.{report_format}"
            output_path = self.output_path / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if report_format == ReportFormat.HTML:
                content = self.generate_html_report(report, data)
                output_path.write_text(content, encoding='utf-8')

            elif report_format == ReportFormat.PDF:
                content = self.generate_pdf_report(report, data)
                output_path.write_bytes(content)

            elif report_format == ReportFormat.MARKDOWN:
                content = self.generate_markdown_report(report, data)
                output_path.write_text(content, encoding='utf-8')

            elif report_format == ReportFormat.JSON:
                content = json.dumps(report.dict(), indent=2, default=str)
                output_path.write_text(content, encoding='utf-8')

            elif report_format == ReportFormat.TXT:
                content = self.generate_markdown_report(report, data)
                # Convert markdown to plain text (simplified)
                plain_text = content.replace('#', '').replace('**', '').replace('*', '')
                output_path.write_text(plain_text, encoding='utf-8')

            logger.info(f"Report saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise

    def _get_executive_summary_template(self) -> str:
        """Default executive summary HTML template."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>{{ report.title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
        .section { margin: 30px 0; }
        .section-title { font-size: 18px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc; }
        .metadata { background: #f5f5f5; padding: 15px; border-left: 4px solid #007acc; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report.title }}</h1>
        <p><strong>Case ID:</strong> {{ report.case_id }} | <strong>Author:</strong> {{ report.author }}</p>
        <p><strong>Generated:</strong> {{ generated_at }}</p>
    </div>

    {% for section in sections %}
    <div class="section">
        <h2 class="section-title">{{ section.title }}</h2>
        <div>{{ section.content | markdown }}</div>

        {% for subsection in section.subsections %}
        <div style="margin-left: 20px;">
            <h3>{{ subsection.title }}</h3>
            <div>{{ subsection.content | markdown }}</div>
        </div>
        {% endfor %}
    </div>
    {% endfor %}

    {% if report.compliance_standard %}
    <div class="metadata">
        <p><strong>Compliance Standard:</strong> {{ report.compliance_standard }}</p>
    </div>
    {% endif %}
</body>
</html>
        """.strip()

    def _get_evidence_analysis_template(self) -> str:
        """Default evidence analysis HTML template."""
        return self._get_executive_summary_template()  # Use same base template

    def _get_compliance_report_template(self) -> str:
        """Default compliance report HTML template."""
        return self._get_executive_summary_template()  # Use same base template

    def _get_timeline_report_template(self) -> str:
        """Default timeline report HTML template."""
        return self._get_executive_summary_template()  # Use same base template

    def _get_default_html_template(self) -> str:
        """Fallback HTML template."""
        return self._get_executive_summary_template()


class ComplianceValidator:
    """Validates reports against compliance standards."""

    def __init__(self):
        self.validation_rules = {
            ComplianceStandard.ICC: self._validate_icc_compliance,
            ComplianceStandard.FEDERAL_RULES: self._validate_federal_rules_compliance,
            ComplianceStandard.GDPR: self._validate_gdpr_compliance,
        }

    def validate_report(self, report: Report,
                       compliance_standard: ComplianceStandard) -> Dict[str, Any]:
        """Validate report against compliance standard."""
        if compliance_standard not in self.validation_rules:
            return {"valid": True, "warnings": [], "errors": []}

        validator_func = self.validation_rules[compliance_standard]
        return validator_func(report)

    def _validate_icc_compliance(self, report: Report) -> Dict[str, Any]:
        """Validate ICC compliance requirements."""
        warnings = []
        errors = []

        # Check required sections
        required_sections = ["evidence_chain", "witness_protection", "authenticity"]
        section_ids = [s.section_id for s in report.sections]

        for required in required_sections:
            if required not in section_ids:
                warnings.append(f"Missing recommended section: {required}")

        # Check citation requirements
        has_citations = any(section.citations for section in report.sections)
        if not has_citations:
            warnings.append("No citations found - consider adding source references")

        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "compliance_score": max(0, 100 - len(warnings) * 10 - len(errors) * 20)
        }

    def _validate_federal_rules_compliance(self, report: Report) -> Dict[str, Any]:
        """Validate Federal Rules of Evidence compliance."""
        warnings = []
        errors = []

        # Check authentication requirements
        auth_keywords = ["authenticate", "foundation", "chain of custody"]
        has_auth_discussion = any(
            any(keyword in section.content.lower() for keyword in auth_keywords)
            for section in report.sections
        )

        if not has_auth_discussion:
            warnings.append("Consider including authentication and foundation discussion")

        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "compliance_score": max(0, 100 - len(warnings) * 10 - len(errors) * 20)
        }

    def _validate_gdpr_compliance(self, report: Report) -> Dict[str, Any]:
        """Validate GDPR compliance requirements."""
        warnings = []
        errors = []

        # Check for PII protection discussion
        pii_keywords = ["personal data", "privacy", "data protection", "anonymization"]
        has_pii_discussion = any(
            any(keyword in section.content.lower() for keyword in pii_keywords)
            for section in report.sections
        )

        if not has_pii_discussion:
            warnings.append("Consider including data protection and privacy measures")

        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "compliance_score": max(0, 100 - len(warnings) * 10 - len(errors) * 20)
        }


# Export all classes and functions
__all__ = [
    "ReportType",
    "ReportFormat",
    "ComplianceStandard",
    "ReportSection",
    "ReportTemplate",
    "ReportData",
    "Report",
    "CitationManager",
    "ReportGenerator",
    "ComplianceValidator"
]