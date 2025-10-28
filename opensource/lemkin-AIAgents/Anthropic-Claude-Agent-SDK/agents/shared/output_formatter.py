"""
Output Formatter for LemkinAI Agents
Provides standardized templates for reports, memos, and structured outputs.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json


class OutputFormatter:
    """
    Formats agent outputs into standardized legal and investigative formats.

    Supports:
    - Investigation memos
    - Evidence analysis reports
    - Legal research briefs
    - Case summaries
    - Chain-of-custody reports
    - Structured JSON exports
    """

    @staticmethod
    def format_investigation_memo(
        title: str,
        prepared_by: str,
        case_id: Optional[str],
        summary: str,
        findings: List[Dict[str, Any]],
        recommendations: Optional[List[str]] = None,
        evidence_cited: Optional[List[str]] = None,
        legal_analysis: Optional[str] = None,
        next_steps: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format an investigation memo.

        Args:
            title: Memo title
            prepared_by: Agent or person preparing the memo
            case_id: Associated case ID
            summary: Executive summary
            findings: List of findings with details
            recommendations: Optional recommendations
            evidence_cited: Optional list of evidence IDs
            legal_analysis: Optional legal analysis section
            next_steps: Optional next steps
            metadata: Optional metadata

        Returns:
            Formatted memo as markdown
        """
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        memo = f"""# INVESTIGATION MEMO

**MEMORANDUM**

**TO:** Investigation Team
**FROM:** {prepared_by}
**DATE:** {date}
**RE:** {title}
"""

        if case_id:
            memo += f"**CASE ID:** {case_id}  \n"

        memo += f"\n---\n\n## EXECUTIVE SUMMARY\n\n{summary}\n\n"

        # Findings
        memo += "## FINDINGS\n\n"
        for i, finding in enumerate(findings, 1):
            memo += f"### {i}. {finding.get('title', 'Finding ' + str(i))}\n\n"
            memo += f"{finding.get('description', '')}\n\n"

            if finding.get('evidence'):
                memo += f"**Evidence:** {finding.get('evidence')}\n\n"

            if finding.get('significance'):
                memo += f"**Significance:** {finding.get('significance')}\n\n"

        # Legal Analysis
        if legal_analysis:
            memo += f"## LEGAL ANALYSIS\n\n{legal_analysis}\n\n"

        # Recommendations
        if recommendations:
            memo += "## RECOMMENDATIONS\n\n"
            for i, rec in enumerate(recommendations, 1):
                memo += f"{i}. {rec}\n"
            memo += "\n"

        # Next Steps
        if next_steps:
            memo += "## NEXT STEPS\n\n"
            for i, step in enumerate(next_steps, 1):
                memo += f"{i}. {step}\n"
            memo += "\n"

        # Evidence Cited
        if evidence_cited:
            memo += "## EVIDENCE CITED\n\n"
            for evidence_id in evidence_cited:
                memo += f"- Evidence ID: `{evidence_id}`\n"
            memo += "\n"

        # Metadata
        if metadata:
            memo += "---\n\n## METADATA\n\n"
            memo += f"```json\n{json.dumps(metadata, indent=2)}\n```\n"

        return memo

    @staticmethod
    def format_evidence_analysis(
        evidence_id: str,
        evidence_type: str,
        analysis_type: str,
        findings: Dict[str, Any],
        confidence_level: Optional[str] = None,
        limitations: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
        analyst_notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format evidence analysis report.

        Args:
            evidence_id: ID of evidence analyzed
            evidence_type: Type of evidence
            analysis_type: Type of analysis performed
            findings: Analysis findings
            confidence_level: Confidence in analysis (low/medium/high)
            limitations: Known limitations of analysis
            recommendations: Recommendations for further analysis
            analyst_notes: Additional notes
            metadata: Optional metadata

        Returns:
            Formatted analysis as markdown
        """
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        report = f"""# EVIDENCE ANALYSIS REPORT

**EVIDENCE ID:** `{evidence_id}`
**EVIDENCE TYPE:** {evidence_type}
**ANALYSIS TYPE:** {analysis_type}
**DATE:** {date}
"""

        if confidence_level:
            report += f"**CONFIDENCE LEVEL:** {confidence_level.upper()}  \n"

        report += "\n---\n\n## FINDINGS\n\n"

        # Format findings based on structure
        for key, value in findings.items():
            report += f"### {key.replace('_', ' ').title()}\n\n"

            if isinstance(value, list):
                for item in value:
                    report += f"- {item}\n"
                report += "\n"
            elif isinstance(value, dict):
                report += f"```json\n{json.dumps(value, indent=2)}\n```\n\n"
            else:
                report += f"{value}\n\n"

        # Limitations
        if limitations:
            report += "## LIMITATIONS\n\n"
            for limitation in limitations:
                report += f"- {limitation}\n"
            report += "\n"

        # Recommendations
        if recommendations:
            report += "## RECOMMENDATIONS\n\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
            report += "\n"

        # Analyst Notes
        if analyst_notes:
            report += f"## ANALYST NOTES\n\n{analyst_notes}\n\n"

        # Metadata
        if metadata:
            report += "---\n\n## METADATA\n\n"
            report += f"```json\n{json.dumps(metadata, indent=2)}\n```\n"

        return report

    @staticmethod
    def format_chain_of_custody(
        evidence_id: str,
        custody_events: List[Dict[str, Any]],
        integrity_verified: bool,
        summary: Optional[str] = None
    ) -> str:
        """
        Format chain-of-custody report.

        Args:
            evidence_id: Evidence identifier
            custody_events: List of custody events from audit log
            integrity_verified: Whether integrity has been verified
            summary: Optional summary

        Returns:
            Formatted chain-of-custody as markdown
        """
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        report = f"""# CHAIN-OF-CUSTODY REPORT

**EVIDENCE ID:** `{evidence_id}`
**REPORT DATE:** {date}
**INTEGRITY VERIFIED:** {'✓ YES' if integrity_verified else '✗ NO'}

---

"""

        if summary:
            report += f"## SUMMARY\n\n{summary}\n\n"

        report += f"## CUSTODY EVENTS\n\n**Total Events:** {len(custody_events)}\n\n"

        for i, event in enumerate(custody_events, 1):
            report += f"### Event {i}: {event.get('event_type', 'Unknown').replace('_', ' ').title()}\n\n"
            report += f"- **Timestamp:** {event.get('timestamp', 'N/A')}\n"
            report += f"- **Agent/Actor:** {event.get('agent_id', 'N/A')}\n"

            if event.get('details'):
                report += f"- **Details:** {json.dumps(event['details'], indent=2)}\n"

            report += f"- **Event Hash:** `{event.get('event_hash', 'N/A')[:16]}...`\n"
            report += "\n"

        return report

    @staticmethod
    def format_gap_analysis(
        case_id: Optional[str],
        required_elements: List[Dict[str, Any]],
        available_evidence: List[str],
        gaps_identified: List[Dict[str, Any]],
        priority_actions: List[Dict[str, Any]],
        alternative_approaches: Optional[List[str]] = None
    ) -> str:
        """
        Format evidence gap analysis.

        Args:
            case_id: Case identifier
            required_elements: Legal elements required
            available_evidence: Evidence currently available
            gaps_identified: Identified gaps
            priority_actions: Prioritized actions to fill gaps
            alternative_approaches: Alternative evidentiary approaches

        Returns:
            Formatted gap analysis as markdown
        """
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        report = f"""# EVIDENCE GAP ANALYSIS

**DATE:** {date}
"""

        if case_id:
            report += f"**CASE ID:** {case_id}  \n"

        report += "\n---\n\n## OVERVIEW\n\n"
        report += f"- **Required Elements:** {len(required_elements)}\n"
        report += f"- **Available Evidence Items:** {len(available_evidence)}\n"
        report += f"- **Gaps Identified:** {len(gaps_identified)}\n\n"

        # Required Elements
        report += "## REQUIRED LEGAL ELEMENTS\n\n"
        for i, element in enumerate(required_elements, 1):
            report += f"{i}. **{element.get('name', 'Element ' + str(i))}**\n"
            report += f"   - Status: {element.get('status', 'Unknown')}\n"
            if element.get('description'):
                report += f"   - {element.get('description')}\n"
            report += "\n"

        # Gaps
        report += "## IDENTIFIED GAPS\n\n"
        for i, gap in enumerate(gaps_identified, 1):
            report += f"### {i}. {gap.get('element', 'Gap ' + str(i))}\n\n"
            report += f"**Severity:** {gap.get('severity', 'Unknown').upper()}  \n"
            report += f"**Description:** {gap.get('description', '')}\n\n"

            if gap.get('impact'):
                report += f"**Impact:** {gap.get('impact')}\n\n"

        # Priority Actions
        report += "## PRIORITY ACTIONS\n\n"
        for i, action in enumerate(priority_actions, 1):
            report += f"{i}. **{action.get('action', 'Action ' + str(i))}** "
            report += f"(Priority: {action.get('priority', 'Normal').upper()})\n"

            if action.get('rationale'):
                report += f"   - Rationale: {action.get('rationale')}\n"

            if action.get('expected_outcome'):
                report += f"   - Expected Outcome: {action.get('expected_outcome')}\n"

            report += "\n"

        # Alternative Approaches
        if alternative_approaches:
            report += "## ALTERNATIVE APPROACHES\n\n"
            for i, approach in enumerate(alternative_approaches, 1):
                report += f"{i}. {approach}\n"

        return report

    @staticmethod
    def format_structured_json(
        data: Dict[str, Any],
        schema_version: str = "1.0",
        include_metadata: bool = True
    ) -> str:
        """
        Format output as structured JSON with metadata.

        Args:
            data: Data to format
            schema_version: Schema version for the output
            include_metadata: Whether to include metadata wrapper

        Returns:
            JSON string
        """
        if include_metadata:
            output = {
                "schema_version": schema_version,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data": data
            }
        else:
            output = data

        return json.dumps(output, indent=2, default=str)

    @staticmethod
    def format_comparison_matrix(
        items: List[Dict[str, Any]],
        comparison_criteria: List[str],
        title: str = "Comparison Matrix"
    ) -> str:
        """
        Format comparison matrix in markdown table format.

        Args:
            items: Items to compare
            comparison_criteria: Criteria to compare on
            title: Title for the matrix

        Returns:
            Markdown table
        """
        output = f"# {title}\n\n"

        if not items or not comparison_criteria:
            return output + "No data to compare.\n"

        # Header
        output += "| Item | " + " | ".join(comparison_criteria) + " |\n"
        output += "|------|" + "|".join(["------" for _ in comparison_criteria]) + "|\n"

        # Rows
        for item in items:
            item_name = item.get('name', 'Unknown')
            output += f"| {item_name} |"

            for criterion in comparison_criteria:
                value = item.get(criterion, 'N/A')
                output += f" {value} |"

            output += "\n"

        return output
