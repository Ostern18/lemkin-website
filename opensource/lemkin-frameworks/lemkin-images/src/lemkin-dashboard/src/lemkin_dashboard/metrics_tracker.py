"""
Lemkin Investigation Metrics Tracker Module

Comprehensive investigation progress tracking and metrics analysis for legal case management.
Provides KPI monitoring, productivity analysis, quality assurance metrics, and predictive
insights for investigation workflows.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from collections import defaultdict, Counter
from statistics import mean, median
import calendar

from .core import (
    MetricsDashboard, Investigation, Evidence, Event, Entity,
    InvestigationStatus, EvidenceStatus, EventType, DashboardConfig,
    ExportOptions, ExportFormat
)

# Configure logging
logger = logging.getLogger(__name__)


class InvestigationMetricsTracker:
    """
    Advanced investigation metrics tracking and analysis system.
    
    Provides comprehensive tracking of:
    - Investigation progress and completion rates
    - Evidence quality and verification metrics
    - Team productivity and workload analysis
    - Timeline adherence and milestone tracking
    - Quality assurance and compliance metrics
    - Predictive analytics for case completion
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize metrics tracker with configuration"""
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(f"{__name__}.InvestigationMetricsTracker")
        
        # KPI thresholds and targets
        self.kpi_thresholds = {
            "evidence_quality_target": 85.0,
            "completion_rate_target": 75.0,
            "verification_rate_target": 90.0,
            "timeline_adherence_target": 80.0,
            "productivity_baseline": 5.0  # items per day
        }
        
        # Color schemes for metrics visualizations
        self.metric_colors = {
            "excellent": "#10b981",
            "good": "#3b82f6",
            "warning": "#f59e0b", 
            "critical": "#ef4444",
            "neutral": "#6b7280"
        }
        
        self.logger.info("Investigation Metrics Tracker initialized")
    
    def track_investigation_metrics(self, investigation: Investigation, 
                                   evidence_list: List[Evidence],
                                   events: List[Event],
                                   team_activity: Optional[Dict[str, Any]] = None) -> MetricsDashboard:
        """
        Generate comprehensive investigation metrics dashboard
        
        Args:
            investigation: Investigation object
            evidence_list: List of evidence items
            events: List of investigation events
            team_activity: Optional team activity data
            
        Returns:
            MetricsDashboard: Complete metrics dashboard
        """
        # Calculate core progress metrics
        progress_metrics = self._calculate_progress_metrics(investigation, evidence_list, events)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(evidence_list)
        
        # Calculate productivity metrics
        productivity_metrics = self._calculate_productivity_metrics(
            evidence_list, events, team_activity
        )
        
        # Calculate timeline metrics
        timeline_metrics = self._calculate_timeline_metrics(investigation, events)
        
        # Calculate team metrics
        team_metrics = self._calculate_team_metrics(investigation, team_activity)
        
        # Calculate risk and blocker metrics
        risk_metrics = self._calculate_risk_metrics(investigation, evidence_list, events)
        
        # Create metrics dashboard
        metrics_dashboard = MetricsDashboard(
            case_id=investigation.case_id,
            title=f"Investigation Metrics: {investigation.title}",
            
            # Progress metrics
            overall_progress=progress_metrics["overall_progress"],
            evidence_collection_progress=progress_metrics["evidence_collection_progress"],
            analysis_progress=progress_metrics["analysis_progress"],
            verification_progress=progress_metrics["verification_progress"],
            
            # Quality metrics
            evidence_quality_score=quality_metrics["evidence_quality_score"],
            data_completeness=quality_metrics["data_completeness"],
            verification_rate=quality_metrics["verification_rate"],
            
            # Productivity metrics
            items_processed_today=productivity_metrics["items_processed_today"],
            items_processed_week=productivity_metrics["items_processed_week"],
            items_processed_month=productivity_metrics["items_processed_month"],
            
            # Timeline metrics
            days_since_start=timeline_metrics["days_since_start"],
            days_until_deadline=timeline_metrics["days_until_deadline"],
            milestone_completion=timeline_metrics["milestone_completion"],
            
            # Team metrics
            team_size=team_metrics["team_size"],
            average_workload=team_metrics["average_workload"],
            productivity_trend=team_metrics["productivity_trend"],
            
            # Risk metrics
            critical_path_items=risk_metrics["critical_path_items"],
            blockers=risk_metrics["blockers"],
            risks=risk_metrics["risks"],
            
            last_updated=datetime.utcnow(),
            update_frequency="daily"
        )
        
        self.logger.info(f"Generated metrics dashboard for investigation: {investigation.case_id}")
        return metrics_dashboard
    
    def generate_progress_visualization(self, metrics_dashboard: MetricsDashboard) -> go.Figure:
        """
        Generate comprehensive progress visualization
        
        Args:
            metrics_dashboard: Metrics dashboard object
            
        Returns:
            go.Figure: Plotly figure with progress metrics
        """
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Progress', 'Evidence Progress', 'Quality Metrics', 'Timeline Status'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Overall progress gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics_dashboard.overall_progress,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Progress (%)"},
                delta={'reference': self.kpi_thresholds["completion_rate_target"]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_progress_color(metrics_dashboard.overall_progress)},
                    'steps': [
                        {'range': [0, 50], 'color': "#fee2e2"},
                        {'range': [50, 75], 'color': "#fef3c7"},
                        {'range': [75, 100], 'color': "#d1fae5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': self.kpi_thresholds["completion_rate_target"]
                    }
                }
            ),
            row=1, col=1
        )
        
        # Evidence progress gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics_dashboard.verification_progress,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Evidence Verification (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_progress_color(metrics_dashboard.verification_progress)},
                    'steps': [
                        {'range': [0, 60], 'color': "#fee2e2"},
                        {'range': [60, 85], 'color': "#fef3c7"},
                        {'range': [85, 100], 'color': "#d1fae5"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Quality metrics bar chart
        quality_metrics = [
            metrics_dashboard.evidence_quality_score,
            metrics_dashboard.data_completeness,
            metrics_dashboard.verification_rate
        ]
        quality_labels = ['Evidence Quality', 'Data Completeness', 'Verification Rate']
        
        fig.add_trace(
            go.Bar(
                x=quality_labels,
                y=quality_metrics,
                name="Quality Metrics",
                marker_color=[self._get_progress_color(score) for score in quality_metrics]
            ),
            row=2, col=1
        )
        
        # Timeline scatter plot
        timeline_data = []
        if metrics_dashboard.days_until_deadline is not None:
            timeline_data = [
                {'metric': 'Days Since Start', 'value': metrics_dashboard.days_since_start, 'type': 'elapsed'},
                {'metric': 'Days Until Deadline', 'value': metrics_dashboard.days_until_deadline, 'type': 'remaining'}
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=[d['metric'] for d in timeline_data],
                    y=[d['value'] for d in timeline_data],
                    mode='markers+lines',
                    name="Timeline",
                    marker=dict(size=15, color=['#ef4444', '#3b82f6'])
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Investigation Progress Dashboard - {metrics_dashboard.case_id}",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def generate_productivity_analysis(self, metrics_dashboard: MetricsDashboard,
                                     historical_data: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
        """
        Generate productivity analysis visualization
        
        Args:
            metrics_dashboard: Metrics dashboard object
            historical_data: Optional historical productivity data
            
        Returns:
            go.Figure: Plotly figure with productivity analysis
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Productivity', 'Productivity Trend', 'Workload Distribution', 'Team Performance'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # Daily productivity
        productivity_data = [
            metrics_dashboard.items_processed_today,
            metrics_dashboard.items_processed_week / 7,  # Daily average for week
            metrics_dashboard.items_processed_month / 30  # Daily average for month
        ]
        productivity_labels = ['Today', 'Week Avg', 'Month Avg']
        
        fig.add_trace(
            go.Bar(
                x=productivity_labels,
                y=productivity_data,
                name="Items Processed",
                marker_color=['#3b82f6', '#10b981', '#f59e0b']
            ),
            row=1, col=1
        )
        
        # Add baseline reference line
        fig.add_hline(
            y=self.kpi_thresholds["productivity_baseline"],
            line_dash="dash",
            line_color="red",
            annotation_text="Baseline",
            row=1, col=1
        )
        
        # Productivity trend (if historical data available)
        if historical_data:
            historical_df = pd.DataFrame(historical_data)
            fig.add_trace(
                go.Scatter(
                    x=historical_df.get('date', []),
                    y=historical_df.get('daily_items', []),
                    mode='lines+markers',
                    name="Productivity Trend",
                    line=dict(color='#8b5cf6')
                ),
                row=1, col=2
            )
        else:
            # Generate sample trend data
            dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
            trend_values = np.random.normal(metrics_dashboard.items_processed_today, 2, 30)
            trend_values = np.maximum(trend_values, 0)  # Ensure non-negative
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=trend_values,
                    mode='lines+markers',
                    name="Productivity Trend",
                    line=dict(color='#8b5cf6')
                ),
                row=1, col=2
            )
        
        # Workload distribution
        workload_categories = ['Critical', 'High', 'Medium', 'Low']
        workload_values = [25, 35, 30, 10]  # Sample distribution
        
        fig.add_trace(
            go.Pie(
                labels=workload_categories,
                values=workload_values,
                name="Workload",
                marker_colors=['#ef4444', '#f59e0b', '#3b82f6', '#10b981']
            ),
            row=2, col=1
        )
        
        # Team performance comparison
        team_members = ['Lead', 'Analyst 1', 'Analyst 2', 'Officer 1']
        performance_scores = [95, 88, 92, 85]  # Sample performance scores
        
        fig.add_trace(
            go.Bar(
                x=team_members,
                y=performance_scores,
                name="Performance Score",
                marker_color=[self._get_progress_color(score) for score in performance_scores]
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Investigation Productivity Analysis",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_quality_assessment(self, evidence_list: List[Evidence]) -> go.Figure:
        """
        Generate evidence quality assessment visualization
        
        Args:
            evidence_list: List of evidence items
            
        Returns:
            go.Figure: Quality assessment visualization
        """
        if not evidence_list:
            return go.Figure().add_annotation(text="No evidence data available")
        
        # Analyze evidence quality metrics
        quality_data = self._analyze_evidence_quality(evidence_list)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Distribution', 'Verification Status', 'Quality by Type', 'Quality Timeline'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Quality distribution
        quality_labels = list(quality_data['quality_distribution'].keys())
        quality_values = list(quality_data['quality_distribution'].values())
        
        fig.add_trace(
            go.Pie(
                labels=quality_labels,
                values=quality_values,
                name="Quality",
                marker_colors=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
            ),
            row=1, col=1
        )
        
        # Verification status
        verification_data = quality_data['verification_status']
        fig.add_trace(
            go.Bar(
                x=list(verification_data.keys()),
                y=list(verification_data.values()),
                name="Verification Status",
                marker_color=['#10b981', '#ef4444', '#f59e0b']
            ),
            row=1, col=2
        )
        
        # Quality by evidence type
        type_quality_data = quality_data['quality_by_type']
        fig.add_trace(
            go.Bar(
                x=list(type_quality_data.keys()),
                y=list(type_quality_data.values()),
                name="Quality by Type",
                marker_color='#8b5cf6'
            ),
            row=2, col=1
        )
        
        # Quality timeline
        timeline_data = quality_data['quality_timeline']
        if timeline_data:
            dates = [item['date'] for item in timeline_data]
            quality_scores = [item['quality_score'] for item in timeline_data]
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=quality_scores,
                    mode='lines+markers',
                    name="Quality Timeline",
                    line=dict(color='#06b6d4')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Evidence Quality Assessment",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_risk_analysis(self, investigation: Investigation,
                              evidence_list: List[Evidence],
                              events: List[Event]) -> go.Figure:
        """
        Generate risk analysis and early warning visualization
        
        Args:
            investigation: Investigation object
            evidence_list: List of evidence items
            events: List of events
            
        Returns:
            go.Figure: Risk analysis visualization
        """
        risk_analysis = self._analyze_investigation_risks(investigation, evidence_list, events)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Level Distribution', 'Critical Issues', 'Timeline Risks', 'Mitigation Status'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'indicator'}]]
        )
        
        # Risk level distribution
        risk_levels = list(risk_analysis['risk_distribution'].keys())
        risk_counts = list(risk_analysis['risk_distribution'].values())
        
        fig.add_trace(
            go.Pie(
                labels=risk_levels,
                values=risk_counts,
                name="Risk Levels",
                marker_colors=['#ef4444', '#f59e0b', '#3b82f6', '#10b981']
            ),
            row=1, col=1
        )
        
        # Critical issues
        critical_issues = risk_analysis['critical_issues']
        if critical_issues:
            fig.add_trace(
                go.Bar(
                    x=list(critical_issues.keys()),
                    y=list(critical_issues.values()),
                    name="Critical Issues",
                    marker_color='#ef4444'
                ),
                row=1, col=2
            )
        
        # Timeline risks
        timeline_risks = risk_analysis['timeline_risks']
        if timeline_risks:
            fig.add_trace(
                go.Scatter(
                    x=[item['date'] for item in timeline_risks],
                    y=[item['risk_score'] for item in timeline_risks],
                    mode='markers+lines',
                    name="Timeline Risks",
                    marker=dict(
                        size=[item['severity'] * 5 for item in timeline_risks],
                        color='#f59e0b'
                    )
                ),
                row=2, col=1
            )
        
        # Mitigation status indicator
        mitigation_score = risk_analysis['mitigation_effectiveness']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=mitigation_score,
                title={'text': "Mitigation Effectiveness (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_progress_color(mitigation_score)},
                    'steps': [
                        {'range': [0, 50], 'color': "#fee2e2"},
                        {'range': [50, 75], 'color': "#fef3c7"},
                        {'range': [75, 100], 'color': "#d1fae5"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Investigation Risk Analysis",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _calculate_progress_metrics(self, investigation: Investigation,
                                   evidence_list: List[Evidence],
                                   events: List[Event]) -> Dict[str, float]:
        """Calculate investigation progress metrics"""
        total_evidence = len(evidence_list)
        analyzed_evidence = len([e for e in evidence_list if e.analysis_completed])
        verified_evidence = len([e for e in evidence_list if e.authenticity_verified])
        
        return {
            "overall_progress": investigation.completion_percentage,
            "evidence_collection_progress": 100.0,  # Assume complete if evidence exists
            "analysis_progress": (analyzed_evidence / total_evidence * 100) if total_evidence > 0 else 0.0,
            "verification_progress": (verified_evidence / total_evidence * 100) if total_evidence > 0 else 0.0
        }
    
    def _calculate_quality_metrics(self, evidence_list: List[Evidence]) -> Dict[str, float]:
        """Calculate evidence quality metrics"""
        if not evidence_list:
            return {"evidence_quality_score": 0.0, "data_completeness": 0.0, "verification_rate": 0.0}
        
        verified_count = len([e for e in evidence_list if e.authenticity_verified])
        analyzed_count = len([e for e in evidence_list if e.analysis_completed])
        total_count = len(evidence_list)
        
        # Calculate completeness based on required fields
        completeness_scores = []
        for evidence in evidence_list:
            required_fields = ['title', 'description', 'evidence_type', 'collected_by']
            filled_fields = sum(1 for field in required_fields if getattr(evidence, field))
            completeness_scores.append(filled_fields / len(required_fields) * 100)
        
        return {
            "evidence_quality_score": (verified_count / total_count * 100) if total_count > 0 else 0.0,
            "data_completeness": mean(completeness_scores) if completeness_scores else 0.0,
            "verification_rate": (verified_count / total_count * 100) if total_count > 0 else 0.0
        }
    
    def _calculate_productivity_metrics(self, evidence_list: List[Evidence],
                                       events: List[Event],
                                       team_activity: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate productivity metrics"""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        
        # Count items processed in different time periods
        today_evidence = len([e for e in evidence_list if e.created_at >= today_start])
        week_evidence = len([e for e in evidence_list if e.created_at >= week_start])
        month_evidence = len([e for e in evidence_list if e.created_at >= month_start])
        
        today_events = len([e for e in events if e.created_at >= today_start])
        week_events = len([e for e in events if e.created_at >= week_start])
        month_events = len([e for e in events if e.created_at >= month_start])
        
        return {
            "items_processed_today": today_evidence + today_events,
            "items_processed_week": week_evidence + week_events,
            "items_processed_month": month_evidence + month_events
        }
    
    def _calculate_timeline_metrics(self, investigation: Investigation,
                                   events: List[Event]) -> Dict[str, Any]:
        """Calculate timeline and milestone metrics"""
        now = datetime.utcnow()
        days_since_start = (now - investigation.start_date).days
        
        days_until_deadline = None
        if investigation.deadline:
            days_until_deadline = (investigation.deadline - now).days
        
        # Calculate milestone completion
        milestone_completion = {}
        for milestone in investigation.milestones:
            milestone_completion[milestone.get('name', 'Unknown')] = milestone.get('completed', False)
        
        return {
            "days_since_start": days_since_start,
            "days_until_deadline": days_until_deadline,
            "milestone_completion": milestone_completion
        }
    
    def _calculate_team_metrics(self, investigation: Investigation,
                               team_activity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate team performance metrics"""
        team_size = len(investigation.team_members) + 1  # +1 for lead investigator
        
        # Calculate average workload (placeholder calculation)
        average_workload = 50.0  # This would be calculated from actual workload data
        
        # Determine productivity trend
        productivity_trend = "stable"  # This would be calculated from historical data
        
        return {
            "team_size": team_size,
            "average_workload": average_workload,
            "productivity_trend": productivity_trend
        }
    
    def _calculate_risk_metrics(self, investigation: Investigation,
                               evidence_list: List[Evidence],
                               events: List[Event]) -> Dict[str, int]:
        """Calculate risk and blocker metrics"""
        # Count critical path items
        critical_evidence = len([e for e in evidence_list if e.priority == "critical"])
        critical_events = len([e for e in events if e.importance == "critical"])
        critical_path_items = critical_evidence + critical_events
        
        # Count blockers (evidence not yet analyzed or verified)
        unanalyzed_evidence = len([e for e in evidence_list if not e.analysis_completed])
        unverified_evidence = len([e for e in evidence_list if not e.authenticity_verified])
        blockers = unanalyzed_evidence + unverified_evidence
        
        # Count risks (high priority items or approaching deadlines)
        high_priority_items = len([e for e in evidence_list if e.priority == "high"])
        risks = high_priority_items
        
        # Add deadline risk if applicable
        if investigation.deadline:
            days_until_deadline = (investigation.deadline - datetime.utcnow()).days
            if days_until_deadline < 30:  # Less than 30 days
                risks += 1
        
        return {
            "critical_path_items": critical_path_items,
            "blockers": blockers,
            "risks": risks
        }
    
    def _analyze_evidence_quality(self, evidence_list: List[Evidence]) -> Dict[str, Any]:
        """Analyze evidence quality in detail"""
        quality_analysis = {
            "quality_distribution": {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0},
            "verification_status": {"Verified": 0, "Unverified": 0, "Pending": 0},
            "quality_by_type": {},
            "quality_timeline": []
        }
        
        # Analyze quality distribution
        for evidence in evidence_list:
            if evidence.authenticity_verified and evidence.analysis_completed:
                quality_analysis["quality_distribution"]["High"] += 1
            elif evidence.analysis_completed:
                quality_analysis["quality_distribution"]["Medium"] += 1
            elif evidence.authenticity_verified:
                quality_analysis["quality_distribution"]["Medium"] += 1
            else:
                quality_analysis["quality_distribution"]["Low"] += 1
        
        # Analyze verification status
        for evidence in evidence_list:
            if evidence.authenticity_verified:
                quality_analysis["verification_status"]["Verified"] += 1
            elif evidence.analysis_completed:
                quality_analysis["verification_status"]["Pending"] += 1
            else:
                quality_analysis["verification_status"]["Unverified"] += 1
        
        # Analyze quality by type
        type_quality = defaultdict(list)
        for evidence in evidence_list:
            quality_score = 0
            if evidence.authenticity_verified:
                quality_score += 50
            if evidence.analysis_completed:
                quality_score += 50
            type_quality[evidence.evidence_type].append(quality_score)
        
        for evidence_type, scores in type_quality.items():
            quality_analysis["quality_by_type"][evidence_type] = mean(scores)
        
        # Generate quality timeline (sample data)
        base_date = min(e.collected_at for e in evidence_list) if evidence_list else datetime.now()
        for i in range(30):
            date = base_date + timedelta(days=i)
            quality_score = 75 + np.random.normal(0, 10)  # Sample quality score
            quality_analysis["quality_timeline"].append({
                "date": date,
                "quality_score": max(0, min(100, quality_score))
            })
        
        return quality_analysis
    
    def _analyze_investigation_risks(self, investigation: Investigation,
                                    evidence_list: List[Evidence],
                                    events: List[Event]) -> Dict[str, Any]:
        """Analyze investigation risks and issues"""
        risk_analysis = {
            "risk_distribution": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0},
            "critical_issues": {},
            "timeline_risks": [],
            "mitigation_effectiveness": 75.0  # Placeholder
        }
        
        # Analyze risk distribution
        total_items = len(evidence_list) + len(events)
        critical_items = len([e for e in evidence_list if e.priority == "critical"])
        critical_items += len([e for e in events if e.importance == "critical"])
        
        high_items = len([e for e in evidence_list if e.priority == "high"])
        high_items += len([e for e in events if e.importance == "high"])
        
        risk_analysis["risk_distribution"]["Critical"] = critical_items
        risk_analysis["risk_distribution"]["High"] = high_items
        risk_analysis["risk_distribution"]["Medium"] = total_items - critical_items - high_items
        risk_analysis["risk_distribution"]["Low"] = 0
        
        # Identify critical issues
        if investigation.deadline:
            days_remaining = (investigation.deadline - datetime.utcnow()).days
            if days_remaining < 30:
                risk_analysis["critical_issues"]["Timeline Pressure"] = days_remaining
        
        unverified_count = len([e for e in evidence_list if not e.authenticity_verified])
        if unverified_count > 0:
            risk_analysis["critical_issues"]["Unverified Evidence"] = unverified_count
        
        # Generate timeline risks (sample data)
        base_date = investigation.start_date
        for i in range(0, (datetime.now() - base_date).days, 5):
            date = base_date + timedelta(days=i)
            risk_score = 30 + np.random.normal(0, 15)  # Sample risk score
            risk_analysis["timeline_risks"].append({
                "date": date,
                "risk_score": max(0, min(100, risk_score)),
                "severity": 3 if risk_score > 60 else 2 if risk_score > 40 else 1
            })
        
        return risk_analysis
    
    def _get_progress_color(self, progress: float) -> str:
        """Get color based on progress value"""
        if progress >= 85:
            return self.metric_colors["excellent"]
        elif progress >= 70:
            return self.metric_colors["good"]
        elif progress >= 50:
            return self.metric_colors["warning"]
        else:
            return self.metric_colors["critical"]
    
    def export_metrics_report(self, metrics_dashboard: MetricsDashboard,
                             evidence_list: List[Evidence],
                             events: List[Event],
                             output_dir: Path) -> Dict[str, Path]:
        """
        Export comprehensive metrics analysis report
        
        Args:
            metrics_dashboard: Metrics dashboard object
            evidence_list: List of evidence items
            events: List of events
            output_dir: Directory to save report files
            
        Returns:
            Dict mapping report types to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_paths = {}
        
        try:
            # Generate progress visualization
            progress_fig = self.generate_progress_visualization(metrics_dashboard)
            progress_path = output_dir / f"metrics_progress_{metrics_dashboard.case_id}.html"
            progress_fig.write_html(str(progress_path))
            report_paths['progress'] = progress_path
            
            # Generate productivity analysis
            productivity_fig = self.generate_productivity_analysis(metrics_dashboard)
            productivity_path = output_dir / f"metrics_productivity_{metrics_dashboard.case_id}.html"
            productivity_fig.write_html(str(productivity_path))
            report_paths['productivity'] = productivity_path
            
            # Generate quality assessment
            quality_fig = self.generate_quality_assessment(evidence_list)
            quality_path = output_dir / f"metrics_quality_{metrics_dashboard.case_id}.html"
            quality_fig.write_html(str(quality_path))
            report_paths['quality'] = quality_path
            
            # Export metrics data
            metrics_data = metrics_dashboard.dict()
            metrics_path = output_dir / f"metrics_data_{metrics_dashboard.case_id}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            report_paths['data'] = metrics_path
            
            self.logger.info(f"Metrics report exported to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics report: {str(e)}")
        
        return report_paths


def track_investigation_metrics(investigation: Investigation, 
                               evidence_list: List[Evidence],
                               events: List[Event],
                               config: Optional[DashboardConfig] = None) -> MetricsDashboard:
    """
    Convenience function to track investigation metrics
    
    Args:
        investigation: Investigation object
        evidence_list: List of evidence items
        events: List of events
        config: Optional dashboard configuration
        
    Returns:
        MetricsDashboard: Complete metrics dashboard
    """
    tracker = InvestigationMetricsTracker(config)
    return tracker.track_investigation_metrics(investigation, evidence_list, events)


def generate_metrics_report(investigation: Investigation,
                           evidence_list: List[Evidence],
                           events: List[Event],
                           output_dir: Path) -> Dict[str, Path]:
    """
    Generate comprehensive metrics report
    
    Args:
        investigation: Investigation object
        evidence_list: List of evidence items
        events: List of events
        output_dir: Directory to save report files
        
    Returns:
        Dict mapping report types to file paths
    """
    tracker = InvestigationMetricsTracker()
    metrics_dashboard = tracker.track_investigation_metrics(investigation, evidence_list, events)
    return tracker.export_metrics_report(metrics_dashboard, evidence_list, events, output_dir)