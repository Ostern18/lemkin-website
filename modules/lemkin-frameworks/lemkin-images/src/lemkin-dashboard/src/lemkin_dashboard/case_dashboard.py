"""
Lemkin Case Dashboard Module

Streamlit-based case overview dashboard for professional legal case presentation.
Provides comprehensive case visualization, evidence tracking, and interactive
analysis tools for legal proceedings.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .core import (
    Dashboard, DashboardGenerator, CaseOverview, Investigation, Evidence, 
    Event, Entity, Relationship, DashboardConfig, InvestigationStatus,
    EvidenceStatus, EventType, EntityType
)

# Configure logging
logger = logging.getLogger(__name__)


def generate_case_dashboard(case_id: str) -> Dashboard:
    """
    Generate a comprehensive case overview dashboard
    
    Args:
        case_id: Unique identifier for the case
        
    Returns:
        Dashboard: Complete case dashboard with all visualizations
    """
    try:
        # Initialize dashboard generator
        config = DashboardConfig(
            theme="professional",
            enable_filtering=True,
            enable_search=True,
            enable_export=True
        )
        generator = DashboardGenerator(config)
        
        # Load case data (in real implementation, this would come from database)
        investigation, evidence_list, entities, events, relationships = _load_case_data(case_id)
        
        # Generate comprehensive dashboard
        dashboard = generator.generate_case_dashboard(
            case_id=case_id,
            investigation=investigation,
            evidence_list=evidence_list,
            entities=entities,
            events=events,
            relationships=relationships
        )
        
        logger.info(f"Generated case dashboard for {case_id}")
        return dashboard
        
    except Exception as e:
        logger.error(f"Failed to generate case dashboard for {case_id}: {str(e)}")
        raise


def create_streamlit_case_dashboard(case_id: str):
    """
    Create Streamlit interface for case dashboard
    
    Args:
        case_id: Case identifier to display
    """
    st.set_page_config(
        page_title=f"Case Dashboard - {case_id}",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .evidence-status {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .status-verified { background: #10b981; color: white; }
    .status-pending { background: #f59e0b; color: white; }
    .status-disputed { background: #ef4444; color: white; }
    .sidebar-section {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Generate dashboard data
        dashboard = generate_case_dashboard(case_id)
        case_overview = dashboard.case_overview
        investigation = case_overview.investigation
        
        # Header section
        st.markdown(f"""
        <div class="main-header">
            <h1>📊 Case Dashboard</h1>
            <h2>{investigation.title}</h2>
            <p><strong>Case ID:</strong> {case_id} | <strong>Status:</strong> {investigation.status.value.title()} | <strong>Lead:</strong> {investigation.lead_investigator}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar filters and controls
        _create_sidebar_controls(dashboard)
        
        # Main dashboard content
        _create_main_dashboard_content(dashboard)
        
    except Exception as e:
        st.error(f"Error loading case dashboard: {str(e)}")
        logger.error(f"Streamlit dashboard error for case {case_id}: {str(e)}")


def _create_sidebar_controls(dashboard: Dashboard):
    """Create sidebar controls for filtering and navigation"""
    st.sidebar.header("🔍 Dashboard Controls")
    
    # Case information section
    with st.sidebar.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Case Information")
        
        if dashboard.case_overview:
            overview = dashboard.case_overview
            investigation = overview.investigation
            
            st.write(f"**Status:** {investigation.status.value.title()}")
            st.write(f"**Priority:** {investigation.case_priority.title()}")
            st.write(f"**Start Date:** {investigation.start_date.strftime('%Y-%m-%d')}")
            
            if investigation.deadline:
                days_until_deadline = (investigation.deadline - datetime.utcnow()).days
                st.write(f"**Deadline:** {investigation.deadline.strftime('%Y-%m-%d')} ({days_until_deadline} days)")
            
            st.write(f"**Progress:** {investigation.completion_percentage:.1f}%")
            st.progress(investigation.completion_percentage / 100.0)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filters section
    with st.sidebar.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Filters")
        
        # Date range filter
        st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            key="date_range"
        )
        
        # Evidence status filter
        evidence_status_filter = st.multiselect(
            "Evidence Status",
            options=[status.value for status in EvidenceStatus],
            default=[status.value for status in EvidenceStatus],
            key="evidence_status_filter"
        )
        
        # Event type filter
        event_type_filter = st.multiselect(
            "Event Types",
            options=[event_type.value for event_type in EventType],
            default=[event_type.value for event_type in EventType],
            key="event_type_filter"
        )
        
        # Entity type filter
        entity_type_filter = st.multiselect(
            "Entity Types",
            options=[entity_type.value for entity_type in EntityType],
            default=[entity_type.value for entity_type in EntityType],
            key="entity_type_filter"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Export section
    with st.sidebar.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Export Options")
        
        export_format = st.selectbox(
            "Export Format",
            options=["HTML", "PDF", "JSON"],
            key="export_format"
        )
        
        if st.button("📥 Export Dashboard", key="export_button"):
            _handle_dashboard_export(dashboard, export_format)
        
        st.markdown('</div>', unsafe_allow_html=True)


def _create_main_dashboard_content(dashboard: Dashboard):
    """Create main dashboard content with multiple tabs and sections"""
    
    # Key metrics overview
    _create_key_metrics_section(dashboard)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "📋 Evidence", "👥 Entities", "⏰ Timeline", "📊 Analytics"
    ])
    
    with tab1:
        _create_overview_tab(dashboard)
    
    with tab2:
        _create_evidence_tab(dashboard)
    
    with tab3:
        _create_entities_tab(dashboard)
    
    with tab4:
        _create_timeline_tab(dashboard)
    
    with tab5:
        _create_analytics_tab(dashboard)


def _create_key_metrics_section(dashboard: Dashboard):
    """Create key metrics display section"""
    if not dashboard.case_overview:
        return
    
    overview = dashboard.case_overview
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1e3a8a;">📄 Evidence</h3>
            <h2 style="margin: 5px 0; color: #3b82f6;">{overview.total_evidence}</h2>
            <p style="margin: 0; color: #64748b;">{overview.verified_evidence} verified</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1e3a8a;">👥 Entities</h3>
            <h2 style="margin: 5px 0; color: #3b82f6;">{overview.total_entities}</h2>
            <p style="margin: 0; color: #64748b;">{len(overview.key_entities)} key entities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1e3a8a;">⏰ Events</h3>
            <h2 style="margin: 5px 0; color: #3b82f6;">{overview.total_events}</h2>
            <p style="margin: 0; color: #64748b;">{len(overview.critical_events)} critical</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1e3a8a;">🔗 Links</h3>
            <h2 style="margin: 5px 0; color: #3b82f6;">{overview.total_relationships}</h2>
            <p style="margin: 0; color: #64748b;">relationships</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        progress = overview.investigation.completion_percentage
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1e3a8a;">📊 Progress</h3>
            <h2 style="margin: 5px 0; color: #3b82f6;">{progress:.1f}%</h2>
            <p style="margin: 0; color: #64748b;">complete</p>
        </div>
        """, unsafe_allow_html=True)


def _create_overview_tab(dashboard: Dashboard):
    """Create case overview tab content"""
    if not dashboard.case_overview:
        st.warning("No case overview data available")
        return
    
    overview = dashboard.case_overview
    investigation = overview.investigation
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📖 Case Summary")
        st.write(f"**Description:** {investigation.description}")
        st.write(f"**Jurisdiction:** {investigation.jurisdiction or 'Not specified'}")
        st.write(f"**Legal Basis:** {investigation.legal_basis or 'Not specified'}")
        
        # Team information
        st.subheader("👨‍💼 Investigation Team")
        st.write(f"**Lead Investigator:** {investigation.lead_investigator}")
        
        if investigation.team_members:
            st.write("**Team Members:**")
            for member in investigation.team_members:
                st.write(f"• {member}")
        
        if investigation.legal_counsel:
            st.write("**Legal Counsel:**")
            for counsel in investigation.legal_counsel:
                st.write(f"• {counsel}")
    
    with col2:
        st.subheader("📅 Key Dates")
        st.write(f"**Case Start:** {investigation.start_date.strftime('%Y-%m-%d')}")
        
        if investigation.end_date:
            st.write(f"**Case End:** {investigation.end_date.strftime('%Y-%m-%d')}")
        
        if investigation.deadline:
            st.write(f"**Deadline:** {investigation.deadline.strftime('%Y-%m-%d')}")
        
        if overview.last_activity_date:
            st.write(f"**Last Activity:** {overview.last_activity_date.strftime('%Y-%m-%d')}")
        
        if overview.next_deadline:
            st.write(f"**Next Deadline:** {overview.next_deadline.strftime('%Y-%m-%d')}")
    
    # Progress visualization
    if investigation.milestones:
        st.subheader("🎯 Milestones")
        milestones_df = pd.DataFrame(investigation.milestones)
        
        if not milestones_df.empty:
            fig = px.bar(
                milestones_df, 
                x="name", 
                y="completion",
                title="Milestone Progress",
                color="completion",
                color_continuous_scale="Blues"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def _create_evidence_tab(dashboard: Dashboard):
    """Create evidence management tab"""
    st.subheader("📄 Evidence Management")
    
    # Sample evidence data for demonstration
    evidence_data = []
    if dashboard.case_overview and dashboard.case_overview.high_priority_evidence:
        for evidence in dashboard.case_overview.high_priority_evidence:
            evidence_data.append({
                "ID": str(evidence.id)[:8] + "...",
                "Title": evidence.title,
                "Type": evidence.evidence_type,
                "Status": evidence.status.value,
                "Priority": evidence.priority,
                "Collected By": evidence.collected_by,
                "Date Collected": evidence.collected_at.strftime('%Y-%m-%d'),
                "Verified": "✅" if evidence.authenticity_verified else "❌",
                "Analyzed": "✅" if evidence.analysis_completed else "❌"
            })
    
    if evidence_data:
        evidence_df = pd.DataFrame(evidence_data)
        
        # Evidence status distribution
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = evidence_df["Status"].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Evidence by Status"
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        with col2:
            priority_counts = evidence_df["Priority"].value_counts()
            fig_priority = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Evidence by Priority"
            )
            st.plotly_chart(fig_priority, use_container_width=True)
        
        # Evidence table with filtering
        st.subheader("Evidence Details")
        
        # Add search functionality
        search_term = st.text_input("🔍 Search evidence", key="evidence_search")
        
        if search_term:
            filtered_df = evidence_df[
                evidence_df["Title"].str.contains(search_term, case=False, na=False) |
                evidence_df["Type"].str.contains(search_term, case=False, na=False)
            ]
        else:
            filtered_df = evidence_df
        
        # Display evidence table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Evidence collection timeline
        st.subheader("📈 Evidence Collection Timeline")
        evidence_timeline = evidence_df.copy()
        evidence_timeline["Date Collected"] = pd.to_datetime(evidence_timeline["Date Collected"])
        evidence_timeline = evidence_timeline.sort_values("Date Collected")
        evidence_timeline["Cumulative Count"] = range(1, len(evidence_timeline) + 1)
        
        fig_timeline = px.line(
            evidence_timeline,
            x="Date Collected",
            y="Cumulative Count",
            title="Cumulative Evidence Collection",
            markers=True
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
    else:
        st.info("No evidence data available for display")


def _create_entities_tab(dashboard: Dashboard):
    """Create entities and relationships tab"""
    st.subheader("👥 Entities and Relationships")
    
    if not dashboard.case_overview or not dashboard.case_overview.key_entities:
        st.info("No entity data available for display")
        return
    
    entities = dashboard.case_overview.key_entities
    
    # Entity type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        entity_types = [entity.entity_type.value for entity in entities]
        type_counts = pd.Series(entity_types).value_counts()
        
        fig_types = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Entities by Type"
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        importance_levels = [entity.importance for entity in entities]
        importance_counts = pd.Series(importance_levels).value_counts()
        
        fig_importance = px.bar(
            x=importance_counts.index,
            y=importance_counts.values,
            title="Entities by Importance"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Entity details table
    st.subheader("Entity Details")
    
    entity_data = []
    for entity in entities:
        entity_data.append({
            "Name": entity.name,
            "Type": entity.entity_type.value,
            "Importance": entity.importance,
            "Role": entity.role_in_case or "Not specified",
            "Location": entity.primary_location or "Not specified",
            "Subject": "✅" if entity.subject_of_investigation else "❌",
            "Witness": "✅" if entity.witness else "❌"
        })
    
    entity_df = pd.DataFrame(entity_data)
    st.dataframe(entity_df, use_container_width=True)
    
    # Network visualization placeholder
    st.subheader("🕸️ Entity Network")
    st.info("Interactive network visualization would be displayed here using networkx/plotly")


def _create_timeline_tab(dashboard: Dashboard):
    """Create timeline visualization tab"""
    st.subheader("⏰ Case Timeline")
    
    if not dashboard.case_overview or not dashboard.case_overview.critical_events:
        st.info("No timeline data available for display")
        return
    
    events = dashboard.case_overview.critical_events
    
    # Prepare timeline data
    timeline_data = []
    for event in events:
        timeline_data.append({
            "Date": event.timestamp,
            "Event": event.title,
            "Type": event.event_type.value,
            "Description": event.description,
            "Importance": event.importance,
            "Location": event.location or "Not specified"
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df = timeline_df.sort_values("Date")
    
    # Timeline visualization
    fig_timeline = px.scatter(
        timeline_df,
        x="Date",
        y="Event",
        color="Type",
        size="Importance",
        hover_data=["Description", "Location"],
        title="Case Events Timeline"
    )
    
    fig_timeline.update_traces(marker=dict(size=12))
    fig_timeline.update_layout(height=600)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Event type distribution over time
    col1, col2 = st.columns(2)
    
    with col1:
        timeline_df["Month"] = timeline_df["Date"].dt.to_period("M").astype(str)
        monthly_events = timeline_df.groupby(["Month", "Type"]).size().reset_index(name="Count")
        
        fig_monthly = px.bar(
            monthly_events,
            x="Month",
            y="Count",
            color="Type",
            title="Events by Month and Type"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        event_type_counts = timeline_df["Type"].value_counts()
        
        fig_event_types = px.pie(
            values=event_type_counts.values,
            names=event_type_counts.index,
            title="Event Types Distribution"
        )
        st.plotly_chart(fig_event_types, use_container_width=True)


def _create_analytics_tab(dashboard: Dashboard):
    """Create analytics and insights tab"""
    st.subheader("📊 Case Analytics")
    
    # Case metrics
    if dashboard.case_overview:
        overview = dashboard.case_overview
        investigation = overview.investigation
        
        # Progress analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Overall Progress",
                value=f"{investigation.completion_percentage:.1f}%",
                delta=f"{investigation.completion_percentage - 75:.1f}%" if investigation.completion_percentage > 75 else None
            )
            
            st.metric(
                label="Evidence Quality",
                value=f"{(overview.verified_evidence / overview.total_evidence * 100):.1f}%" if overview.total_evidence > 0 else "0%",
                delta="High quality" if overview.verified_evidence / overview.total_evidence > 0.8 else "Needs improvement"
            )
        
        with col2:
            days_active = (datetime.utcnow() - investigation.start_date).days
            st.metric(label="Days Active", value=days_active)
            
            if investigation.deadline:
                days_remaining = (investigation.deadline - datetime.utcnow()).days
                st.metric(
                    label="Days to Deadline",
                    value=days_remaining,
                    delta="On track" if days_remaining > 30 else "Urgent"
                )
        
        # Activity heatmap
        st.subheader("📅 Activity Heatmap")
        
        # Generate sample activity data
        dates = pd.date_range(start=investigation.start_date, end=datetime.now(), freq='D')
        activity_data = []
        
        for date in dates:
            # Simulate activity levels
            activity_level = max(0, min(10, int((date.weekday() + 1) * 1.2 + (date.day % 7))))
            activity_data.append({
                "Date": date,
                "Activity": activity_level,
                "Weekday": date.strftime("%A"),
                "Week": date.isocalendar()[1]
            })
        
        activity_df = pd.DataFrame(activity_data)
        
        fig_heatmap = px.density_heatmap(
            activity_df,
            x="Week",
            y="Weekday",
            z="Activity",
            title="Investigation Activity Heatmap",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    else:
        st.warning("No analytics data available")


def _handle_dashboard_export(dashboard: Dashboard, export_format: str):
    """Handle dashboard export functionality"""
    try:
        generator = DashboardGenerator()
        
        # Create export path
        export_path = Path(f"case_dashboard_{dashboard.case_id}.{export_format.lower()}")
        
        # Export dashboard
        if export_format == "HTML":
            success = generator.export_dashboard(dashboard, export_path, export_format="html")
        elif export_format == "JSON":
            success = generator.export_dashboard(dashboard, export_path, export_format="json")
        else:
            st.warning(f"Export format {export_format} not yet implemented")
            return
        
        if success:
            st.success(f"Dashboard exported successfully to {export_path}")
            
            # Provide download link
            with open(export_path, "rb") as file:
                st.download_button(
                    label=f"📥 Download {export_format}",
                    data=file.read(),
                    file_name=export_path.name,
                    mime="text/html" if export_format == "HTML" else "application/json"
                )
        else:
            st.error("Export failed. Please try again.")
    
    except Exception as e:
        st.error(f"Export error: {str(e)}")


def _load_case_data(case_id: str) -> tuple:
    """Load case data (placeholder for database integration)"""
    # This would typically load from a database
    # For demonstration, we'll create sample data
    
    from uuid import uuid4
    
    # Sample investigation
    investigation = Investigation(
        case_id=case_id,
        title=f"Investigation Case {case_id}",
        description="Sample investigation for demonstration purposes",
        status=InvestigationStatus.ACTIVE,
        lead_investigator="Detective Smith",
        team_members=["Officer Johnson", "Analyst Brown"],
        case_priority="high",
        completion_percentage=65.0
    )
    
    # Sample evidence
    evidence_list = [
        Evidence(
            case_id=case_id,
            title="Digital Evidence - Smartphone",
            description="iPhone recovered from suspect",
            evidence_type="Digital Device",
            collected_by="Officer Johnson",
            authenticity_verified=True,
            analysis_completed=True,
            priority="high"
        ),
        Evidence(
            case_id=case_id,
            title="Document - Contract",
            description="Signed contract found at scene",
            evidence_type="Document",
            collected_by="Detective Smith",
            authenticity_verified=False,
            analysis_completed=False,
            priority="medium"
        )
    ]
    
    # Sample entities
    entities = [
        Entity(
            case_id=case_id,
            name="John Doe",
            entity_type=EntityType.PERSON,
            importance="high",
            subject_of_investigation=True
        ),
        Entity(
            case_id=case_id,
            name="ABC Corporation",
            entity_type=EntityType.ORGANIZATION,
            importance="medium",
            witness=True
        )
    ]
    
    # Sample events
    events = [
        Event(
            case_id=case_id,
            title="Initial Report Filed",
            description="Case officially opened",
            event_type=EventType.INCIDENT,
            timestamp=datetime.now() - timedelta(days=30),
            importance="critical"
        ),
        Event(
            case_id=case_id,
            title="Evidence Collection",
            description="Physical evidence collected from scene",
            event_type=EventType.EVIDENCE_COLLECTED,
            timestamp=datetime.now() - timedelta(days=25),
            importance="high"
        )
    ]
    
    # Sample relationships
    relationships = [
        Relationship(
            case_id=case_id,
            source_entity_id=entities[0].id,
            target_entity_id=entities[1].id,
            relationship_type="EMPLOYED_BY",
            strength=0.8,
            confidence=0.9
        )
    ]
    
    return investigation, evidence_list, entities, events, relationships


# Main Streamlit app entry point
if __name__ == "__main__":
    st.sidebar.header("Case Selection")
    case_id = st.sidebar.text_input("Enter Case ID", value="CASE-2024-001")
    
    if case_id:
        create_streamlit_case_dashboard(case_id)
    else:
        st.info("Please enter a case ID to view the dashboard")