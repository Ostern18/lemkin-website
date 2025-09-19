"""
Communication Pattern Detection Module

Advanced pattern detection and anomaly analysis for communication data.
Includes temporal pattern analysis, behavioral profiling, sentiment analysis,
and suspicious activity detection for forensic investigations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import json
import re
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from .core import (
    Communication, PatternAnalysis, PatternMatch, AnomalyIndicator,
    Anomaly, TemporalPattern, CommsConfig, PatternType, AnomalyType
)

logger = logging.getLogger(__name__)


@dataclass
class BehavioralProfile:
    """User behavioral profile"""
    user_id: str
    message_frequency: float  # messages per day
    active_hours: List[int]   # preferred hours of activity
    response_time_avg: float  # average response time in minutes
    message_length_avg: float # average message length
    platform_preferences: Dict[str, float]  # platform usage distribution
    social_pattern: str       # social interaction pattern
    anomaly_score: float      # behavioral anomaly score


@dataclass 
class CommunicationBurst:
    """Represents a burst of communication activity"""
    start_time: datetime
    end_time: datetime
    participants: List[str]
    message_count: int
    intensity: float  # messages per minute
    platforms: List[str]
    trigger_event: Optional[str] = None


class TemporalAnalyzer:
    """Analyzes temporal patterns in communications"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_temporal_patterns(self, communications: List[Communication]) -> List[TemporalPattern]:
        """Analyze comprehensive temporal communication patterns"""
        if not communications:
            return []
        
        # Convert to DataFrame for easier analysis
        df = self._create_temporal_dataframe(communications)
        
        patterns = []
        
        # Hourly patterns
        patterns.extend(self._analyze_hourly_patterns(df))
        
        # Daily patterns  
        patterns.extend(self._analyze_daily_patterns(df))
        
        # Weekly patterns
        patterns.extend(self._analyze_weekly_patterns(df))
        
        # Burst detection
        patterns.extend(self._detect_communication_bursts(df))
        
        # Seasonal patterns
        patterns.extend(self._analyze_seasonal_patterns(df))
        
        return patterns
    
    def _create_temporal_dataframe(self, communications: List[Communication]) -> pd.DataFrame:
        """Create DataFrame with temporal features"""
        data = []
        
        for comm in communications:
            data.append({
                'timestamp': comm.timestamp,
                'sender': comm.sender_id,
                'platform': comm.platform.value,
                'hour': comm.timestamp.hour,
                'day_of_week': comm.timestamp.weekday(),
                'day_of_month': comm.timestamp.day,
                'month': comm.timestamp.month,
                'is_weekend': comm.timestamp.weekday() >= 5,
                'message_length': len(comm.content or ''),
                'recipient_count': len(comm.recipient_ids),
                'has_attachments': len(comm.attachments) > 0
            })
        
        return pd.DataFrame(data)
    
    def _analyze_hourly_patterns(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze hourly communication patterns"""
        hourly_counts = df.groupby('hour').size()
        
        # Calculate statistics
        peak_hours = hourly_counts.nlargest(3).index.tolist()
        quiet_hours = hourly_counts.nsmallest(3).index.tolist()
        
        # Calculate regularity (inverse coefficient of variation)
        cv = hourly_counts.std() / hourly_counts.mean()
        regularity = max(0, 1 - cv)
        
        # Detect patterns
        pattern_type = PatternType.REGULAR if regularity > 0.7 else PatternType.TEMPORAL
        
        return [TemporalPattern(
            time_unit='hour',
            frequency_distribution={str(h): int(c) for h, c in hourly_counts.items()},
            peak_times=[f"{h:02d}:00" for h in peak_hours],
            quiet_periods=[f"{h:02d}:00" for h in quiet_hours],
            regularity_score=regularity,
            trend_direction=self._calculate_trend(hourly_counts.values)
        )]
    
    def _analyze_daily_patterns(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze daily communication patterns"""
        daily_counts = df.groupby('day_of_week').size()
        
        # Day names for readability
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        peak_days = [day_names[i] for i in daily_counts.nlargest(2).index]
        quiet_days = [day_names[i] for i in daily_counts.nsmallest(2).index]
        
        # Weekend vs weekday analysis
        weekend_avg = df[df['is_weekend']].shape[0] / 2  # 2 weekend days
        weekday_avg = df[~df['is_weekend']].shape[0] / 5  # 5 weekdays
        
        cv = daily_counts.std() / daily_counts.mean()
        regularity = max(0, 1 - cv)
        
        return [TemporalPattern(
            time_unit='day_of_week',
            frequency_distribution={day_names[i]: int(c) for i, c in daily_counts.items()},
            peak_times=peak_days,
            quiet_periods=quiet_days,
            regularity_score=regularity,
            seasonal_patterns={
                'weekend_vs_weekday_ratio': weekend_avg / max(1, weekday_avg)
            },
            trend_direction=self._calculate_trend(daily_counts.values)
        )]
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze weekly communication patterns"""
        # Group by week
        df_copy = df.copy()
        df_copy['week'] = df_copy['timestamp'].dt.isocalendar().week
        df_copy['year_week'] = df_copy['timestamp'].dt.strftime('%Y-W%U')
        
        weekly_counts = df_copy.groupby('year_week').size()
        
        if len(weekly_counts) < 2:
            return []
        
        # Find peak and quiet weeks
        peak_weeks = weekly_counts.nlargest(2).index.tolist()
        quiet_weeks = weekly_counts.nsmallest(2).index.tolist()
        
        cv = weekly_counts.std() / weekly_counts.mean()
        regularity = max(0, 1 - cv)
        
        return [TemporalPattern(
            time_unit='week',
            frequency_distribution={w: int(c) for w, c in weekly_counts.items()},
            peak_times=peak_weeks,
            quiet_periods=quiet_weeks,
            regularity_score=regularity,
            trend_direction=self._calculate_trend(weekly_counts.values)
        )]
    
    def _detect_communication_bursts(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Detect bursts of communication activity"""
        # Group by hour windows
        df_sorted = df.sort_values('timestamp')
        df_sorted['hour_window'] = df_sorted['timestamp'].dt.floor('H')
        
        hourly_counts = df_sorted.groupby('hour_window').size()
        
        # Use peak detection to find bursts
        counts = hourly_counts.values
        mean_activity = counts.mean()
        std_activity = counts.std()
        
        # Define burst threshold (mean + 2 * std)
        burst_threshold = mean_activity + 2 * std_activity
        
        # Find peaks above threshold
        peaks, _ = find_peaks(counts, height=burst_threshold, distance=3)
        
        burst_periods = []
        for peak_idx in peaks:
            peak_time = hourly_counts.index[peak_idx]
            intensity = counts[peak_idx]
            
            burst_periods.append(f"{peak_time.strftime('%Y-%m-%d %H:%M')} (intensity: {intensity})")
        
        if burst_periods:
            return [TemporalPattern(
                time_unit='burst',
                frequency_distribution={'bursts_detected': len(burst_periods)},
                peak_times=burst_periods,
                quiet_periods=[],
                regularity_score=0.0,  # Bursts are irregular by definition
                trend_direction='burst_activity'
            )]
        
        return []
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze seasonal patterns (monthly)"""
        if df['timestamp'].dt.year.nunique() < 2:
            # Need at least 1 year of data for seasonal analysis
            return []
        
        monthly_counts = df.groupby('month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        peak_months = [month_names[i-1] for i in monthly_counts.nlargest(3).index]
        quiet_months = [month_names[i-1] for i in monthly_counts.nsmallest(3).index]
        
        cv = monthly_counts.std() / monthly_counts.mean()
        regularity = max(0, 1 - cv)
        
        return [TemporalPattern(
            time_unit='month',
            frequency_distribution={month_names[i-1]: int(c) for i, c in monthly_counts.items()},
            peak_times=peak_months,
            quiet_periods=quiet_months,
            regularity_score=regularity,
            trend_direction=self._calculate_trend(monthly_counts.values)
        )]
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction from time series values"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Linear regression to determine trend
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        if abs(r_value) < 0.3:  # Weak correlation
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'


class AnomalyDetector:
    """Detects anomalies in communication patterns"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sensitivity = config.anomaly_sensitivity
    
    def detect_anomalies(self, communications: List[Communication]) -> List[Anomaly]:
        """Detect various types of communication anomalies"""
        if not communications:
            return []
        
        anomalies = []
        
        # Time-based anomalies
        anomalies.extend(self._detect_temporal_anomalies(communications))
        
        # Frequency anomalies
        anomalies.extend(self._detect_frequency_anomalies(communications))
        
        # Content anomalies
        anomalies.extend(self._detect_content_anomalies(communications))
        
        # Network anomalies
        anomalies.extend(self._detect_network_anomalies(communications))
        
        # Platform switching anomalies
        anomalies.extend(self._detect_platform_anomalies(communications))
        
        return anomalies
    
    def _detect_temporal_anomalies(self, communications: List[Communication]) -> List[Anomaly]:
        """Detect unusual timing patterns"""
        anomalies = []
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': c.timestamp,
            'sender': c.sender_id,
            'hour': c.timestamp.hour,
            'is_weekend': c.timestamp.weekday() >= 5
        } for c in communications])
        
        # Detect unusual hours
        unusual_hours = []
        for sender in df['sender'].unique():
            sender_df = df[df['sender'] == sender]
            
            # Check for activity in unusual hours (late night/early morning)
            night_activity = sender_df[(sender_df['hour'] >= 23) | (sender_df['hour'] <= 5)]
            
            if len(night_activity) > 0:
                ratio = len(night_activity) / len(sender_df)
                if ratio > 0.3:  # More than 30% of messages in unusual hours
                    unusual_hours.append({
                        'sender': sender,
                        'night_ratio': ratio,
                        'total_messages': len(sender_df)
                    })
        
        # Create anomaly indicators
        for unusual in unusual_hours:
            anomaly_indicator = AnomalyIndicator(
                anomaly_type=AnomalyType.UNUSUAL_TIME,
                description=f"High proportion of messages sent during unusual hours by {unusual['sender']}",
                severity='medium',
                confidence=min(0.9, unusual['night_ratio'] * 2),
                affected_contacts=[unusual['sender']],
                time_range=(df['timestamp'].min(), df['timestamp'].max()),
                actual_value=unusual['night_ratio'],
                baseline_value=0.1,  # Expected ratio
                deviation_score=unusual['night_ratio'] / 0.1
            )
            
            anomalies.append(Anomaly(
                anomaly=anomaly_indicator,
                context={'total_messages': unusual['total_messages']},
                related_messages=[]
            ))
        
        return anomalies
    
    def _detect_frequency_anomalies(self, communications: List[Communication]) -> List[Anomaly]:
        """Detect unusual communication frequency patterns"""
        anomalies = []
        
        # Group by sender and analyze frequency
        sender_stats = defaultdict(list)
        
        for comm in communications:
            sender_stats[comm.sender_id].append(comm.timestamp)
        
        for sender, timestamps in sender_stats.items():
            if len(timestamps) < 5:  # Need minimum messages for analysis
                continue
            
            timestamps.sort()
            
            # Calculate intervals between messages
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
                intervals.append(interval)
            
            if not intervals:
                continue
            
            # Detect bursts (many messages in short time)
            short_intervals = [i for i in intervals if i < 1]  # Less than 1 hour
            
            if len(short_intervals) > len(intervals) * 0.5:  # More than 50% burst activity
                anomaly_indicator = AnomalyIndicator(
                    anomaly_type=AnomalyType.BURST_ACTIVITY,
                    description=f"Burst communication pattern detected for {sender}",
                    severity='high',
                    confidence=0.8,
                    affected_contacts=[sender],
                    time_range=(timestamps[0], timestamps[-1]),
                    actual_value=len(short_intervals) / len(intervals),
                    baseline_value=0.2,  # Expected burst ratio
                    deviation_score=(len(short_intervals) / len(intervals)) / 0.2
                )
                
                anomalies.append(Anomaly(
                    anomaly=anomaly_indicator,
                    context={'total_messages': len(timestamps), 'avg_interval_hours': np.mean(intervals)},
                    related_messages=[]
                ))
            
            # Detect sudden silence
            recent_msgs = [t for t in timestamps if (datetime.now() - t).days < 7]
            if len(recent_msgs) == 0 and len(timestamps) > 20:  # Was active but now silent
                anomaly_indicator = AnomalyIndicator(
                    anomaly_type=AnomalyType.SUDDEN_SILENCE,
                    description=f"Sudden communication silence detected for {sender}",
                    severity='medium',
                    confidence=0.7,
                    affected_contacts=[sender],
                    time_range=(timestamps[-1], datetime.now()),
                    actual_value=0,
                    baseline_value=len(timestamps) / max(1, (timestamps[-1] - timestamps[0]).days),
                    deviation_score=1.0
                )
                
                anomalies.append(Anomaly(
                    anomaly=anomaly_indicator,
                    context={'last_message': timestamps[-1].isoformat()},
                    related_messages=[]
                ))
        
        return anomalies
    
    def _detect_content_anomalies(self, communications: List[Communication]) -> List[Anomaly]:
        """Detect unusual content patterns"""
        anomalies = []
        
        # Analyze message lengths
        sender_lengths = defaultdict(list)
        
        for comm in communications:
            if comm.content:
                sender_lengths[comm.sender_id].append(len(comm.content))
        
        for sender, lengths in sender_lengths.items():
            if len(lengths) < 10:  # Need sufficient data
                continue
            
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            # Find unusually long or short messages
            outliers = [l for l in lengths if abs(l - mean_length) > 3 * std_length]
            
            if len(outliers) > len(lengths) * 0.1:  # More than 10% outliers
                anomaly_indicator = AnomalyIndicator(
                    anomaly_type=AnomalyType.UNUSUAL_CONTENT,
                    description=f"Unusual message length patterns for {sender}",
                    severity='low',
                    confidence=0.6,
                    affected_contacts=[sender],
                    time_range=(communications[0].timestamp, communications[-1].timestamp),
                    actual_value=len(outliers) / len(lengths),
                    baseline_value=0.05,  # Expected outlier ratio
                    deviation_score=(len(outliers) / len(lengths)) / 0.05
                )
                
                anomalies.append(Anomaly(
                    anomaly=anomaly_indicator,
                    context={'mean_length': mean_length, 'std_length': std_length},
                    related_messages=[]
                ))
        
        return anomalies
    
    def _detect_network_anomalies(self, communications: List[Communication]) -> List[Anomaly]:
        """Detect network-based anomalies"""
        anomalies = []
        
        # Track new contacts appearing
        contact_first_seen = {}
        
        for comm in communications:
            # Track when each contact first appears
            if comm.sender_id not in contact_first_seen:
                contact_first_seen[comm.sender_id] = comm.timestamp
            
            for recipient in comm.recipient_ids:
                if recipient not in contact_first_seen:
                    contact_first_seen[recipient] = comm.timestamp
        
        # Find contacts that appeared recently with high activity
        recent_threshold = datetime.now() - timedelta(days=7)
        new_active_contacts = []
        
        for contact, first_seen in contact_first_seen.items():
            if first_seen > recent_threshold:
                # Count activity for this new contact
                contact_activity = len([
                    c for c in communications 
                    if c.sender_id == contact or contact in c.recipient_ids
                ])
                
                if contact_activity > 10:  # High activity for new contact
                    new_active_contacts.append({
                        'contact': contact,
                        'first_seen': first_seen,
                        'activity': contact_activity
                    })
        
        for new_contact in new_active_contacts:
            anomaly_indicator = AnomalyIndicator(
                anomaly_type=AnomalyType.NEW_CONTACT,
                description=f"New contact with high activity: {new_contact['contact']}",
                severity='medium',
                confidence=0.7,
                affected_contacts=[new_contact['contact']],
                time_range=(new_contact['first_seen'], datetime.now()),
                actual_value=new_contact['activity'],
                baseline_value=5,  # Expected activity for new contacts
                deviation_score=new_contact['activity'] / 5
            )
            
            anomalies.append(Anomaly(
                anomaly=anomaly_indicator,
                context={'first_seen': new_contact['first_seen'].isoformat()},
                related_messages=[]
            ))
        
        return anomalies
    
    def _detect_platform_anomalies(self, communications: List[Communication]) -> List[Anomaly]:
        """Detect unusual platform switching patterns"""
        anomalies = []
        
        # Track platform usage by sender
        sender_platforms = defaultdict(Counter)
        
        for comm in communications:
            sender_platforms[comm.sender_id][comm.platform.value] += 1
        
        # Detect sudden platform switches
        for sender, platforms in sender_platforms.items():
            total_messages = sum(platforms.values())
            
            if len(platforms) > 1 and total_messages > 20:
                # Calculate platform diversity
                entropy = -sum((count/total_messages) * np.log2(count/total_messages) 
                             for count in platforms.values() if count > 0)
                
                if entropy > 1.5:  # High platform switching
                    anomaly_indicator = AnomalyIndicator(
                        anomaly_type=AnomalyType.PLATFORM_SWITCH,
                        description=f"Frequent platform switching detected for {sender}",
                        severity='low',
                        confidence=0.6,
                        affected_contacts=[sender],
                        time_range=(communications[0].timestamp, communications[-1].timestamp),
                        actual_value=entropy,
                        baseline_value=0.5,  # Expected platform entropy
                        deviation_score=entropy / 0.5
                    )
                    
                    anomalies.append(Anomaly(
                        anomaly=anomaly_indicator,
                        context={'platforms': dict(platforms), 'entropy': entropy},
                        related_messages=[]
                    ))
        
        return anomalies


class SentimentAnalyzer:
    """Analyzes sentiment and emotional patterns in communications"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Simple sentiment keywords (in practice, would use more sophisticated models)
        self.positive_words = set([
            'good', 'great', 'excellent', 'happy', 'pleased', 'satisfied',
            'wonderful', 'amazing', 'fantastic', 'love', 'like', 'enjoy'
        ])
        
        self.negative_words = set([
            'bad', 'terrible', 'awful', 'hate', 'angry', 'upset', 'disappointed',
            'frustrated', 'annoyed', 'disgusted', 'worried', 'concerned', 'problem'
        ])
        
        self.urgent_words = set([
            'urgent', 'emergency', 'asap', 'immediately', 'critical', 'important',
            'rush', 'quickly', 'now', 'deadline'
        ])
    
    def analyze_sentiment_patterns(self, communications: List[Communication]) -> Dict[str, Any]:
        """Analyze sentiment patterns across communications"""
        sentiment_data = []
        
        for comm in communications:
            if not comm.content:
                continue
            
            content = comm.content.lower()
            words = re.findall(r'\b\w+\b', content)
            
            # Simple sentiment scoring
            positive_score = sum(1 for word in words if word in self.positive_words)
            negative_score = sum(1 for word in words if word in self.negative_words)
            urgent_score = sum(1 for word in words if word in self.urgent_words)
            
            total_words = len(words)
            if total_words == 0:
                continue
            
            sentiment_data.append({
                'message_id': comm.message_id,
                'sender': comm.sender_id,
                'timestamp': comm.timestamp,
                'positive_ratio': positive_score / total_words,
                'negative_ratio': negative_score / total_words,
                'urgent_ratio': urgent_score / total_words,
                'sentiment_score': (positive_score - negative_score) / max(1, total_words),
                'total_words': total_words
            })
        
        if not sentiment_data:
            return {}
        
        df = pd.DataFrame(sentiment_data)
        
        # Analyze patterns
        analysis = {
            'overall_sentiment': df['sentiment_score'].mean(),
            'sentiment_std': df['sentiment_score'].std(),
            'urgency_level': df['urgent_ratio'].mean(),
            'sentiment_by_sender': df.groupby('sender')['sentiment_score'].mean().to_dict(),
            'urgency_by_sender': df.groupby('sender')['urgent_ratio'].mean().to_dict(),
            'temporal_sentiment': self._analyze_temporal_sentiment(df)
        }
        
        return analysis
    
    def _analyze_temporal_sentiment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how sentiment changes over time"""
        # Group by day and calculate daily sentiment
        df['date'] = df['timestamp'].dt.date
        daily_sentiment = df.groupby('date')['sentiment_score'].mean()
        
        # Calculate trend
        if len(daily_sentiment) > 1:
            x = np.arange(len(daily_sentiment))
            slope, _, r_value, _, _ = stats.linregress(x, daily_sentiment.values)
            
            trend = 'improving' if slope > 0.01 else ('declining' if slope < -0.01 else 'stable')
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'daily_sentiment': {str(k): float(v) for k, v in daily_sentiment.items()},
            'sentiment_volatility': daily_sentiment.std() if len(daily_sentiment) > 1 else 0
        }


class BehavioralProfiler:
    """Creates behavioral profiles for communication participants"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_behavioral_profiles(self, communications: List[Communication]) -> Dict[str, BehavioralProfile]:
        """Create behavioral profiles for all participants"""
        profiles = {}
        
        # Group communications by sender
        sender_comms = defaultdict(list)
        for comm in communications:
            sender_comms[comm.sender_id].append(comm)
            # Also track as recipient
            for recipient in comm.recipient_ids:
                if recipient not in sender_comms:
                    sender_comms[recipient] = []
        
        for sender, comms in sender_comms.items():
            if len(comms) < 3:  # Need minimum data for profiling
                continue
            
            profile = self._create_individual_profile(sender, comms, communications)
            profiles[sender] = profile
        
        return profiles
    
    def _create_individual_profile(
        self, 
        user_id: str, 
        user_comms: List[Communication],
        all_comms: List[Communication]
    ) -> BehavioralProfile:
        """Create behavioral profile for individual user"""
        
        if not user_comms:
            return BehavioralProfile(
                user_id=user_id,
                message_frequency=0,
                active_hours=[],
                response_time_avg=0,
                message_length_avg=0,
                platform_preferences={},
                social_pattern='inactive',
                anomaly_score=0
            )
        
        # Calculate message frequency (messages per day)
        time_span = (max(c.timestamp for c in user_comms) - 
                    min(c.timestamp for c in user_comms)).days
        frequency = len(user_comms) / max(1, time_span)
        
        # Determine active hours
        hours = [c.timestamp.hour for c in user_comms]
        hour_counts = Counter(hours)
        active_hours = [h for h, count in hour_counts.most_common(5)]
        
        # Calculate average message length
        lengths = [len(c.content or '') for c in user_comms]
        avg_length = np.mean(lengths) if lengths else 0
        
        # Platform preferences
        platforms = [c.platform.value for c in user_comms]
        platform_counts = Counter(platforms)
        total_msgs = sum(platform_counts.values())
        platform_prefs = {p: c/total_msgs for p, c in platform_counts.items()}
        
        # Response time analysis
        response_times = self._calculate_response_times(user_id, all_comms)
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Social pattern classification
        social_pattern = self._classify_social_pattern(user_id, all_comms)
        
        # Anomaly score
        anomaly_score = self._calculate_behavioral_anomaly_score(
            user_id, user_comms, frequency, avg_response_time, platform_prefs
        )
        
        return BehavioralProfile(
            user_id=user_id,
            message_frequency=frequency,
            active_hours=active_hours,
            response_time_avg=avg_response_time,
            message_length_avg=avg_length,
            platform_preferences=platform_prefs,
            social_pattern=social_pattern,
            anomaly_score=anomaly_score
        )
    
    def _calculate_response_times(self, user_id: str, all_comms: List[Communication]) -> List[float]:
        """Calculate response times for user"""
        response_times = []
        
        # Sort all communications by timestamp
        sorted_comms = sorted(all_comms, key=lambda c: c.timestamp)
        
        for i, comm in enumerate(sorted_comms):
            if comm.sender_id != user_id:
                continue
            
            # Look for previous message this could be responding to
            for j in range(i-1, max(-1, i-10), -1):  # Look back up to 10 messages
                prev_comm = sorted_comms[j]
                
                # Check if this could be a response
                if (user_id in prev_comm.recipient_ids and
                    (comm.timestamp - prev_comm.timestamp).total_seconds() < 3600):  # Within 1 hour
                    
                    response_time = (comm.timestamp - prev_comm.timestamp).total_seconds() / 60  # minutes
                    response_times.append(response_time)
                    break
        
        return response_times
    
    def _classify_social_pattern(self, user_id: str, all_comms: List[Communication]) -> str:
        """Classify user's social interaction pattern"""
        sent_count = len([c for c in all_comms if c.sender_id == user_id])
        received_count = len([c for c in all_comms if user_id in c.recipient_ids])
        
        # Calculate unique contacts
        contacts = set()
        for comm in all_comms:
            if comm.sender_id == user_id:
                contacts.update(comm.recipient_ids)
            elif user_id in comm.recipient_ids:
                contacts.add(comm.sender_id)
        
        total_activity = sent_count + received_count
        
        if total_activity < 5:
            return 'inactive'
        elif sent_count > received_count * 2:
            return 'broadcaster'  # Sends much more than receives
        elif received_count > sent_count * 2:
            return 'listener'     # Receives much more than sends
        elif len(contacts) > 20:
            return 'connector'    # Communicates with many people
        elif len(contacts) < 3:
            return 'isolated'     # Few communication partners
        else:
            return 'balanced'     # Balanced communication pattern
    
    def _calculate_behavioral_anomaly_score(
        self,
        user_id: str,
        user_comms: List[Communication], 
        frequency: float,
        avg_response_time: float,
        platform_prefs: Dict[str, float]
    ) -> float:
        """Calculate behavioral anomaly score"""
        anomaly_factors = []
        
        # Frequency anomaly
        if frequency > 50:  # Very high frequency
            anomaly_factors.append(min(1.0, frequency / 100))
        elif frequency < 0.1:  # Very low frequency
            anomaly_factors.append(0.3)
        
        # Response time anomaly
        if avg_response_time < 1:  # Very fast responses (< 1 minute)
            anomaly_factors.append(0.4)
        elif avg_response_time > 1440:  # Very slow responses (> 1 day)
            anomaly_factors.append(0.3)
        
        # Platform switching anomaly
        if len(platform_prefs) > 3:  # Uses many platforms
            anomaly_factors.append(0.3)
        
        # Timing anomalies (night activity)
        night_msgs = [c for c in user_comms 
                     if c.timestamp.hour >= 23 or c.timestamp.hour <= 5]
        night_ratio = len(night_msgs) / len(user_comms)
        if night_ratio > 0.4:  # High night activity
            anomaly_factors.append(night_ratio)
        
        return min(1.0, np.mean(anomaly_factors)) if anomaly_factors else 0.0


class PatternDetector:
    """Main pattern detection class that orchestrates all pattern analysis"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.temporal_analyzer = TemporalAnalyzer(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.behavioral_profiler = BehavioralProfiler(config)
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, communications: List[Communication]) -> PatternAnalysis:
        """Perform comprehensive pattern analysis"""
        
        if not communications:
            return PatternAnalysis(
                detected_patterns=[],
                anomalies=[],
                temporal_analysis={},
                behavioral_profiles={},
                risk_assessment={},
                recommendations=[],
                confidence_summary={}
            )
        
        self.logger.info("Starting comprehensive pattern analysis...")
        
        # Temporal pattern analysis
        self.logger.info("Analyzing temporal patterns...")
        temporal_patterns = self.temporal_analyzer.analyze_temporal_patterns(communications)
        
        # Convert temporal patterns to pattern matches
        detected_patterns = []
        for temp_pattern in temporal_patterns:
            pattern_match = PatternMatch(
                pattern_type=PatternType.TEMPORAL,
                description=f"Temporal pattern in {temp_pattern.time_unit}",
                confidence=temp_pattern.regularity_score,
                participants=list(set(c.sender_id for c in communications)),
                time_range=(
                    min(c.timestamp for c in communications),
                    max(c.timestamp for c in communications)
                ),
                platforms=list(set(c.platform for c in communications)),
                evidence=[c.message_id for c in communications[:10]],  # Sample evidence
                significance='high' if temp_pattern.regularity_score > 0.7 else 'medium'
            )
            detected_patterns.append(pattern_match)
        
        # Anomaly detection
        self.logger.info("Detecting anomalies...")
        anomalies = self.anomaly_detector.detect_anomalies(communications)
        
        # Sentiment analysis
        self.logger.info("Analyzing sentiment patterns...")
        sentiment_analysis = self.sentiment_analyzer.analyze_sentiment_patterns(communications)
        
        # Behavioral profiling
        self.logger.info("Creating behavioral profiles...")
        behavioral_profiles = self.behavioral_profiler.create_behavioral_profiles(communications)
        
        # Risk assessment
        self.logger.info("Performing risk assessment...")
        risk_assessment = self._perform_risk_assessment(anomalies, behavioral_profiles, communications)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies, risk_assessment)
        
        # Calculate confidence metrics
        confidence_summary = self._calculate_confidence_summary(
            detected_patterns, anomalies, behavioral_profiles
        )
        
        self.logger.info("Pattern analysis completed")
        
        return PatternAnalysis(
            detected_patterns=detected_patterns,
            anomalies=anomalies,
            temporal_analysis={
                'patterns': temporal_patterns,
                'sentiment_analysis': sentiment_analysis
            },
            behavioral_profiles={p.user_id: p for p in behavioral_profiles.values()},
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            confidence_summary=confidence_summary
        )
    
    def _perform_risk_assessment(
        self, 
        anomalies: List[Anomaly], 
        profiles: Dict[str, BehavioralProfile],
        communications: List[Communication]
    ) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        
        # Risk scoring
        high_risk_contacts = []
        medium_risk_contacts = []
        
        for user_id, profile in profiles.items():
            risk_score = 0.0
            risk_factors = []
            
            # Behavioral anomaly score
            risk_score += profile.anomaly_score * 0.4
            if profile.anomaly_score > 0.5:
                risk_factors.append("unusual_behavior")
            
            # Social pattern risks
            if profile.social_pattern in ['broadcaster', 'isolated']:
                risk_score += 0.2
                risk_factors.append(f"social_pattern_{profile.social_pattern}")
            
            # Activity level risks
            if profile.message_frequency > 20:  # Very high frequency
                risk_score += 0.2
                risk_factors.append("high_activity")
            
            # Response time risks
            if profile.response_time_avg < 2:  # Very fast responses
                risk_score += 0.1
                risk_factors.append("fast_responses")
            
            # Platform diversity risks
            if len(profile.platform_preferences) > 3:
                risk_score += 0.1
                risk_factors.append("platform_switching")
            
            if risk_score > 0.7:
                high_risk_contacts.append({
                    'contact': user_id,
                    'risk_score': risk_score,
                    'risk_factors': risk_factors
                })
            elif risk_score > 0.4:
                medium_risk_contacts.append({
                    'contact': user_id,
                    'risk_score': risk_score,
                    'risk_factors': risk_factors
                })
        
        # Anomaly severity assessment
        critical_anomalies = [a for a in anomalies if a.anomaly.severity == 'high']
        moderate_anomalies = [a for a in anomalies if a.anomaly.severity == 'medium']
        
        # Overall risk level
        overall_risk = 'low'
        if len(critical_anomalies) > 2 or len(high_risk_contacts) > 0:
            overall_risk = 'high'
        elif len(moderate_anomalies) > 3 or len(medium_risk_contacts) > 2:
            overall_risk = 'medium'
        
        return {
            'overall_risk_level': overall_risk,
            'high_risk_contacts': high_risk_contacts,
            'medium_risk_contacts': medium_risk_contacts,
            'critical_anomalies': len(critical_anomalies),
            'moderate_anomalies': len(moderate_anomalies),
            'risk_factors_summary': {
                'behavioral_anomalies': sum(1 for p in profiles.values() if p.anomaly_score > 0.5),
                'temporal_anomalies': len([a for a in anomalies if a.anomaly.anomaly_type in [AnomalyType.UNUSUAL_TIME, AnomalyType.BURST_ACTIVITY]]),
                'network_anomalies': len([a for a in anomalies if a.anomaly.anomaly_type == AnomalyType.NEW_CONTACT]),
                'platform_anomalies': len([a for a in anomalies if a.anomaly.anomaly_type == AnomalyType.PLATFORM_SWITCH])
            }
        }
    
    def _generate_recommendations(
        self, 
        anomalies: List[Anomaly], 
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate investigation recommendations based on analysis"""
        recommendations = []
        
        # High-level recommendations
        if risk_assessment['overall_risk_level'] == 'high':
            recommendations.append("PRIORITY: Immediate investigation recommended due to high-risk patterns")
            recommendations.append("Deploy additional monitoring for high-risk contacts")
            recommendations.append("Cross-reference with external intelligence sources")
        
        # Specific anomaly recommendations
        anomaly_types = [a.anomaly.anomaly_type for a in anomalies]
        
        if AnomalyType.BURST_ACTIVITY in anomaly_types:
            recommendations.append("Investigate communication bursts for coordinated activity")
        
        if AnomalyType.NEW_CONTACT in anomaly_types:
            recommendations.append("Verify identity and legitimacy of new high-activity contacts")
        
        if AnomalyType.PLATFORM_SWITCH in anomaly_types:
            recommendations.append("Monitor for potential operational security practices")
        
        if AnomalyType.UNUSUAL_TIME in anomaly_types:
            recommendations.append("Correlate unusual timing with external events")
        
        # Risk-based recommendations
        if len(risk_assessment['high_risk_contacts']) > 0:
            recommendations.append("Prioritize surveillance of identified high-risk contacts")
        
        if risk_assessment['risk_factors_summary']['temporal_anomalies'] > 2:
            recommendations.append("Analyze temporal patterns for operational planning indicators")
        
        if risk_assessment['risk_factors_summary']['network_anomalies'] > 1:
            recommendations.append("Map extended network connections for new contacts")
        
        # General recommendations
        recommendations.append("Regular monitoring and pattern updates recommended")
        recommendations.append("Consider correlation with other intelligence sources")
        
        return recommendations
    
    def _calculate_confidence_summary(
        self,
        patterns: List[PatternMatch],
        anomalies: List[Anomaly],
        profiles: Dict[str, BehavioralProfile]
    ) -> Dict[str, float]:
        """Calculate confidence metrics for analysis results"""
        
        if not patterns and not anomalies:
            return {'overall_confidence': 0.0}
        
        pattern_confidences = [p.confidence for p in patterns]
        anomaly_confidences = [a.anomaly.confidence for a in anomalies]
        
        confidence_metrics = {
            'pattern_detection_confidence': np.mean(pattern_confidences) if pattern_confidences else 0.0,
            'anomaly_detection_confidence': np.mean(anomaly_confidences) if anomaly_confidences else 0.0,
            'behavioral_profiling_confidence': min(1.0, len(profiles) / 5.0),  # Higher confidence with more profiles
            'data_completeness': 1.0,  # Assume complete for now
        }
        
        # Overall confidence is weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        overall_confidence = sum(w * v for w, v in zip(weights, confidence_metrics.values()))
        
        return {
            **confidence_metrics,
            'overall_confidence': overall_confidence
        }


# Main function for CLI
def detect_communication_patterns(
    communications: List[Communication], 
    config: CommsConfig = None
) -> PatternAnalysis:
    """Detect communication patterns and return analysis results"""
    if config is None:
        config = CommsConfig()
    
    detector = PatternDetector(config)
    return detector.analyze(communications)