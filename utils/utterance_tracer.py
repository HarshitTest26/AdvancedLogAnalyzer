"""
Utterance Tracer Module for Automotive Log Analysis Platform

This module provides functionality to trace voice assistant utterance flows
from when a voice command is made until response or failure.
"""

import regex as re
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Patterns for identifying utterance-related logs
# These patterns match common voice assistant log formats
UTTERANCE_PATTERNS = {
    'utterance_start': [
        r'(?i)utterance\s*(?:started|begin|received)',
        r'(?i)voice\s*(?:command|input|request)\s*(?:received|started)',
        r'(?i)speech\s*(?:recognition|input)\s*(?:started|begin)',
        r'(?i)listening\s*(?:started|active)',
        r'(?i)audio\s*(?:capture|recording)\s*(?:started|begin)',
    ],
    'utterance_processing': [
        r'(?i)processing\s*(?:utterance|command|speech)',
        r'(?i)(?:nlu|asr)\s*(?:processing|analyzing)',
        r'(?i)intent\s*(?:recognition|detection|classification)',
        r'(?i)entity\s*(?:extraction|recognition)',
        r'(?i)command\s*(?:parsing|interpretation)',
    ],
    'utterance_response': [
        r'(?i)(?:response|reply)\s*(?:generated|sent|completed)',
        r'(?i)tts\s*(?:started|playing|completed)',
        r'(?i)audio\s*(?:playback|output)\s*(?:started|completed)',
        r'(?i)utterance\s*(?:completed|finished|ended)',
        r'(?i)command\s*(?:executed|completed|finished)',
    ],
    'utterance_error': [
        r'(?i)utterance\s*(?:failed|error|timeout)',
        r'(?i)speech\s*(?:recognition|processing)\s*(?:failed|error)',
        r'(?i)command\s*(?:failed|rejected|invalid)',
        r'(?i)(?:nlu|asr)\s*(?:failed|error|timeout)',
        r'(?i)timeout\s*(?:waiting|processing)\s*(?:utterance|command)',
    ],
}

# Patterns for extracting session/request IDs
ID_PATTERNS = [
    r'(?:session|request|utterance)[_\s]*(?:id|ID)[:\s]*([a-zA-Z0-9\-_]+)',
    r'(?:reqId|sessionId|uttId)[:\s]*([a-zA-Z0-9\-_]+)',
    r'\[([a-fA-F0-9\-]{8,})\]',  # UUID-like patterns in brackets
    r'ID[:\s=]*([a-zA-Z0-9\-_]+)',
    r'(?:for|of)\s+(voice_\w+)',  # Match "for voice_001" or "of voice_002"
    r'(?:for|of)\s+(req_\w+)',    # Match "for req_001" or "of req_002"
]

# Components typically involved in voice assistant processing
VOICE_COMPONENTS = [
    'SpeechRecognizer', 'VoiceAssistant', 'AudioManager', 'TC', 'AHE-NLU',
    'NLU', 'ASR', 'TTS', 'DialogManager', 'IntentRecognizer', 'CommandProcessor',
    'VoiceInput', 'VoiceOutput', 'SpeechService', 'VoiceService'
]


class UtteranceTracer:
    """
    Traces voice assistant utterance flows through log entries
    """
    
    def __init__(self):
        """Initialize the utterance tracer"""
        self.compiled_patterns = self._compile_patterns()
        self.compiled_id_patterns = [re.compile(pattern) for pattern in ID_PATTERNS]
        logger.info("UtteranceTracer initialized")
    
    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for better performance"""
        compiled = {}
        for category, patterns in UTTERANCE_PATTERNS.items():
            compiled[category] = [re.compile(pattern) for pattern in patterns]
        return compiled
    
    def extract_session_id(self, message: str) -> Optional[str]:
        """
        Extract session/request ID from a log message
        
        Args:
            message (str): Log message text
            
        Returns:
            Optional[str]: Extracted ID or None
        """
        for pattern in self.compiled_id_patterns:
            match = pattern.search(message)
            if match:
                return match.group(1)
        return None
    
    def classify_utterance_event(self, message: str, component: str) -> Optional[str]:
        """
        Classify a log message as an utterance event type
        
        Args:
            message (str): Log message text
            component (str): Component name
            
        Returns:
            Optional[str]: Event type ('start', 'processing', 'response', 'error') or None
        """
        # Check if component is voice-related
        is_voice_component = any(vc.lower() in component.lower() for vc in VOICE_COMPONENTS)
        
        # Check message patterns
        for event_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    # Map event types to simpler categories
                    if 'start' in event_type:
                        return 'start'
                    elif 'processing' in event_type:
                        return 'processing'
                    elif 'response' in event_type:
                        return 'response'
                    elif 'error' in event_type:
                        return 'error'
        
        # If no pattern matched but it's a voice component with error/warning
        if is_voice_component:
            return 'voice_related'
        
        return None
    
    def is_utterance_related(self, entry: Dict) -> bool:
        """
        Check if a log entry is related to utterance processing
        
        Args:
            entry (dict): Log entry dictionary
            
        Returns:
            bool: True if utterance-related
        """
        message = entry.get('message', '')
        component = entry.get('component', '')
        
        # Check if it has utterance classification
        event_type = self.classify_utterance_event(message, component)
        return event_type is not None
    
    def enrich_with_utterance_data(self, issue_entries: List[Dict]) -> List[Dict]:
        """
        Enrich log entries with utterance metadata
        
        Args:
            issue_entries (list): List of log entry dictionaries
            
        Returns:
            list: Enriched log entries
        """
        enriched_entries = []
        
        for entry in issue_entries:
            entry_copy = entry.copy()
            
            # Extract session ID if present
            session_id = self.extract_session_id(entry['message'])
            if session_id:
                entry_copy['utterance_session_id'] = session_id
            
            # Classify utterance event
            event_type = self.classify_utterance_event(entry['message'], entry['component'])
            if event_type:
                entry_copy['utterance_event_type'] = event_type
                entry_copy['is_utterance_related'] = True
            else:
                entry_copy['is_utterance_related'] = False
            
            enriched_entries.append(entry_copy)
        
        logger.info(f"Enriched {len(enriched_entries)} entries with utterance metadata")
        return enriched_entries
    
    def trace_utterance_flows(self, issue_entries: List[Dict]) -> List[Dict]:
        """
        Group log entries by utterance session to trace complete flows
        
        Args:
            issue_entries (list): List of log entry dictionaries (should be enriched first)
            
        Returns:
            list: List of utterance flow dictionaries
        """
        # Group by session ID
        sessions = defaultdict(list)
        orphan_utterances = []
        
        for entry in issue_entries:
            if not entry.get('is_utterance_related', False):
                continue
            
            session_id = entry.get('utterance_session_id')
            if session_id:
                sessions[session_id].append(entry)
            else:
                # Utterance-related but no session ID
                orphan_utterances.append(entry)
        
        # Create flow objects
        flows = []
        
        for session_id, entries in sessions.items():
            # Sort by timestamp
            sorted_entries = sorted(entries, key=lambda x: x.get('timestamp', ''))
            
            if not sorted_entries:
                continue
            
            # Analyze the flow
            flow = self._analyze_utterance_flow(session_id, sorted_entries)
            flows.append(flow)
        
        # Try to group orphan utterances by temporal proximity
        if orphan_utterances:
            orphan_flows = self._group_orphan_utterances(orphan_utterances)
            flows.extend(orphan_flows)
        
        logger.info(f"Traced {len(flows)} utterance flows")
        return flows
    
    def _analyze_utterance_flow(self, session_id: str, entries: List[Dict]) -> Dict:
        """
        Analyze a complete utterance flow
        
        Args:
            session_id (str): Session/request ID
            entries (list): List of log entries for this session
            
        Returns:
            dict: Flow analysis
        """
        if not entries:
            return {}
        
        # Extract flow characteristics
        start_time = entries[0].get('timestamp')
        end_time = entries[-1].get('timestamp')
        
        # Calculate duration if timestamps are parseable
        duration = None
        try:
            if start_time and end_time:
                # Try to parse with different formats (log format is MM-DD HH:MM:SS.mmm)
                # Add current year to make it parseable
                from datetime import datetime as dt
                current_year = dt.now().year
                
                # Try to parse with year prepended
                try:
                    start_str = f"{current_year}-{start_time}"
                    end_str = f"{current_year}-{end_time}"
                    start_dt = pd.to_datetime(start_str, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                    end_dt = pd.to_datetime(end_str, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                except:
                    # Fallback to default parsing
                    start_dt = pd.to_datetime(start_time, errors='coerce')
                    end_dt = pd.to_datetime(end_time, errors='coerce')
                
                if pd.notna(start_dt) and pd.notna(end_dt):
                    duration = (end_dt - start_dt).total_seconds()
        except Exception as e:
            logger.debug(f"Could not calculate duration: {e}")
        
        # Count event types
        event_types = [e.get('utterance_event_type') for e in entries]
        has_start = 'start' in event_types
        has_response = 'response' in event_types
        has_error = 'error' in event_types or any(e.get('level') in ['E', 'F'] for e in entries)
        
        # Determine status
        if has_error:
            status = 'failed'
        elif has_response:
            status = 'completed'
        elif has_start:
            status = 'incomplete'
        else:
            status = 'unknown'
        
        # Get involved components
        components = list(set(e.get('component') for e in entries))
        
        # Create timeline
        timeline = []
        for entry in entries:
            timeline.append({
                'timestamp': entry.get('timestamp'),
                'event_type': entry.get('utterance_event_type'),
                'component': entry.get('component'),
                'level': entry.get('level'),
                'message': entry.get('message')
            })
        
        return {
            'session_id': session_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': duration,
            'status': status,
            'has_errors': has_error,
            'entry_count': len(entries),
            'components': components,
            'timeline': timeline,
            'entries': entries
        }
    
    def _group_orphan_utterances(self, orphan_entries: List[Dict]) -> List[Dict]:
        """
        Group utterance entries that don't have session IDs by temporal proximity
        
        Args:
            orphan_entries (list): List of utterance-related entries without session IDs
            
        Returns:
            list: List of inferred flows
        """
        # Sort by timestamp
        sorted_entries = sorted(orphan_entries, key=lambda x: x.get('timestamp', ''))
        
        flows = []
        current_flow = []
        last_timestamp = None
        time_threshold = timedelta(seconds=30)  # Group entries within 30 seconds
        
        for entry in sorted_entries:
            try:
                timestamp = pd.to_datetime(entry.get('timestamp'), errors='coerce')
                
                if pd.isna(timestamp):
                    continue
                
                # Start new flow if time gap is too large or this is the first entry
                if last_timestamp is None or (timestamp - last_timestamp) > time_threshold:
                    if current_flow:
                        # Save previous flow
                        flow_id = f"inferred_{len(flows)+1}"
                        flow = self._analyze_utterance_flow(flow_id, current_flow)
                        flow['is_inferred'] = True
                        flows.append(flow)
                    
                    # Start new flow
                    current_flow = [entry]
                else:
                    # Add to current flow
                    current_flow.append(entry)
                
                last_timestamp = timestamp
                
            except Exception as e:
                logger.debug(f"Error processing orphan entry: {e}")
                continue
        
        # Don't forget the last flow
        if current_flow:
            flow_id = f"inferred_{len(flows)+1}"
            flow = self._analyze_utterance_flow(flow_id, current_flow)
            flow['is_inferred'] = True
            flows.append(flow)
        
        return flows
    
    def create_utterance_timeline(self, flow: Dict) -> str:
        """
        Create a visual text timeline for an utterance flow
        
        Args:
            flow (dict): Utterance flow dictionary
            
        Returns:
            str: Text representation of timeline
        """
        if not flow or 'timeline' not in flow:
            return "No timeline data available"
        
        lines = []
        lines.append(f"Utterance Flow: {flow.get('session_id', 'Unknown')}")
        lines.append(f"Status: {flow.get('status', 'unknown').upper()}")
        lines.append(f"Duration: {flow.get('duration_seconds', 'N/A')}s")
        lines.append(f"Components: {', '.join(flow.get('components', []))}")
        lines.append("")
        lines.append("Timeline:")
        lines.append("-" * 80)
        
        for i, event in enumerate(flow['timeline']):
            # Create visual indicator based on event type
            event_type = event.get('event_type', 'unknown')
            level = event.get('level', 'I')
            
            # Choose icon
            if event_type == 'start':
                icon = '🎤'
            elif event_type == 'processing':
                icon = '⚙️'
            elif event_type == 'response':
                icon = '✅'
            elif event_type == 'error':
                icon = '❌'
            elif level in ['E', 'F']:
                icon = '⚠️'
            else:
                icon = '•'
            
            timestamp = event.get('timestamp', 'N/A')
            component = event.get('component', 'Unknown')
            message = event.get('message', '')[:100]  # Truncate long messages
            
            lines.append(f"{icon} [{timestamp}] {component}")
            lines.append(f"   {message}")
            
            if i < len(flow['timeline']) - 1:
                lines.append("   |")
        
        lines.append("-" * 80)
        
        return "\n".join(lines)
    
    def analyze_utterance_patterns(self, flows: List[Dict]) -> Dict:
        """
        Analyze patterns across multiple utterance flows
        
        Args:
            flows (list): List of utterance flow dictionaries
            
        Returns:
            dict: Pattern analysis results
        """
        if not flows:
            return {
                'total_flows': 0,
                'completed': 0,
                'failed': 0,
                'incomplete': 0,
                'avg_duration': None,
                'error_rate': 0,
                'common_failure_components': []
            }
        
        # Aggregate statistics
        total = len(flows)
        completed = sum(1 for f in flows if f.get('status') == 'completed')
        failed = sum(1 for f in flows if f.get('status') == 'failed')
        incomplete = sum(1 for f in flows if f.get('status') == 'incomplete')
        
        # Calculate average duration
        durations = [f.get('duration_seconds') for f in flows if f.get('duration_seconds') is not None]
        avg_duration = sum(durations) / len(durations) if durations else None
        
        # Calculate error rate
        error_rate = (failed / total * 100) if total > 0 else 0
        
        # Find common failure components
        failure_components = []
        for flow in flows:
            if flow.get('has_errors'):
                failure_components.extend(flow.get('components', []))
        
        component_counts = defaultdict(int)
        for comp in failure_components:
            component_counts[comp] += 1
        
        common_failures = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_flows': total,
            'completed': completed,
            'failed': failed,
            'incomplete': incomplete,
            'avg_duration': round(avg_duration, 2) if avg_duration else None,
            'error_rate': round(error_rate, 2),
            'common_failure_components': [{'component': c, 'count': n} for c, n in common_failures]
        }


def analyze_utterances(issue_entries: List[Dict]) -> Dict:
    """
    Main function to perform complete utterance analysis
    
    Args:
        issue_entries (list): List of log entry dictionaries
        
    Returns:
        dict: Complete utterance analysis
    """
    tracer = UtteranceTracer()
    
    # Enrich entries with utterance data
    enriched_entries = tracer.enrich_with_utterance_data(issue_entries)
    
    # Trace flows
    flows = tracer.trace_utterance_flows(enriched_entries)
    
    # Analyze patterns
    patterns = tracer.analyze_utterance_patterns(flows)
    
    return {
        'enriched_entries': enriched_entries,
        'flows': flows,
        'patterns': patterns,
        'total_utterance_related': sum(1 for e in enriched_entries if e.get('is_utterance_related', False))
    }
