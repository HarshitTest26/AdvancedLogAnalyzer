import regex as re
import pandas as pd
import streamlit as st
from pathlib import Path
import logging
from datetime import datetime

from .utterance_tracer import UtteranceTracer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define severity levels and their priorities
SEVERITY_LEVELS = {
    'F': 1,  # Fatal
    'E': 2,  # Error
    'W': 3,  # Warning
    'I': 4,  # Info
    'D': 5,  # Debug
    'V': 6   # Verbose
}

# Define error keywords for Level 2 filtering
ERROR_KEYWORDS = [
    'fail', 'error', 'exception', 'crash', 'timeout', 'deadlock', 'fault',
    'invalid', 'denied', 'unable', 'cannot', 'not found', 'rejected',
    'abort', 'fatal', 'panic', 'leaked', 'corruption', 'violation',
    'broken', 'overflow', 'underflow'
]

# Define critical components for Level 3 filtering
CRITICAL_COMPONENTS = [
    'TC', 'AHE-NLU', 'MqttNotificationManager', 'AudioManager', 'BluetoothAdapter',
    'CarService', 'NavigationManager', 'CameraManager', 'SpeechRecognizer',
    'SecurityManager', 'ConnectivityService', 'PowerManager', 'TelephonyManager'
]

# Compile the canonical log pattern with regex
# Format: MM-DD HH:MM:SS.mmm PID TID Level Tag: Message
LOG_PATTERN = re.compile(
    r'(?P<date>\d{2}-\d{2})\s+'
    r'(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+'
    r'(?P<pid>\d+)\s+'
    r'(?P<tid>\d+)\s+'
    r'(?P<level>[FEWID])\s+'
    r'(?P<tag>[^:]+):\s+'
    r'(?P<message>.*)'
)

@st.cache_data
def analyze_logs_refactored(log_file_path):
    """
    Analyzes log files using a three-level filtering approach with improved memory handling.
    
    Args:
        log_file_path (Path): Path object to the log file
    
    Returns:
        dict: Dictionary containing analysis results and filtered log entries
    """
    logger.info(f"Starting analysis of {log_file_path.name}")
    
    if not log_file_path.exists():
        logger.error(f"Log file {log_file_path} does not exist")
        return {"error": f"Log file {log_file_path} does not exist"}
    
    # Initialize counters and storage
    total_lines = 0
    processed_lines = 0
    issue_entries = []
    component_counts = {}
    level_counts = {'F': 0, 'E': 0, 'W': 0, 'I': 0, 'D': 0, 'V': 0}
    
    # Calculate file size for progress updates
    file_size = log_file_path.stat().st_size
    chunk_size = 1024 * 1024  # 1MB chunks
    
    try:
        with log_file_path.open('r', encoding='utf-8', errors='ignore') as file:
            # Process the file in chunks to handle large files
            buffer = ""
            chunk = file.read(chunk_size)
            
            while chunk:
                buffer += chunk
                lines = buffer.split('\n')
                
                # Keep the last line which might be incomplete
                buffer = lines.pop()
                
                for line in lines:
                    total_lines += 1
                    match = LOG_PATTERN.search(line)
                    
                    if match:
                        processed_lines += 1
                        log_entry = match.groupdict()
                        
                        # Extract components
                        component = log_entry['tag'].strip()
                        level = log_entry['level']
                        message = log_entry['message']
                        
                        # Update component and level counts
                        if component not in component_counts:
                            component_counts[component] = 0
                        component_counts[component] += 1
                        
                        if level in level_counts:
                            level_counts[level] += 1
                        
                        # Apply the three-level filtering
                        should_include = False
                        matched_keywords = []
                        is_qa_relevant = False
                        
                        # Level 1: Explicit severities (E, F, W)
                        if level in ['F', 'E', 'W']:
                            should_include = True
                            is_qa_relevant = True
                            matched_keywords.append(f"Severity: {level}")
                        
                        # Level 2: Messages containing error keywords
                        elif any(keyword.lower() in message.lower() for keyword in ERROR_KEYWORDS):
                            should_include = True
                            is_qa_relevant = True
                            matched_keywords.extend([kw for kw in ERROR_KEYWORDS if kw.lower() in message.lower()])
                        
                        # Level 3: Info/Debug logs from critical components
                        elif level in ['I', 'D'] and any(crit_comp in component for crit_comp in CRITICAL_COMPONENTS):
                            should_include = True
                            matched_keywords.append(f"Critical Component: {component}")
                        
                        # Include the log entry if it passed any filter
                        if should_include:
                            entry_dict = {
                                'date': log_entry['date'],
                                'time': log_entry['time'],
                                'timestamp': f"{log_entry['date']} {log_entry['time']}",
                                'pid': log_entry['pid'],
                                'tid': log_entry['tid'],
                                'level': level,
                                'component': component,
                                'message': message,
                                'is_qa_relevant': is_qa_relevant,
                                'matched_qa_keywords': ', '.join(matched_keywords) if matched_keywords else None
                            }
                            issue_entries.append(entry_dict)
                
                # Read the next chunk
                chunk = file.read(chunk_size)
                
            # Process any remaining buffer
            if buffer:
                match = LOG_PATTERN.search(buffer)
                if match:
                    # Process the last line similar to above logic
                    log_entry = match.groupdict()
                    
                    # Extract components
                    component = log_entry['tag'].strip()
                    level = log_entry['level']
                    message = log_entry['message']
                    
                    # Update component and level counts
                    if component not in component_counts:
                        component_counts[component] = 0
                    component_counts[component] += 1
                    
                    if level in level_counts:
                        level_counts[level] += 1
                    
                    # Apply the three-level filtering
                    should_include = False
                    matched_keywords = []
                    is_qa_relevant = False
                    
                    # Level 1: Explicit severities (E, F, W)
                    if level in ['F', 'E', 'W']:
                        should_include = True
                        is_qa_relevant = True
                        matched_keywords.append(f"Severity: {level}")
                    
                    # Level 2: Messages containing error keywords
                    elif any(keyword.lower() in message.lower() for keyword in ERROR_KEYWORDS):
                        should_include = True
                        is_qa_relevant = True
                        matched_keywords.extend([kw for kw in ERROR_KEYWORDS if kw.lower() in message.lower()])
                    
                    # Level 3: Info/Debug logs from critical components
                    elif level in ['I', 'D'] and any(crit_comp in component for crit_comp in CRITICAL_COMPONENTS):
                        should_include = True
                        matched_keywords.append(f"Critical Component: {component}")
                    
                    # Include the log entry if it passed any filter
                    if should_include:
                        entry_dict = {
                            'date': log_entry['date'],
                            'time': log_entry['time'],
                            'timestamp': f"{log_entry['date']} {log_entry['time']}",
                            'pid': log_entry['pid'],
                            'tid': log_entry['tid'],
                            'level': level,
                            'component': component,
                            'message': message,
                            'is_qa_relevant': is_qa_relevant,
                            'matched_qa_keywords': ', '.join(matched_keywords) if matched_keywords else None
                        }
                        issue_entries.append(entry_dict)
        
        # Create analysis summary
        analysis_results = {
            'log_file_name': log_file_path.name,
            'log_file_path': str(log_file_path),
            'total_lines': total_lines,
            'processed_lines': processed_lines,
            'issues_found': len(issue_entries),
            'component_counts': component_counts,
            'level_counts': level_counts,
            'issue_entries': issue_entries,
            'qa_relevant_issues': sum(1 for entry in issue_entries if entry['is_qa_relevant']),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Completed analysis of {log_file_path.name}. Found {len(issue_entries)} issues.")
        return analysis_results
    
    except Exception as e:
        logger.error(f"Error analyzing log file {log_file_path.name}: {str(e)}")
        return {"error": f"Error analyzing log file: {str(e)}"}


def filter_issues_by_relevance(issue_entries, qa_only=False):
    """
    Filter issues based on QA relevance.
    
    Args:
        issue_entries (list): List of issue dictionaries
        qa_only (bool): If True, return only QA-relevant issues
    
    Returns:
        list: Filtered list of issues
    """
    if qa_only:
        return [entry for entry in issue_entries if entry['is_qa_relevant']]
    return issue_entries


def enhance_log_analysis(parsed_df):
    """
    Enhances the parsed log dataframe with utterance tracking
    and correlation features.
    
    Args:
        parsed_df: DataFrame of parsed log entries
        
    Returns:
        Enhanced DataFrame with session IDs and correlation info
    """
    # Initialize utterance tracer
    tracer = UtteranceTracer()
    
    # Identify all utterance sessions
    sessions = tracer.identify_utterance_sessions(parsed_df)
    
    # Extract detailed session information
    session_details = tracer.extract_session_details(sessions)
    
    # Add session ID to the original dataframe
    parsed_df['session_id'] = None
    for session_id, session_df in sessions.items():
        for idx in session_df.index:
            parsed_df.at[idx, 'session_id'] = session_id
    
    return parsed_df, session_details


def extract_key_utterance_components(parsed_df):
    """
    Extracts key components related to the voice assistant workflow
    to help understand the processing pipeline.
    
    Args:
        parsed_df: DataFrame of parsed log entries
        
    Returns:
        Dict of component information
    """
    components = {
        'voice_activation': [],
        'nlu_processing': [],
        'action_execution': [],
        'response_generation': []
    }
    
    # Voice activation components (AHE-AHAP, etc.)
    voice_components = parsed_df[parsed_df['tag'].str.contains('AHAP|Voice|Audio', case=False, na=False)]
    if not voice_components.empty:
        components['voice_activation'] = voice_components['tag'].unique().tolist()
    
    # NLU components
    nlu_components = parsed_df[parsed_df['tag'].str.contains('NLU|Intent|LRO', case=False, na=False)]
    if not nlu_components.empty:
        components['nlu_processing'] = nlu_components['tag'].unique().tolist()
    
    # Action execution components
    action_components = parsed_df[parsed_df['tag'].str.contains('vehicle|HAL|MQTT|TC', case=False, na=False)]
    if not action_components.empty:
        components['action_execution'] = action_components['tag'].unique().tolist()
    
    # Response generation
    response_components = parsed_df[parsed_df['tag'].str.contains('TTS|Media|Player|AACS', case=False, na=False)]
    if not response_components.empty:
        components['response_generation'] = response_components['tag'].unique().tolist()
    
    return components