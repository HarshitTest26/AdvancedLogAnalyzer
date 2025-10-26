import pandas as pd
import re
import os
import logging
from pathlib import Path

# Add imports for utterance tracing
try:
    from utils.utterance_tracer import UtteranceTracer
except ImportError:
    UtteranceTracer = None

logger = logging.getLogger(__name__)

LOG_PATTERN = r'^(INFO|ERROR|DEBUG|WARN|FATAL|V)\s+(.*)$'  # Updated to include 'V' level

def parse_timestamp(timestamp):
    # Function to parse multiple timestamp formats
    formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S']  # Example formats
    for fmt in formats:
        try:
            return pd.to_datetime(timestamp, format=fmt)
        except ValueError:
            continue
    # Try pandas auto-parsing as fallback
    try:
        return pd.to_datetime(timestamp)
    except:
        return None

def validate_log_entry(entry):
    # Validate the log entry against the LOG_PATTERN
    return bool(re.match(LOG_PATTERN, entry))

def check_file_size(file_path):
    # Check if the file size exceeds 500MB
    return os.path.getsize(file_path) <= 500 * 1024 * 1024  # 500MB

def parse_line(line, line_number):
    """Parse a single log line and extract structured data"""
    try:
        # Basic log parsing - can be enhanced based on log format
        parts = line.split(' ', 2)  # Split into timestamp, level, message
        if len(parts) >= 3:
            timestamp_str = parts[0] + ' ' + parts[1] if len(parts) > 2 else parts[0]
            level = parts[2] if len(parts) > 2 else 'INFO'
            message = ' '.join(parts[3:]) if len(parts) > 3 else ''
            
            return {
                'line_number': line_number,
                'timestamp': parse_timestamp(timestamp_str),
                'level': level,
                'message': message,
                'raw_line': line
            }
        else:
            return {
                'line_number': line_number,
                'timestamp': None,
                'level': 'UNKNOWN',
                'message': line,
                'raw_line': line
            }
    except Exception as e:
        logger.error(f"Error parsing line {line_number}: {e}")
        return None

def apply_filters(df, filters):
    """Apply filters to the dataframe"""
    try:
        if 'level' in filters and 'level' in df.columns:
            df = df[df['level'].isin(filters['level'])]
        
        if 'message_contains' in filters and 'message' in df.columns:
            df = df[df['message'].str.contains(filters['message_contains'], na=False, case=False)]
        
        if 'start_time' in filters and 'timestamp' in df.columns:
            df = df[df['timestamp'] >= filters['start_time']]
        
        if 'end_time' in filters and 'timestamp' in df.columns:
            df = df[df['timestamp'] <= filters['end_time']]
        
        return df
    except Exception as e:
        logger.error(f"Error applying filters: {e}")
        return df

def analyze_logs_refactored(file_path, filters=None, sort_by='timestamp', ascending=True):
    """Enhanced analyze_logs function with utterance tracing support"""
    try:
        log_data = []
        total_lines = 0
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line_number, line in enumerate(file, 1):
                total_lines += 1
                
                if line.strip():
                    parsed_data = parse_line(line.strip(), line_number)
                    if parsed_data:
                        log_data.append(parsed_data)
        
        df = pd.DataFrame(log_data)
        
        if df.empty:
            return df, {}
        
        # Apply filters if provided
        if filters:
            df = apply_filters(df, filters)
        
        # Sort the data
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Initialize UtteranceTracer if available
        utterance_data = {}
        if UtteranceTracer:
            try:
                tracer = UtteranceTracer()
                sessions = tracer.identify_utterance_sessions(df)
                session_details = tracer.extract_session_details(sessions)
                utterance_data = {
                    'sessions': sessions,
                    'session_details': session_details,
                    'total_sessions': len(sessions)
                }
            except Exception as e:
                logger.error(f"Error in utterance tracing: {e}")
                utterance_data = {}
        
        # Extract log statistics
        stats = {
            'total_entries': len(df),
            'total_lines': total_lines,
            'unique_levels': df['level'].unique().tolist() if 'level' in df.columns else [],
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns and not df.empty else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns and not df.empty else None
            },
            'utterances': utterance_data
        }
        
        return df, stats
        
    except Exception as e:
        logger.error(f"Error analyzing logs: {e}")
        return pd.DataFrame(), {}


def filter_issues_by_relevance(df, relevance_threshold=0.5):
    """Filter log entries by relevance score for issue detection"""
    if df.empty:
        return df
    
    # Define relevance keywords and their weights
    error_keywords = {
        'error': 3.0,
        'exception': 3.0,
        'fail': 2.5,
        'crash': 3.0,
        'timeout': 2.0,
        'warning': 1.5,
        'alert': 2.0,
        'critical': 3.0,
        'fatal': 3.0,
        'disconnect': 2.0,
        'connection': 1.5,
        'retry': 1.0,
        'abort': 2.5,
        'denied': 2.0,
        'refused': 2.0,
        'invalid': 2.0,
        'corrupt': 2.5,
        'missing': 2.0,
        'not found': 2.0,
        'unavailable': 2.0
    }
    
    # Calculate relevance scores
    def calculate_relevance(text):
        if pd.isna(text):
            return 0.0
        
        text_lower = str(text).lower()
        score = 0.0
        
        for keyword, weight in error_keywords.items():
            if keyword in text_lower:
                score += weight
        
        # Boost score for ERROR level logs
        if 'level' in df.columns and any(level in text_lower for level in ['error', 'fatal', 'critical']):
            score *= 1.5
        
        return min(score, 10.0)  # Cap at 10.0
    
    # Apply relevance scoring
    df['relevance_score'] = df['message'].apply(calculate_relevance)
    
    # Filter by threshold
    filtered_df = df[df['relevance_score'] >= relevance_threshold]
    
    return filtered_df.sort_values('relevance_score', ascending=False)


def enhance_log_analysis(df):
    """Enhance log analysis with additional insights and patterns"""
    if df.empty:
        return {}
    
    insights = {
        'error_patterns': {},
        'time_patterns': {},
        'severity_distribution': {},
        'common_issues': [],
        'recommendations': []
    }
    
    try:
        # Analyze error patterns
        if 'message' in df.columns:
            error_df = df[df['level'].isin(['ERROR', 'FATAL', 'CRITICAL'])] if 'level' in df.columns else df
            
            if not error_df.empty:
                # Extract common error patterns
                error_patterns = {}
                for msg in error_df['message'].dropna():
                    # Simple pattern extraction - find repeated error types
                    words = re.findall(r'\b\w+\b', str(msg).lower())
                    for word in words:
                        if len(word) > 3:  # Skip short words
                            error_patterns[word] = error_patterns.get(word, 0) + 1
                
                # Keep top 10 patterns
                insights['error_patterns'] = dict(sorted(error_patterns.items(), 
                                                       key=lambda x: x[1], reverse=True)[:10])
        
        # Analyze time patterns
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour
            hourly_counts = df['hour'].value_counts().to_dict()
            insights['time_patterns'] = {
                'hourly_distribution': hourly_counts,
                'peak_hour': max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
            }
        
        # Severity distribution
        if 'level' in df.columns:
            insights['severity_distribution'] = df['level'].value_counts().to_dict()
        
        # Generate recommendations
        recommendations = []
        if insights['error_patterns']:
            top_error = max(insights['error_patterns'].items(), key=lambda x: x[1])
            recommendations.append(f"Most frequent error pattern: '{top_error[0]}' - Consider investigating this issue")
        
        if insights['time_patterns'].get('peak_hour'):
            peak_hour = insights['time_patterns']['peak_hour']
            recommendations.append(f"Peak activity at hour {peak_hour} - Monitor system resources during this time")
        
        insights['recommendations'] = recommendations
        
    except Exception as e:
        logger.error(f"Error in enhance_log_analysis: {e}")
    
    return insights


def extract_key_utterance_components(utterance_text):
    """Extract key components from utterance text for analysis"""
    if not utterance_text or pd.isna(utterance_text):
        return {}
    
    text = str(utterance_text).strip()
    components = {
        'wake_words': [],
        'commands': [],
        'entities': [],
        'intent': None,
        'confidence': None,
        'session_id': None,
        'request_id': None
    }
    
    try:
        # Extract wake words (Alexa specific)
        wake_patterns = [
            r'\b(alexa|amazon|echo)\b',
            r'\bwake\s+word\b',
            r'\bactivation\s+detected\b'
        ]
        
        for pattern in wake_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            components['wake_words'].extend(matches)
        
        # Extract common voice commands
        command_patterns = [
            r'\b(play|stop|pause|resume|next|previous|skip)\b',
            r'\b(turn\s+on|turn\s+off|set|adjust)\b',
            r'\b(what|when|where|how|who)\b',
            r'\b(tell\s+me|ask|search|find)\b'
        ]
        
        for pattern in command_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            components['commands'].extend(matches)
        
        # Extract entities (times, numbers, names)
        entity_patterns = {
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b',
            'number': r'\b\d+\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                components['entities'].append({
                    'type': entity_type,
                    'values': matches
                })
        
        # Extract intent from common patterns
        if any(word in text.lower() for word in ['music', 'play', 'song']):
            components['intent'] = 'music'
        elif any(word in text.lower() for word in ['weather', 'temperature', 'forecast']):
            components['intent'] = 'weather'
        elif any(word in text.lower() for word in ['timer', 'alarm', 'remind']):
            components['intent'] = 'timer'
        elif any(word in text.lower() for word in ['smart home', 'lights', 'thermostat']):
            components['intent'] = 'smart_home'
        
        # Extract confidence if present
        confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', text, re.IGNORECASE)
        if confidence_match:
            components['confidence'] = float(confidence_match.group(1))
        
        # Extract session/request IDs
        session_match = re.search(r'session[:\s]+([a-zA-Z0-9-]+)', text, re.IGNORECASE)
        if session_match:
            components['session_id'] = session_match.group(1)
        
        request_match = re.search(r'request[:\s]+([a-zA-Z0-9-]+)', text, re.IGNORECASE)
        if request_match:
            components['request_id'] = request_match.group(1)
        
    except Exception as e:
        logger.error(f"Error extracting utterance components: {e}")
    
    return components
