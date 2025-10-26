import re
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class UtteranceTracer:
    """
    Traces utterance flows from voice command to system response
    by tracking dialog request IDs or session IDs across log entries.
    """
    
    def __init__(self):
        # CRITICAL FIX #1: Support custom patterns (initialize empty custom patterns)
        self.utterance_start_patterns = [
            r'handleUtterance\(\) started',
            r'Received Pryon FirstPassRecognition',
            r'AHE-NLU.*processing utterance',
            r'WakewordDetected.*payload.*wakeword.*ALEXA',
            r'topic=SpeechRecognizer.*action=WakewordDetected',
            r'DialogStateChanged.*state.*LISTENING',
            r'payload=\{"wakeword":"ALEXA"\}',
            r'DialogStateChangedMessages.*state LISTENING',
            r'AutoVoiceChrome state changed LISTENING'
        ]
        
        # Add generic voice assistant patterns
        self.utterance_start_patterns.extend([
            r'voice.*command.*received',
            r'utterance.*processing',
            r'speech.*detected'
        ])
        
        # Message ID extraction patterns
        self.message_id_pattern = re.compile(r'(\d{10,})')
        self.message_id_extractors = [
            re.compile(r'messageId=([0-9a-f-]+)', re.IGNORECASE),
            re.compile(r'"messageId"\s*:\s*"([0-9a-f-]+)"', re.IGNORECASE),
            re.compile(r'dialogId[=:]?\s*["\']?([0-9a-f-]+)', re.IGNORECASE)
        ]
        
        # Dialog state patterns
        self.dialog_state_patterns = {
            'LISTENING': re.compile(r'(DialogState.*LISTENING|setAnimationState\s+LISTENING|state[=:]?\s*LISTENING|AutoVoiceChrome.*state changed LISTENING|DialogStateChangedMessages.*state LISTENING)', re.IGNORECASE),
            'THINKING': re.compile(r'(DialogState.*THINKING|setAnimationState\s+THINKING|state[=:]?\s*THINKING|AutoVoiceChrome.*state changed THINKING|DialogStateChangedMessages.*state THINKING)', re.IGNORECASE),
            'SPEAKING': re.compile(r'(DialogState.*SPEAKING|setAnimationState\s+SPEAKING|state[=:]?\s*SPEAKING|AutoVoiceChrome.*state changed SPEAKING|DialogStateChangedMessages.*state SPEAKING)', re.IGNORECASE),
            'IDLE': re.compile(r'(DialogState.*IDLE|setAnimationState\s+IDLE|state[=:]?\s*IDLE|AutoVoiceChrome.*state changed IDLE|DialogStateChangedMessages.*state IDLE)', re.IGNORECASE)
        }
        
        # Session ID patterns
        self.session_id_pattern = re.compile(r'(\d{10,})')
    
    def identify_utterance_sessions(self, df):
        """
        Identifies all utterance sessions in the log dataframe.
        """
        sessions = {}
        current_session = None
        
        try:
            for idx, row in df.iterrows():
                message = row.get('message', '')
                tag = row.get('tag', '')
                
                # CRITICAL FIX #3: Validate patterns before using them
                is_start = False
                for pattern in self.utterance_start_patterns:
                    try:
                        if re.search(pattern, message):
                            is_start = True
                            break
                    except re.error as e:
                        logger.warning(f"Invalid regex pattern {pattern}: {str(e)}")
                        continue
                
                if is_start:
                    message_id = self.extract_message_id(message)
                    
                    if not message_id:
                        sid_match = self.session_id_pattern.search(message)
                        if sid_match:
                            message_id = sid_match.group(1)
                    
                    if message_id:
                        if message_id not in sessions:
                            sessions[message_id] = []
                        current_session = message_id
                
                if current_session:
                    sessions[current_session].append(idx)
                    
                    if "response complete" in message.lower() or "timeout" in message.lower() or re.search(r"DialogState.*IDLE", message):
                        current_session = None
            
            # Second pass - associate entries by PID/TID and timestamp correlation
            for session_id, indices in list(sessions.items()):
                if indices:
                    base_entries = df.iloc[indices]
                    pids = set(base_entries['pid'].dropna())
                    tids = set(base_entries['tid'].dropna())
                    
                    try:
                        base_timestamps = pd.to_datetime(base_entries['timestamp'], errors='coerce')
                        min_time = base_timestamps.min() - pd.Timedelta(seconds=2)
                        max_time = base_timestamps.max() + pd.Timedelta(seconds=5)
                        
                        for idx, row in df.iterrows():
                            if idx not in indices:
                                try:
                                    row_timestamp = pd.to_datetime(row['timestamp'], errors='coerce')
                                    if (row['pid'] in pids or row['tid'] in tids) and \
                                       pd.notna(row_timestamp) and min_time <= row_timestamp <= max_time:
                                        sessions[session_id].append(idx)
                                except (TypeError, ValueError):
                                    continue
                    except (TypeError, ValueError):
                        continue
            
            # Clean up abandoned sessions
            sessions = self.cleanup_abandoned_sessions(sessions)
            
            return {sid: df.iloc[indices].sort_values('timestamp') for sid, indices in sessions.items()}
        
        except Exception as e:
            logger.error(f"Error identifying utterance sessions: {str(e)}")
            return {}
    
    def extract_session_details(self, sessions):
        """
        Extracts key information from each session.
        """
        session_details = []
        
        try:
            for session_id, session_df in sessions.items():
                if session_df.empty:
                    continue
                
                utterance_text = "Unknown"
                for _, row in session_df.iterrows():
                    if "utterance text:" in row['message'].lower():
                        match = re.search(r'utterance text:\s*[\'"]?(.*?)[\'"]?$', row['message'], re.IGNORECASE)
                        if match:
                            utterance_text = match.group(1)
                            break
                
                has_errors = any(row['level'] in ['E', 'F', 'W'] for _, row in session_df.iterrows())
                
                components = session_df['tag'].unique().tolist()
                
                dialog_states = []
                current_state = None
                for _, row in session_df.iterrows():
                    message = row.get('message', '')
                    state = self.extract_dialog_state(message)
                    if state and state != current_state:
                        dialog_states.append({
                            'state': state,
                            'timestamp': row.get('timestamp')
                        })
                        current_state = state
                
                final_state = dialog_states[-1]['state'] if dialog_states else 'UNKNOWN'
                
                try:
                    timestamps = pd.to_datetime(session_df['timestamp'], errors='coerce')
                    valid_timestamps = timestamps.dropna()
                    
                    if len(valid_timestamps) >= 2:
                        timestamp_start = valid_timestamps.min()
                        timestamp_end = valid_timestamps.max()
                        duration_ms = (timestamp_end - timestamp_start).total_seconds() * 1000
                    elif len(valid_timestamps) == 1:
                        timestamp_start = timestamp_end = valid_timestamps.iloc[0]
                        duration_ms = 0
                    else:
                        timestamp_start = session_df['timestamp'].iloc[0] if len(session_df) > 0 else 'Unknown'
                        timestamp_end = session_df['timestamp'].iloc[-1] if len(session_df) > 0 else 'Unknown'
                        duration_ms = 0
                except Exception:
                    timestamp_start = session_df['timestamp'].iloc[0] if len(session_df) > 0 else 'Unknown'
                    timestamp_end = session_df['timestamp'].iloc[-1] if len(session_df) > 0 else 'Unknown'
                    duration_ms = 0
                
                session_details.append({
                    'session_id': session_id,
                    'utterance': utterance_text,
                    'timestamp_start': timestamp_start,
                    'timestamp_end': timestamp_end,
                    'duration_ms': duration_ms,
                    'has_errors': has_errors,
                    'components': components,
                    'entries': len(session_df),
                    'dialog_states': dialog_states,
                    'final_state': final_state
                })
        
        except Exception as e:
            logger.error(f"Error extracting session details: {str(e)}")
        
        return session_details
    
    def extract_dialog_states(self, session_df):
        """
        Extract dialog state transitions from a session dataframe.
        """
        state_transitions = []
        current_state = None
        
        try:
            for _, row in session_df.iterrows():
                message = row.get('message', '')
                timestamp = row.get('timestamp')
                
                for state_name, pattern in self.dialog_state_patterns.items():
                    try:
                        if pattern.search(message) and state_name != current_state:
                            state_transitions.append({
                                'timestamp': timestamp,
                                'state': state_name,
                                'message': message,
                                'tag': row.get('tag', '')
                            })
                            current_state = state_name
                            break
                    except re.error as e:
                        logger.debug(f"Invalid pattern for state {state_name}: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error extracting dialog states: {str(e)}")
        
        return state_transitions
    
    def extract_message_id(self, message):
        """
        CRITICAL FIX #4: Add validation for message ID extraction
        Extract message ID from log message using multiple patterns.
        """
        if not message:
            return None
        
        try:
            for extractor in self.message_id_extractors:
                try:
                    match = extractor.search(message)
                    if match:
                        return match.group(1)
                except re.error:
                    continue
            
            match = self.session_id_pattern.search(message)
            if match:
                return match.group(1)
        
        except Exception as e:
            logger.debug(f"Error extracting message ID: {str(e)}")
        
        return None

    def extract_dialog_state(self, message):
        """
        Extract dialog state from message with pattern validation.
        """
        if not message:
            return None
        
        try:
            for state, pattern in self.dialog_state_patterns.items():
                try:
                    if pattern.search(message):
                        return state
                except re.error:
                    logger.debug(f"Invalid regex for state {state}")
                    continue
        except Exception as e:
            logger.debug(f"Error extracting dialog state: {str(e)}")
        
        return None

    def cleanup_abandoned_sessions(self, sessions, max_duration_seconds=120):
        """
        Cleans up sessions that exceed a maximum duration
        """
        cleaned_sessions = {}
        
        try:
            for session_id, session_df in sessions.items():
                try:
                    timestamps = pd.to_datetime(session_df['timestamp'], errors='coerce')
                    valid_timestamps = timestamps.dropna()
                    
                    if len(valid_timestamps) >= 2:
                        min_time = valid_timestamps.min()
                        max_time = valid_timestamps.max()
                        duration_seconds = (max_time - min_time).total_seconds()
                        
                        if duration_seconds <= max_duration_seconds:
                            cleaned_sessions[session_id] = session_df
                    else:
                        cleaned_sessions[session_id] = session_df
                except Exception:
                    cleaned_sessions[session_id] = session_df
        
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {str(e)}")
            return sessions
        
        return cleaned_sessions
    
    def add_custom_pattern(self, pattern):
        """
        CRITICAL FIX #5: Add support for custom patterns
        Add a custom pattern to utterance detection
        """
        try:
            # Validate pattern first
            re.compile(pattern)
            self.utterance_start_patterns.append(pattern)
            logger.info(f"Added custom pattern: {pattern}")
            return True
        except re.error as e:
            logger.error(f"Invalid regex pattern: {str(e)}")
            return False