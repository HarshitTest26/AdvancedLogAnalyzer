import re
from pathlib import Path
import pandas as pd

class UtteranceTracer:
    """
    Traces utterance flows from voice command to system response
    by tracking dialog request IDs or session IDs across log entries.
    """
    
    def __init__(self):
        # Enhanced patterns to identify the start of an utterance
        self.utterance_start_patterns = [
            r'handleUtterance\(\) started',
            r'Received Pryon FirstPassRecognition',
            r'AHE-NLU.*processing utterance',
            r'WakewordDetected.*payload.*wakeword.*ALEXA',  # Add for Alexa detection
            r'topic=SpeechRecognizer.*action=WakewordDetected',  # Common in Alexa logs
            r'DialogStateChanged.*state.*LISTENING'         # Add for dialog state changes
        ]
        
        # Add message ID extraction patterns for different formats
        self.message_id_pattern = re.compile(r'(\d{10,})')
        self.message_id_extractors = [
            re.compile(r'messageId=([0-9a-f-]+)', re.IGNORECASE),
            re.compile(r'"messageId"\s*:\s*"([0-9a-f-]+)"', re.IGNORECASE),
            re.compile(r'dialogId[=:]?\s*["\']?([0-9a-f-]+)', re.IGNORECASE)
        ]
        
        # Enhanced dialog state patterns for Alexa logs
        self.dialog_state_patterns = {
            'LISTENING': re.compile(r'(DialogState.*LISTENING|setAnimationState\s+LISTENING|state[=:]?\s*LISTENING|AutoVoiceChrome.*state changed LISTENING)', re.IGNORECASE),
            'THINKING': re.compile(r'(DialogState.*THINKING|setAnimationState\s+THINKING|state[=:]?\s*THINKING|AutoVoiceChrome.*state changed THINKING)', re.IGNORECASE),
            'SPEAKING': re.compile(r'(DialogState.*SPEAKING|setAnimationState\s+SPEAKING|state[=:]?\s*SPEAKING|AutoVoiceChrome.*state changed SPEAKING)', re.IGNORECASE),
            'IDLE': re.compile(r'(DialogState.*IDLE|setAnimationState\s+IDLE|state[=:]?\s*IDLE|AutoVoiceChrome.*state changed IDLE)', re.IGNORECASE)
        }
        
        # Session ID patterns - looking for long integer IDs like 520618634416
        self.session_id_pattern = re.compile(r'(\d{10,})')
        
    def identify_utterance_sessions(self, df):
        """
        Identifies all utterance sessions in the log dataframe.
        
        Args:
            df: Pandas DataFrame with parsed log entries
            
        Returns:
            Dictionary mapping session IDs to all related log entries
        """
        sessions = {}
        current_session = None
        
        # First pass - identify utterance starts and extract session IDs
        for idx, row in df.iterrows():
            message = row.get('message', '')
            tag = row.get('tag', '')
            
            # Check if this is the start of an utterance
            is_start = any(re.search(pattern, message) for pattern in self.utterance_start_patterns)
            
            if is_start:
                # First try to extract message ID
                message_id = self.extract_message_id(message)
                
                # If no message ID, use session ID
                if not message_id:
                    sid_match = self.session_id_pattern.search(message)
                    if sid_match:
                        message_id = sid_match.group(1)
                
                # If we found an ID, create a session
                if message_id:
                    if message_id not in sessions:
                        sessions[message_id] = []
                    current_session = message_id
                
            # If we're in a session, add this entry
            if current_session:
                sessions[current_session].append(idx)
                
                # Check if this entry ends the session (response or timeout)
                if "response complete" in message.lower() or "timeout" in message.lower() or "DialogState.*IDLE" in message:
                    current_session = None
        
        # Second pass - associate entries by PID/TID and timestamp correlation
        for session_id, indices in list(sessions.items()):
            if indices:
                base_entries = df.iloc[indices]
                pids = set(base_entries['pid'].dropna())
                tids = set(base_entries['tid'].dropna())
                
                # Find timestamp range (with buffer)
                min_time = min(base_entries['timestamp']) - pd.Timedelta(seconds=2)
                max_time = max(base_entries['timestamp']) + pd.Timedelta(seconds=5)
                
                # Add any entries that match PID/TID within the time window
                for idx, row in df.iterrows():
                    if idx not in indices:
                        if (row['pid'] in pids or row['tid'] in tids) and \
                           min_time <= row['timestamp'] <= max_time:
                            sessions[session_id].append(idx)
        
        return {sid: df.iloc[indices].sort_values('timestamp') for sid, indices in sessions.items()}
    
    def extract_session_details(self, sessions):
        """
        Extracts key information from each session.
        
        Args:
            sessions: Dictionary of session dataframes
            
        Returns:
            List of session summaries with key information
        """
        session_details = []
        
        for session_id, session_df in sessions.items():
            # Find the utterance text if available
            utterance_text = "Unknown"
            for _, row in session_df.iterrows():
                if "utterance text:" in row['message'].lower():
                    match = re.search(r'utterance text:\s*[\'"]?(.*?)[\'"]?$', row['message'], re.IGNORECASE)
                    if match:
                        utterance_text = match.group(1)
                        break
            
            # Check if there were errors in this session
            has_errors = any(row['level'] in ['E', 'F', 'W'] for _, row in session_df.iterrows())
            
            # Get all components involved in this session
            components = session_df['tag'].unique().tolist()
            
            # Track dialog states in this session
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
            
            # Determine final dialog state
            final_state = dialog_states[-1]['state'] if dialog_states else 'UNKNOWN'
            
            session_details.append({
                'session_id': session_id,
                'utterance': utterance_text,
                'timestamp_start': session_df['timestamp'].min(),
                'timestamp_end': session_df['timestamp'].max(),
                'duration_ms': (session_df['timestamp'].max() - session_df['timestamp'].min()).total_seconds() * 1000,
                'has_errors': has_errors,
                'components': components,
                'entries': len(session_df),
                'dialog_states': dialog_states,
                'final_state': final_state
            })
        
        return session_details
    
    def extract_dialog_states(self, session_df):
        """
        Extract dialog state transitions from a session dataframe.
        
        Args:
            session_df: DataFrame containing log entries for a single session
            
        Returns:
            List of dialog state transitions with timestamps
        """
        state_transitions = []
        current_state = None
        
        for _, row in session_df.iterrows():
            message = row.get('message', '')
            timestamp = row.get('timestamp')
            
            # Check for each dialog state pattern
            for state_name, pattern in self.dialog_state_patterns.items():
                if pattern.search(message) and state_name != current_state:
                    state_transitions.append({
                        'timestamp': timestamp,
                        'state': state_name,
                        'message': message,
                        'tag': row.get('tag', '')
                    })
                    current_state = state_name
                    break
        
        return state_transitions
    
    def extract_message_id(self, message):
        """
        Extract message ID from log message using multiple patterns.
        
        Args:
            message: String containing the log message
            
        Returns:
            Extracted message ID or None
        """
        if not message:
            return None
            
        # Try all extractors
        for extractor in self.message_id_extractors:
            match = extractor.search(message)
            if match:
                return match.group(1)
                
        # Try session ID pattern as fallback
        match = self.session_id_pattern.search(message)
        if match:
            return match.group(1)
            
        return None

    def extract_dialog_state(self, message):
        """
        Extract dialog state from message.
        
        Args:
            message: Log message text
            
        Returns:
            Dialog state name or None
        """
        if not message:
            return None
            
        for state, pattern in self.dialog_state_patterns.items():
            if pattern.search(message):
                return state
                
        return None