import re
from pathlib import Path
import pandas as pd

class UtteranceTracer:
    """
    Traces utterance flows from voice command to system response
    by tracking dialog request IDs or session IDs across log entries.
    """
    
    def __init__(self):
        # Patterns to identify the start of an utterance
        self.utterance_start_patterns = [
            r'handleUtterance\(\) started',
            r'Received Pryon FirstPassRecognition',
            r'AHE-NLU.*processing utterance'
        ]
        
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
                # Look for session ID in this entry or nearby entries
                sid_match = self.session_id_pattern.search(message)
                if sid_match:
                    session_id = sid_match.group(1)
                    if session_id not in sessions:
                        sessions[session_id] = []
                    current_session = session_id
                    
            # If we're in a session, add this entry
            if current_session:
                sessions[current_session].append(idx)
                
                # Check if this entry ends the session (response or timeout)
                if "response complete" in message.lower() or "timeout" in message.lower():
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
            
            session_details.append({
                'session_id': session_id,
                'utterance': utterance_text,
                'timestamp_start': session_df['timestamp'].min(),
                'timestamp_end': session_df['timestamp'].max(),
                'duration_ms': (session_df['timestamp'].max() - session_df['timestamp'].min()).total_seconds() * 1000,
                'has_errors': has_errors,
                'components': components,
                'entries': len(session_df)
            })
        
        return session_details