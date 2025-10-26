import re
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class LogParser:
    """Base class for log parsing"""
    
    def parse_line(self, line: str) -> Optional[Dict]:
        """Parse a single log line. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def parse_file(self, file_path: str) -> pd.DataFrame:
        """Parse an entire log file."""
        entries = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = self.parse_line(line)
                        if entry:
                            entry['line_number'] = line_num
                            entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
        
        return pd.DataFrame(entries) if entries else pd.DataFrame()


class CanonicalLogParser(LogParser):
    """Parser for canonical automotive log format: MM-DD HH:MM:SS.mmm PID TID Level Tag: Message"""
    
    def __init__(self):
        self.pattern = re.compile(
            r'(?P<date>\d{2}-\d{2})\s+'
            r'(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+'
            r'(?P<pid>\d+)\s+'
            r'(?P<tid>\d+)\s+'
            r'(?P<level>[FEWIDV])\s+'
            r'(?P<tag>[^:]+):\s+'
            r'(?P<message>.*)'
        )
    
    def parse_line(self, line: str) -> Optional[Dict]:
        match = self.pattern.search(line)
        if match:
            return match.groupdict()
        return None


class ISO8601LogParser(LogParser):
    """Parser for ISO 8601 timestamp format: YYYY-MM-DDTHH:MM:SS.fffZ Level Tag: Message"""
    
    def __init__(self):
        self.pattern = re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z\s+'
            r'(?P<level>[FEWIDV])\s+'
            r'(?P<tag>[^:]+):\s+'
            r'(?P<message>.*)'
        )
    
    def parse_line(self, line: str) -> Optional[Dict]:
        match = self.pattern.search(line)
        if match:
            data = match.groupdict()
            data['pid'] = 'N/A'
            data['tid'] = 'N/A'
            return data
        return None


class UnixTimestampLogParser(LogParser):
    """Parser for Unix timestamp format: 1234567890.123 PID TID Level Tag: Message"""
    
    def __init__(self):
        self.pattern = re.compile(
            r'(?P<unix_timestamp>\d+\.\d+)\s+'
            r'(?P<pid>\d+)\s+'
            r'(?P<tid>\d+)\s+'
            r'(?P<level>[FEWIDV])\s+'
            r'(?P<tag>[^:]+):\s+'
            r'(?P<message>.*)'
        )
    
    def parse_line(self, line: str) -> Optional[Dict]:
        match = self.pattern.search(line)
        if match:
            return match.groupdict()
        return None


class JSONLogParser(LogParser):
    """Parser for JSON-formatted logs"""
    
    def parse_line(self, line: str) -> Optional[Dict]:
        import json
        try:
            data = json.loads(line)
            # Normalize to common format
            normalized = {
                'timestamp': data.get('timestamp', data.get('time', '')),
                'level': data.get('level', data.get('severity', 'I')).upper()[:1],
                'tag': data.get('tag', data.get('component', data.get('logger', ''))),
                'message': data.get('message', data.get('msg', str(data))),
                'pid': str(data.get('pid', 'N/A')),
                'tid': str(data.get('tid', data.get('thread_id', 'N/A')))
            }
            return normalized
        except Exception:
            return None


class LogParserFactory:
    """Factory for creating appropriate log parser based on format detection"""
    
    PARSERS = [
        CanonicalLogParser(),
        ISO8601LogParser(),
        UnixTimestampLogParser(),
        JSONLogParser(),
    ]
    
    @staticmethod
    def detect_format(file_path: str, sample_lines: int = 50) -> LogParser:
        """Detect log format by analyzing sample lines"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = [f.readline().strip() for _ in range(sample_lines)]
            
            # Try each parser
            for parser in LogParserFactory.PARSERS:
                success_count = 0
                for line in lines:
                    if line and parser.parse_line(line):
                        success_count += 1
                
                # If parser successfully parses >70% of sample lines, use it
                if success_count / max(len([l for l in lines if l]), 1) > 0.7:
                    logger.info(f"Detected log format: {parser.__class__.__name__}")
                    return parser
            
            # Default to canonical format
            logger.warning("Could not detect log format, using canonical format")
            return CanonicalLogParser()
        
        except Exception as e:
            logger.error(f"Error detecting format: {str(e)}")
            return CanonicalLogParser()
    
    @staticmethod
    def parse(file_path: str, parser: Optional[LogParser] = None) -> pd.DataFrame:
        """Parse log file with optional parser specification"""
        if parser is None:
            parser = LogParserFactory.detect_format(file_path)
        
        return parser.parse_file(file_path)