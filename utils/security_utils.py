import os
import re


def sanitize_path(path):
    """Sanitizes the file path to prevent directory traversal attacks."""
    # Normalize the path and remove any directory traversal components
    normalized_path = os.path.normpath(path)
    # Ensure the path doesn't contain any illegal characters
    if re.search(r'[^a-zA-Z0-9_/\\.-]', normalized_path):
        raise ValueError("Invalid path characters.")
    return normalized_path


def validate_file(file_path):
    """Validates if the specified file exists and is a file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return True
