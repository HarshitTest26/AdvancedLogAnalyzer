# input_validator.py

def validate_string(input_string):
    """
    Validate if the input is a string.
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")
    return True


def validate_integer(input_integer):
    """
    Validate if the input is an integer.
    """
    if not isinstance(input_integer, int):
        raise ValueError("Input must be an integer.")
    return True


def validate_positive_integer(input_integer):
    """
    Validate if the input is a positive integer.
    """
    validate_integer(input_integer)
    if input_integer <= 0:
        raise ValueError("Input must be a positive integer.")
    return True


def validate_email(input_email):
    """
    Validate if the input is a valid email address.
    """
    import re
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, input_email):
        raise ValueError("Input must be a valid email address.")
    return True