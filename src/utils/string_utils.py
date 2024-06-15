
def replace_special_chars_with_whitespace(s: str) -> str:
    '''
    Replace special characters with whitespaces in a string.

    Args:
        s (str): Input string.

    Returns:
        str: String with special characters replaced by whitespaces.
    '''
    return ''.join(' ' if not e.isalnum() and not e.isspace() else e for e in s)