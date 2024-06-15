
def replace_special_chars_with_whitespace(s: str) -> str:
    '''
    Replace special characters with whitespaces in a string.

    Args:
        s (str): Input string.

    Returns:
        str: String with special characters replaced by whitespaces.
    '''
    return ''.join(' ' if not e.isalnum() and not e.isspace() else e for e in s)


def remove_extra_whitespaces(s: str) -> str:
    '''
    Remove extra whitespaces from a string.

    Args:
        s (str): Input string.

    Returns:
        str: String with extra whitespaces removed.
    '''
    return ' '.join(s.split())


def add_whitespace_between_attached_words(s: str) -> str:
    '''
    Add whitespace between attached words in a string.
    It considers that words are attached when a lowercase letter is followed by an uppercase letter or a number.

    Args:
        s (str): Input string.

    Returns:
        str: String with whitespace added between attached words.
    '''
    return re.sub(r'([a-z])([A-Z0-9])', r'\1 \2', s)