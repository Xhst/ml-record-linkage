
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


def find_longest_common_prefix(strings, min_percentage = 0.5):
    '''
    Find the longest common prefix among a list of strings.

    Args:
        strings (List[str]): List of strings.
        min_percentage (float): Minimum percentage of strings that must contain the prefix.

    Returns:
        str: Longest common prefix among the strings.
    '''
    min_occurrences = len(strings) * min_percentage

    shortest_string = min(strings, key=len)

    for length in reversed(range(len(shortest_string))):
        sample = shortest_string[:length]
        if sum(1 for string in strings if string.startswith(sample)) >= min_occurrences:
            return sample
    
    return ""
            

def find_longest_common_suffix(strings, min_percentage = 0.5):
    '''
    Find the longest common suffix among a list of strings.

    Args:
        strings (List[str]): List of strings.
        min_percentage (float): Minimum percentage of strings that must contain the suffix.

    Returns:
        str: Longest common suffix among the strings.
    '''
    min_occurrences = len(strings) * min_percentage

    shortest_string = min(strings, key=len)

    for length in reversed(range(len(shortest_string))):
        sample = shortest_string[-length:]
        if sum(1 for string in strings if string.endswith(sample)) >= min_occurrences:
            return sample

    return ""
