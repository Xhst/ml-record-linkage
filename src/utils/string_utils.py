
import re

def replace_special_chars_with_whitespace(s: str, exclude: list[str] = []) -> str:
    '''
    Replace special characters with whitespace in a string.

    Args:
        s (str): Input string.
        exclude (List[str]): List of characters to exclude from replacement.

    Returns:
        str: String with special characters replaced with whitespace.
    '''
    return ''.join(c if c.isalnum() or c in exclude else ' ' for c in s)


def remove_special_chars(s: str, exclude: list[str] = []) -> str:
    '''
    Remove special characters from a string.

    Args:
        s (str): Input string.
        exclude (List[str]): List of characters to exclude from removal.

    Returns:
        str: String with special characters removed.
    '''
    return ''.join(c for c in s if c.isalnum() or c in exclude)


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


def find_longest_common_prefix(strings: list[str], min_percentage: float = 0.5) -> str:
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

    for length in reversed(range(1, len(shortest_string))):
        sample = shortest_string[:length]
        if sum(1 for string in strings if string.startswith(sample)) >= min_occurrences:
            return sample
    
    return ""
            

def find_longest_common_suffix(strings: list[str], min_percentage: float = 0.5) -> str:
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

    for length in reversed(range(1, len(shortest_string))):
        sample = shortest_string[-length:]
        if sum(1 for string in strings if string.endswith(sample)) >= min_occurrences:
            return sample

    return ""


def remove_whitespaces_between_letters_and_numbers(s: str) -> str:
    '''
    Remove whitespaces between letters and numbers in a string.

    Args:
        s (str): Input string.

    Returns:
        str: String with whitespaces removed between letters and numbers.
    '''
    return re.sub(r'(\D)\s+(\d)', r'\1\2', s)


def find_longest_alphanumeric_word(s):
    '''
    Find the longest alphanumeric word in a string.

    Args:
        s (str): Input string.

    Returns:
        str: Longest alphanumeric word in the string.
    '''
    words = re.findall(r'\w+', s)
    
    alphanumeric_words = [word for word in words if re.search(r'[A-Za-z]', word) and re.search(r'\d', word)]
    
    if alphanumeric_words:
        return max(alphanumeric_words, key=len)
    
    return None