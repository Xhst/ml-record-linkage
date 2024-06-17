import os
import json
import re
import paths
import utils.string_utils as string_utils
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


def clean_text_data(text: str) -> str:
    '''    
    Args:
        text (str): The text data to be cleaned.
    
    Returns:
        str: The cleaned text data.
    '''
    text = text.lower()
    text = string_utils.replace_special_chars_with_whitespace(text)
    #text = string_utils.remove_whitespaces_between_letters_and_numbers(text)
    #text = string_utils.remove_whitespaces_between_numbers(text)
    text = remove_non_relevant_words(text)
    text = string_utils.remove_extra_whitespaces(text)

    return text


def remove_non_relevant_words(string: str) -> str:
    # remove words containing numbers followed by units of measurement
    string = re.sub(r'\d+(?:\.\d+)?(?:inch|in|cm|mm|ms|dc)', '', string)

    # remove words representing resolutions (like 1920x1080)
    string = re.sub(r'\d+x\d+', '', string)

    # remove frequent words like led, lcd, monitor
    string = re.sub(r'\b(?:led|lcd|monitor|monitors|vga|hdmi|led|display|tft|tv|cable|cables)', '', string)

    return string


def remove_common_words(source: str, item2pagetitle: dict[str, str], word_counter: Counter, min_percentage: float = 0.1):
    '''
    Removes common words from the page titles based on the minimum percentage of occurrences.

    Args:
        source (str): The source name.
        item2pagetitle (Dict[str, str]): Dictionary containing item names and their corresponding page titles.
        word_counter (Counter): Counter object containing the count of words in the page titles.
        min_percentage (float): Minimum percentage of occurrences for a word to be considered common.
    '''
    min_occurrences = len(item2pagetitle) * min_percentage

    words_to_remove = []
    for word, count in word_counter.most_common():
        if count < min_occurrences: continue
        words_to_remove.append(word)
    
    print(f"Words to remove for {source}: {words_to_remove}")

    for key, value in item2pagetitle.items():
        for word in value.split():
            if word not in words_to_remove: continue
            value = value.replace(word, '')
            value = string_utils.remove_extra_whitespaces(value)
            item2pagetitle[key] = value


def get_relevant_text(data: dict) -> str:
    '''
    Extracts the relevant text from the JSON data.

    Args:
        data (Dict): JSON data.

    Returns:
        str: Relevant text from the JSON data.
    '''
    relevant_labels = ['model', 'model name', 'product model', 'model number', 'product name', 'part']

    for label in relevant_labels:
        if data.get(label):
            if isinstance(data[label], list):
                return data[label][0]
            return data[label]
    
    return data['<page title>']


def try_find_model_name(string: str) -> str:
    alphanumeric_words = string_utils.find_alphanumeric_words(string)
    val = ' '.join(word if len(word) > 3 else '' for word in alphanumeric_words)
    val = string_utils.remove_extra_whitespaces(val)
    
    return val


def process_directory(root_dir, source_dir):
    item2pagetitle = {}
    word_counter = Counter()

    for _, _, files in os.walk(root_dir + '/' + source_dir):
        for file in files:
            filepath = os.path.join(root_dir + '/' + source_dir, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data:
                        item_name = source_dir + '//' + file.replace('.json', '')

                        text_to_process = get_relevant_text(data)

                        cleaned_text = clean_text_data(text_to_process)

                        if len(cleaned_text.split()) > 1:
                            model = try_find_model_name(cleaned_text)
                            if model != '':
                                cleaned_text = model

                        word_counter.update(cleaned_text.split())

                        item2pagetitle[item_name] = cleaned_text
                    else:
                        print(f"Empty JSON file: {filepath}")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    remove_common_words(source_dir, item2pagetitle, word_counter)  
        
    return item2pagetitle


def process_sources(root_dir):
    item2pagetitle = {}

    # Process each directory in a different thread
    with ThreadPoolExecutor() as executor:
        futures = []
        for _, dirnames, _ in os.walk(root_dir):
            for dirname in dirnames:
                futures.append(executor.submit(process_directory, root_dir, dirname))

        for future in futures:
            item2pagetitle.update(future.result())
    
    try:
        output_filepath = os.path.join(paths.RESULTS_DIR + '/preprocessing/preprocessed_dataset.json')
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(item2pagetitle, f, ensure_ascii=False, indent=4)
        print(f"Written to {output_filepath}")
    except Exception as e:
        print(f"Error writing {output_filepath}: {e}")
    