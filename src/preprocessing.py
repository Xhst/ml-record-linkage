import os
import json
import torch
import os
import json
from difflib import SequenceMatcher 


DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset/2013_monitor_specs')

def get_dataset_dir():
    return DATASET_DIR
             
def common_prefix_two_strings(str1, str2):
    matcher = SequenceMatcher(None, str1, str2)
    match = matcher.find_longest_match(0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]


def find_longest_common_string(strings):
 
    if not strings:
        return ""
    prefix = strings[0]
    for string in strings[1:]:
        prefix = common_prefix_two_strings(prefix, string)
        if not prefix:
            break
    return prefix


def process_directory(root_dir, source_dir):
    item2pagetitle = {}
    for _, _, files in os.walk(root_dir + '/' + source_dir):
        for file in files:
            filepath = os.path.join(root_dir + '/' + source_dir, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data:
                        item_name = source_dir + '//' + file.replace('.json', '')
                        page_title = data['<page title>']
                        
                        item2pagetitle[item_name] = page_title
                    else:
                        print(f"Empty JSON file: {filepath}")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
    longest_common_string = find_longest_common_string(list(item2pagetitle.values()))
    
    for key, value in item2pagetitle.items():
        item2pagetitle[key] = value.replace(longest_common_string, '')
        
    return item2pagetitle


def process_sources(root_dir):
    item2pagetitle = {}
    for _, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            item2pagetitle.update(process_directory(root_dir, dirname))
            
    try:
        output_filepath = os.path.join(root_dir + '/../output/preprocessed.json')
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(item2pagetitle, f, ensure_ascii=False, indent=4)
        print(f"Written to {output_filepath}")
    except Exception as e:
        print(f"Error writing {output_filepath}: {e}")
    