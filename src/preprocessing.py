import os
import json
import torch
import os
import json
import paths
from difflib import SequenceMatcher 
from collections import defaultdict


from collections import defaultdict
from difflib import SequenceMatcher

def common_prefix_two_strings(str1, str2):
    matcher = SequenceMatcher(None, str1, str2)
    match = matcher.find_longest_match(0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]

def find_longest_common_string(strings, percentage=0.2):
    if not strings or percentage <= 0:
        return ""

    num_to_consider = int(len(strings) * percentage)
    strings_to_compare = strings[:num_to_consider] 

    if not strings_to_compare:
        return ""

    prefix = strings_to_compare[0]
    for string in strings_to_compare[1:]:
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
        output_filepath = os.path.join(paths.RESULTS_DIR + '/preprocessing/truncated_pagetitles.json')
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(item2pagetitle, f, ensure_ascii=False, indent=4)
        print(f"Written to {output_filepath}")
    except Exception as e:
        print(f"Error writing {output_filepath}: {e}")
    