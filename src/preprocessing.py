import os
import json
import torch
import paths
import utils.string_utils as string_utils
from collections import defaultdict


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
                
    string_utils.remove_longest_common_prefix_suffix(item2pagetitle.values())
    
    for key, value in item2pagetitle.items():
        new_val = string_utils.replace_special_chars_with_whitespace(value)
        new_val = string_utils.remove_extra_whitespaces(new_val)

        item2pagetitle[key] = new_val
        
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
    