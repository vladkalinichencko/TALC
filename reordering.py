import json
# import nltk
from nltk.tokenize import word_tokenize

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
# nltk.download('punkt')

def value_function(path):
    print(path)
    
    return len(path)

def extract_descriptions(filename):
    descriptions = []
    
    with open(filename, 'r') as file:
        json_data = json.load(file)
        
        for item in json_data:
            description = item.get('description', '')
            
            if description:
                words = description.split()
                
                if 'Parameter:' in words:
                    trimmed_description = ' '.join(words[:words.index('Parameter:')])
                    descriptions.append(trimmed_description)
                else:
                    descriptions.append(description)

    return descriptions

# function which inserts array into dictionary without duplication
def recursive_insert(sub_tree, array_part):
    if len(array_part) == 0:
        return sub_tree
    
    current_item = array_part[0]
    
    if not current_item in sub_tree.keys():
        new_sub_tree = {current_item: {}}
        sub_tree.update(new_sub_tree)
        
    return recursive_insert(sub_tree[current_item], array_part[1:])

def traverse_tree(tree, value_function):
    def depth_first_search(sub_tree, current_path):
        if not sub_tree:
            return current_path

        max_value = -float('inf')
        next_vertex = None
        next_path = None

        for key, subtree in sub_tree.items():
            new_path = current_path + [key]
            value = value_function(new_path)

            if value > max_value:
                max_value = value
                next_vertex = key
                next_path = new_path

        if next_vertex is None:
            return current_path

        return depth_first_search(sub_tree[next_vertex], next_path)

    return depth_first_search(tree, [])

descs = extract_descriptions('tools/settings_app.json')
# print(descs)

# use more appropriate tokenizer instead 
descs_tokenized = []
for d in descs:
    descs_tokenized.append(word_tokenize(d.lower()))

# print(descs_tokenized)

descs_tree = {}
for d in descs_tokenized:
    recursive_insert(descs_tree, d)

path = traverse_tree(descs_tree, value_function)
print(path)
    
# print(descs_tree)