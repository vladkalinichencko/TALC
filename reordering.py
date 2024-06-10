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

descs = extract_descriptions('tools/settings_app.json')
print(descs)

# use more appropriate tokenizer instead 
descs_tokenized = []
for d in descs:
    descs_tokenized.append(word_tokenize(d.lower()))

# print(descs_tokenized)

descs_tree = {}
for d in descs_tokenized:
    recursive_insert(descs_tree, d)
    
print(descs_tree)