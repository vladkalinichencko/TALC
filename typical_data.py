from mlx_lm import generate
import mlx.core as mx
import mlx.nn.losses as losses
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def trim_description(description):
    words = description.split()
                
    if 'Parameter:' in words:
        trimmed_description = ' '.join(words[:words.index('Parameter:')])
        return trimmed_description
    elif 'Parameters:' in words:
        trimmed_description = ' '.join(words[:words.index('Parameters:')])
        return trimmed_description
    elif 'parameter:' in words:
        trimmed_description = ' '.join(words[:words.index('parameter:')])
        return trimmed_description
    elif 'parameters:' in words:
        trimmed_description = ' '.join(words[:words.index('parameters:')])
        return trimmed_description
    else:
        return description

def extract_descriptions(filename):
    descriptions = []
    
    with open(filename, 'r') as file:
        json_data = json.load(file)
        
        for item in json_data:
            description = item.get('description', '')
            
            if description:
                descriptions.append(trim_description(description))

    # random.shuffle(descriptions)

    return descriptions

def get_prompts_from_description(filename, description):
    prompts = []
    
    with open(filename, 'r') as file:
        json_data = json.load(file)
        
        for item in json_data:
            
            if 'description' in item and trim_description(item['description'].lower()) == trim_description(description.lower()):
                prompts = item.get('prompts', [])
            
    return prompts

def tokenize_api_descriptions(descriptions, tokenizer):
    tokenized_descriptions = []
    
    for description in descriptions:
        tokenized_description = tokenizer.encode(description)
        tokenized_descriptions.append(tokenized_description[1:])
    
    return tokenized_descriptions

def description_to_name(description):
    name = description.replace(' ', '_').lower()
    
    return name



def generate_tool_prompt(model_name, descriptions):
    if model_name == 'mistral':
        tools = []
        
        for description in descriptions:
            name = description_to_name(description)
            
            tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
            tools.append(tool)

        json_tools = "[\n" + ",\n".join([json.dumps(t, indent=2).replace('\n', '\n  ') for t in tools]) + "\n]"
        
        return "[AVAILABLE_TOOLS]\n" + json_tools + "[/AVAILABLE_TOOLS]\n"
    elif model_name == 'command r':
        tools = []
        
        for description in descriptions:
            name = description_to_name(description)
            
            tool = f'''```python
                def {name}() -> List[Dict]:
                    """{description}
                    """
                    pass
                ```\n'''
            tools.append(tool)
        
        return "\<|START_OF_TURN_TOKEN|>\<|SYSTEM_TOKEN|>\n" + tools + "<|END_OF_TURN_TOKEN|>\n"

def find_longest_tokenized_description(tokenized_descriptions):
    max_length = 0

    for description in tokenized_descriptions:
        if len(description) > max_length:
            max_length = len(description)

    return max_length

def evaluate_typical_method(model_name, model, tokenizer, list_path):
    count = 0
    right = 0
    descriptions = extract_descriptions(list_path)
    base_prompt = generate_tool_prompt(model_name, descriptions)
    
    for description in descriptions:
        prompts = get_prompts_from_description(list_path, description)

        for user_prompt in prompts:
            prompt = base_prompt + "[INST]" + "You are a self-driving car AI assistant. You have to output one and only most relevant function out of given to comply with the user's needs. Output strictly only proper function name. \n" + user_prompt + "\n Output one most relevant function name: " + "[/INST]"
            output = generate(model, tokenizer, prompt=prompt, verbose=False)
            
            print('PROMPT:', prompt)
            print('OUTPUT:', output)
            
            if description_to_name(description)[:-1] in output:
                right += 1
                
                print('RIGHT')
            else:
                print('WRONG')
            
            count += 1
    
    final_score = right / count
    
    return final_score

