from mlx_lm import generate
import mlx.core as mx
import mlx.nn.losses as losses
import json
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# data preparation

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

# classification methods

def legacy_tokenwise_relationship_classification(model, prompt, relationship_phrase, tokenizer, list_path):
    descriptions = extract_descriptions(list_path)
    
    augmented_prompt = "A user asks to " + prompt + " " + relationship_phrase + ""
    tokenized_prompt = tokenizer.encode(augmented_prompt)
    
    tokenized_descriptions = tokenize_api_descriptions(descriptions, tokenizer)
    logits_for_descriptions = []

    for description_index in range(len(tokenized_descriptions)):
        description = tokenized_descriptions[description_index]
        logits_for_descriptions.append([])
        tokenized_api_prompt = tokenized_prompt.copy() + description
        logits = model(mx.array(tokenized_api_prompt)[None])
        
        logits_to_print = []
        norm_logits_to_print = []
        
        for index in range(len(description)):
            token = description[index]

            token_logits = logits[:, -len(description) - 1 + index, :]
            token_logits = token_logits[0]
            
            token_logits = mx.softmax(token_logits)
            
            logits_to_print.append(token_logits[token].item())
            
            token_logits = mx.log(token_logits)
            
            norm_logits_to_print.append(token_logits[token].item())
            
            logit = token_logits[token].item()
            
            logits_for_descriptions[description_index].append(logit)
 
            # print('token', token)
            # print('logit', logit)
            # print('logits for token', token_logits)
            # print(max(token_logits))
            # print(min(token_logits))
            # print(sum(token_logits))
            # print('')
            
            # print("Tokenized description:", description)
            # print("Description:", tokenizer.decode(description))
            # print("Token index:", index)
            # print("Token:", token)
            # print("Symbol:", tokenizer.decode(token))
            # print("Logits:", logits)
            # print("Logit:", logit)
            # print("Logits list:", logits_for_descriptions)
            # print("New tokenized prompt:", tokenized_api_prompt)
            # print("New prompt:", tokenizer.decode(tokenized_api_prompt))
        
        # print('')
        # print('    ', tokenizer.decode(description))
        # print('    ', *logits_to_print)
        # print('    ', *norm_logits_to_print)
        # print('    ', reduce(operator.mul, logits_to_print, 1))
        # print('    ', mx.log(reduce(operator.mul, logits_to_print, 1)))
        # print('    ', sum(norm_logits_to_print))
    
    final_probabilities = []
    
    for probabilities_list in logits_for_descriptions:
        final_probabilities.append(sum(probabilities_list))

    largest_probability_index = mx.argmax(mx.array(final_probabilities)).item()
    
    most_probable_api_name = descriptions[largest_probability_index]
    
    # print("Softmax probabilities list:", mx.softmax(mx.array(final_probabilities)))
    # print("Final probabilities:", final_probabilities)
    # print("Largest probability index:", largest_probability_index)
    # print("Most probable API name:", most_probable_api_name)
    print(max(mx.array(final_probabilities)))
    
    return most_probable_api_name

def get_top_similar(probabilities, similarity_threshold):
    sorted_probs_with_indices = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    top_prob = sorted_probs_with_indices[0][1]
    
    similar_probs = [(prob, index) for index, prob in sorted_probs_with_indices if prob >= top_prob * similarity_threshold]
    top_probabilities, top_indices = zip(*similar_probs)
    
    return list(top_probabilities), list(top_indices)


def right_tokenwise_relationship_classification(model, prompt, relationship_phrase, tokenizer, list_path):
    descriptions = extract_descriptions(list_path)
    
    augmented_prompt = "A user asks to " + prompt + " " + relationship_phrase + " "
    tokenized_prompt = tokenizer.encode(augmented_prompt)
    
    tokenized_descriptions = tokenize_api_descriptions(descriptions, tokenizer)
    
    logits_for_position = []
    for i in tokenized_descriptions:
        logits_for_position.append((tokenizer.decode(i), 0, True))
    
    prob_threshold = 0
    
    for position in range(find_longest_tokenized_description(tokenized_descriptions)):
        for index in range(len(tokenized_descriptions)):
            description = tokenized_descriptions[index]
            tokenized_api_prompt = tokenized_prompt.copy() + description
            logits = model(mx.array(tokenized_api_prompt)[None])[0]
            
            token = description[min(position, len(description) - 1)]
            
            token_logits = logits[(-len(description) + position):]
            token_logits = token_logits[0]
            token_logits = mx.softmax(token_logits)
            
            print(mx.argmax(token_logits))
            
            token_logits = mx.log(token_logits)
            logit = token_logits[token].item()
            
            prev_prob = logits_for_position[index][1]
            new_prob = (prev_prob + logit) / (position + 1)
            b = logits_for_position[index][2]
            
            logits_for_position[index] = (tokenizer.decode(description), new_prob, b)
        
        filtered_logits = [x for x in logits_for_position if x[2]]
        sorted_filtered_logits = sorted(filtered_logits, key=lambda x: x[1], reverse=True)
        
        prob_threshold = sorted_filtered_logits[len(sorted_filtered_logits)//2][1]
        
        updated_logits_for_position = []
        for i in logits_for_position:
            decoded_description, prob, flag = i
            if prob < prob_threshold:
                new_tuple = (decoded_description, prob, False)
            else:
                new_tuple = i
            updated_logits_for_position.append(new_tuple)
        logits_for_position = updated_logits_for_position
        
        print(position, 'position')
        print(*sorted_filtered_logits)
        print('')
    
    print('')
    sorted_filtered_logits = sorted(logits_for_position, key=lambda x: x[1], reverse=True)
    print(*sorted_filtered_logits)
    print('')
    
    return 'Finished!'

def legacy_cosine_similarity_classification(model, prompt, tokenizer, list_path):
    descriptions = extract_descriptions(list_path)
    
    for i in range(len(descriptions)):
        descriptions[i] = "A user asks to " + descriptions[i]
    
    tokenized_prompt = tokenizer.encode(prompt)[1:]
    tokenized_descriptions = tokenize_api_descriptions(descriptions, tokenizer)
    
    prompt_embedding = model(mx.array(tokenized_prompt)[None])
    prompt_embedding = prompt_embedding[0].mean(axis=0)
    prompt_embedding = prompt_embedding.reshape(1, -1)
    
    similarities = []
    
    for description in tokenized_descriptions:
        description_embedding = model(mx.array(description)[None])
        description_embedding = description_embedding[0].mean(axis=0)
        description_embedding = description_embedding.reshape(1, -1)
        
        sim = losses.cosine_similarity_loss(prompt_embedding, description_embedding)[0]
        similarities.append(sim)
    
    largest_similarity_index = mx.argmax(mx.array(similarities)).item()
    
    most_probable_api_name = descriptions[largest_similarity_index]
    
    print(max(mx.array(similarities)))
    
    return most_probable_api_name

def right_cosine_similarity_classification(model, prompt, list_path):
    descriptions = extract_descriptions(list_path)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = embedding_model.tokenizer
    
    for i in range(len(descriptions)):
        descriptions[i] = "A user asks to " + descriptions[i]
    
    tokenized_prompt = tokenizer.tokenize(prompt)
    
    tokenized_descriptions = []
    
    for description in descriptions:
        tokenized_description = tokenizer.tokenize(description)
        tokenized_descriptions.append(tokenized_description)
        
    prompt_embedding = embedding_model.encode(tokenized_prompt)
    prompt_embedding = np.mean(prompt_embedding, axis=0)
    
    similarities = []
    
    # print(*prompt_embedding)
    
    for description in tokenized_descriptions:
        description_embedding = embedding_model.encode(description)
        description_embedding = np.mean(description_embedding, axis=0)
        
        # print('    ', description_embedding)
        
        sim = cosine_similarity([prompt_embedding], [description_embedding])
        similarities.append(sim[0][0])
        
        # print('    ', sim)
        # print('')
    
    largest_similarity_index = np.argmax(np.array(similarities)).item()
    most_probable_api_name = descriptions[largest_similarity_index]
    
    # print(max(mx.array(similarities)))
    
    return most_probable_api_name

# benchmarks

def evaluate_talc(model, relationship_phrase, tokenizer, list_path):
    count = 0
    right = 0
    descriptions = extract_descriptions(list_path)
        
    for description in descriptions:
        prompts = get_prompts_from_description(list_path, description)
        
        for prompt in prompts:
            talc_output = right_tokenwise_relationship_classification(model, prompt, relationship_phrase, tokenizer, list_path)
            
            print("Prompt:", "A user asks: " + prompt, "Tool:", description, "Output tool:", talc_output)
            
            if talc_output == description:
                right += 1
            
            count += 1
    
    final_score = right / count
    
    return final_score

def evaluate_cos_sim(model, tokenizer, list_path):
    count = 0
    right = 0
    descriptions = extract_descriptions(list_path)
    
    for description in descriptions:
        prompts = get_prompts_from_description(list_path, description)
        
        for prompt in prompts:
            cos_sim_output = right_cosine_similarity_classification(model, "A user asks: " + prompt, list_path)
            
            print("Prompt:", "A user asks: " + prompt, "Tool:", description, "Output tool:", cos_sim_output)
            
            if cos_sim_output == "A user asks to " + description:
                right += 1
            
            count += 1
    
    final_score = right / count
    
    return final_score

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
