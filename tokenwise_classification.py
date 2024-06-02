from mlx_lm import load, generate, convert
import mlx.core as mx
import mlx.nn.losses as losses
import json
import numpy as np

# data preparation

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

def get_prompts_from_description(filename, description):
    prompts = []
    
    with open(filename, 'r') as file:
        json_data = json.load(file)
        
        for item in json_data:
            if 'description' in item and item['description'].lower() == description.lower():
                prompts = item.get('prompts', [])
            
    return prompts

def tokenize_api_descriptions(descriptions, tokenizer):
    tokenized_descriptions = []
    
    for description in descriptions:
        tokenized_description = tokenizer.encode(description)
        tokenized_descriptions.append(tokenized_description[1:])
    
    return tokenized_descriptions

def description_to_name(description):
    name = description.join(' ', '_')
    
    return name

def generate_tool_prompt(model_name, filename):
    if model_name == 'mistral':
        pass
    elif model_name == 'command r':
        pass

# classification methods

def tokenwise_relationship_classification(model, prompt, relationship_phrase, tokenizer, list_path):
    descriptions = extract_descriptions(list_path)
    
    augmented_prompt = prompt + " " + relationship_phrase + ""
    tokenized_prompt = tokenizer.encode(augmented_prompt)[1:]
    
    tokenized_descriptions = tokenize_api_descriptions(descriptions, tokenizer)
    logits_for_descriptions = []

    for description_index in range(len(tokenized_descriptions)):
        description = tokenized_descriptions[description_index]
        logits_for_descriptions.append([])
        tokenized_api_prompt = tokenized_prompt.copy() # + description
        logits = model(mx.array(tokenized_api_prompt)[None])
        
        for index in range(len(description)):
            token = description[index]

            token_logits = logits[:, -len(description) - 1 + index, :]
            token_logits = token_logits[0]
            logit = token_logits[token].item()
            
            # print("Tokenized description:", description)
            # print("Description:", tokenizer.decode(description))
            # print("Token index:", index)
            # print("Token:", token)
            # print("Symbol:", tokenizer.decode(token))
            # print("Logits:", logits)
            # print("Logit:", logit)
            
            logits_for_descriptions[description_index].append(logit)
            tokenized_api_prompt.append(token)
            
            # print("Logits list:", logits_for_descriptions)
            # print("New tokenized prompt:", tokenized_api_prompt)
            # print("New prompt:", tokenizer.decode(tokenized_api_prompt))
        
        # print("")
    
    final_probabilities = []
    
    for probabilities_list in logits_for_descriptions:
        
        final_probabilities.append(sum(probabilities_list) / len(probabilities_list))

    largest_probability_index = mx.argmax(mx.softmax(mx.array(final_probabilities), axis=-1)).item()
    
    most_probable_api_name = descriptions[largest_probability_index]
    
    # print("Softmax probabilities list:", mx.softmax(mx.array(final_probabilities), axis=-1))
    
    # print("Final probabilities:", final_probabilities)
    # print("Largest probability index:", largest_probability_index)
    # print("Most probable API name:", most_probable_api_name)
    
    return most_probable_api_name

def cosine_similarity_classification(model, prompt, tokenizer, list_path):
    descriptions = extract_descriptions(list_path)
    
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
        
        # print(description)
        # print(sim)
        
        similarities.append(sim)
    
    largest_similarity_index = mx.argmax(mx.softmax(mx.array(similarities), axis=-1)).item()
    
    most_probable_api_name = descriptions[largest_similarity_index]
    
    return most_probable_api_name

# benchmarks

def evaluate_talc(model, relationship_phrase, tokenizer, list_path):
    count = 0
    right = 0
    descriptions = extract_descriptions(list_path)
    
    # print(descriptions)
    
    for description in descriptions:
        prompts = get_prompts_from_description(list_path, description)
        
        for prompt in prompts:
            talc_output = tokenwise_relationship_classification(model, "A user asks: " + prompt, relationship_phrase, tokenizer, list_path)
            
            print("Prompt:", prompt, "Tool:", description, "Output tool:", talc_output)
            
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
            cos_sim_output = cosine_similarity_classification(model, prompt, tokenizer, list_path)
            
            print("Prompt:", prompt, "Tool:", description, "Output tool:", cos_sim_output)
            
            if cos_sim_output == description:
                right += 1
            
            count += 1
    
    final_score = right / count
    
    return final_score

def evaluate_typical_method(model_name, model, prompt, tokenizer, list_path):
    count = 0
    right = 0
    descriptions = extract_descriptions(list_path)
    base_prompt = generate_tool_prompt(model_name, list_path)
    
    for description in descriptions:
        prompts = get_prompts_from_description(list_path, description)
        
        for user_prompt in prompts:
            prompt = base_prompt + user_prompt
            output = generate(model, tokenizer, prompt=prompt, verbose=True)
            
            if description_to_name(description) in output:
                right += 1
            
            count += 1
    
    final_score = right / count
    
    return final_score
