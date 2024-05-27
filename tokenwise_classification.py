import mlx.core as mx
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_descriptions(filename):
    descriptions = []
    
    with open(filename, 'r') as file:
        json_data = json.load(file)
        
        for item in json_data:
            description = item.get('description', '')
            if description:
                descriptions.append(description)

    return descriptions

def get_prompts_by_description_from_file(filename, description):
    prompts = []
    
    with open(filename, 'r') as file:
        json_data = json.load(file)
        
        for item in json_data:
            if 'description' in item and item['description'].lower() == description.lower():
                prompts = item.get('prompts', [])
            
    return prompts

def tokenize_api_descriptions(api_list, tokenizer):
    descriptions = []
    for api in api_list:
        description = api["API_call"].get("description", "")
        descriptions.append(description)
        
    print("")
    
    tokenized_descriptions = []
    for description in descriptions:
        tokenized_description = tokenizer.encode(description)
        tokenized_descriptions.append(tokenized_description[1:])
    
    return tokenized_descriptions

def tokenwise_relationship_classification(model, prompt, relationship_phrase, tokenizer, list_path):
    api_list = extract_descriptions(list_path)
    
    augmented_prompt = prompt + " " + relationship_phrase + " "
    tokenized_prompt = tokenizer.encode(augmented_prompt)
    
    tokenized_descriptions = tokenize_api_descriptions(api_list, tokenizer)
    logits_for_descriptions = []
    
    for description_index in range(len(tokenized_descriptions)):
        description = tokenized_descriptions[description_index]
        logits_for_descriptions.append([])
        tokenized_api_prompt = tokenized_prompt.copy() + description
        logits, _ = model(mx.array(tokenized_api_prompt)[None])
        
        for index in range(len(description)):
            token = description[index]

            token_logits = logits[:, -len(description) - 1 + index, :]
            token_logits = token_logits[0]
            logit = token_logits[token].item()
            
            # print("Tokenized description:", description)
            # print("Token index:", index)
            # print("Token:", token)
            # print("Logits:", logits)
            # print("Logit:", logit)
            
            logits_for_descriptions[description_index].append(logit)
            tokenized_api_prompt.append(token)
            
            # print("Logits list:", logits_for_descriptions)
            # print("New prompt:", tokenized_api_prompt)
        
        print("")
    
    final_probabilities = []
    
    for probabilities_list in logits_for_descriptions:
        final_probabilities.append(sum(probabilities_list) / len(probabilities_list))

    largest_probability_index = mx.argmax(mx.softmax(mx.array(final_probabilities), axis=-1)).item()
    
    most_probable_api_name = api_list[largest_probability_index]
    
    # print("Softmax probabilities list:", mx.softmax(mx.array(final_probabilities), axis=-1))
    
    # print("Final probabilities:", final_probabilities)
    # print("Largest probability index:", largest_probability_index)
    # print("Most probable API name:", most_probable_api_name)
    
    return most_probable_api_name

def cosine_similarity_classification(model, prompt, tokenizer, list_path):
    api_list = extract_descriptions(list_path)
    
    tokenized_prompt = tokenizer.encode(prompt)
    tokenized_descriptions = tokenize_api_descriptions(api_list, tokenizer)
    
    prompt_embedding, _ = model(mx.array(tokenized_prompt)[None])
    prompt_embedding = prompt_embedding[0].mean(axis=0)
    
    similarities = []
    
    for description in tokenized_descriptions:
        description_embedding, _ = model(mx.array(description)[None])
        description_embedding = description_embedding[0].mean(axis=0)  # Mean pooling
        
        sim = cosine_similarity([prompt_embedding], [description_embedding])[0][0]
        similarities.append(sim)
    
    return similarities

# benchmark
def evaluate_talc(results, description):
    for prompt, scores in results.items():
        most_probable_description = max(scores, key=scores.get)
        print(f"Prompt: {prompt}")
        print(f"Most Probable API Description: {most_probable_description}")
        print(f"Actual Description: {description}")
        print(f"Match: {most_probable_description == description}\n")

# def trivial_llm_classification(api_list, prompts):
#     # Assuming you have a local LLM server or using a Hugging Face pipeline
#     classifier = pipeline("text-classification", model="distilbert-base-uncased")  # Replace with actual model if needed
#     results = {}
    
#     for prompt in prompts:
#         results[prompt] = []
#         for api in api_list:
#             api_name = api["API_call"].get("name", "")
#             api_desc = api["API_call"].get("description", "")
#             full_prompt = f"Does the following prompt: '{prompt}' match any of the API functions? Reply with only the function's name."
#             result = classifier(full_prompt)
#             results[prompt].append((api_name, result))
    
#     return results

# def main():
#     file_path = '/mnt/data/input.json'  # Path to the JSON file
#     list_path = '/mnt/data/api_list.jsonl'  # Path to the JSONL file containing API descriptions
#     data = read_json_file(file_path)
    
#     description = data["description"]
#     prompts = data["prompts"]
    
#     results = classify_prompts(model, tokenizer, description, prompts, list_path)
    
#     evaluate_results(results, description)
