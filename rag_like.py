from mlx_lm import generate
import mlx.core as mx
import mlx.nn.losses as losses
import json
from sentence_transformers import SentenceTransformer
from typical_data import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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