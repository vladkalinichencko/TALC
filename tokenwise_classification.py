from mlx_lm import generate
import mlx.core as mx
import mlx.nn.losses as losses
import json
from sentence_transformers import SentenceTransformer
from typical_data import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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

