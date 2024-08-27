import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

def tokenize_text(text):
    return word_tokenize(text)

def get_embeddings(tokens, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    
    text = ' '.join(tokens)
    
    embedding = model.encode([text])[0]
    
    return embedding

text = "This is a sample sentence for tokenization and embedding."
tokens = tokenize_text(text)
embedding = get_embeddings(tokens)

print(f"Tokens: {tokens}")
print(f"Embedding shape: {embedding.shape}")