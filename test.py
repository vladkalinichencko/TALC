from mlx_lm import load, generate, convert
import mlx_lm.utils as utils
from pathlib import Path
from huggingface_hub import snapshot_download, login
from token_file import HF_TOKEN
from tokenwise_classification import *

login(token=HF_TOKEN)

repo_name = "mistralai/Mistral-7B-v0.3"

model, tokenizer = load(repo_name)

print(evaluate_talc(model, "Therefore, you have to", tokenizer, "tools/music_app_new.json"))