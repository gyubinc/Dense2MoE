from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# MMLU = 4개 choices만 존재
ds = load_dataset("cais/mmlu", 'all')



subject = ds['test']['subject']


from collections import Counter

# Count the total subject distribution
subject_counts = Counter(subject)
# print(subject_counts)

breakpoint()