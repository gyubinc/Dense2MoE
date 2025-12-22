from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

import os 
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '3'




sub_list = ['high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', \
    'global_facts', 'prehistory', 'world_religions', 'sociology', 'human_sexuality']


#Index(['id', 'question', 'train_answer', 'metric_answer'], dtype='object')

df_id = []
df_question = []
df_train_answer = []
df_metric_answer = []






x = 0
for sub in sub_list:
    ds = load_dataset("cais/mmlu", sub)

    question = ds['test']['question']
    choices = ds['test']['choices']
    answer = ds['test']['answer']
    

    answer_letter = ['A', 'B', 'C', 'D']
    for i in tqdm(range(min(len(ds["test"]), 100))):
        df_id.append(x)
        x += 1
        prompt = question[i]
        prompts = "Question: " + prompt + "\n\nOptions:\nA. " + choices[i][0] + "\nB. " + choices[i][1] + "\nC. " + choices[i][2] + "\nD. " + choices[i][3] + "\n"
        df_question.append(prompts)
        
        metric_answer = answer_letter[answer[i]]
        train_answer = "Answer: " + metric_answer
        df_train_answer.append(train_answer)
        df_metric_answer.append(metric_answer)

        # breakpoint()


# DataFrame 생성
df = pd.DataFrame({'id': df_id, 'question': df_question, 'train_answer': df_train_answer, 'metric_answer': df_metric_answer})
os.makedirs('processed/MMLU', exist_ok=True)
df.to_json('processed/MMLU/MMLU_test.json', orient='records', force_ascii=False, indent=2)
df.to_json('processed/MMLU/MMLU_train.json', orient='records', force_ascii=False, indent=2)