from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

instruction_template = '''Answer the following multiple-choice question.
    Choose only one option: A, B, C, or D.
    Your response MUST follow this format:

    Answer: [letter]
    Explanation: [your reasoning here]

    Example:
    Answer: C
    Explanation: Because ...

    Now, here is the question:\n\n{question}'''

# MMLU = 4개 choices만 존재
sub = 'high_school_biology'

sub_list = ['high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', \
    'global_facts', 'prehistory', 'world_religions', 'sociology', 'human_sexuality']

accuracy_list = []
no_alphabet_list = []
another_list = []
for sub in sub_list:
    ds = load_dataset("cais/mmlu", sub)

    question = ds['test']['question']
    subject = ds['test']['subject']
    choices = ds['test']['choices']
    answer = ds['test']['answer']
    
    correct = 0
    no_alphabet = 0
    another = 0
    answer_letter = ['A', 'B', 'C', 'D']
    for i in tqdm(range(min(len(ds["test"]), 100))):

        prompt = question[i]
        prompt = "Question: " + prompt + "\n\nOptions:\nA. " + choices[i][0] + "\nB. " + choices[i][1] + "\nC. " + choices[i][2] + "\nD. " + choices[i][3] + "\n\nAnswer: "
        prompt = instruction_template.format(question=prompt)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        output = model.generate(input_ids, max_new_tokens=10, num_return_sequences=1)
        pred = tokenizer.decode(output[0][len(input_ids[0]):]).strip()

        breakpoint()

        if pred[0] == answer_letter[answer[i]]:
            correct += 1
        if pred[0] not in answer_letter:
            no_alphabet += 1
        if choices[i][answer[i]] in pred:
            another += 1
    
    accuracy_list.append(correct / 100)
    no_alphabet_list.append(no_alphabet / 100)
    another_list.append(another / 100)
print('--------------------------------')
print('The Accuracy of MMLU is')
for sub, accuracy in zip(sub_list, accuracy_list):
    print(f'The Accuracy of {sub} is {accuracy}')
print('--------------------------------')
# no_alphabet_list
print('The Accuracy of no_alphabet is')
for sub, accuracy in zip(sub_list, no_alphabet_list):
    print(f'The no_alphabet of {sub} is {accuracy}')
print('--------------------------------')
print('The Accuracy of another is')
for sub, accuracy in zip(sub_list, another_list):
    print(f'The another of {sub} is {accuracy}')
print('--------------------------------')
for i in range(len(accuracy_list)):
    print("The accuracy of alphabet and another is", accuracy_list[i] + another_list[i])




# 또는
# ds = load_dataset("tasksource/mmlu")