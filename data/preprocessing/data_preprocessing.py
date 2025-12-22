#!/usr/bin/env python3
"""Data Preprocessing Script - Converted from data_preprocessing.ipynb"""

from datasets import load_dataset

# 1) 기본 load
ds = load_dataset("openlifescienceai/medmcqa")

# 2) 특정 split만 불러오기
ds_train = load_dataset("openlifescienceai/medmcqa", split="train")
ds_test = load_dataset("openlifescienceai/medmcqa", split="validation")  # 혹은 "test" 여부 확인

# 3) 컬럼들을 미리 확인하기
print(ds_train.column_names)
# 예: ['id', 'question', 'opa', 'opb', 'opc', 'opd', 'cop', 'choice_type', 'exp', 'subject_name', 'topic_name']

# # 4) parquet 파일 직접 경로로 불러오기 (필요 시)
# # 만약 로컬 혹은 원격 parquet 파일로 접근 가능하다면 다음과 같이:
# ds_local = load_dataset(
#     "parquet",
#     data_files={
#         "train": "path/to/train-00000-of-00001.parquet",
#         "validation": "path/to/validation-00000-of-00001.parquet",
#         "test": "path/to/test-00000-of-00001.parquet",
#     }
# )


single_list = []
for i in range(len(ds_train)):
    if ds_train[i]['choice_type'] != 'single':
        single_list.append(ds_train[i])
len(single_list)
single_list[0]


x_single_list = []

for i in range(len(single_list)):
    if single_list[i]['exp'] != None:
        x_single_list.append(single_list[i])

print(len(single_list))
print(len(x_single_list))

import json
import os 

single_list = x_single_list
df_train = []
for i in range(0, 20000):
    question = "Question: " + single_list[i]['question'] + "\n\nOptions:\nA. " + single_list[i]['opa'] + "\nB. " + single_list[i]['opb'] + "\nC. " + single_list[i]['opc'] + "\nD. " + single_list[i]['opd'] + '\n'
    if single_list[i]['cop'] == 0:
        ans = 'A. ' + single_list[i]['opa']
    elif single_list[i]['cop'] == 1:
        ans = 'B. ' + single_list[i]['opb']
    elif single_list[i]['cop'] == 2:
        ans = 'C. ' + single_list[i]['opc']
    else:
        ans = 'D. ' + single_list[i]['opd']
    train_answer = "Answer: " + ans + "\n\nExplanation: " + single_list[i]['exp']
    metric_answer = ans
    df_train.append([question, train_answer, metric_answer])

df_test = []
for i in range(20000, 21000):
    question = "Question: " + single_list[i]['question'] + "\n\nOptions:\nA. " + single_list[i]['opa'] + "\nB. " + single_list[i]['opb'] + "\nC. " + single_list[i]['opc'] + "\nD. " + single_list[i]['opd'] + '\n'
    if single_list[i]['cop'] == 0:
        ans = 'A. ' + single_list[i]['opa']
    elif single_list[i]['cop'] == 1:
        ans = 'B. ' + single_list[i]['opb']
    elif single_list[i]['cop'] == 2:
        ans = 'C. ' + single_list[i]['opc']
    else:
        ans = 'D. ' + single_list[i]['opd']
    train_answer = "Answer: " + ans + "\n\nExplanation: " + single_list[i]['exp']
    metric_answer = ans
    df_test.append([question, train_answer, metric_answer])

os.makedirs('./medical', exist_ok=True)

# medical_train.json 저장 (2차원 리스트)
with open('./medical/medical_train.json', 'w', encoding='utf-8') as f:
    json.dump(df_train, f, ensure_ascii=False, indent=2)

# medical_test.json 저장 (2차원 리스트)
with open('./medical/medical_test.json', 'w', encoding='utf-8') as f:
    json.dump(df_test, f, ensure_ascii=False, indent=2)    

# 1. MEDICAL 도메인 수정 (이미 로드된 데이터 사용)
print("=== MEDICAL 도메인 구조 수정 ===")

# medical_train.json 수정
with open('./medical/medical_train.json', 'r', encoding='utf-8') as f:
    medical_train_list = json.load(f)

medical_train_dict = []
for i, item in enumerate(medical_train_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"medical_train_{i}",
            "question": item[0],
            "train_answer": item[1], 
            "metric_answer": item[2]
        }
        medical_train_dict.append(dict_item)

# medical_test.json 수정
with open('./medical/medical_test.json', 'r', encoding='utf-8') as f:
    medical_test_list = json.load(f)

medical_test_dict = []
for i, item in enumerate(medical_test_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"medical_test_{i}",
            "question": item[0],
            "train_answer": item[1],
            "metric_answer": item[2]
        }
        medical_test_dict.append(dict_item)

# 수정된 파일 저장
with open('./processed/medical/medical_train.json', 'w', encoding='utf-8') as f:
    json.dump(medical_train_dict, f, ensure_ascii=False, indent=2)

with open('./processed/medical/medical_test.json', 'w', encoding='utf-8') as f:
    json.dump(medical_test_dict, f, ensure_ascii=False, indent=2)

print(f"Medical train: {len(medical_train_dict)} samples")
print(f"Medical test: {len(medical_test_dict)} samples")



df_train[0]

--------------------------------
Answer: C. Atrophy

Explanation: Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine Refer Robbins 7yh/9,1012,9/e. P950
--------------------------------
C. Atrophy
--------------------------------

# load each domain data
import json
import pandas as pd



medical_dir = "../../raw_data/medical/medmcqa_train.json"
medical_test_dir = "../../raw_data/medical/medmcqa_test.json"
math_dir = "../../raw_data/math/mathqa_train.json"
math_test_dir = "../../raw_data/math/mathqa_test.json"
law_dir = "../../raw_data/law/case_hold_train.json"
law_test_dir = "../../raw_data/law/case_hold_test.json"
code_dir = "../../raw_data/code/coding_mcq_reasoning_train.jsonl"


medical_df = pd.read_json(medical_dir)
medical_test_df = pd.read_json(medical_test_dir)
math_df = pd.read_json(math_dir)
math_test_df = pd.read_json(math_test_dir)
law_df = pd.read_json(law_dir)
law_test_df = pd.read_json(law_test_dir)
# read jsonl file
code_df = pd.read_json(code_dir, lines=True)

print(len(medical_df))
print(len(medical_test_df))
print(len(math_df))
print(len(math_test_df))
print(len(law_df))
print(len(law_test_df))
print(len(code_df))


medical_df.iloc[0]['formatted_question']

i=0
question = medical_df.iloc[i]['formatted_question'][:-8] + ' '   
train_answer = "Answer: " + medical_df.iloc[i]['formatted_answer'] + "\n\n" + "Explanation: " + medical_df.iloc[i]['explanation']
metric_answer = medical_df.iloc[i]['formatted_answer']


print('--------------------------------')
print(question)
print('--------------------------------')
print(train_answer)
print('--------------------------------')
print(metric_answer)
print('--------------------------------')

# medical_df에서 앞에서부터 2만개 뽑아서 question, train_answer, metric_answer 추출
# medical_test_df에서 앞에서부터 1000개 뽑아서 question, train_answer, metric_answer 추출

# 각각 ./medical폴더에 medical_train.json, medical_test.json 파일 생성
# 각 파일은 2차원 리스트로 저장
# 예시 : [["question", "train_answer", "metric_answer"], ["question", "train_answer", "metric_answer"], ...]

import os
import json

# medical_df에서 앞에서부터 2만개 뽑아서 question, train_answer, metric_answer 추출
medical_train_samples = []
for i in range(min(20000, len(medical_df))):
    question = medical_df.iloc[i]['formatted_question'][:-8] + ' '
    train_answer = "Answer: " + str(medical_df.iloc[i]['formatted_answer']) + "\n\n" + "Explanation: " + str(medical_df.iloc[i]['explanation'])
    metric_answer = medical_df.iloc[i]['formatted_answer']
    medical_train_samples.append([question, train_answer, metric_answer])


medical_test_samples = []
for i in range(20000, 21000):
    question = medical_df.iloc[i]['formatted_question'][:-8] + ' '
    train_answer = "Answer: " + str(medical_df.iloc[i]['formatted_answer']) + "\n\n" + "Explanation: " + str(medical_df.iloc[i]['explanation'])
    metric_answer = medical_df.iloc[i]['formatted_answer']
    medical_test_samples.append([question, train_answer, metric_answer])

# ./medical 폴더 생성 (없으면)
os.makedirs('./medical', exist_ok=True)

# medical_train.json 저장 (2차원 리스트)
with open('./medical/medical_train.json', 'w', encoding='utf-8') as f:
    json.dump(medical_train_samples, f, ensure_ascii=False, indent=2)

# medical_test.json 저장 (2차원 리스트)
with open('./medical/medical_test.json', 'w', encoding='utf-8') as f:
    json.dump(medical_test_samples, f, ensure_ascii=False, indent=2)

# 1. MEDICAL 도메인 수정 (이미 로드된 데이터 사용)
print("=== MEDICAL 도메인 구조 수정 ===")

# medical_train.json 수정
with open('./medical/medical_train.json', 'r', encoding='utf-8') as f:
    medical_train_list = json.load(f)

medical_train_dict = []
for i, item in enumerate(medical_train_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"medical_train_{i}",
            "question": item[0],
            "train_answer": item[1], 
            "metric_answer": item[2]
        }
        medical_train_dict.append(dict_item)

# medical_test.json 수정
with open('./medical/medical_test.json', 'r', encoding='utf-8') as f:
    medical_test_list = json.load(f)

medical_test_dict = []
for i, item in enumerate(medical_test_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"medical_test_{i}",
            "question": item[0],
            "train_answer": item[1],
            "metric_answer": item[2]
        }
        medical_test_dict.append(dict_item)

# 수정된 파일 저장
with open('./processed/medical/medical_train.json', 'w', encoding='utf-8') as f:
    json.dump(medical_train_dict, f, ensure_ascii=False, indent=2)

with open('./processed/medical/medical_test.json', 'w', encoding='utf-8') as f:
    json.dump(medical_test_dict, f, ensure_ascii=False, indent=2)

print(f"Medical train: {len(medical_train_dict)} samples")
print(f"Medical test: {len(medical_test_dict)} samples")

medical_df.columns

medical_train_samples[0]

math_df.columns

len(math_test_df)

i=0
question = math_df.iloc[i]['formatted_question'] + ' '   
train_answer = math_df.iloc[i]['formatted_answer'] + "\n\n" + "Explanation: " + medical_df.iloc[i]['explanation']
metric_answer = math_df.iloc[i]['formatted_answer']

# math_df에서 앞에서부터 2만개 뽑아서 question, train_answer, metric_answer 추출
# math_test_df에서 앞에서부터 1000개 뽑아서 question, train_answer, metric_answer 추출

# 각각 ./math 폴더에 math_train.json, math_test.json 파일 생성
# 각 파일은 2차원 리스트로 저장
# 예시 : [["question", "train_answer", "metric_answer"], ["question", "train_answer", "metric_answer"], ...]

import os
import json

# math_df에서 앞에서부터 2만개 뽑아서 question, train_answer, metric_answer, id 추출
math_train_samples = []
for i in range(min(20000, len(math_df))):
    question = math_df.iloc[i]['formatted_question']
    train_answer = ' ' + math_df.iloc[i]['formatted_answer'] + "\n\n" + "Explanation: " + str(math_df.iloc[i]['explanation'])
    metric_answer = math_df.iloc[i]['formatted_answer']
    sample = {
        "id": math_df.iloc[i]['id'],
        "question": question,
        "train_answer": train_answer,
        "metric_answer": metric_answer
    }
    math_train_samples.append(sample)

# math_test_df에서 앞에서부터 1000개 뽑아서 question, train_answer, metric_answer, id 추출
math_test_samples = []
for i in range(min(1000, len(math_test_df))):
    question = math_test_df.iloc[i]['formatted_question']
    train_answer = ' ' + math_test_df.iloc[i]['formatted_answer'] + "\n\n" + "Explanation: " + str(math_test_df.iloc[i]['explanation'])
    metric_answer = math_test_df.iloc[i]['formatted_answer']
    sample = {
        "id": math_test_df.iloc[i]['id'],
        "question": question,
        "train_answer": train_answer,
        "metric_answer": metric_answer
    }
    math_test_samples.append(sample)

# ./math 폴더 생성 (없으면)
os.makedirs('./math', exist_ok=True)

# math_train.json 저장
with open('./math/math_train.json', 'w', encoding='utf-8') as f:
    json.dump(math_train_samples, f, ensure_ascii=False, indent=2)

# math_test.json 저장
with open('./math/math_test.json', 'w', encoding='utf-8') as f:
    json.dump(math_test_samples, f, ensure_ascii=False, indent=2)


# 2. MATH 도메인 수정
print("=== MATH 도메인 구조 수정 ===")

# math_train.json 수정
with open('./processed/math/math_train.json', 'r', encoding='utf-8') as f:
    math_train_list = json.load(f)

math_train_dict = []
for i, item in enumerate(math_train_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"math_train_{i}",
            "question": item[0],
            "train_answer": item[1],
            "metric_answer": item[2]
        }
        math_train_dict.append(dict_item)

# math_test.json 수정
with open('./processed/math/math_test.json', 'r', encoding='utf-8') as f:
    math_test_list = json.load(f)

math_test_dict = []
for i, item in enumerate(math_test_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"math_test_{i}",
            "question": item[0],
            "train_answer": item[1],
            "metric_answer": item[2]
        }
        math_test_dict.append(dict_item)

# 수정된 파일 저장
with open('./processed/math/math_train.json', 'w', encoding='utf-8') as f:
    json.dump(math_train_dict, f, ensure_ascii=False, indent=2)

with open('./processed/math/math_test.json', 'w', encoding='utf-8') as f:
    json.dump(math_test_dict, f, ensure_ascii=False, indent=2)

print(f"Math train: {len(math_train_dict)} samples")
print(f"Math test: {len(math_test_dict)} samples")

law_df.columns

law_df['formatted_question'][0]

import os
import json

law_train_samples = []
law_test_samples = []

labels = ["A", "B", "C", "D", "E"]  # 5지선다 고정

# Train 샘플 생성
for i in range(min(20000, len(law_df))):
    row = law_df.iloc[i]
    question = row['formatted_question'] + ' \n\nOptions:\n' + \
               "".join(f"{labels[j]}. {opt}\1n" for j, opt in enumerate(row['endings']))
    train_answer = f"Answer: {labels[row['correct_ending_idx']]}. {row['correct_ending']}\n\n" + \
                   f"Explanation: According to the case, {row['correct_ending']}."
    metric_answer = f"{labels[row['correct_ending_idx']]}. {row['correct_ending']}"

    sample = {
        "id": i + 1,
        "question": question,
        "train_answer": train_answer,
        "metric_answer": metric_answer
    }
    law_train_samples.append(sample)

# Test 샘플 생성 (앞에서부터 1000개)
for i in range(min(1000, len(law_test_df))):
    row = law_test_df.iloc[i]
    question = row['formatted_question'] + ' \n\nOptions:\n' + \
               "".join(f"{labels[j]}. {opt}\n" for j, opt in enumerate(row['endings']))
    train_answer = f"Answer: {labels[row['correct_ending_idx']]}. {row['correct_ending']}\n\n" + \
                   f"Explanation: According to the case, {row['correct_ending']}."
    metric_answer = f"{labels[row['correct_ending_idx']]}. {row['correct_ending']}"

    sample = {
        "id": i + 1,
        "question": question,
        "train_answer": train_answer,
        "metric_answer": metric_answer
    }
    law_test_samples.append(sample)

print(len(law_train_samples))  # 20000
print(law_train_samples[0])    # 첫 번째 샘플 확인
print(len(law_test_samples))   # 1000
print(law_test_samples[0])     # 첫 번째 테스트 샘플 확인

# ./law 폴더 생성 (없으면)
os.makedirs('./law', exist_ok=True)

# law_train.json 저장
with open('./law/law_train.json', 'w', encoding='utf-8') as f:
    json.dump(law_train_samples, f, ensure_ascii=False, indent=2)

# law_test.json 저장
with open('./law/law_test.json', 'w', encoding='utf-8') as f:
    json.dump(law_test_samples, f, ensure_ascii=False, indent=2)


len(law_df.iloc[0]['endings'])

len(law_df)

import json
import pandas as pd
code_dir = "./code/codeqa_train.json"

code_df = pd.read_json(code_dir)

for id in code_df.columns:
    print("columns: ", id)
    print(code_df.iloc[0][id])
    print("--------------------------------")


# 1) 라이브러리 설치 (처음 한 번만)
# pip install -U datasets

from datasets import load_dataset

# 2) 로드 (스플릿은 train 하나만 제공)
ds = load_dataset("tuandunghcmut/coding-mcq-reasoning")
print(ds)               # DatasetDict({'train': Dataset})
print(ds["train"][0])   # 샘플 확인

# 3) 필요하면 로컬로 저장 (jsonl/parquet 등)
ds["train"].to_json("coding_mcq_reasoning_train.jsonl", lines=True, force_ascii=False)
# 또는
ds["train"].to_parquet("coding_mcq_reasoning_train.parquet")


for i in range(len(ds['train'])):
    if (ds['train'][i]['choices'] == ds['train'][i]['list_choices']):
        print(1)

for key in ds['train'][0].keys():
    print(key)
    print(ds['train'][0][key])
    print("--------------------------------")


ds['train'][0].keys()

# 이미 로컬에 저장된 code, medical 데이터를 바로 불러옵니다.

import json

code_dir = "/data/disk5/internship_disk/gyubin/raw_data/code/coding_mcq_reasoning_train.jsonl"
medical_dir = "/data/disk5/internship_disk/gyubin/Qwen_MoE/data/medical/medical_train.json"

# code 데이터 로드 (jsonl)
code_data = []
with open(code_dir, "r", encoding="utf-8") as f:
    for line in f:
        code_data.append(json.loads(line))


len(code_data)



code_data[0]

len(code_data)

len(code_data[0])

import os

save_dir = "/data/disk5/internship_disk/gyubin/Qwen_MoE/data/code"
save_name_train = "code_train.json"
save_name_test = "code_test.json"
train_size = 3000
test_size = 300
code_data = [item for item in code_data if len(item['list_choices']) == 4]
os.makedirs(save_dir, exist_ok=True)

def make_code_item(item):
    return {
        'id': item['task_id'],
        'question': item['question'] + "\n\nOptions:\nA. " + item['list_choices'][0] + '\nB. ' + item['list_choices'][1] + '\nC. ' + item['list_choices'][2] + '\nD. ' + item['list_choices'][3],
        'train_answer': 'Answer: ' + item['answer'] + '\n\nExplanation: ' +'\n' + item['teacher_conclusion'],
        'metric_answer': item['answer']
    }
    
print(len(code_data[:train_size]))

# 앞에서부터 3000개는 train, 그 다음 500개는 test
code_train = [make_code_item(item) for item in code_data[:train_size]]
code_test = [make_code_item(item) for item in code_data[train_size:train_size+test_size]]

# 저장
with open(os.path.join(save_dir, save_name_train), "w", encoding="utf-8") as f:
    json.dump(code_train, f, ensure_ascii=False, indent=2)

with open(os.path.join(save_dir, save_name_test), "w", encoding="utf-8") as f:
    json.dump(code_test, f, ensure_ascii=False, indent=2)

3549-223

# list_choices가 4개가 아닌 항목을 code_data에서 삭제


# MATH와 MEDICAL 도메인을 CODE, LAW와 동일한 딕셔너리 구조로 수정
import os
import json

# 1. MEDICAL 도메인 수정
print("=== MEDICAL 도메인 구조 수정 ===")

# medical_train.json 수정
with open('./medical/medical_train.json', 'r', encoding='utf-8') as f:
    medical_train_list = json.load(f)

medical_train_dict = []
for i, item in enumerate(medical_train_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"medical_train_{i}",
            "question": item[0],
            "train_answer": item[1], 
            "metric_answer": item[2]
        }
        medical_train_dict.append(dict_item)

# medical_test.json 수정
with open('./medical/medical_test.json', 'r', encoding='utf-8') as f:
    medical_test_list = json.load(f)

medical_test_dict = []
for i, item in enumerate(medical_test_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"medical_test_{i}",
            "question": item[0],
            "train_answer": item[1],
            "metric_answer": item[2]
        }
        medical_test_dict.append(dict_item)

# 수정된 파일 저장
with open('./medical/medical_train.json', 'w', encoding='utf-8') as f:
    json.dump(medical_train_dict, f, ensure_ascii=False, indent=2)

with open('./medical/medical_test.json', 'w', encoding='utf-8') as f:
    json.dump(medical_test_dict, f, ensure_ascii=False, indent=2)

print(f"Medical train: {len(medical_train_dict)} samples")
print(f"Medical test: {len(medical_test_dict)} samples")


# 2. MATH 도메인 수정
print("=== MATH 도메인 구조 수정 ===")

# math_train.json 수정
with open('./math/math_train.json', 'r', encoding='utf-8') as f:
    math_train_list = json.load(f)

math_train_dict = []
for i, item in enumerate(math_train_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"math_train_{i}",
            "question": item[0],
            "train_answer": item[1],
            "metric_answer": item[2]
        }
        math_train_dict.append(dict_item)

# math_test.json 수정
with open('./math/math_test.json', 'r', encoding='utf-8') as f:
    math_test_list = json.load(f)

math_test_dict = []
for i, item in enumerate(math_test_list):
    if isinstance(item, list) and len(item) == 3:
        # 리스트 형태 [question, train_answer, metric_answer]를 딕셔너리로 변환
        dict_item = {
            "id": f"math_test_{i}",
            "question": item[0],
            "train_answer": item[1],
            "metric_answer": item[2]
        }
        math_test_dict.append(dict_item)

# 수정된 파일 저장
with open('./math/math_train.json', 'w', encoding='utf-8') as f:
    json.dump(math_train_dict, f, ensure_ascii=False, indent=2)

with open('./math/math_test.json', 'w', encoding='utf-8') as f:
    json.dump(math_test_dict, f, ensure_ascii=False, indent=2)

print(f"Math train: {len(math_train_dict)} samples")
print(f"Math test: {len(math_test_dict)} samples")


