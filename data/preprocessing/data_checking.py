#!/usr/bin/env python3
"""Data Checking Script - Converted from data_checking.ipynb"""

import pandas as pd

domain = 'law'

data_dir = f'./processed/{domain}/{domain}_train.json'
data_test_dir = f'./processed/{domain}/{domain}_test.json'

df = pd.read_json(data_dir)
df_test = pd.read_json(data_test_dir)

# df와 df_test의 question 컬럼에서 겹치는 질문이 있는지 확인
# 겹치는 경우, 해당 question과 각 데이터프레임의 id를 함께 출력

# set을 이용해 겹치는 question 추출
questions_train = set(df['train_answer'])
questions_test = set(df_test['train_answer'])
overlap_questions = questions_train & questions_test

print(f"겹치는 question 개수: {len(overlap_questions)}")

if overlap_questions:
    # 겹치는 question에 대해 각 데이터프레임에서 id 추출
    overlap_info = []
    for q in overlap_questions:
        train_id = df.loc[df['question'] == q, 'id'].values
        test_id = df_test.loc[df_test['question'] == q, 'id'].values
        overlap_info.append({
            'question': q,
            'train_id': train_id[0] if len(train_id) > 0 else None,
            'test_id': test_id[0] if len(test_id) > 0 else None
        })
    # DataFrame으로 보기 좋게 출력
    overlap_df = pd.DataFrame(overlap_info)
    print(overlap_df)
else:
    print("겹치는 question이 없습니다.")


a_set = set()
for i in range(len(df)):
    a_set.add(df.iloc[i]['metric_answer'][0])

print(f'len(a_set): {len(a_set)}')






import pandas as pd

domain = 'code'

data_dir = f'./processed/{domain}/{domain}_train.json'
data_test_dir = f'./processed/{domain}/{domain}_test.json'

# 데이터 불러오기
df = pd.read_json(data_dir)
df_test = pd.read_json(data_test_dir)
print(df.iloc[0]['question'])



df.iloc[0]['train_answer']

import pandas as pd

domain = 'medical'

data_dir = f'./processed/{domain}/{domain}_train.json'
data_test_dir = f'./processed/{domain}/{domain}_test.json'

# 데이터 불러오기
df = pd.read_json(data_dir)
df_test = pd.read_json(data_test_dir)



i=0
print(df.iloc[i]['question'])
print('--------------------------------')
print(df.iloc[i]['train_answer'])
print('--------------------------------')
print(df.iloc[i]['metric_answer'])




import json

# 로컬 경로에서 불러오기

# dir = "/data/disk5/internship_disk/gyubin/med_qa/data_clean/data_clean/questions/US/metamap_extracted_phrases/train/phrases_train.jsonl"
# dir = "/data/disk5/internship_disk/gyubin/med_qa/data_clean/data_clean/questions/US/train.jsonl"
dir = "/data/disk5/internship_disk/gyubin/med_qa/data_clean/data_clean/questions/US/train.jsonl"
dataset = []
with open(dir, 'r') as f:
    for line in f:
        data = json.loads(line)
        dataset.append(data)
# 이제 전체 데이터셋이 dataset 리스트에 저장됨
print(f'len(dataset): {len(dataset)}')


dataset[0]['options']

a = 0
b = 0

for i in range(len(dataset)):
    if len(dataset[i]['options']) != 4:
        a += 1
    else:
        b += 1
print(f'a: {a}, b: {b}')


import pandas as pd

# code 도메인에 대해서만 적용
domain = 'code'

data_dir = f'./processed/{domain}/{domain}_train.json'
data_test_dir = f'./processed/{domain}/{domain}_test.json'

df = pd.read_json(data_dir)
df_test = pd.read_json(data_test_dir)

# 각 데이터프레임의 'question' 컬럼에 '\n' 추가
df['question'] = df['question'].astype(str) + '\n'
df_test['question'] = df_test['question'].astype(str) + '\n'

# 변경사항을 파일에 저장
df.to_json(data_dir, force_ascii=False, orient='records', lines=False)
df_test.to_json(data_test_dir, force_ascii=False, orient='records', lines=False)

# 결과 확인
print(df.iloc[0]['question'])
print('--------------------------------')
print(df.iloc[0]['train_answer'])
print('--------------------------------')
print(df.iloc[0]['metric_answer'])


import pandas as pd
import re

domains = ['medical', 'law', 'math', 'code', "MMLU"]
i = 4
domain = domains[i]

data_dir = f'./processed/{domain}/{domain}_train.json'
data_test_dir = f'./processed/{domain}/{domain}_test.json'

df = pd.read_json(data_dir)
df_test = pd.read_json(data_test_dir)

print(df.keys())


print(df.iloc[i]['question'])

print(df.iloc[i]['metric_answer'])

print(df.iloc[i]['train_answer'])

