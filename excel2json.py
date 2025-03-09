import pandas as pd
import json
from groq import Groq
from difflib import SequenceMatcher
from tqdm import tqdm  # Import tqdm

# 엑셀 파일 읽기
file_path = "ko-fr.xlsx"
df = pd.read_excel(file_path, header=None)

# A열(카테고리) 채우기 (왼쪽 병합된 셀만)
df.iloc[:, 0] = df.iloc[:, 0].fillna(method="ffill")

# "식사" 카테고리 필터링
filtered_df = df[df.iloc[:, 0] == "담화 표지"]

# Groq API 설정
with open("apikey/groq.txt", 'r', encoding='utf-8') as file:
    groq_key = file.read()

GroqClient = Groq(api_key=groq_key)

# 예시 문장 생성 함수
def get_example(word, korean):
    response = GroqClient.chat.completions.create(
        messages=[{'role': 'user', 'content': f'Make only one French sentence with a Korean translation using the word {word}. Use {korean} for translation of {source}. Use one of pronouns like je, tu, elle, il, ils, elles, on, nous, vous in level of B1. No details or additional explanation. \nex. Je vais au cafe. (나는 카페에 간다.)'}],
        model="llama-3.3-70b-versatile",  # Use the appropriate model ID
        temperature=0.0,
        max_tokens=1024,
        top_p=1,
        stream=False
    )
    return response.choices[0].message.content.strip()

# JSON 구조 만들기
json_list = []
prev_source = None  # 이전 source 저장
current_json = None  # 현재 JSON 객체

# tqdm을 이용하여 진행 상황 표시
for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0], desc="Processing Rows"):
    source = row[1] if pd.notna(row[1]) else ""
    
    # 새로운 source가 나오면 새로운 JSON 객체 생성
    if source and source != prev_source:
        if current_json:
            json_list.append(current_json)  # 기존 JSON 저장
        subsource = row[2] if pd.notna(row[2]) else ""  # 새로운 subsource 설정
        current_json = {
            "source": source,
            "subsource": subsource,
            "target": {},
            "description": [],
            "example": {}
        }

    prev_source = source  # 현재 source를 이전 source로 저장

    # target 데이터 구성
    words = str(row[3]).split("/") if pd.notna(row[3]) else []
    formalities = str(row[4]).split("/") if pd.notna(row[4]) else ["0"] * len(words)

    # 단어 개수와 formality 개수를 맞춤
    if len(words) > len(formalities):
        formalities += ["0"] * (len(words) - len(formalities))

    target = {word.strip(): int(form) if form.isdigit() else 0 for word, form in zip(words, formalities)}

    # example 초기화 및 예시 문장 추가
    example = {}
    for word in target.keys():
        example[word.strip()] = "" # get_example(word.strip(), source)  # 예시 문장을 가져옴

    if current_json:
        current_json["target"].update(target)
        current_json["example"].update(example)

# 마지막 JSON 객체 저장
if current_json:
    json_list.append(current_json)

# JSON 파일 저장
output_file = "output.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)

# 출력 확인
print(json.dumps(json_list, ensure_ascii=False, indent=2))
