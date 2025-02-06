from groq import Groq
import xlrd
import pandas as pd
import warnings
import sys
import re

# 워닝 무시
warnings.filterwarnings('ignore')

import pandas as pd
import xlrd

workbook = xlrd.open_workbook('ko-id.xlsx')
sheet = workbook.sheet_by_index(0)

# 병합된 셀 처리 (이전 코드에서 사용한 함수들)
def get_merged_cells(sheet):
    merged_cells = []
    for merged_range in sheet.merged_cells:
        r1, r2, c1, c2 = merged_range
        for row in range(r1, r2):
            for col in range(c1, c2):
                merged_cells.append((row, col))
    return merged_cells

def fill_merged_cells(sheet, df):
    merged_cells = get_merged_cells(sheet)
    for merged_range in sheet.merged_cells:
        r1, r2, c1, c2 = merged_range
        value = sheet.cell_value(r1, c1)
        for row in range(r1, r2):
            for col in range(c1, c2):
                df.iloc[row, col] = value

# DataFrame 생성
df = pd.DataFrame('', index=range(sheet.nrows), columns=range(sheet.ncols))

# 데이터 채우기
for r in range(sheet.nrows):
    for c in range(sheet.ncols):
        df.iloc[r, c] = sheet.cell_value(r, c)

fill_merged_cells(sheet, df)

def add_dynamic_keywords(df):
    new_rows = []  # 새로운 행들을 저장할 리스트

    for index, row in df.iterrows():
        word = row.iloc[1]  # 단어 (B열)
        formality = row.iloc[2]  # 형식성 (C열)
        sector = row.iloc[3]  # 분야 (D열)
        keyword = row.iloc[4]  # 키워드 (E열)
        group = row.iloc[0]  # A열 (Group)

        # B열에서 {키워드} 부분 찾기
        if isinstance(word, str) and '{' in word and '}' in word:
            # { } 안에 있는 모든 키워드 추출
            keywords_in_word = re.findall(r'\{([^}]+)\}', word)
            replacements_list = []  # 키워드에 대한 치환 후보들을 저장할 리스트

            # 각 키워드에 대해 치환 가능한 단어 가져오기
            for keyword_in_word in keywords_in_word:
                # 키워드에 대응되는 치환 단어들 가져오기
                replacements = df[df.iloc[:, 4] == keyword_in_word].iloc[:, 1].values
                if replacements.size > 0:
                    replacements_list.append(replacements)  # 치환 후보를 리스트에 추가
                else:
                    # 치환 가능한 단어가 없으면 원래 키워드 유지
                    replacements_list.append([f"{{{keyword_in_word}}}"])

            print(replacements_list)
            # 가능한 모든 치환 조합 생성
            from itertools import product
            replacement_combinations = list(product(*replacements_list))

            # 각 조합에 대해 새로운 단어 생성
            for combination in replacement_combinations:
                new_word = word
                for original, replacement in zip(keywords_in_word, combination):
                    new_word = new_word.replace(f"{{{original}}}", replacement)  # 원래 키워드를 대체

                # 새로운 행 추가
                try:
                    # 치환 단어들에 대한 최대 formality 계산
                    replacement_formality = max(
                        [df[df.iloc[:, 1] == r].iloc[0, 2] for r in combination if r in df.iloc[:, 1].values]
                    )
                except Exception as e:
                    # 치환 단어에 formality가 없으면 기본 formality 유지
                    replacement_formality = formality

                new_row = [group, new_word.strip(), replacement_formality, sector, keyword]
                new_rows.append(new_row)
        else:
            # 키워드가 없거나 치환할 필요가 없는 경우 그대로 복사
            new_row = [group, word, formality, sector, keyword]
            new_rows.append(new_row)

    # 새로운 단어들을 DataFrame에 추가
    new_df = pd.DataFrame(new_rows, columns=[0, 1, 2, 3, 4])  # A, B, C, D, E열 모두 포함
    return new_df




# 동적 키워드 처리 함수 실행
new_df = add_dynamic_keywords(df)

# 기존 df는 버리고 새로운 df만 유지
df = new_df

# 결과 출력
print(df.to_string())


def get_closest_word(df, source, formality):
    # source_input에 해당하는 단어 찾기
    source_rows = df[df.iloc[:, 1] == source]  # Target 열에서 source_input에 해당하는 행 찾기
    
    # 만약 해당 source가 없다면 None 반환
    if source_rows.empty:
        return None

    # 해당 source가 속한 그룹을 찾기 (Source 열을 기준으로)
    source_group = source_rows.iloc[0, 0]  # 동일한 Source 그룹을 찾기

    # 해당 그룹에 속한 단어들만 필터링
    group_rows = df[df.iloc[:, 0] == source_group]

    closest_word = None
    min_diff = float('inf')

    # formality 값과 가까운 단어 찾기
    for index, row in group_rows.iterrows():
        try:
            # formality 값을 실수로 변환
            formality_value = float(row.iloc[2])
            diff = abs(formality_value - formality)  # formality와의 차이 계산
            print(f"Comparing {row.iloc[1]}: formality = {formality_value}, diff = {diff}")
            if diff < min_diff:
                min_diff = diff
                closest_word = row.iloc[1]  # 가장 가까운 단어 저장
        except ValueError:
            # formality 값이 변환할 수 없는 경우 (예: 공백 또는 비정상적인 값)
            continue

    return closest_word


import httpx, json
import re

deeplx_api = "http://127.0.0.1:1188/translate"

data = {
	#"text": "할머니께서는 우리에게 매일 사과를 보내신다.",
            "text": "징징아! 난 너가 싫어!",
	"source_lang": "KO",
	"target_lang": "ID"
}

post_data = json.dumps(data)
r = httpx.post(url = deeplx_api, data = post_data).text

match = re.search(r'"alternatives":\["(.*?)"\]', r)
if match:
    translated_text = match.group(1)

translated_text = translated_text.split(",")[0].strip('"')
print(translated_text)


import nltk
from nltk.tokenize import word_tokenize

def process_text(df, text, formality):
    # 문장에서 문장부호를 분리하지 않고 처리
    words = re.findall(r'\w+|[^\s\w]', text, re.UNICODE)
    result = []
    skip = 0  # 단어 스킵 인덱스

    for i in range(len(words)):
        if skip > 0:  # 이미 그룹 처리된 단어는 건너뛰기
            skip -= 1
            continue
        
        matched = False
        # 4단어, 3단어, 2단어, 1단어 순으로 검사
        for n in range(4, 0, -1):
            if i + n <= len(words):
                group = ' '.join(words[i:i + n])  # n개의 단어 그룹 생성
                replacement = get_closest_word(df, group, formality)
                if replacement:
                    result.append(replacement)
                    skip = n - 1  # n-1 단어는 건너뛴다
                    matched = True
                    break
        
        if not matched:
            result.append(words[i])  # 대체되지 않은 단어는 그대로 추가

    # 결과 합치기 (문장부호와 단어 사이의 공백 제거)
    final_result = ''.join(
        [result[i] + (' ' if i + 1 < len(result) and re.match(r'\w', result[i + 1]) else '') for i in range(len(result))]
    )
    return final_result.strip()

formality = 1.8
final_result = process_text(df, translated_text, formality)
print(final_result)