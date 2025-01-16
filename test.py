import httpx, json
import re

deeplx_api = "http://127.0.0.1:1188/translate"

data = {
	"text": "아이스 아메리카노랑 초콜렛 라떼 한잔 주세요.",
	"source_lang": "KO",
	"target_lang": "ID"
}

post_data = json.dumps(data)
r = httpx.post(url = deeplx_api, data = post_data).text

match = re.search(r'"alternatives":\["(.*?)"\]', r)
if match:
    translated_text = match.group(1)

print(translated_text)