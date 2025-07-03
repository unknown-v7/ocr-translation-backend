import requests
import base64

# 读取图片并转换为 base64
with open('test_image.png', 'rb') as f:
    img_bytes = f.read()
img_base64 = base64.b64encode(img_bytes).decode('utf-8')

url = 'http://127.0.0.1:5050/ocr'

payload = {
    "images_base64": [img_base64],
    "ocr_lang": "eng",
    "target_lang": "zh-CN"
}

response = requests.post(url, json=payload)

print("状态码:", response.status_code)
print("响应头:", response.headers)
print("响应内容:", response.text)

try:
    data = response.json()
    print(data)
except Exception as e:
    print("解析JSON失败:", e)
