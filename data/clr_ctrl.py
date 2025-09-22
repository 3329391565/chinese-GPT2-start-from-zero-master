import json
import re

# 定义用于去除控制字符的正则表达式
control_chars = re.compile(r'[\x00-\x1F\x7F]')

# 定义函数用于清理控制字符
def remove_control_characters(s):
    return control_chars.sub('', s)

# 初始化空字符串存储清理后的内容
cleaned_content = ''

# 逐行读取并清理文件
with open('data\\input.json', 'r', encoding='utf-8') as f:
    for line in f:
        cleaned_content += remove_control_characters(line)

# 打印部分内容进行调试
print(cleaned_content[1234170:1234200])  # 打印清理后的前1000个字符以检查是否有问题

# 解析为 JSON 对象
try:
    data = json.loads(cleaned_content)
    print("JSON 文件成功解析")
except json.JSONDecodeError as e:
    print(f"解析 JSON 时出错: {e}")
    exit(1)

# 定义递归函数清理数据中的控制字符
def clean_data(obj):
    if isinstance(obj, str):
        return remove_control_characters(obj)
    elif isinstance(obj, list):
        return [clean_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_data(value) for key, value in obj.items()}
    else:
        return obj

# 清理 JSON 数据中的控制字符
cleaned_data = clean_data(data)

# 保存清理后的 JSON 文件
with open('data\\train.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
