import json

# 定义情绪映射字典
emotion_mapping = {
    # 错误分类的词
    'normal': 'neutral',         # 普通 -> 中性
    'negative': 'sadness',       # 消极 -> 悲伤
    
    # 积极情绪映射
    'adventure': 'excitement',    # 冒险 -> 兴奋
    'inspiration': 'awe',        # 启发 -> 敬畏
    'hope': 'contentment',       # 希望 -> 满足
    'confidence': 'contentment', # 自信 -> 满足
    'curiosity': 'excitement',   # 好奇 -> 兴奋
    'interest': 'excitement',    # 兴趣 -> 兴奋
    'comfort': 'contentment',    # 舒适 -> 满足
    'relaxation': 'contentment', # 放松 -> 满足
    'respect': 'awe',           # 尊重 -> 敬畏
    
    # 消极情绪映射
    'anger': 'disgust',         # 愤怒 -> 厌恶
    'disappointment': 'sadness', # 失望 -> 悲伤
    'unease': 'fear',           # 不安 -> 恐惧
    'loneliness': 'sadness',    # 孤独 -> 悲伤
    'stress': 'fear',           # 压力 -> 恐惧
    'nostalgia': 'contentment', # 怀旧 -> 满足
    'frustration': 'disgust',   # 沮丧 -> 厌恶
    'anxiety': 'fear',          # 焦虑 -> 恐惧
    'confusion': 'neutral',     # 困惑 -> 中性
    'concern': 'fear',          # 担忧 -> 恐惧
}

def normalize_emotions(data):
    # 遍历所有图片
    for image_name, image_data in data.items():
        # 遍历每个群组的评估
        for group, assessments in image_data.items():
            # 如果情绪在映射字典中，则替换为目标情绪
            if assessments['emotion'] in emotion_mapping:
                assessments['emotion'] = emotion_mapping[assessments['emotion']]
    
    return data

# 读取JSON文件
with open('process/gemini_restructured_results.json', 'r') as f:
    data = json.load(f)

# 规范化情绪值
normalized_data = normalize_emotions(data)

# 保存更新后的JSON文件
with open('process/gemini_normalized_results.json', 'w') as f:
    json.dump(normalized_data, f, indent=4)

# 验证更新后的结果
def verify_emotions(data):
    valid_emotions = {'amusement', 'excitement', 'contentment', 'awe', 
                     'disgust', 'sadness', 'fear', 'neutral'}
    invalid_emotions = set()
    
    for image_name, image_data in data.items():
        for group, assessments in image_data.items():
            if assessments['emotion'] not in valid_emotions:
                invalid_emotions.add(assessments['emotion'])
    
    return invalid_emotions

# 检查是否还有非目标情绪词
remaining_invalid = verify_emotions(normalized_data)
print("Remaining invalid emotions:", remaining_invalid if remaining_invalid else "None")
