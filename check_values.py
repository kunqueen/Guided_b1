import json

def check_values(data):
    # 定义目标值集合
    valid_quality_aesthetic = {'positive', 'normal', 'negative'}
    valid_emotions = {'amusement', 'excitement', 'contentment', 'awe', 
                     'disgust', 'sadness', 'fear', 'neutral'}
    
    # 存储异常值
    invalid_values = {
        'quality': set(),
        'aesthetic': set(),
        'emotion': set()
    }
    
    # 遍历所有图片
    for image_name, image_data in data.items():
        # 遍历每个群组的评估
        for group, assessments in image_data.items():
            # 检查quality
            if assessments['quality'] not in valid_quality_aesthetic:
                invalid_values['quality'].add(assessments['quality'])
            
            # 检查aesthetic
            if assessments['aesthetic'] not in valid_quality_aesthetic:
                invalid_values['aesthetic'].add(assessments['aesthetic'])
            
            # 检查emotion
            if assessments['emotion'] not in valid_emotions:
                invalid_values['emotion'].add(assessments['emotion'])
    
    return invalid_values

# 读取JSON文件
with open('process/gemini_normalized_results.json', 'r') as f:
    data = json.load(f)

# 检查非目标值
invalid_values = check_values(data)

# 打印结果
print("Invalid values found:")
print("Quality:", invalid_values['quality'] if invalid_values['quality'] else "None")
print("Aesthetic:", invalid_values['aesthetic'] if invalid_values['aesthetic'] else "None")
print("Emotion:", invalid_values['emotion'] if invalid_values['emotion'] else "None")
