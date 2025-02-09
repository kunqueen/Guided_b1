import requests
import base64
import json
import os
from tqdm import tqdm
import time
import random
from datetime import datetime

def extract_json_from_content(content):
    """Extract and parse JSON from the content string"""
    try:
        # Find JSON between ```json and ``` markers
        json_str = content.split('```json\n')[1].split('\n```')[0]
        return json.loads(json_str)
    except Exception as e:
        return {
            "error": f"Failed to parse JSON: {str(e)}", 
            "original_content": content
        }

def process_images(base_folder, output_file="results_clean.json"):
    headers = {
        'Authorization': 'sk-p3vtbOEfIjot9urO42865a10A8A24aAaAc2b8164152e748c',  # api-key
        'Content-Type': 'application/json'
    }
    
    proxy = {
        "http": "http://l50041666:Lk50041666.@proxyhk.huawei.com:8080",
        "https": "http://l50041666:Lk50041666.@proxyhk.huawei.com:8080"
    }

    # Load existing results if any
    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            temp_results = json.load(f)
            # Handle both old and new format results
            if isinstance(temp_results, dict):
                if any(isinstance(v, dict) and "session" in str(k).lower() for k, v in temp_results.items()):
                    # Old format with sessions
                    for session in temp_results.values():
                        if isinstance(session, dict):
                            results.update(session)
                else:
                    # New format or partially processed results
                    results = temp_results

    # Get all folders and images
    folders = sorted([f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))])

    def make_api_request(json_data, max_retries=10, initial_delay=5):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.post(
                    'http://114.119.180.232:3000/v1/chat/completions',
                    headers=headers,
                    json=json_data,
                    proxies=proxy,
                    timeout=30
                )
                
                response_data = response.json()
                
                # Extract only the content from the response
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    content = response_data['choices'][0]['message']['content']
                    return extract_json_from_content(content)
                else:
                    raise requests.exceptions.RequestException("Invalid response format")

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                retry_count += 1
                delay = initial_delay * (2 ** retry_count) + random.uniform(0, 3)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Error occurred: {str(e)}")
                
                if retry_count < max_retries:
                    print(f"Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Skipping this request.")
                    return {"error": str(e)}

    try:
        for folder in tqdm(folders, desc="Processing folders"):
            folder_path = os.path.join(base_folder, folder)
            images = sorted([img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))])

            for image in tqdm(images, desc=f"Processing {folder}", leave=False):
                # Skip if image already processed successfully
                if image in results and 'error' not in results[image]:
                    continue

                image_path = os.path.join(folder_path, image)

                try:
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')

                    json_data = {
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": 
            """
### Task
Please analyze the provide image based on the following three image assessment tasks:
1. Image quality assessment: Evaluate the image based on low-level elements like technical quality, composition, color balance, lighting, sharpness, exposure, contrast, and overall visual impact. Choose one of the following: "positive", "normal", or "negative".
2. Image aesthetic assessment: Evaluate the aesthetic appeal of the image, focusing on elements like composition, color harmony, visual balance, and overall attractiveness. Choose one of the following: "positive", "normal", "negative".
3. Image emotional perception: analyze the emotions the image evokes or conveys to viewers. Choose one of the following: "amusement", "excitement", "contentment", "awe", "disgust", "sadness", "fear" or "neutral".

### Information
There are the following 12 types of individuals. How do you understand these groups of people? Based on your understanding, please role-play each group to complete the three tasks mentioned above.
Categorization by age: Age group 18 to 21, Age group 22 to 25, Age group 26 to 29, Age group 30 to 34, Age group 35 to 40.
Categorization by education: Junior College graduates, Junior High School graduates, Senior High School graduates, Technical Secondary School graduates, University graduates
Categorization by gender: Female, Male.
please proceed step by step, ensuring that the evaluation for wach group is based on their unique background and perspective. Make sure the results for each group do not interfere with or influence one another.
After completing the tasks for each group, please disregard these identity charateristics and objectively complete the three tasks without considering the group attributes.

### Response Format
Your output should include the following content and be in JSON format:
```json
{
    "age_18_to_21":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"age_22_to_25":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"age_26_to_29":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"age_30_to_34":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"age_35_to_40":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"JuniorCollege":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"JuniorHigh":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"SeniorHigh":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"TechnicalSecondarySchool":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"University":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"female":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"male":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},"id_free":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
}
```

### Notes
- Do not indicate that you use additional information/context in your answer. Only use it implicitly to answer the questions.
            """
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 450
                    }

                    # Make API request and get cleaned response
                    response_data = make_api_request(json_data)
                    results[image] = response_data

                    # Save results after each successful image processing
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)

                    # Add a small delay between requests
                    time.sleep(random.uniform(0.5, 2))

                except Exception as e:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    error_msg = f"[{timestamp}] Error processing {image_path}: {str(e)}"
                    print(f"\n{error_msg}")
                    
                    results[image] = {"error": str(e)}
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress saved.")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    base_folder = "/home/kk/dataset/imgs_downsample/" # Update this path
    results = process_images(base_folder)





# ```json
# {
#     "age_18_to_21":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "age_22_to_25":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "age_26_to_29":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "age_30_to_34":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "age_35_to_40":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "JuniorCollege":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "JuniorHigh":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "SeniorHigh":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "TechnicalSecondarySchool":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "University":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "female":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "male":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
#     "id_free":{"quality":"positive/normal/negative", "aesthetic":"positive/normal/negative", "emotion":"amusement/excitement/contentment/awe/disgust/sadness/fear/neutral"},
# }
# ```
