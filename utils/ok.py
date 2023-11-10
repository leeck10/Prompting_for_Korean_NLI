import json

input_path = '/home/cyh/GKP-master/data/kor_nli/inference/10shot.baseline.gpt-3.5-turbo-16k.json'

# api를 돌리고 생성된 inference 파일을 불러와 ok(맞은 항목)의 개수를 카운트
with open(input_path, 'r', encoding='utf-8') as f:
    inference_json = json.load(f)
    total = len(inference_json)

    correct = 0
    error = 0
    error_list = []
    for i in range(total):
        correct += 1 if inference_json[i]['ok'] == 1 else 0
        error += 1 if inference_json[i]['pred'] not in ['entailment', 'neutral', 'contradiction'] else 0
        if inference_json[i]['pred'] not in ['entailment', 'neutral', 'contradiction']:
            error_list.append(inference_json[i]['pred'])

    print(correct, error, error_list)