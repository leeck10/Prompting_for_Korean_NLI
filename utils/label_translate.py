import json

input_path = '/home/cyh/GKP-master/data/kor_nli/knowledge/knowledge_gpt3.dev5.json'
output_path = '/home/cyh/GKP-master/data/kor_nli/knowledge/kor_knowledge_gpt3.dev5.json'

with open(input_path, 'r', encoding='utf-8') as f:
    json_file = json.load(f)

    for i in range(len(json_file)):
        if json_file[i]["answer"] == "entailment":
            json_file[i]["answer"] = "함의"
        elif json_file[i]["answer"] == "neutral":
            json_file[i]["answer"] = "중립"
        elif json_file[i]["answer"] == "contradiction":
            json_file[i]["answer"] = "모순"

with open(output_path, 'w') as fo:
    json.dump(json_file, fo, ensure_ascii=False, indent=4)

