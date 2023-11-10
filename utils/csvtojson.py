import pandas as pd
import json

from collections import OrderedDict

#.csv 형태의 파일을 json으로 변환
output_path = '/home/cyh/GKP-master/train.json'
nli_dev = OrderedDict()
nli_dev_csv = pd.read_csv('/home/cyh/GKP-master/data/kor_nli/klue-nli-v1.1_train.csv', encoding='utf-8')

baselist = []

for i in range(len(nli_dev_csv)):
    baselist.append({})
    baselist[i]["premise"] = nli_dev_csv.loc[i]['premise']
    baselist[i]["hypothesis"] = nli_dev_csv.loc[i]['hypothesis']
    baselist[i]["answer"] = nli_dev_csv.loc[i]['gold_label']

nli_dev = baselist

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nli_dev, f, ensure_ascii=False, indent=4)