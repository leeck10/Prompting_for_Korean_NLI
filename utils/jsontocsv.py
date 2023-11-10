import pandas as pd
import json

#baseline or knowledge
mode = 'baseline'



if mode == 'baseline':
    jsonlead = open('/home/cyh/GKP-master/data/kor_nli/train.json', encoding='utf-8')
    json_file = json.load(jsonlead)

    csv_file = pd.DataFrame(columns=['premise', 'hypothesis', 'label'])

    for i in range(len(json_file)):
        csv_file.loc[i] = [json_file[i]['premise'], json_file[i]['hypothesis'], json_file[i]['answer']]
    csv_file.to_csv('/home/cyh/GKP-master/data/kor_nli/train_baseline.csv')     

elif mode == 'knowledge':
    jsonlead = open('/home/cyh/GKP-master/data/kor_nli/knowledge/knowledge_gpt3.train.json', encoding='utf-8')
    json_file = json.load(jsonlead)

    csv_file = pd.DataFrame(columns=['premise', 'hypothesis', 'label', 'knowledge'])

    for i in range(len(json_file)):
        csv_file.loc[i] = [json_file[i]['premise'], json_file[i]['hypothesis'], json_file[i]['answer'], json_file[i]['knowledges'][0]]
    csv_file.to_csv('/home/cyh/GKP-master/data/kor_nli/train_knowledge.csv')    


