import argparse
import json
import numpy as np
import torch
import transformers
import openai

from utils.constants import OPENAI_API_KEY, NLI_ANSWERS
from tqdm import tqdm
from collections import Counter

openai.api_key = OPENAI_API_KEY

def checker(args, answer, pred):
    return 1 if answer == pred else 0


def score_for_input(args, premise, hypothesis, cands=[], knowledge=None):
    with open(args.example_path) as f:
        example_string = f.read().split('{shot}')[:(args.shot)]
    if args.shot > 0:
        example = ''.join(example_string)
    
    prompts = None
    prefix = 'NLI 태스크의 레이블 분류'
    postfix = knowledge

    # Completion 기반 모델
    if args.model_type == 'text-davinci-003':
        if args.task == 'baseline':
            prompts = [f'{prefix} premise: {premise} hypothesis: {hypothesis} answer: {cand}' for cand in cands]
        elif args.task == 'knowledge':
            prompts = [f'{prefix} premise: {premise} hypothesis: {hypothesis} knowledge: {postfix} answer: {cand}' for cand in cands]
        
        if prompts is None:
            raise Exception(f'score_for_input() not implemented for {args.task}!')

        while True:
            try:
                response = openai.Completion.create(
                    model='text-davinci-003',
                    prompt=prompts,
                    max_tokens=64, # suppress continuation
                    #temperature=1.,
                    #top_p=0.5,
                    #n=1,
                    stop='\n',
                    logprobs=0,
                    echo=True, # so the logprobs of prompt tokens are shown
                )
                break
            except Exception as e:
                print(e)
                import time
                time.sleep(1)

        scores = []
        for c, cand in enumerate(cands):
            query_offset_tokens = 0
            while response['choices'][c]['logprobs']['text_offset'][query_offset_tokens] < len(prefix):
                query_offset_tokens += 1
            logprobs = response['choices'][c]['logprobs']['token_logprobs'][query_offset_tokens:]
            if args.average_loss:
                score = np.mean(logprobs)
            else:
                score = np.sum(logprobs)
            scores.append(score)
        scores = torch.tensor(scores)
        probs = torch.softmax(scores, dim=0)

        return scores, probs

    # Chat 기반 모델 
    elif args.model_type == 'gpt-3.5-turbo-16k':
        prefix = 'NLI 태스크 분류\n 다음의 예시와 같이 순서를 나누어 [entailment, neutral, contradiction] 선택'
        if args.task =='baseline':
            if args.shot == 0:
                prompts = [{"role": "system", "content": prefix},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} answer:'}]
            elif args.shot > 0:
                prompts = [{"role": "system", "content": f'{prefix}\n예시\n{example}'},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} 위의 예시와 같이 문제를 쪼개어 순서대로 위 문장의 관계를 유추한다.'}]

        elif args.task == 'knowledge':
            if args.shot == 0:
                prompts = [{"role": "system", "content": prefix},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} knowledge: {postfix} answer:'}]
            elif args.shot > 0:
                prompts = [{"role": "system", "content": f'{prefix}\n예시\n{example} '},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} knowledge: {postfix} answer:'}]

        if prompts is None:
            raise Exception(f'score_for_input() not implemented for {args.task}!')
        
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-16k',
                    messages=prompts,
                    max_tokens=512,
                    temperature=0.5,
                    #top_p=1,
                    n=1,
                    #stop='\n',
                )
                break
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")
                import time
                time.sleep(60)
            
        _pred = response.choices[0].message.content
        pred = 'entailment' if 'entailment' in _pred else 'error'
        pred = 'neutral' if 'neutral' in _pred else 'error'
        pred = 'contradiction' if 'contradiction' in _pred else 'error'

        return pred


def score_for_query(args, premise, hypothesis, cands=[], knowledges=[]):
    scores_, probs_ = [], []
    v, h = args.v, args.h

    if args.model_type == 'text-davinci-003':
        # without knowledge by cyh
        if args.task == 'baseline':
            scores, probs = score_for_input(args, premise, hypothesis, cands)
            scores_.append(scores)
            probs_.append(probs)

        # with knowledge
        elif args.task == 'knowledge':
            for i in range(0, h, 4):
                knowledge = ' '.join(knowledges[i:i+h])
                scores, probs = score_for_input(args, premise, hypothesis, cands=cands, knowledge=knowledge)
                scores_.append(scores)
                probs_.append(probs)

        return torch.stack(scores_), torch.stack(probs_)
    
    elif args.model_type == 'gpt-3.5-turbo-16k':
        if args.task == 'baseline':
            pred = score_for_input(args, premise, hypothesis)
        
        elif args.task == 'knowledge':
            pred_cands = []
            label_cands = ['entailment', 'neutral', 'contradiction']

            for i in range(0, len(knowledges), h):
                knowledge = ' '.join(knowledges[i:i+h])
                pred_cands.append(score_for_input(args, premise, hypothesis, knowledge=knowledge))

            pred_counter = Counter(pred_cands)
            _pred = pred_counter.most_common(n=1)[0][0]
            pred = _pred

        return pred


def process_item(args, item):
    premise = item['premise']
    hypothesis = item['hypothesis']
    knowledges = item['knowledges'] if 'knowledges' in item else []

    if args.model_type == 'text-davinci-003':
        cands = NLI_ANSWERS
        scores_, probs_ = score_for_query(args, premise, hypothesis, cands, knowledges)
        scores, _ = torch.max(scores_, dim=0)
        probs, _ = torch.max(probs_, dim=0)

        if args.aggfunc == 'best_score':
            p = scores.argmax().item()
        elif args.aggfunc == 'best_prob':
            p = probs.argmax().item()
        pred = cands[p]

        #item['scores_'] = scores_.tolist()
        #item['probs_'] = probs_.tolist()
        item['scores'] = scores.tolist()
        item['probs'] = probs.tolist()
        item['pred'] = pred

        if 'answer' in item:
            answer = item['answer']
            ok = checker(args, answer, pred)
            item['ok'] = ok

        if args.submit:
            item['output'] = {
                'premise': premise,
                'hypothesis': hypothesis,
                'result_list': [{'word': _[1]} for _ in sorted(zip(probs.tolist(), NLI_ANSWERS), reverse=True)],
            }
    
    elif args.model_type == 'gpt-3.5-turbo-16k':
        pred = score_for_query(args, premise, hypothesis, knowledges=knowledges)

        if 'answer' in item:
            answer = item['answer']
            ok = checker(args, answer, pred)
            item['ok'] = ok
        item['pred'] = pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--model-type', type=str, default='gpt3')
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--average-loss', type=bool, default=True)
    parser.add_argument('--h', type=int, default=1, help="한 번에 몇 개의 지식을 포함해 추론할 것인지")
    parser.add_argument('--v', type=int, default=-1) # ignore
    parser.add_argument('--aggfunc', type=str, default='best_prob', choices=['best_score', 'best_prob'])
    parser.add_argument('--submit', type=bool, default=False)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--shot', type=int, default=0, help="In-Context의 개수")
    parser.add_argument('--num-k', type=int, default=5, help="사용할 지식의 개수")
    parser.add_argument('--example-path', type=str, default='/home/cyh/GKP-master/data/kor_nli/CoT_example.txt', help="In-Context 환경, 태스크 예시 경로")
    args = parser.parse_args()
    
    args.input_path = '/home/cyh/GKP-master/data/kor_nli/dev.json'
    args.output_path = f'data/kor_nli/inference/CoT.{args.shot}shot.{args.task}.{args.model_type}.json'
    with open(args.input_path) as f:
        ds = json.load(f)
        if args.n is not None:
            ds = ds[:args.n]

    pbar = tqdm(ds)
    num, den = 0, 0

    #stop = 0
    for item in pbar:
        process_item(args, item)
        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})
        #stop += 1
        #if stop == 20:
            #break
    
    with open(args.output_path, 'w') as f:
        json.dump(ds, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    main()