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
    if args.shot > 0:
        with open(args.example_path) as f:
            example_string = f.read().split('{shot}')[:(args.shot)]
        example = ''.join(example_string)
    with open(args.sum_path) as ff:
        sum_string = ff.read()
    
    if len(premise) > 60:
        response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=[{"role": "system", "content": f"다음의 예시와 같이 문장 요약\n{sum_string}"},
                                {"role": "user", "content": f'premise: {premise}\n 요약: '}],
                        max_tokens=256,
                        # temperature=temperature,
                        # top_p=top_p,
                        n=1,
                        stop='\n',
                    )
        premise = response.choices[0].message.content
    
    prompts = None
    prefix = 'NLI 태스크의 레이블 분류'
    postfix = knowledge

    # Chat 기반 모델 
    if args.model_type == 'gpt-3.5-turbo':
        prefix = 'NLI 태스크 분류 [entailment, neutral, contradiction] 중 선택'
        if args.task =='baseline':
            if args.shot == 0:
                prompts = [{"role": "system", "content": prefix},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} answer:'}]
            elif args.shot > 0:
                prompts = [{"role": "system", "content": f'{prefix} Exapmle {example}'},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} answer:'}]

        elif args.task == 'knowledge':
            if args.shot == 0:
                prompts = [{"role": "system", "content": prefix},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} knowledge: {postfix} answer:'}]
            elif args.shot > 0:
                prompts = [{"role": "system", "content": f'{prefix} Example {example} '},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} knowledge: {postfix} answer:'}]

        if prompts is None:
            raise Exception(f'score_for_input() not implemented for {args.task}!')
        
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=prompts,
                    max_tokens=3,
                    # temperature=temperature,
                    # top_p=top_p,
                    n=1,
                    stop='\n',
                )
                break
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")
                import time
                time.sleep(60)
            
        _pred = response.choices[0].message.content
        pred = _pred.lower().strip(' ''(''는''와''.')

        return pred, premise


def score_for_query(args, premise, hypothesis, cands=[], knowledges=[]):
    scores_, probs_ = [], []
    
    if args.model_type == 'gpt-3.5-turbo':
        if args.task == 'baseline':
            pred, sum_premise = score_for_input(args, premise, hypothesis)

        return pred, sum_premise


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
    
    elif args.model_type == 'gpt-3.5-turbo':
        pred, sum_premise = score_for_query(args, premise, hypothesis, knowledges=knowledges)

        if 'answer' in item:
            answer = item['answer']
            ok = checker(args, answer, pred)
            item['ok'] = ok
        item['premise'] = sum_premise
        item['pred'] = pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="baseline")
    parser.add_argument('--model-type', type=str, default='gpt3')
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--average-loss', type=bool, default=True)
    parser.add_argument('--v', type=int, default=-1) # ignore
    parser.add_argument('--aggfunc', type=str, default='best_prob', choices=['best_score', 'best_prob'])
    parser.add_argument('--submit', type=bool, default=False)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--shot', type=int, default=0, help="In-Context의 개수")
    parser.add_argument('--example-path', type=str, default='/home/cyh/GKP-master/data/kor_nli/baseline_example.txt', help="In-Context 환경, 태스크 예시 경로")
    parser.add_argument('--sum-path', type=str, default='/home/cyh/GKP-master/data/kor_nli/sum_example.txt')
    args = parser.parse_args()
    

    args.input_path = '/home/cyh/GKP-master/data/kor_nli/dev.json'

    args.output_path = f'data/kor_nli/inference/{args.shot}shot.summary.{args.model_type}.json'
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