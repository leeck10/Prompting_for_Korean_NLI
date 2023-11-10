import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import json
import torch
import click
from pathlib import Path
from typing import List
import openai
from utils.constants import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

#davinci 모델용 프롬프트 변환 by cyh
def prompt_format(prompt_path: str, keywords: List[str], query1: str, query2: str):
    with open(prompt_path) as f:
        context_string = f.read().strip('\n')
    if keywords is not None:
        n = np.random.choice(range(1, len(keywords)+1))      # number of keywords
        keywords = random.sample(keywords, n)                # subset of keywords
        context_string = context_string.replace('{keywords}', ', '.join(keywords))
    if query1 is not None:
        context_string = context_string.replace('{premise}', query1)
        context_string = context_string.replace('{hypothesis}', query2)
    return context_string

#turbo 모델용 프롬프트 변환 by cyh
def turbo_prompt_format(prompt_path: str, keywords: List[str], query1: str, query2: str):
    with open(prompt_path) as f:
        context_string = []
        context_string.append({})
        context_string[0]["role"] = "system"
        context_string[0]["content"] = f.readline()
    with open(prompt_path) as ff:
        lines = ff.readlines()
        context_string.append({})
        context_string[1]["role"] = "user"
        lines = lines[1:]
        content = ''.join(lines)
        context_string[1]["content"] = content
    if query1 is not None:
        context_string[1]["content"] = context_string[1]["content"].replace('{premise}', query1)
        context_string[1]["content"] = context_string[1]["content"].replace('{hypothesis}', query2)
    return context_string

@click.command()
@click.option('--task', type=str, default=None)
@click.option('--input_path', type=str, default=None)
@click.option('--output_path', type=str, default=None)
@click.option('--prompt_path', type=str, default=None)
@click.option('--num_knowledge', type=int, default=1)
@click.option('--top_p', default=1.0, type=float)
@click.option('--temperature', default=1.0, type=float)
@click.option('--max_tokens', default=128, type=int)
@click.option('--n', default=None, type=int)
def main(
    task: str,
    input_path: str,
    output_path: str,
    prompt_path: bool,
    num_knowledge: int,
    top_p: float,
    temperature: float,
    max_tokens: int,
    n: int,
):
    # read examples for inference
    eval_df = pd.read_json(input_path)
    eval_df = eval_df[:n]

    # generate knowledge!
    generated_examples = []
    #stop = 0
    for i, row in tqdm(eval_df.iterrows(), desc="Knowledge generate", total=n):
        context_string = turbo_prompt_format(
            prompt_path,
            keywords=None,
            query1=row['premise'],
            query2=row['hypothesis'])

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-16k',
                    messages=context_string,
                    max_tokens=64,
                    # temperature=temperature,
                    # top_p=top_p,
                    n=2,
                    #stop='\n',
                )
                break
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")
                import time
                time.sleep(60)

        keywords = response.choices[0].message.content
        row['Keywords'] = keywords
        generated_examples.append(row.to_dict())
        #stop += 1
        #if stop == 15:
            #break

    with open(output_path, 'w') as fo:
        json.dump(generated_examples, fo, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
