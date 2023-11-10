import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import openai
from typing import List

from utils.constants import OPENAI_API_KEY
from tqdm import tqdm

openai.api_key = OPENAI_API_KEY

def request(
    prompt: str,
    model="gpt-3.5-turbo",
    max_tokens=256,
    temperature=1.0,
    top_p=1.0,
    n=1,
    stop='\n',
    presence_penalty=0.0,
    frequency_penalty=0.0,
    ):
    # retry request (handles connection errors, timeouts, and overloaded API)

    #Completion 모델
    if model == "text-davinci-003":
        while True:
            try:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                break
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")
                import time
                time.sleep(60)

        generations = [gen['text'].lstrip() for gen in response['choices']]
        generations = [_ for _ in generations if _ != '']
        return generations
    
    #Chat 기반 모델 by cyh
    elif model == "gpt-3.5-turbo":
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                break
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")
                import time
                time.sleep(60)
        generations = [gen['message']['content'].lstrip() for gen in response['choices']]
        generations = [_ for _ in generations if _ != '']
        return generations
    