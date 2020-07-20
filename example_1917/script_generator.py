import random
import time

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def prepare_data(input_f):
    d = {}
    for idx in range(len(input_f)):
        d[f'{str(idx)}-{str(idx+5)}'] = input_f[idx:idx + 5]
    return d


def generate_script(key, model, max_length, num_samples):
    start = time.time()
    input_ids = tokenizer.encode(inp[key], return_tensors='pt')
    output = model.generate(
        input_ids=input_ids,
        do_sample=True,
        top_k=50,
        max_length=max_length,
        top_p=0.95,
        num_return_sequences=num_samples)

    for index, sample in enumerate(output):
        text_file = open(f'1917_({key})_{index}.txt', "w")
        text_file.write(tokenizer.decode(sample, skip_special_tokens=True))
        text_file.close()
        print(f'Total time: {time.time()-start}\n')


model_path = './script_gpt/script_generation/models'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

model.eval()
max_length = 1000
num_samples = 3

# inp = prepare_data('./1917.txt')
f = open('./1917.json', 'r')
inp = json.load(f)
for key in list(inp.keys()):
    print(key)
    generate_script(key, model, max_length, num_samples)
