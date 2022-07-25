from lib2to3.pgen2 import token
from pickletools import optimize
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import random
import os
from tqdm.auto import trange
import numpy as np



logging.basicConfig(format='[%(asctime)s] - %(message)s', datefmt='%H:%M:%S')
logging.root.setLevel(logging.NOTSET)

logging.info("Loading dataset")

train_df = pd.read_csv("DATA/dev/tatoeba-dev.eng-rus.tsv", sep='\t')

logging.info("Generating input-target pairs")

input_df = list(train_df['inp_txt'])
target_df = list(train_df['out_txt'])
pairs = list(zip(input_df, target_df))

logging.info("Loading raw T5 model")

raw_model_name = 'translate_eng-ru_model'

model = T5ForConditionalGeneration.from_pretrained(raw_model_name).cuda()
tokenizer = T5Tokenizer.from_pretrained(raw_model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()

batch_size = 8
report_steps = 100
epochs = 10

logging.info("Starting training")

losses = []

for epoch in range(epochs):

    logging.info(f"EPOCH: {epoch}")

    random.shuffle(pairs)

    for i in trange(0, int(len(pairs) / batch_size)):

        batch = pairs[i * batch_size: (i + 1) * batch_size]

        x = tokenizer([p[0] for p in batch], return_tensors='pt', padding=True).to(model.device)
        y = tokenizer([p[1] for p in batch], return_tensors='pt', padding=True).to(model.device)

        loss = model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            labels=y.input_ids,
            decoder_attention_mask=y.attention_mask,
            return_dict=True
        ).loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if i % report_steps == 0:
            print('step', i, 'loss', np.mean(losses[-report_steps:]))

logging.info("Finished training, saving model")

trained_model_path = "translate_eng-ru_model"

model.save_pretrained(trained_model_path)
tokenizer.save_pretrained(trained_model_path)

logging.info("Loading evaluation")

model.eval()

def answer(x, **kwargs):
    inputs = tokenizer(x, return_tensors='pt').to(model.device)
    with torch.no_grad():
        hypotheses = model.generate(**inputs, **kwargs)
    return [tokenizer.decode(h, skip_special_tokens=True) for h in hypotheses]

while True:
    user_input = input(">>> ")
    print(answer(user_input))
