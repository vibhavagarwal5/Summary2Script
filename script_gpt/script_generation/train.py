import argparse
import os
import random

import numpy as np

import torch
from dataset import ScriptData
from torch.utils.data import DataLoader, Dataset
from transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW, GPT2LMHeadModel,
                          GPT2Tokenizer, get_linear_schedule_with_warmup)


def train(model, tokenizer, args):
    dataset = ScriptData(tokenizer=tokenizer,
                         file_path=args.dataset_path)
    script_loader = DataLoader(dataset,
                               batch_size=args.batch_size,
                               shuffle=True)

    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps, num_training_steps=-1)
    script_count = 0
    sum_loss = 0.0

    for epoch in range(args.epochs):
        print(f"EPOCH {epoch} started" + '=' * 30)
        for idx, script in enumerate(script_loader):
            script = script.to(args.device)
            outputs = model(script, labels=script)
            loss, logits = outputs[:2]

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            sum_loss += loss.item()

            if idx % 200 == 0:
                model.eval()
                print(f"sum loss {sum_loss}")
                sample_outputs = model.generate(
                    bos_token_id=random.randint(1, 30000),
                    do_sample=True,
                    top_k=50,
                    max_length=1000,
                    top_p=0.95,
                    num_return_sequences=1
                )
                print("Output:\n" + 100 * '-')
                for i, sample_output in enumerate(sample_outputs):
                    decoded_output = tokenizer.decode(sample_output,
                                                      skip_special_tokens=True)
                    print(f"{i}: {decoded_output}")
                sum_loss = 0.0
                model.train()

    torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
    model.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
    tokenizer.save_vocabulary(args.output_dir)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="./models/", help="Output directory for saving models etc")
    parser.add_argument("--model_checkpoint", type=str,
                        default="gpt2-medium", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int,
                        default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float,
                        default=0.00002, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int,
                        default=10000, help="Number of warmup steps")
    parser.add_argument("--dataset_path", type=str,
                        default="./storage/data/film_text.txt", help="Path for the dataset")
    parser.add_argument("--no_generate", action="store_true",
                        help="No generation task")
    parser.add_argument("--no_train", action="store_true",
                        help="No training task")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    model = model.to(device)
    if not args.no_train:
        train(model, tokenizer, args)

    if not args.no_generate:
        # ----------GENERATE-----------------
        model.eval()

        input_ids = tokenizer.encode('He kisses her softly and takes out his gun.',
                                     return_tensors='pt')
        sample_outputs = model.generate(
            input_ids=input_ids,
            num_beams=5,
            max_length=1000,
            top_p=0.85,
            num_return_sequences=3
        )
        for i, sample_output in enumerate(sample_outputs):
            decoded_output = tokenizer.decode(
                sample_output, skip_special_tokens=True)
            print(f"{i}: {decoded_output}")
