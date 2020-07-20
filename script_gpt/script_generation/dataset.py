import logging
import os
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer


class ScriptData(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512, overwrite_cache=False):
        assert os.path.isfile(file_path)
        block_size -= (tokenizer.max_len - tokenizer.max_len_single_sentence)
        directory, filename = os.path.split(file_path)
        # change if args are added at later point
        cached_features_file = os.path.join(directory,
                                            f"gpt2_{str(block_size)}_{filename}")

        if os.path.exists(cached_features_file) and not overwrite_cache:
            print(
                f"Loading features from your cached file {cached_features_file}")
            with open(cached_features_file, "rb") as cache:
                self.examples = pickle.load(cache)
        else:
            print(f"Creating features from file {filename} at {directory}")
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            tokenized_text = tokenizer.encode(text)

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i: i + block_size]))

            print(f"Saving features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as cache:
                pickle.dump(self.examples, cache,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    sc = ScriptData(tokenizer,
                    file_path="script_generation/data/film_text.txt",
                    overwrite_cache=True)
