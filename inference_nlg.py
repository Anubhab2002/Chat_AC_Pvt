# import required libraries

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import argparse
import os
import random

print("PROCESS STARTED")

# Initialise wandb

# wandb.init(project="t5_chat_autocompletion", entity="anu2002")

# use GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# create custom dataset for Training

'''
class AutocompleteDataset(Dataset):
    def __init__(self, tokenizer, sentences, max_length=512):
        self.tokenizer = tokenizer
        self.pairs = []

        for sentence in sentences:
            input_text = sentence[: 2 * len(sentence) // 3]
            target_text = sentence[2 * len(sentence) // 3 :]

            input_encoded = tokenizer(
                input_text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            target_encoded = tokenizer(
                target_text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            self.pairs.append((input_encoded, target_encoded))
            """
            words = sentence.split()
            for i in range(1, len(words)):
                input_text = ' '.join(words[:i])
                target_text = ' '.join(words[i:])
                input_encoded = tokenizer(input_text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
                target_encoded = tokenizer(target_text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
                self.pairs.append((input_encoded, target_encoded))
            """

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
'''

# setup cli args list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--model_dir", type=str)
    """
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--bs',  type=int, default=4)
    """
    args = parser.parse_args()
    return args


# get training data
def main(args):
    with open(args.test_data, "r") as f:
        data = f.read()

    dataset = data.split("\n")

    test_data = pd.DataFrame(dataset)
    sentences = test_data.values.flatten().tolist()

    # load model from ckpt

    def load_model_from_checkpoint(checkpoint_path, model, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer

    # helper function

    def generate_completion_with_top_p(
        model, tokenizer, text, max_length=50, top_p=0.92
    ):
        model.eval()
        with torch.no_grad():
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

            # Prepare output probabilities list
            output_probs = []

            # Prepare first input for generation
            curr_input_ids = input_ids

            for _ in range(max_length):
                # Get logits for the next token
                outputs = model(curr_input_ids, return_dict=True)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply top-p filtering
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float("Inf")

                # Sample the next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Save the probability of the chosen token
                chosen_prob = probs[0, next_token].item()
                output_probs.append(chosen_prob)

                # Append the new token to the input
                curr_input_ids = torch.cat(
                    [curr_input_ids, next_token.unsqueeze(-1)], dim=-1
                )

                # Stop if the model generates the end of sequence token
                if next_token == tokenizer.eos_token_id:
                    break

            # Decode the output sequence
            output_text = tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)
            return output_text, output_probs

    # Load pre-trained model and tokenizer

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    model, _ = load_model_from_checkpoint(args.model_dir, model)
    model.to(device)

    print("STARTING INFERENCING")

    def process_sentences(sentences, model, tokenizer):
        for sentence in sentences:
            num_splits = min(57, len(sentence))
            for _ in range(num_splits):
                split_index = random.randint(1, len(sentence))  # Random split point
                input_part = sentence[:split_index]
                ground_truth = sentence[split_index:]

                # Generate completion and probabilities
                output_text, output_probs = generate_completion_with_top_p(
                    model, tokenizer, input_part
                )

                # Calculate the average probability of the generated output
                avg_prob = sum(output_probs) / len(output_probs) if output_probs else 0

                # Print the required information
                print(input_part, ground_truth, output_text, avg_prob, 1)

    process_sentences(sentences, model, tokenizer)


if __name__ == "__main__":
    main(get_args())
