# import required libraries

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import argparse
import os
import random

#print("PROCESS STARTED")

# Initialise wandb

# wandb.init(project="t5_chat_autocompletion", entity="anu2002")

# use GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", device)

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
    parser.add_argument("--inference_method", type=str, choices=["top-p", "beam"], default="beam")
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
            
            pad_token_id = tokenizer.pad_token_id
            decoder_input_ids = torch.full((input_ids.shape[0], 1), pad_token_id, dtype=torch.long, device=input_ids.device)
            
            # Prepare output probabilities list
            output_probs = []

            for _ in range(max_length):
                # Get logits for the next token
                outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
                #print(outputs)
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
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

                # Save the probability of the chosen token
                chosen_prob = probs[0, next_token].item()
                output_probs.append(chosen_prob)

                # Stop if the model generates the end of sequence token
                if next_token == tokenizer.eos_token_id:
                    break

            # Decode the output sequence
            output_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
            return output_text, output_probs
        
    from torch.nn.functional import log_softmax

    def generate_completion_with_beam_search(
        model, tokenizer, text, max_length=50, beam_size=3
    ):
        model.eval()
        with torch.no_grad():
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            
            pad_token_id = tokenizer.pad_token_id
            decoder_input_ids = torch.full((input_ids.shape[0], 1), pad_token_id, dtype=torch.long, device=input_ids.device)
            
            # Prepare output probabilities list and beam list
            output_probs = []
            beams = [(decoder_input_ids, 1.0)]
            
            for _ in range(max_length):
                new_beams = []
                for beam_input_ids, beam_prob in beams:
                    # Get logits for the next token
                    outputs = model(input_ids=input_ids, decoder_input_ids=beam_input_ids, return_dict=True)
                    next_token_logits = outputs.logits[:, -1, :]

                    # Apply log_softmax to get log probabilities
                    log_probs = log_softmax(next_token_logits, dim=-1)

                    # Get top K tokens and their corresponding log probabilities
                    topk_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

                    for i in range(beam_size):
                        next_token = topk_indices[0, i].unsqueeze(0)
                        next_token_prob = topk_probs[0, i].item()

                        new_beam_input_ids = torch.cat([beam_input_ids, next_token], dim=1)
                        new_beam_prob = beam_prob * next_token_prob

                        new_beams.append((new_beam_input_ids, new_beam_prob))

                # Select top K beams based on their probabilities
                sorted_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
                beams = sorted_beams[:beam_size]

                # Check if any of the beams generated the end of sequence token
                eos_beam = next((beam for beam in beams if beam[0][0, -1] == tokenizer.eos_token_id), None)
                if eos_beam:
                    decoder_input_ids, _ = eos_beam
                    break

            # Decode the output sequence
            output_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

            # Calculate the total output probability for the predicted beam
            total_output_prob = beams[0][1]

            return output_text, total_output_prob

    # Load pre-trained model and tokenizer

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    model, _ = load_model_from_checkpoint(args.model_dir, model)
    model.to(device)

    #print("STARTING INFERENCING")

    def generate_completion(model, tokenizer, text, max_length=50):
        model.eval()
        with torch.no_grad():
            input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
            output_ids = model.generate(input_ids, max_length=max_length)

            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def process_sentences(sentences, model, tokenizer):
        for sentence in sentences:
            num_splits = min(57, len(sentence))
            for _ in range(num_splits):
                split_index = random.randint(1, len(sentence))  # Random split point
                input_part = sentence[:split_index]
                ground_truth = sentence[split_index:]
                #output_text = generate_completion(model, tokenizer, input_part)
                #print(input_part,"\t", ground_truth, "\t", output_text)
                
                # Generate completion and probabilities
                if args.inference_method is "beam":
                    output_text, output_probs = generate_completion_with_beam_search(
                        model, tokenizer, input_part
                    )
                elif args.inference_method is "top_p":
                    output_text, output_probs = generate_completion_with_top_p(
                        model, tokenizer, input_part
                    )

                # Calculate the average probability of the generated output
                avg_prob = sum(output_probs) / len(output_probs) if output_probs else 0

                # Print the required information
                print(f"{input_part}\t{ground_truth}\t{output_text}\t{ avg_prob}\t{1}")
                
    process_sentences(sentences, model, tokenizer)


if __name__ == "__main__":
    main(get_args())
