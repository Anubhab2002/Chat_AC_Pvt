# import required libraries

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import argparse
import os
import wandb

print("PROCESS STARTED")

# Initialise wandb

wandb.init(project="t5_chat_autocompletion", entity="anu2002")

# use GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# create custom dataset for Training

class AutocompleteDataset(Dataset):
    def __init__(self, tokenizer, sentences, max_length=512):
        self.tokenizer = tokenizer
        self.pairs = []
        
        for sentence in sentences:
            input_text = sentence[:2*len(sentence)//3]
            target_text = sentence[2*len(sentence)//3:]

            input_encoded = tokenizer(input_text, padding="max_length", max_length=max_length, truncation=True, return_tensors='pt')
            target_encoded = tokenizer(target_text, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
            self.pairs.append((input_encoded, target_encoded))
            '''
            words = sentence.split()
            for i in range(1, len(words)):
                input_text = ' '.join(words[:i])
                target_text = ' '.join(words[i:])
                input_encoded = tokenizer(input_text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
                target_encoded = tokenizer(target_text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
                self.pairs.append((input_encoded, target_encoded))
            '''

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# setup cli args list

def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--bs',  type=int, default=4)
    args = parser.parse_args()
    return args

# get training data
def main(args):
    with open(args.train_data, "r") as f:
      data = f.read()

    dataset = data.split("\n")

    train_data = pd.DataFrame(dataset)
    sentences = train_data.values.flatten().tolist()

    # Load pre-trained model and tokenizer

    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model.to(device)
    # setup dataloader and optimizer

    dataset = AutocompleteDataset(tokenizer, sentences)
    print("total size of train dataset: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.bs)
    if args.val_data is not None:
        with open(args.val_data, "r") as f:
          data = f.read()

        dataset = data.split("\n")

        val_data = pd.DataFrame(dataset)
        val_sentences = val_data.values.flatten().tolist()
        val_dataset = AutocompleteDataset(tokenizer, val_sentences)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs)
        
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    if not os.path.exists(args.model_dir):
        # Create a new directory because it does not exist
        os.makedirs(args.model_dir)
        print("The model directory is created!")

    print("STARTING TRAINING")
    for epoch in range(args.num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            iterations = 0
            for batch in dataloader:
                inputs, targets = batch

                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = {k: v.to(device) for k, v in targets.items()}

                # Prepare data
                input_ids = inputs['input_ids'].squeeze(1)
                attention_mask = inputs['attention_mask'].squeeze(1)
                labels = targets['input_ids'].squeeze(1)

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                if iterations%200 == 0:
                    # Decode and print input text
                    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    print(f"Input Text (Epoch: {epoch}, Iteration {iterations}): {input_text}")
                    
                    # Generate model output and decode
                    with torch.no_grad():
                        model_output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
                        output_text = tokenizer.decode(model_output[0], skip_special_tokens=True)
                        print(f"Model Output (Epoch: {epoch}, Iteration {iterations}): {output_text}")
                    
                    wandb.log({
                        "training loss": loss.item(),
                        })
                
                total_train_loss += loss.item()
                iterations = iterations + 1

            avg_train_loss = total_train_loss / len(dataloader)

            if args.val_data is not None:
                # Validation phase
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        inputs, targets = batch

                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        targets = {k: v.to(device) for k, v in targets.items()}

                        # Prepare data
                        input_ids = inputs['input_ids'].squeeze(1)
                        attention_mask = inputs['attention_mask'].squeeze(1)
                        labels = targets['input_ids'].squeeze(1)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss

                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)

                print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
                wandb.log({
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss
                    })
                # Save model checkpoint at the end of each epoch
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                }
                checkpoint_path = os.path.join(args.model_dir, f'T5_chat_ac_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at '{checkpoint_path}'")
            else:
                print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}")
                wandb.log({
                    "avg_train_loss": avg_train_loss
                    })
                            # Save model checkpoint at the end of each epoch
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                }
                checkpoint_path = os.path.join(args.model_dir, f'T5_chat_ac_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at '{checkpoint_path}'")

if __name__ == "__main__":
    main(get_args())
