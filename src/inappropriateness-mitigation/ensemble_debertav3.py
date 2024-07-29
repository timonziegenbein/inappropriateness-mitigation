import argparse
import os
import warnings
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

import torch
from torch.nn import functional as F

from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"


class HFDataset(Dataset):
    def __init__(self, df, tokenizer, text_col, id_col, label_col=None):
        self.df = df
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.id_col = id_col
        self.label_col = label_col

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row[self.text_col].strip()
        encoding = self.tokenizer(
            text,
            truncation=True,
        )

        out = {
            "id": row[self.id_col],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            }

        if self.label_col is not None:
            label = row[self.label_col]
            out["label"] = label

        return out

    def __len__(self):
        return len(self.df)


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if 'label' in batch[0].keys():
            output["labels"] = [sample["label"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
        output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if 'label' in batch[0].keys():
            output["labels"] = torch.tensor(output["labels"], dtype=torch.long)

        return output


class DebertaPredictor_old:
    def __init__(self, checkpoint, tokenizer, batch_size=7, device_map=None, device="cuda:0"):
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer

        self.training_args = TrainingArguments(
            output_dir=checkpoint,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=4
        )

        if device_map:
            self.trainer = Trainer(
                args=self.training_args,
                data_collator=Collate(tokenizer=self.tokenizer),
                tokenizer=self.tokenizer,
                model=AutoModelForSequenceClassification.from_pretrained(checkpoint, device_map=device_map)
            )
        else:
            self.trainer = Trainer(
                args=self.training_args,
                data_collator=Collate(tokenizer=self.tokenizer),
                tokenizer=self.tokenizer,
                model=AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
            )

    def predict(self, x):
        tmp_df = pd.DataFrame({"text": x, "id_col": [i for i in range(len(x))]})
        inference_dataset = HFDataset(tmp_df, self.tokenizer, "text", "id_col")

        # adjust batch size to number of samples if batch size is larger than number of samples
        #if len(x) < self.training_args.per_device_eval_batch_size:
            # TODO
        print(f"Predicting {len(x)} samples...")
        with torch.no_grad():
            preds = self.trainer.predict(inference_dataset).predictions
        return preds


class DebertaPredictor:
    def __init__(self, checkpoint, tokenizer, batch_size=7, device_map=None, device="cuda:0"):
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(device)
        self.model.half()
        self.model.eval()

        self.batch_size = batch_size
        self.device = device

    def predict(self, samples):
        scores_list = []
        for i in range(0, len(samples), self.batch_size):
            sub_samples = samples[i:i + self.batch_size]
            encodings_dict = self.tokenizer(
                sub_samples,
                truncation=True,
                #max_length=config.train.seq_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(self.device)
            attn_masks = encodings_dict["attention_mask"].to(self.device)
            with torch.no_grad():
                sub_scores = self.model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list += sub_scores.logits.cpu().numpy().tolist()
        return np.array(scores_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=0, required=False)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--text_col", type=str, required=True)
    parser.add_argument("--label_col", type=str, required=False, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_count", type=int, required=True)
    parser.add_argument("--id_col", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input, sep='\t')

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    inference_dataset = HFDataset(df, tokenizer, args.text_col, args.id_col, args.label_col)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output, f"fold{args.repeat}/ensemble"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        dataloader_num_workers=8
    )

    # ensemble the prediction probabilities of 5 models
    probs = np.zeros((len(inference_dataset), 2))
    for i in range(args.model_count):
        trainer = Trainer(
                args=training_args,
                data_collator=Collate(tokenizer=tokenizer),
                tokenizer=tokenizer,
                model=AutoModelForSequenceClassification.from_pretrained(os.path.join(args.output, f"fold{args.repeat}/{i}", args.checkpoint))
        )
        # get the prediction probabilities of the model using softmax
        probs += F.softmax(torch.tensor(trainer.predict(inference_dataset).predictions), dim=1).numpy()
    # average the probabilities
    probs /= args.model_count


    # with dataset name
    with open(os.path.join(args.output, f"fold{args.repeat}/ensemble_predictions_{args.dataset_name}.txt"), "w") as f:
        for post_id, prediction in zip(inference_dataset.df[args.id_col].tolist(), probs.tolist()):
            f.write("{}\t{}\n".format(post_id, prediction))
