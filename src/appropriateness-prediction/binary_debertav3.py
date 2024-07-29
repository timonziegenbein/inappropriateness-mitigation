import argparse
from functools import partial
import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from spacy.lang.en import English

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"


class AppropriatenessDataset:
    def __init__(self, df, tokenizer, text_col, shuffle):
        self.df = df
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.shuffle = shuffle
        if self.shuffle:
            self.spacy_tokenizer = English().tokenizer

    def __getitem__(self, idx):
        argument = self.df.iloc[idx]
        text = argument[self.text_col].strip()
        if self.shuffle:
            spacy_tokenized_text = [x.text for x in self.spacy_tokenizer(text)]
            text = ' '.join(random.sample(spacy_tokenized_text, len(spacy_tokenized_text)))
        encoding = self.tokenizer(
            text,
            truncation=True,
        )

        out = {
            "id": argument["post_id"],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        if 'Inappropriateness' in argument.keys():
            label = argument['Inappropriateness']
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, required=False)
    parser.add_argument("--repeat", type=int, default=0, required=False)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--issue", dest='issue', action='store_true', default=False, required=False)
    parser.add_argument("--shuffle", dest='shuffle', action='store_true', default=False, required=False)

    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def model_init(df):
    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/deberta-v3-large', problem_type="single_label_classification", num_labels=2)
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    out_dict = {}
    scores = precision_recall_fscore_support(labels, predictions, average='macro')
    out_dict['Inappropriateness_macroF1'] = scores[2]
    scores = precision_recall_fscore_support(labels, predictions, average='micro')
    out_dict['Inappropriateness_microF1'] = scores[2]
    scores = precision_recall_fscore_support(labels, predictions, average='binary', pos_label=1)
    out_dict['Inappropriateness_binaryF1'] = scores[2]
    scores = precision_recall_fscore_support(labels, predictions, average='binary', pos_label=0)
    out_dict['Appropriateness_binaryF1'] = scores[2]

    return out_dict


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.input)

    if args.issue:
        text_col = 'arg_issue'
    else:
        text_col = 'post_text'

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    train_dataset = AppropriatenessDataset(
        df[df["fold{}.{}".format(args.repeat, args.fold)] == 'TRAIN'], tokenizer, text_col, args.shuffle)
    valid_dataset = AppropriatenessDataset(
        df[df["fold{}.{}".format(args.repeat, args.fold)] == 'VALID'], tokenizer, text_col, args.shuffle)
    test_dataset = AppropriatenessDataset(
        df[df["fold{}.{}".format(args.repeat, args.fold)] == 'TEST'], tokenizer, text_col, args.shuffle)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output, f"fold{args.repeat}/{args.fold}"),
        report_to=None,
        learning_rate=3e-6,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=150,
        save_steps=150,
        load_best_model_at_end=True,
        num_train_epochs=10,
        max_steps=-1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.1,
        logging_dir=os.path.join(args.output, f"fold{args.repeat}/{args.fold}")+'/logs',
        skip_memory_metrics=True,
        disable_tqdm=False,
        metric_for_best_model='Inappropriateness_macroF1',
        greater_is_better=True,
        warmup_ratio=0.1,
        lr_scheduler_type="polynomial",
    )

    trainer = Trainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=Collate(tokenizer=tokenizer),
        tokenizer=tokenizer,
        model_init=partial(model_init, df),
        #model=AutoModelForSequenceClassification.from_pretrained(os.path.join(
        #    args.output, f"fold{args.repeat}/{args.fold}/checkpoint-1800"), problem_type="single_label_classification", num_labels=2),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.eval_dataset = test_dataset
    metrics = trainer.evaluate()
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    # get predictions from trainer
    #predictions = trainer.predict(test_dataset).predictions
    #predictions = np.argmax(predictions, axis=1)
    ## save predictions and post_ids from dataset to file
    #with open(os.path.join(args.output, f"fold{args.repeat}/{args.fold}/predictions.txt"), "w") as f:
    #    for post_id, prediction in zip(test_dataset.df.post_id.tolist(), predictions):
    #        f.write("{}\t{}\n".format(post_id, prediction))

    ## read re-written test dataframe
    #df = pd.read_csv('../../data/style-transfer/appropriateness_corpus_conservative_prompt_gpt_3.5.csv')
    ## filter df for test posts by post_id
    #df = df[df.post_id.isin(test_dataset.df.post_id.tolist())]
    ## create column that combines issue and prompt_gpt_3.5 columns in df to single column
    #df['prompt_gpt_3.5'] = df['issue'] + ' ' + df['prompt_gpt_3.5']
    ## create AppropriatenessDataset with re-written test dataframe
    #test_dataset = AppropriatenessDataset(df, tokenizer, 'prompt_gpt_3.5', False)
    ## get predictions from trainer
    #predictions = trainer.predict(test_dataset).predictions
    #predictions = np.argmax(predictions, axis=1)
    ## save predictions and post_ids from dataset to file
    #with open(os.path.join(args.output, f"fold{args.repeat}/{args.fold}/predictions_gpt_3.5.txt"), "w") as f:
    #    for post_id, prediction in zip(test_dataset.df.post_id.tolist(), predictions):
    #        f.write("{}\t{}\n".format(post_id, prediction))
