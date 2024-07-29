import pandas as pd
import numpy as np
import math
from bert_score import BERTScorer
from transformers import AutoTokenizer
from perplexity import Perplexity
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from ensemble_debertav3 import DebertaPredictor
from hirschberg import *



class MetricsCalculator:
    def __init__(self,
                 semantic_similarity=True,
                 token_edit_distance=True,
                 perplexity=True,
                 classifier_prediction=True,
                 batch_size=4,
                 classifier_fold=0,
                 device_map=None,
                 device='cuda:0'
                 ):
        self.semantic_similarity = semantic_similarity
        self.token_edit_distance = token_edit_distance
        self.perplexity = perplexity
        self.classifier_prediction = classifier_prediction
        self.batch_size = batch_size
        self.device_map = device_map
        self.device = device

        if semantic_similarity:
            self.semantic_similarity_model = BERTScorer(
                model_type="microsoft/deberta-xlarge-mnli", rescale_with_baseline=True, lang="en", batch_size=self.batch_size, device=self.device)

        if token_edit_distance:
            nlp = English()
            self.word_tokenizer = Tokenizer(nlp.vocab)

        if perplexity:
            self.perplexity_model = Perplexity(model_id='gpt2', device='gpu')

        if classifier_prediction:
            self.tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/deberta-v3-large")
            self.set_classifier_based_on_fold(classifier_fold)

    def set_classifier_based_on_fold(self, fold):
        self.classifier = DebertaPredictor(
            '/bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue/fold0/'+str(fold)+'/checkpoint-1800', self.tokenizer, self.batch_size, device=self.device, device_map=self.device_map)
        self.classifier_fold = fold

    def calculate_metrics(self, x, y):
        output = {}
        print('Calculating metrics...')

        if self.semantic_similarity:
            print('Calculating semantic similarity...')
            _, _, semantic_similarities = self.semantic_similarity_model.score(x, y)
            output['semantic_similarities'] = semantic_similarities.numpy()
            output['mean_semantic_similarity'] = np.mean(semantic_similarities.numpy())

        if self.token_edit_distance:
            print('Calculating token edit distance...')
            token_edit_distances = []

            def normalized_edit_similarity(m, d):
                # d : edit distance between the two strings
                # m : length of the shorter string
                if m == d:
                    return 0.0
                elif d == 0:
                    return 1.0
                else:
                    return (1.0 / math.exp( d / (m - d)))

            for i in range(len(x)):
                y_tokens = [token.text for token in self.word_tokenizer(y[i])]
                x_tokens = [token.text for token in self.word_tokenizer(x[i])]
                Z, W, S = Hirschberg(x_tokens,  y_tokens)
                num_edits = len([s for s in S if s != '<KEEP>'])
                #token_edit_distances.append(num_edits)
                normalized_num_edits = normalized_edit_similarity(len(S), num_edits)
                token_edit_distances.append(normalized_num_edits)
            output['token_edit_distances'] = token_edit_distances
            output['mean_token_edit_distance'] = np.mean(token_edit_distances)

        if self.perplexity:
            print('Calculating perplexity...')
            gt_perplexities = self.perplexity_model._compute(
                data=y, batch_size=self.batch_size)['perplexities']
            prompt_perplexities = self.perplexity_model._compute(
                data=x, batch_size=self.batch_size)['perplexities']
            output['gt_perplexities'] = gt_perplexities
            output['prompt_perplexities'] = prompt_perplexities
            output['mean_gt_perplexity'] = np.mean(gt_perplexities)
            output['mean_prompt_perplexity'] = np.mean(prompt_perplexities)

        if self.classifier_prediction:
            print('Calculating classifier prediction...')
            classifier_predictions = self.classifier.predict(x)
            classifier_predictions = np.exp(
                classifier_predictions) / np.sum(np.exp(classifier_predictions),
                                                 axis=1, keepdims=True)
            output['classifier_predictions_app'] = classifier_predictions[:, 0]
            output['classifier_predictions_inapp'] = classifier_predictions[:, 1]
            output['classifier_predictions'] = np.argmax(
                classifier_predictions, axis=1)
            output['classifier_predictions'] = [1.0 if x == 1 else 0.0 for x in output['classifier_predictions']]
            output['mean_classifier_prediction_app'] = np.mean(
                classifier_predictions[:, 0])
            output['mean_classifier_prediction_inapp'] = np.mean(
                classifier_predictions[:, 1])
            output['mean_classifier_prediction'] = np.mean(
                output['classifier_predictions'])
            print('Done calculating metrics.')

        return output


if __name__ == '__main__':
    df = pd.read_csv(
        '../../data/style-transfer/appropriateness_corpus_conservative_prompt_chatGPT.csv')
    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df_folds = pd.read_csv(ds_path)
    df['fold0.0'] = df_folds['fold0.0']
    df = df[df['fold0.0'] == 'TRAIN']
    df = df[df['Inappropriateness'] == 1]
    metric_calculator = MetricsCalculator()
    tmp_predictions = [' '.join([x, y]) for x, y in zip(df['issue'], df['prompt_gpt_3.5'])]
    tmp_prompts = [' '.join([x, y]) for x, y in zip(df['issue'], df['post_text'])]
    metrics = metric_calculator.calculate_metrics(
        tmp_predictions, tmp_prompts)
    # other way around
    # metrics = metric_calculator.calculate_metrics(df.post_text.tolist(), df['prompt_gpt_3.5'].tolist())
    print('Semantic Similarity: '+str(metrics['mean_semantic_similarity']))
    print('# of Edits: '+str(metrics['mean_token_edit_distance']))
    print('GT Perplexity: '+str(metrics['mean_gt_perplexity']))
    print('Perplexity: '+str(metrics['mean_prompt_perplexity']))
    print('Classifier Prediction (Appropriateness): ' +
          str(metrics['mean_classifier_prediction_app']))
    print('Classifier Prediction (Inappropriateness): ' +
          str(metrics['mean_classifier_prediction_inapp']))
    print('Classifier Prediction: '+str(metrics['mean_classifier_prediction']))
    df['semantic_similarity'] = metrics['semantic_similarities']
    df['token_edit_distance'] = metrics['token_edit_distances']
    df['gt_perplexity'] = metrics['gt_perplexities']
    df['prompt_perplexity'] = metrics['prompt_perplexities']
    df['classifier_predictions_app'] = metrics['classifier_predictions_app']
    df['classifier_predictions_inapp'] = metrics['classifier_predictions_inapp']
    df['classifier_predictions'] = metrics['classifier_predictions']
    #df.to_csv('../../data/style-transfer/appropriateness_corpus_conservative_prompt_gpt_3.5_analysis.csv', index=False)
