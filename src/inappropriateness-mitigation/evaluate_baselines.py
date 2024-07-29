import pandas as pd
import numpy as np
from calculate_metrics import MetricsCalculator

MODEL_LIST = [
    "EleutherAI/gpt-j-6B",
    "bigscience/bloom-7b1",
    "huggyllama/llama-7b",
    "facebook/opt-6.7b",
    "../../data/models/instruction-finetuning/llama-7b-instruct",
    "../../data/models/instruction-finetuning/gpt-j-6b-instruct",
    "../../data/models/instruction-finetuning/bloom-7b1-instruct",
    "../../data/models/instruction-finetuning/opt-6.7b-instruct",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-04a-06ss/best_checkpoint/",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-045a-055ss/best_checkpoint/",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-05a-05ss/best_checkpoint/",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-055a-045ss/best_checkpoint/",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-06a-04ss/best_checkpoint/",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-10a-00ss/best_checkpoint/",
    "gram-large-paraphrase-rewrite",
    "gram-large-formality-rewrite",
    "gram-large-neutral-rewrite",
    "gram-large-politeness-rewrite",
    "gram-xlarge-paraphrase-rewrite",
    "gram-xlarge-formality-rewrite",
    "gram-xlarge-neutral-rewrite",
    "gram-xlarge-politeness-rewrite"
]

NUM_SHOTS = [0,1,4,9]


def evaluate(metric_calculator, prompts, predictions, ids):
    fold_dict = get_fold_dict()

    metrics_input_dict = {}
    for prompt, prediction, id in zip(prompts, predictions, ids):
        fold = fold_dict[id]
        if fold not in metrics_input_dict:
            metrics_input_dict[fold] = {'prompts': [], 'predictions': [], 'ids': []}
        metrics_input_dict[fold]['prompts'].append(prompt)
        metrics_input_dict[fold]['predictions'].append(prediction)
        metrics_input_dict[fold]['ids'].append(id)

    tmp_metrics = []
    new_id_order = []
    for i in range(5):
        if i != metric_calculator.classifier_fold:
            metric_calculator.set_classifier_based_on_fold(i)
        tmp_prompts = metrics_input_dict[i]['prompts']
        tmp_predictions = metrics_input_dict[i]['predictions']
        tmp_metrics.append(metric_calculator.calculate_metrics(tmp_prompts, tmp_predictions))
        new_id_order += metrics_input_dict[i]['ids']

    # metrics is of type dict of lists
    metrics = {}
    for key in tmp_metrics[0].keys():
        for i in range(5):
            if key == 'model_name':
                metrics[key] = tmp_metrics[i][key]
            else:
                if 'mean' not in key:
                    if metrics.get(key) is None:
                        metrics[key] = []
                    metrics[key] += list(tmp_metrics[i][key])

    metrics['ids_order'] = new_id_order

    return metrics


def combine_metrics(mode):
    eval_dict = {
        'model_name': [],
        'mean_semantic_similarity': [],
        'mean_token_edit_distance': [],
        'mean_gt_perplexity': [],
        'mean_prompt_perplexity': [],
        'mean_classifier_prediction_app': [],
        'mean_classifier_prediction_inapp': [],
        'mean_classifier_prediction': [],
        'mean_gt_inapp': []
    }

    for model_name in MODEL_LIST:
        for shot in NUM_SHOTS:
            if ('checkpoint' in model_name or 'gram' in model_name) and shot != 0:
                continue
            else:
                combined_df = pd.read_csv(
                     '../../data/style-transfer/baselines/eval_{}_{}-shot.csv'.format(model_name.replace('/', '_'), shot))

            # select every 5th row starting from 0
            combined_df = combined_df.iloc[:2191]
            ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
            #print index of rows where Inappropriateness is 1

            combined_df['fold0.0'] = pd.read_csv(ds_path)['fold0.0'].tolist()
            combined_df['post_text2'] = pd.read_csv(ds_path)['post_text'].tolist()
            # print where post_text2 is not equal to post_text
            #print(combined_df[combined_df['post_text2'] != combined_df['post_text']].index)
            combined_df = combined_df[combined_df['fold0.0'] == mode]
            combined_df = combined_df[combined_df['Inappropriateness'] == 1]
            #print(combined_df[combined_df['Inappropriateness'] == 1]['post_text'][0:10])

            eval_dict['model_name'].append(model_name + '_{}'.format(shot))
            eval_dict['mean_semantic_similarity'].append(np.mean(combined_df['semantic_similarity']))
            eval_dict['mean_token_edit_distance'].append(np.mean(combined_df['token_edit_distance']))
            #print(combined_df['gt_perplexity'])
            eval_dict['mean_gt_perplexity'].append(np.mean(combined_df['gt_perplexity']))
            eval_dict['mean_prompt_perplexity'].append(np.mean(combined_df['prompt_perplexity']))
            eval_dict['mean_classifier_prediction_app'].append(np.mean(combined_df['classifier_predictions_app']))
            eval_dict['mean_classifier_prediction_inapp'].append(np.mean(combined_df['classifier_predictions_inapp']))
            eval_dict['mean_classifier_prediction'].append(np.mean(combined_df['classifier_predictions']))
            eval_dict['mean_gt_inapp'].append(np.mean(combined_df['Inappropriateness']))

    eval_df = pd.DataFrame(eval_dict)
    eval_df.to_csv('../../data/style-transfer/baselines/eval_baselines_{}.csv'.format(mode.lower()), index=False)


def get_fold_dict():
    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df = pd.read_csv(ds_path)

    fold_dict = {}
    for i in range(5):
        for j in range(len(df)):
            if df['fold0.{}'.format(i)][j] == 'TEST':
                fold_dict[j] = i

    return fold_dict


def calculate_metrics():
    metric_calculator = MetricsCalculator()

    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df = pd.read_csv(ds_path)
    #df = df[df['fold0.0'] == 'TEST']

    eval_dict = {
        'model_name': [],
        'mean_semantic_similarity': [],
        'mean_token_edit_distance': [],
        'mean_gt_perplexity': [],
        'mean_prompt_perplexity': [],
        'mean_classifier_prediction_app': [],
        'mean_classifier_prediction_inapp': [],
        'mean_classifier_prediction': [],
        'mean_gt_inapp': []
    }

    for model_name in MODEL_LIST:
        for shot in NUM_SHOTS:
            df1 = df.copy()
            df2 = df.copy()
            df3 = df.copy()
            df4 = df.copy()
            df5 = df.copy()
            dfs = [df1, df2, df3, df4, df5]

            save_path = '../../data/style-transfer/baselines/predictions_{}_{}-shot.csv'.format(
                model_name.replace('/', '_'), shot)

            predictions = []
            ids = []
            with open(save_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        predictions.append(line.split('\t')[1])
                        ids.append(int(line.split('\t')[0]))
                    except:
                        print(i)
                        print(line)
                        print(line.split('\t'))
                        break

            last_id = 0
            for id_ in ids:
                if id_ != 0:
                    if id_ != last_id+1:
                        print(id_)
                last_id = id_

            for i in range(len(dfs)):
                dfs[i]['predictions'] = predictions[i::5]

                tmp_predictions = [' '.join([x, y]) for x, y in zip(dfs[i]['issue'], dfs[i]['predictions'])]
                tmp_prompts = [' '.join([x, y]) for x, y in zip(dfs[i]['issue'], dfs[i]['post_text'])]
                metrics = evaluate(metric_calculator, tmp_predictions, tmp_prompts, ids)
                # sort values in metrics dict by ids in metrics dict

                dfs[i]['semantic_similarity'] = [x for _, x in sorted(zip(metrics['ids_order'], metrics['semantic_similarities']))]
                dfs[i]['token_edit_distance'] = [x for _, x in sorted(zip(metrics['ids_order'], metrics['token_edit_distances']))]
                dfs[i]['gt_perplexity'] = [x for _, x in sorted(zip(metrics['ids_order'], metrics['gt_perplexities']))]
                dfs[i]['prompt_perplexity'] = [x for _, x in sorted(zip(metrics['ids_order'], metrics['prompt_perplexities']))]
                dfs[i]['classifier_predictions_app'] = [x for _, x in sorted(zip(metrics['ids_order'], metrics['classifier_predictions_app']))]
                dfs[i]['classifier_predictions_inapp'] = [x for _, x in sorted(zip(metrics['ids_order'], metrics['classifier_predictions_inapp']))]
                dfs[i]['classifier_predictions'] = [x for _, x in sorted(zip(metrics['ids_order'], metrics['classifier_predictions']))]

            combined_df = pd.concat(dfs)
            print(combined_df.head())
            combined_df = combined_df[['issue', 'post_text', 'predictions', 'Inappropriateness', 'semantic_similarity', 'token_edit_distance',
                                       'gt_perplexity', 'prompt_perplexity', 'classifier_predictions_app', 'classifier_predictions_inapp', 'classifier_predictions']]
            combined_df.to_csv(
                '../../data/style-transfer/baselines/eval_{}_{}-shot.csv'.format(model_name.replace('/', '_'), shot), index=False)
            print('Done with {} {}-shot'.format(model_name, shot))


if __name__ == '__main__':
    #calculate_metrics()
    combine_metrics('TRAIN')
    combine_metrics('VALID')
    combine_metrics('TEST')
