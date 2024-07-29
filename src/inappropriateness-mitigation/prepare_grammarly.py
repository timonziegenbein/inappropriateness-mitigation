import pandas as pd
from calculate_metrics import MetricsCalculator
from icecream import ic


def get_fold_dict():
    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df = pd.read_csv(ds_path)

    fold_dict = {}
    for i in range(5):
        for j in range(len(df)):
            if df['fold0.{}'.format(i)][j] == 'TEST':
                fold_dict[j] = i

    return fold_dict


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


def write_to_file(file_path, predictions):
    num_written = 0
    with open(file_path, 'w') as f:
        for prediction in predictions:
            prediction = prediction.replace("\n", " ")
            prediction = prediction.replace("\t", " ")
            f.write(f"{num_written}\t{prediction}\n")
            num_written += 1


def evaluate_gram(df, metric_calculator, model):
    for column in ['formality_rewrite', 'paraphrase_rewrite', 'neutral_rewrite', 'politeness_rewrite']:
        column_name = column.replace('_', '-')
        model_name = model+'-'+column_name
        save_path = '../../data/style-transfer/baselines/predictions_{}_{}-shot.csv'.format(model_name, 0)
        write_to_file(save_path, df[column].tolist())

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

        df['predictions'] = predictions

        tmp_predictions = [' '.join([x, y]) for x, y in zip(df['issue'], df['predictions'])]
        tmp_prompts = [' '.join([x, y]) for x, y in zip(df['issue'], df['post_text'])]
        metrics = evaluate(metric_calculator, tmp_predictions, tmp_prompts, ids)
        # sort values in metrics dict by ids in metrics dict

        df['semantic_similarity'] = [x for _, x in sorted(
            zip(metrics['ids_order'], metrics['semantic_similarities']))]
        df['token_edit_distance'] = [x for _, x in sorted(
            zip(metrics['ids_order'], metrics['token_edit_distances']))]
        df['gt_perplexity'] = [x for _, x in sorted(zip(metrics['ids_order'], metrics['gt_perplexities']))]
        df['prompt_perplexity'] = [x for _, x in sorted(
            zip(metrics['ids_order'], metrics['prompt_perplexities']))]
        df['classifier_predictions_app'] = [x for _, x in sorted(
            zip(metrics['ids_order'], metrics['classifier_predictions_app']))]
        df['classifier_predictions_inapp'] = [x for _, x in sorted(
            zip(metrics['ids_order'], metrics['classifier_predictions_inapp']))]
        df['classifier_predictions'] = [x for _, x in sorted(
            zip(metrics['ids_order'], metrics['classifier_predictions']))]

        tmp_df = df[['issue', 'post_text', 'predictions', 'Inappropriateness', 'semantic_similarity', 'token_edit_distance',
                             'gt_perplexity', 'prompt_perplexity', 'classifier_predictions_app', 'classifier_predictions_inapp', 'classifier_predictions']]
        tmp_df.to_csv(
            '../../data/style-transfer/baselines/eval_{}_{}-shot.csv'.format(model_name, 0), index=False)


if __name__ == '__main__':
    metric_calculator = MetricsCalculator()
    df_large = pd.read_csv("../../data/style-transfer/baselines/gram_large_rephrased.csv")
    df_xlarge = pd.read_csv("../../data/style-transfer/baselines/gram_xlarge_rephrased.csv")

    evaluate_gram(df_large, metric_calculator, 'gram-large')
    evaluate_gram(df_xlarge, metric_calculator, 'gram-xlarge')
