import pandas as pd
from icecream import ic
from calculate_metrics import MetricsCalculator


def evaluate():
    df = pd.read_csv(
        '../../data/style-transfer/study_pairs_writing_results.csv')
    df = df.sort_values(by=['post_id'], ascending=True)


    ic(df.head())
    ic(df.columns)
    ic(df[['post_text', 'rewrite']])

    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df_folds = pd.read_csv(ds_path)
    df_folds = df_folds[df_folds['fold0.0'] == 'TEST']
    df_folds = df_folds[df_folds['Inappropriateness'] == 1]

    # print post_texts that are in df_folds but not in df
    ic(df_folds[~df_folds['post_text'].isin(df['post_text'])])
    # print post_texts that are in df but not in df_folds
    ic(df[~df['post_text'].isin(df_folds['post_text'])])

    metric_calculator = MetricsCalculator()
    tmp_predictions = [' '.join([x, y]) for x, y in zip(df['issue'], df['rewrite'])]
    tmp_prompts = [' '.join([x, y]) for x, y in zip(df['issue'], df['post_text'])]
    metrics = metric_calculator.calculate_metrics(
        tmp_predictions, tmp_prompts)
    # other way around
    # metrics = metric_calculator.calculate_metrics(df.post_text.tolist(), df['prompt_gpt_3.5'].tolist())
    ic('Semantic Similarity: '+str(metrics['mean_semantic_similarity']))
    ic('# of Edits: '+str(metrics['mean_token_edit_distance']))
    ic('GT Perplexity: '+str(metrics['mean_gt_perplexity']))
    ic('Perplexity: '+str(metrics['mean_prompt_perplexity']))
    ic('Classifier Prediction (Appropriateness): ' +
       str(metrics['mean_classifier_prediction_app']))
    ic('Classifier Prediction (Inappropriateness): ' +
       str(metrics['mean_classifier_prediction_inapp']))
    ic('Classifier Prediction: '+str(metrics['mean_classifier_prediction']))
    df_eval_baseliens_test = pd.read_csv('../../data/style-transfer/baselines/eval_baselines_test.csv')
    df_eval_baseliens_test = pd.concat([df_eval_baseliens_test, pd.DataFrame({
        'model_name': 'human',
        'mean_semantic_similarity': metrics['mean_semantic_similarity'],
        'mean_token_edit_distance': metrics['mean_token_edit_distance'],
        'mean_gt_perplexity': metrics['mean_gt_perplexity'],
        'mean_prompt_perplexity': metrics['mean_prompt_perplexity'],
        'mean_classifier_prediction_app': metrics['mean_classifier_prediction_app'],
        'mean_classifier_prediction_inapp': metrics['mean_classifier_prediction_inapp'],
        'mean_classifier_prediction': metrics['mean_classifier_prediction']
                                          }, index=[0])], ignore_index=True)

    df_eval_baseliens_test.to_csv('../../data/style-transfer/baselines/eval_baselines_test.csv', index=False)

if __name__ == '__main__':
    evaluate()
