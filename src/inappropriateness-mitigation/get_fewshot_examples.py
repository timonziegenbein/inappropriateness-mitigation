import pandas as pd
from calculate_metrics import MetricsCalculator
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def get_semantic_similarity(doc1, doc2):
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 512
    embeddings1 = model.encode([doc1])
    embeddings2 = model.encode([doc2])
    return cosine_similarity(embeddings1, embeddings2)[0][0]


def get_pairs_of_two(docs):
    pairs = []
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            pairs.append((docs[i], docs[j]))
    return pairs


def get_pagerank(docs):
    pairs = get_pairs_of_two(docs)
    similarities = []
    for pair in pairs:
        similarities.append(get_semantic_similarity(pair[0], pair[1]))
    similarity_matrix = np.zeros((len(docs), len(docs)))
    for i in range(len(pairs)):
        for j in range(len(docs)):
            if pairs[i][0] == docs[j]:
                for k in range(len(docs)):
                    if pairs[i][1] == docs[k]:
                        similarity_matrix[j][k] = similarities[i]
                        similarity_matrix[k][j] = similarities[i]

    pr = nx.pagerank(nx.from_numpy_array(similarity_matrix))
    return pr


def get_top_k_docs(docs, k):
    pr = get_pagerank(docs)
    top_k_docs = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return top_k_docs[:k]


def get_metric_based_ranking(docs, issue, prompt, calculator):
    docs = [x for x in docs if x != '-']
    tmp_prompts = [prompt] * len(docs)
    tmp_predictions = docs
    metrics = metric_calculator.calculate_metrics(
        tmp_predictions, tmp_prompts)
    semantic_similarities = metrics['semantic_similarities']
    token_edit_distances = metrics['token_edit_distances']
    prompt_perplexities = metrics['prompt_perplexities']
    classifier_predictions_app = metrics['classifier_predictions_app']
    print((semantic_similarities, token_edit_distances, prompt_perplexities, classifier_predictions_app))
    # double argsort to get rank
    semantic_similarities_rank = np.argsort(np.argsort(semantic_similarities)[::-1])
    token_edit_distances_rank = np.argsort(np.argsort(token_edit_distances))
    prompt_perplexities_rank = np.argsort(np.argsort(prompt_perplexities))
    classifier_predictions_app_rank = np.argsort(np.argsort(classifier_predictions_app)[::-1])
    # calculate average rank
    avg_rank = []
    for i in range(len(docs)):
        avg_rank.append(np.mean([semantic_similarities_rank[i],
                                 token_edit_distances_rank[i],
                                 prompt_perplexities_rank[i],
                                 classifier_predictions_app_rank[i]]))
    # return top k docs
    top_k_docs = sorted(zip(avg_rank, docs))
    return top_k_docs[:1]


def get_harmonic_mean_based_ranking(docs, issue, prompt, calculator):
    docs = [x for x in docs if x != '-']
    combined_prompt = ' '.join([issue, prompt])
    tmp_prompts = [combined_prompt] * len(docs)
    tmp_predictions = [' '.join([x, y]) for x, y in zip(issue, docs)]
    metrics = metric_calculator.calculate_metrics(
        tmp_predictions, tmp_prompts)
    semantic_similarities = metrics['semantic_similarities']
    token_edit_distances = metrics['token_edit_distances']
    prompt_perplexities = metrics['prompt_perplexities']
    classifier_predictions_app = metrics['classifier_predictions_app']
    print((semantic_similarities, token_edit_distances, prompt_perplexities, classifier_predictions_app))
    # get harmonic mean of semantic similarities and classifier predictions_app
    harmonic_mean = []
    for i in range(len(docs)):
        harmonic_mean.append(2 * semantic_similarities[i] * classifier_predictions_app[i] /
                             (semantic_similarities[i] + classifier_predictions_app[i]))
    top_k_docs = sorted(zip(harmonic_mean, docs))
    return top_k_docs[:1]


if __name__ == '__main__':
    df = pd.read_csv('../../data/style-transfer/few-shot_examples.csv').reset_index(drop=True)
    metric_calculator = MetricsCalculator(semantic_similarity=True,
                                          token_edit_distance=True,
                                          perplexity=True,
                                          classifier_prediction=True,
                                          batch_size=4)
    for i in range(len(df)):
        tmp_docs = [
            #df['Rewritten more appropriate argument (Timon)'][i],
            df['Rewritten more appropriate argument (Max)'][i],
            df['Rewritten more appropriate argument (Maja)'][i],
            df['Rewritten more appropriate argument (Meghdut)'][i],
            df['Rewritten more appropriate argument (Alireza)'][i],
        ]
        #top_k_docs = get_top_k_docs(tmp_docs, 1)
        # print(tmp_docs[top_k_docs[0][0]])
        #top_k_docs = get_metric_based_ranking(tmp_docs, df['Topic'][i], df['Argument'][i], metric_calculator)
        top_k_docs = get_harmonic_mean_based_ranking(tmp_docs, df['Topic'][i], df['Argument'][i], metric_calculator)
        print(top_k_docs[0])

    tmp_docs = [
        df['Rewritten more appropriate argument (Max)'],
        df['Rewritten more appropriate argument (Maja)'],
        df['Rewritten more appropriate argument (Meghdut)'],
        df['Rewritten more appropriate argument (Alireza)'],
    ]

    tmp_arguments = [
        df['Argument'],
        df['Argument'],
        df['Argument'],
        df['Argument'],
    ]

    tmp_issues = [
        df['Topic'],
        df['Topic'],
        df['Topic'],
        df['Topic'],
    ]

    # flatten
    tmp_docs = [item for sublist in tmp_docs for item in sublist]
    tmp_arguments = [item for sublist in tmp_arguments for item in sublist]
    tmp_issues = [item for sublist in tmp_issues for item in sublist]

    keep_idx = [i for i, x in enumerate(tmp_docs) if x != '-']
    tmp_docs = [tmp_docs[i] for i in keep_idx]
    tmp_arguments = [tmp_arguments[i] for i in keep_idx]
    tmp_issues = [tmp_issues[i] for i in keep_idx]

    tmp_prompts = [' '.join([x, y]) for x, y in zip(tmp_issues, tmp_arguments)]
    tmp_predictions = [' '.join([x, y]) for x, y in zip(tmp_issues, tmp_docs)]  

    metrics = metric_calculator.calculate_metrics(tmp_predictions, tmp_prompts)
    print((metrics['mean_semantic_similarity'],metrics['mean_token_edit_distance'],metrics['mean_prompt_perplexity'],metrics['mean_classifier_prediction']))


    # now let's do it for each annotator
    tmp_annotator_metrics = []
    for annotator in ['Max', 'Maja', 'Meghdut', 'Alireza']:
        tmp_docs = [
            df['Rewritten more appropriate argument (%s)' % annotator],
        ]

        tmp_arguments = [
            df['Argument'],
        ]

        tmp_issues = [
            df['Topic'],
        ]

        # flatten
        tmp_docs = [item for sublist in tmp_docs for item in sublist]
        tmp_arguments = [item for sublist in tmp_arguments for item in sublist]
        tmp_issues = [item for sublist in tmp_issues for item in sublist]

        keep_idx = [i for i, x in enumerate(tmp_docs) if x != '-']
        tmp_docs = [tmp_docs[i] for i in keep_idx]
        tmp_arguments = [tmp_arguments[i] for i in keep_idx]
        tmp_issues = [tmp_issues[i] for i in keep_idx]

        tmp_prompts = [' '.join([x, y]) for x, y in zip(tmp_issues, tmp_arguments)]
        tmp_predictions = [' '.join([x, y]) for x, y in zip(tmp_issues, tmp_docs)]

        metrics = metric_calculator.calculate_metrics(tmp_predictions, tmp_prompts)
        print((metrics['mean_semantic_similarity'],metrics['mean_token_edit_distance'],metrics['mean_prompt_perplexity'],metrics['mean_classifier_prediction']))
        tmp_annotator_metrics.append([metrics['mean_semantic_similarity'],metrics['mean_token_edit_distance'],metrics['mean_prompt_perplexity'],metrics['mean_classifier_prediction']])
    #average
    tmp_annotator_metrics = np.array(tmp_annotator_metrics)
    print(np.mean(tmp_annotator_metrics, axis=0))
    #confidence intervaqrt(len(tmp_annotator_metrics)))
    upper = np.mean(tmp_annotator_metrics, axis=0) + 1.96 * np.std(tmp_annotator_metrics, axis=0) / np.sqrt(len(tmp_annotator_metrics))
    lower = np.mean(tmp_annotator_metrics, axis=0) - 1.96 * np.std(tmp_annotator_metrics, axis=0) / np.sqrt(len(tmp_annotator_metrics))
    print(upper)
    print(lower)
