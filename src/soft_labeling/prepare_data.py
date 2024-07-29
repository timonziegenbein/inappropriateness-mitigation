import pandas as pd
import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


def process_issue(x):
    x = x.replace('-', ' ').strip().capitalize()
    if x[-1] != ['.', '!', '?', ':']:
        x = x+':'
    return x


def prepare_GAQ():
    # read csv files
    df_qa_dev = pd.read_csv('../../data/GAQCorpus_split/qa_forums_mixtrain_overlaptest_dev.csv')
    df_qa_train = pd.read_csv('../../data/GAQCorpus_split/qa_forums_mixtrain_overlaptest_train.csv')
    df_qa_dev['type'] = 'qa'
    df_qa_train['type'] = 'qa'
    # read debate files
    df_debate_dev = pd.read_csv('../../data/GAQCorpus_split/debate_forums_mixtrain_overlaptest_dev.csv')
    df_debate_train = pd.read_csv('../../data/GAQCorpus_split/debate_forums_mixtrain_overlaptest_train.csv')
    df_debate_dev['type'] = df_debate_dev['id'].apply(lambda x: 'reddit' if 'reddit' in x else 'convinceme')
    df_debate_train['type'] = df_debate_train['id'].apply(lambda x: 'reddit' if 'reddit' in x else 'convinceme')
    # read review files
    df_review_dev = pd.read_csv('../../data/GAQCorpus_split/review_forums_mixtrain_overlaptest_dev.csv')
    df_review_train = pd.read_csv('../../data/GAQCorpus_split/review_forums_mixtrain_overlaptest_train.csv')
    df_review_dev['type'] = 'review'
    df_review_train['type'] = 'review'

    print('Length of QA: ', len(df_qa_dev)+len(df_qa_train))
    print('Length of CMV: ', len(df_debate_dev[df_debate_dev['type'] == 'reddit'])+len(df_debate_train[df_debate_train['type'] == 'reddit']))
    print('Length of Convinceme: ', len(df_debate_dev[df_debate_dev['type'] == 'convinceme'])+len(df_debate_train[df_debate_train['type'] == 'convinceme']))
    print('Length of Review: ', len(df_review_dev)+len(df_review_train))

    df_dev = pd.concat([df_qa_dev, df_debate_dev, df_review_dev])
    df_train = pd.concat([df_qa_train, df_debate_train, df_review_train])

    df = pd.concat([df_dev, df_train])

    df['text'] = df['text'].apply(lambda x: x.replace('\t', ' '))
    df['title'] = df['title'].apply(lambda x: x.replace('\t', ' '))
    df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))
    df['title'] = df['title'].apply(lambda x: x.replace('\n', ' '))

    df['titel'] = df['title'].apply(process_issue)

    df['arg_issue'] = df[['title','text']].apply(lambda x: ' '.join(x), axis = 1)

    return df


def prepare_iac2():
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    # read csv files
    df_createdebate = pd.read_csv('../../data/iac2/createdebate_post_view.csv')
    df_convinceme = pd.read_csv('../../data/iac2/convinceme_post_view.csv')
    print('Length of createdebate before removal: ', len(df_createdebate))
    print('Length of convinceme before removal: ', len(df_convinceme))

    # filter out UKPv2 and GAQCorpus
    discussion_ids_to_exclude_convinceme = [44, 66, 47, 328, 187]
    df_convinceme = df_convinceme[df_convinceme['discussion_id'].apply(lambda x: x not in discussion_ids_to_exclude_convinceme)]
    print('Length of convinceme after UKPv2 removal: ', len(df_convinceme))
    df_debate_test = pd.read_csv('../../data/GAQCorpus_split/debate_forums_mixtrain_overlaptest_mixtest.csv')
    df_debate_test = df_debate_test[df_debate_test['id'].apply(lambda x: 'reddit' not in x)]
    discussion_ids_to_exclude_convinceme = []
    for id_ in df_debate_test['id']:
        if int(id_.split('-')[-1]) in df_convinceme['text_id'].values.tolist():
            tmp_discussion_id = df_convinceme[df_convinceme['text_id'] == int(id_.split('-')[-1])].discussion_id.values.tolist()[0]
            discussion_ids_to_exclude_convinceme.append(tmp_discussion_id)
    discussion_ids_to_exclude_convinceme = list(set(discussion_ids_to_exclude_convinceme))
    df_convinceme = df_convinceme[df_convinceme['discussion_id'].apply(lambda x: x not in discussion_ids_to_exclude_convinceme)]
    print('Length of convinceme after GAQCorpus removal: ', len(df_convinceme))

    #preprocess discussion title and text 
    df_createdebate['text'] = df_createdebate['text'].apply(lambda x: x[2:-1])
    df_convinceme['text'] = df_convinceme['text'].apply(lambda x: x[2:-1])
    df_createdebate['discussion_title'] = df_createdebate['discussion_title'].apply(lambda x: process_issue(x[2:-1]))
    df_convinceme['discussion_title'] = df_convinceme['discussion_title'].apply(lambda x: process_issue(x[2:-1]))

    df_createdebate['text'] = df_createdebate['text'].apply(lambda x: x.replace('\t', ' '))
    df_createdebate['text'] = df_createdebate['text'].apply(lambda x: x.replace('\n', ' '))
    df_convinceme['text'] = df_convinceme['text'].apply(lambda x: x.replace('\t', ' '))
    df_convinceme['text'] = df_convinceme['text'].apply(lambda x: x.replace('\n', ' '))
    df_createdebate['discussion_title'] = df_createdebate['discussion_title'].apply(lambda x: x.replace('\t', ' '))
    df_createdebate['discussion_title'] = df_createdebate['discussion_title'].apply(lambda x: x.replace('\n', ' '))
    df_convinceme['discussion_title'] = df_convinceme['discussion_title'].apply(lambda x: x.replace('\t', ' '))
    df_convinceme['discussion_title'] = df_convinceme['discussion_title'].apply(lambda x: x.replace('\n', ' '))

    #concatenate discussion title and text
    df_createdebate['arg_issue'] = df_createdebate[['discussion_title', 'text']].apply(lambda x: x[0]+': '+x[1], axis=1)
    df_convinceme['arg_issue'] = df_convinceme[['discussion_title', 'text']].apply(lambda x: x[0]+': '+x[1], axis=1)

    # filter out short and long texts
    df_createdebate['num_words'] = df_createdebate['text'].apply(lambda x: len(tokenizer(x)))
    df_createdebate = df_createdebate[df_createdebate['num_words'] <= 220]
    df_createdebate = df_createdebate[df_createdebate['num_words'] >= 10]
    df_createdebate = df_createdebate[df_createdebate['text'].apply(lambda x: len(x)) <= 1100]

    df_convinceme['num_words'] = df_convinceme['text'].apply(lambda x: len(tokenizer(x)))
    df_convinceme = df_convinceme[df_convinceme['num_words'] <= 220]
    df_convinceme = df_convinceme[df_convinceme['num_words'] >= 10]
    df_convinceme = df_convinceme[df_convinceme['text'].apply(lambda x: len(x)) <= 1100]

    return df_createdebate, df_convinceme


def prepare_datasets():
    df_createdebate, df_convinceme = prepare_iac2()
    print('-'*50)
    df_GAQ = prepare_GAQ()
    print('-'*50)

    # concatenate all texts
    texts = df_createdebate['text'].values.tolist() + df_convinceme['text'].values.tolist() + df_GAQ['text'].values.tolist()
    texts_len = [len(text) for text in texts]

    # filter datasets based on percentile of text Length
    df_createdebate = df_createdebate[df_createdebate['text'].str.len() < int(np.percentile(texts_len, 95))]
    df_createdebate = df_createdebate[df_createdebate['text'].str.len() > int(np.percentile(texts_len, 5))]
    df_convinceme = df_convinceme[df_convinceme['text'].str.len() < int(np.percentile(texts_len, 95))]
    df_convinceme = df_convinceme[df_convinceme['text'].str.len() > int(np.percentile(texts_len, 5))]
    df_GAQ = df_GAQ[df_GAQ['text'].str.len() < int(np.percentile(texts_len, 95))]
    df_GAQ = df_GAQ[df_GAQ['text'].str.len() > int(np.percentile(texts_len, 5))]

    print('Length of createdebate after filtering: ', len(df_createdebate))
    print('Length of convinceme after filtering: ', len(df_convinceme))
    print('Length of QA after filtering: ', len(df_GAQ[df_GAQ['type'] == 'qa']))
    print('Length of CMV after filtering: ', len(df_GAQ[df_GAQ['type'] == 'reddit']))
    print('Length of Convinceme after filtering: ', len(df_GAQ[df_GAQ['type'] == 'convinceme']))
    print('Length of Review after filtering: ', len(df_GAQ[df_GAQ['type'] == 'review']))
    print('-'*50)
    print('Total length: ', len(df_createdebate) + len(df_convinceme) + len(df_GAQ))

    df_createdebate.to_csv('../../data/iac2/createdebate.csv', index = False, sep = '\t')
    df_convinceme.to_csv('../../data/iac2/convinceme.csv', index = False, sep = '\t')
    df_GAQ.to_csv('../../data/GAQCorpus_split/GAQ.csv', index = False, sep = '\t')


def add_prediction(df, predictions):
    # set column names of predictions to post_id and prediction
    predictions.columns = ['post_id', 'prediction']
    df['prediction'] = predictions['prediction'].values.tolist()
    df['prediction'] = df['prediction'].apply(lambda x: [x.split(",")[0][1:], x.split(",")[1][1:-1]])
    df['appropriate'] = df['prediction'].apply(lambda x: float(x[0]))
    df['inappropriate'] = df['prediction'].apply(lambda x: float(x[1]))
    return df


def filter_out_appropriate():
    df_createdebate = pd.read_csv('/bigwork/nhwpziet/appropriateness-style-transfer/data/iac2/createdebate.csv', sep='\t')
    df_convinceme = pd.read_csv('/bigwork/nhwpziet/appropriateness-style-transfer/data/iac2/convinceme.csv', sep='\t')
    df_GAQ = pd.read_csv('/bigwork/nhwpziet/appropriateness-style-transfer/data/GAQCorpus_split/GAQ.csv', sep='\t')

    print(df_GAQ.columns)

    df_createdebate = add_prediction(df_createdebate, pd.read_csv(
        "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_createdebate.txt", sep="\t", header=None))
    df_convinceme = add_prediction(df_convinceme, pd.read_csv(
        "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_convinceme.txt", sep="\t", header=None))
    df_GAQ = add_prediction(df_GAQ, pd.read_csv(
        "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_GAQ.txt", sep="\t", header=None))

    df_createdebate = df_createdebate[df_createdebate['inappropriate'] >= 0.5]
    df_convinceme = df_convinceme[df_convinceme['inappropriate'] >= 0.5]
    df_GAQ = df_GAQ[df_GAQ['inappropriate'] >= 0.5]

    print('Length of createdebate after filtering out appropriate: ', len(df_createdebate))
    print('Length of convinceme after filtering out appropriate: ', len(df_convinceme))
    print('Length of QA after filtering: ', len(df_GAQ[df_GAQ['type'] == 'qa']))
    print('Length of CMV after filtering: ', len(df_GAQ[df_GAQ['type'] == 'reddit']))
    print('Length of Convinceme after filtering: ', len(df_GAQ[df_GAQ['type'] == 'convinceme']))
    print('Length of Review after filtering: ', len(df_GAQ[df_GAQ['type'] == 'review']))
    print('-'*50)
    print('Total length: ', len(df_createdebate) + len(df_convinceme) + len(df_GAQ))


if __name__ == "__main__":
    prepare_datasets()
    #filter_out_appropriate()
