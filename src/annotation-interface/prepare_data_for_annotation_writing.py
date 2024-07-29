import pandas as pd


def prepare_for_interface():
    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df = pd.read_csv(ds_path)

    df = df[df['fold0.0'] == 'TEST']
    df = df[df['Inappropriateness'] == 1]

    df['rewrite_a'] = [''] * len(df)
    df['rewrite_b'] = [''] * len(df)

    df['source'] = df['post_text']

    df['id'] = [i for i in range(len(df))]

    # divide into 5 batches
    df['batch'] = [(i % 5)+1 for i in range(len(df))]

    df = df[['id', 'source', 'rewrite_a', 'rewrite_b', 'issue', 'batch']]

    print(df['batch'].value_counts())
    df.to_csv('../../data/style-transfer/study_pairs_writing.csv', index=False)


if __name__ == '__main__':
    prepare_for_interface()
