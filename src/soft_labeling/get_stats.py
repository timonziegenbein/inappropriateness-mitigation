import pandas as pd
from ast import literal_eval
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


def add_prediction(df, predictions):
    # set column names of predictions to post_id and prediction
    predictions.columns = ['post_id', 'prediction']
    df['prediction'] = predictions['prediction'].values.tolist()
    df['prediction'] = df['prediction'].apply(lambda x: [x.split(",")[0][1:], x.split(",")[1][1:-1]])
    df['appropriate'] = df['prediction'].apply(lambda x: float(x[0]))
    df['inappropriate'] = df['prediction'].apply(lambda x: float(x[1]))
    return df


def print_stats(df, min_confidence):
    print("appropriate > {}: {}".format(min_confidence, len(df[df['appropriate'] > min_confidence])))
    print("inappropriate > {}: {}".format(min_confidence, len(df[df['inappropriate'] > min_confidence])))
    print("-"*100)
    print("-"*100)
    for i in range(5):
        print(df[df['appropriate'] > min_confidence].sample(1)['text'].values[0])
        print("-"*10)
    print("-"*100)
    for i in range(5):
        print(df[df['inappropriate'] > min_confidence].sample(1)['text'].values[0])
        print("-"*10)
    print("+"*100)


if __name__ == "__main__":
    df_createdebate = pd.read_csv("../../data/iac2/createdebate.csv", sep="\t")
    df_convinceme = pd.read_csv("../../data/iac2/convinceme.csv", sep="\t")
    df_gaq = pd.read_csv("../../data/GAQCorpus_split/GAQ.csv", sep="\t")

    df_createdebate = add_prediction(df_createdebate, pd.read_csv(
        "../../data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_createdebate.txt", sep="\t", header=None))
    df_convinceme = add_prediction(df_convinceme, pd.read_csv(
        "../../data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_convinceme.txt", sep="\t", header=None))
    df_gaq = add_prediction(df_gaq, pd.read_csv(
        "../../data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_GAQ.txt", sep="\t", header=None))
    df_createdebate = df_createdebate[['text_id', 'text', 'appropriate', 'inappropriate']]
    df_createdebate['text_id'] = df_createdebate['text_id'].apply(lambda x: "createdebate_" + str(x))
    df_convinceme = df_convinceme[['text_id', 'text', 'appropriate', 'inappropriate']]
    df_convinceme['text_id'] = df_convinceme['text_id'].apply(lambda x: "convinceme_" + str(x))
    df_gaq = df_gaq[['id', 'text', 'appropriate', 'inappropriate']].rename(columns={'id': 'text_id'})
    df_gaq['text_id'] = df_gaq['text_id'].apply(lambda x: "gaq_" + str(x))
    df_combined = pd.concat([df_createdebate, df_convinceme, df_gaq], ignore_index=True)

    # make sure all ids are unique
    assert len(df_combined['text_id'].unique()) == len(df_combined['text_id'])

    print("Createdebate:")
    print_stats(df_createdebate, 0.99)
    print("Convinceme:")
    print_stats(df_convinceme, 0.99)
    print("GAQ:")
    print_stats(df_gaq, 0.99)
    print("Combined:")
    print_stats(df_combined, 0.99)

    df_combined.to_csv("../../data/soft-labels/combined.csv", sep="\t", index=False)
