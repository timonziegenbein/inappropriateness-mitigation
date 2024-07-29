import pandas as pd
import random
import numpy as np
from itertools import combinations
from icecream import ic

MODEL_LIST = [
    "../../data/models/instruction-finetuning/llama-7b-instruct",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-04a-06ss/best_checkpoint/",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-05a-05ss/best_checkpoint/",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-06a-04ss/best_checkpoint/",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-10a-00ss/best_checkpoint/",
]

NUM_SHOTS = [0]

MODEL_ABBREVIATIONS = {
    "../../data/models/instruction-finetuning/llama-7b-instruct": "instruct",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-04a-06ss/best_checkpoint/": "40a-60ss",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-05a-05ss/best_checkpoint/": "50a-50ss",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-06a-04ss/best_checkpoint/": "60a-40ss",
    "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/ppo-finetuning/llama-7b-arithmetic-mean-10a-00ss/best_checkpoint/": "10a-00ss",
}


def prepare_for_relative_assessment():
    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df = pd.read_csv(ds_path)

    id_to_issue = {}
    id_to_source = {}
    id_to_rewrite = {}

    df_writing_study = pd.read_csv('../../data/style-transfer/study_pairs_writing_results.csv')

    df_writing_study['test_id'] = df_writing_study['post_id']
    df_writing_study['test_id'] = df_writing_study['test_id'].astype(str)
    df_writing_study['model'] = ['human'] * len(df_writing_study)
    df_writing_study['id'] = df_writing_study['test_id'] + '_' + 'human'

    id_to_issue.update(dict(zip(df_writing_study['id'].tolist(), df_writing_study['issue'].tolist())))
    id_to_source.update(dict(zip(df_writing_study['id'].tolist(), df_writing_study['post_text'].tolist())))
    id_to_rewrite.update(dict(zip(df_writing_study['id'].tolist(), df_writing_study['rewrite'].tolist())))

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
            with open(save_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        predictions.append(line.split('\t')[1])
                    except:
                        print(i)
                        print(line)
                        print(line.split('\t'))
                        break

            for i in range(len(dfs)):
                dfs[i]['predictions'] = predictions[i::5]
                dfs[i] = dfs[i][dfs[i]['fold0.0'] == 'TEST']
                dfs[i] = dfs[i][dfs[i]['Inappropriateness'] == 1]

                ic(len(dfs[i]))
                dfs[i]['test_id'] = [x for x in range(len(dfs[i]))]
                dfs[i]['test_id'] = dfs[i]['test_id'].astype(str)
                dfs[i]['model'] = [MODEL_ABBREVIATIONS[model_name]] * len(dfs[i])
                dfs[i]['id'] = dfs[i]['test_id'] + '_' + dfs[i]['model']

                dfs[i] = dfs[i][dfs[i]['Inappropriateness'] == 1]

                id_to_issue.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['issue'].tolist())))
                id_to_source.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['post_text'].tolist())))
                id_to_rewrite.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['predictions'].tolist())))

                break

    combo_data = {
        'id': [],
        'arg_id': [],
        'source': [],
        'rewrite_a': [],
        'rewrite_b': [],
        'issue': [],
    }

    ids = list(id_to_issue.keys())
    for test_id in list(set([x.split('_')[0] for x in ids])):
        ic(test_id)
        tmp_ids = []
        for id_ in ids:
            if id_.split('_')[0] == test_id:
                tmp_ids.append(id_)
        tmp_combos = list(combinations(tmp_ids, 2))
        for tmp_combo in tmp_combos:
            combo_data['id'].append(tmp_combo[0] + '_' + '_'.join(tmp_combo[1].split('_')[1:]))
            combo_data['arg_id'].append(tmp_combo[0].split('_')[0])
            combo_data['source'].append(id_to_source[tmp_combo[0]])
            combo_data['rewrite_a'].append(id_to_rewrite[tmp_combo[0]])
            combo_data['rewrite_b'].append(id_to_rewrite[tmp_combo[1]])
            combo_data['issue'].append(id_to_issue[tmp_combo[0]])

    combo_df = pd.DataFrame.from_dict(combo_data)
    combo_df['batch'] = [1] * len(combo_df)
    # select 20 random issues and select all pairs for those issues
    arg_ids = list(set(combo_df['arg_id'].tolist()))
    np.random.shuffle(arg_ids)
    arg_ids_prestudy = arg_ids[:45]
    arg_ids_study = arg_ids[45:]
    ic(combo_df.sample(45))

    prestudy_df = combo_df[combo_df['arg_id'].isin(arg_ids_prestudy)]
    prestudy_df.drop(columns=['arg_id'], inplace=True)
    prestudy_df.to_csv('../../data/style-transfer/prestudy_pairs.csv', index=False)

    study_df = combo_df[combo_df['arg_id'].isin(arg_ids_study)]
    study_df.drop(columns=['arg_id'], inplace=True)
    study_df.to_csv('../../data/style-transfer/study_pairs.csv', index=False)


def sampling_filter(mode=None, num_random=None):
    df = pd.read_csv('../../data/style-transfer/study_pairs.csv')
    df['id_source'] = df['id'].apply(lambda x: x.split('_')[0])
    df['id_model1'] = df['id'].apply(lambda x: x.split('_')[1])
    df['id_model2'] = df['id'].apply(lambda x: x.split('_')[-1])

    l = [0, 1, 2, 3, 4, 5]
    n = len(l)
    circle_ids = [(l[i], l[(i+1) % n]) for i in range(n)]
    if mode == 'extended':
        circle_ids += [(0,3), (1,4), (2,5)]
    if mode == 'reduced':
        circle_ids = circle_ids[:-1]
    if mode == 'random':
        #create num_random combinations of ids
        circle_ids = []
        while len(circle_ids) < num_random:
            circle = random.sample(l, 2)
            if circle not in circle_ids:
                circle_ids.append(circle)

    models = list(set(df.id_model1.unique().tolist()+df.id_model2.unique().tolist()))
    random.shuffle(models)
    model_pos_in_circle = {model: i for i, model in enumerate(models)}
    # only keep rows in tmp_df where both models are in next to each other in the circle
    df = df[df.apply(lambda x: (model_pos_in_circle[x['id_model1']],
                            model_pos_in_circle[x['id_model2']]) in circle_ids or (model_pos_in_circle[x['id_model2']], model_pos_in_circle[x['id_model1']]) in circle_ids, axis=1)]

    df.to_csv('../../data/style-transfer/filtered_study_pairs_{}.csv'.format(mode), index=False)


def prepare_for_absolute_assessment():
    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df = pd.read_csv(ds_path)

    id_to_issue = {}
    id_to_source = {}
    id_to_rewrite = {}

    df_writing_study = pd.read_csv('../../data/style-transfer/study_pairs_writing_results.csv')

    df_writing_study['test_id'] = df_writing_study['post_id']
    df_writing_study['test_id'] = df_writing_study['test_id'].astype(str)
    df_writing_study['model'] = ['human'] * len(df_writing_study)
    df_writing_study['id'] = df_writing_study['test_id'] + '_' + 'human'

    id_to_issue.update(dict(zip(df_writing_study['id'].tolist(), df_writing_study['issue'].tolist())))
    id_to_source.update(dict(zip(df_writing_study['id'].tolist(), df_writing_study['post_text'].tolist())))
    id_to_rewrite.update(dict(zip(df_writing_study['id'].tolist(), df_writing_study['rewrite'].tolist())))

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
            with open(save_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        predictions.append(line.split('\t')[1])
                    except:
                        print(i)
                        print(line)
                        print(line.split('\t'))
                        break

            for i in range(len(dfs)):
                dfs[i]['predictions'] = predictions[i::5]
                dfs[i] = dfs[i][dfs[i]['fold0.0'] == 'TEST']
                dfs[i] = dfs[i][dfs[i]['Inappropriateness'] == 1]

                ic(len(dfs[i]))
                dfs[i]['test_id'] = [x for x in range(len(dfs[i]))]
                dfs[i]['test_id'] = dfs[i]['test_id'].astype(str)
                dfs[i]['model'] = [MODEL_ABBREVIATIONS[model_name]] * len(dfs[i])
                dfs[i]['id'] = dfs[i]['test_id'] + '_' + dfs[i]['model']

                dfs[i] = dfs[i][dfs[i]['Inappropriateness'] == 1]

                id_to_issue.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['issue'].tolist())))
                id_to_source.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['post_text'].tolist())))
                id_to_rewrite.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['predictions'].tolist())))

                break

    data = {
        'id': [],
        'arg_id': [],
        'source': [],
        'rewrite': [],
        'issue': [],
    }

    ids = list(id_to_issue.keys())
    for test_id in list(set([x.split('_')[0] for x in ids])):
        ic(test_id)
        tmp_ids = []
        for id_ in ids:
            if id_.split('_')[0] == test_id:
                tmp_ids.append(id_)
        for tmp_id in tmp_ids:
            data['id'].append(tmp_id)
            data['arg_id'].append(int(tmp_id.split('_')[0]))
            data['source'].append(id_to_source[tmp_id])
            data['rewrite'].append(id_to_rewrite[tmp_id])
            data['issue'].append(id_to_issue[tmp_id])

    study_df = pd.DataFrame.from_dict(data)

    arg_ids = list(set(study_df['arg_id'].tolist()))
    np.random.shuffle(arg_ids)
    arg_ids_half1 = arg_ids[:112]

    study_df['batch'] = study_df['arg_id'].apply(lambda x: 1 if x in arg_ids_half1 else 2)
    study_df = study_df.sort_values(by=['batch', 'arg_id'])
    study_df.drop(columns=['arg_id'], inplace=True)
    study_df.to_csv('../../data/style-transfer/study_pairs_absolute.csv', index=False)


def replace_model():
    df_filtered = pd.read_csv('../../data/style-transfer/filtered_study_pairs_{}.csv'.format(None))
    df_filtered = df_filtered[['id', 'source', 'rewrite_a', 'rewrite_b','issue','batch']]
    df_prestudy = pd.read_csv('../../data/style-transfer/prestudy_pairs.csv')
    df_filtered = pd.concat([df_filtered, df_prestudy])
    ic(df_filtered.columns)
    ic(df_filtered[['id','source','rewrite_a','rewrite_b']].head())
    
    ds_path = '../../../arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv'
    df = pd.read_csv(ds_path)
    id_to_issue = {}
    id_to_source = {}
    id_to_rewrite = {}

    for model_name in [MODEL_LIST[3]]:
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
            with open(save_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        predictions.append(line.split('\t')[1])
                    except:
                        print(i)
                        print(line)
                        print(line.split('\t'))
                        break

            for i in range(len(dfs)):
                dfs[i]['predictions'] = predictions[i::5]
                dfs[i] = dfs[i][dfs[i]['fold0.0'] == 'TEST']
                dfs[i] = dfs[i][dfs[i]['Inappropriateness'] == 1]

                ic(len(dfs[i]))
                dfs[i]['test_id'] = [x for x in range(len(dfs[i]))]
                dfs[i]['test_id'] = dfs[i]['test_id'].astype(str)
                dfs[i]['model'] = [MODEL_ABBREVIATIONS[model_name]] * len(dfs[i])
                dfs[i]['id'] = dfs[i]['test_id'] + '_' + dfs[i]['model']
                dfs[i] = dfs[i][dfs[i]['Inappropriateness'] == 1]

                id_to_issue.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['issue'].tolist())))
                id_to_source.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['post_text'].tolist())))
                id_to_rewrite.update(dict(zip(dfs[i]['id'].tolist(), dfs[i]['predictions'].tolist())))

                break
    df_filtered.reset_index(inplace=True, drop=True)
    ic(len(df_filtered))    
    ic(len(df_filtered[df_filtered['id'].str.contains('_60a-40ss')]))
    counter = 0
    for i, row in df_filtered.iterrows():
        if '45a-55ss' == row['id'].split('_')[2]:
            df_filtered.at[i, 'id'] = row['id'].replace('_45a-55ss', '_60a-40ss')
            df_filtered.at[i, 'rewrite_b'] = id_to_rewrite[row['id'].split('_')[0] + '_60a-40ss']
            df_filtered.at[i, 'source'] = id_to_source[row['id'].split('_')[0] + '_60a-40ss']
            counter += 1
        elif '45a-55ss' == row['id'].split('_')[1]:
            df_filtered.at[i, 'id'] = row['id'].replace('_45a-55ss', '_60a-40ss')
            df_filtered.at[i, 'rewrite_a'] = id_to_rewrite[row['id'].split('_')[0] + '_60a-40ss']
            df_filtered.at[i, 'source'] = id_to_source[row['id'].split('_')[0] + '_60a-40ss']
            counter += 1
    ic(len(df_filtered))
    ic(counter)
    ic(df_filtered[['id','source','rewrite_a','rewrite_b']].head())
    df_filtered.to_csv('../../data/style-transfer/filtered_study_pairs_{}.csv'.format('None_extended'), index=False)

    df_filtered = df_filtered[df_filtered['id'].str.contains('_60a-40ss')]
    ic(len(df_filtered[df_filtered['id'].str.contains('_60a-40ss')]))

    df_filtered.to_csv('../../data/style-transfer/filtered_study_pairs_{}.csv'.format('None_fixed'), index=False)


if __name__ == '__main__':
    #prepare_for_relative_assessment()
    #sampling_filter('extended')
    #sampling_filter('reduced')
    #sampling_filter()
    #prepare_for_absolute_assessment()
    replace_model()
