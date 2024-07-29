import pandas as pd
from sklearn.metrics import ndcg_score
import random
from scipy import stats
import numpy as np
from ast import literal_eval
from scipy.optimize import minimize
from itertools import combinations
import krippendorff

# taken from https://github.com/webis-de/acl20-efficient-argument-quality-annotation
class BradleyTerry:
    def __init__(self, comparisons, parsefunc=None):
        """
        Constructor
        :param comparisons: list of comparisons
        :param parsefunc: optionally pass a custom parsign function to cope with different data formats
        """
        parsefunc = parsefunc if parsefunc is not None else self.__parsefunc__
        self.items, self.comparisons, self.merits = parsefunc(comparisons)

    @staticmethod
    def __parsefunc__(comparisons) -> tuple:
        """
        Function to parse supplied comparison data to the format needed by the model
        :param comparisons: comparison data
        :return
        """
        items = list(set([x[0] for x in comparisons]+[x[1] for x in comparisons]))

        # Mapping
        items_parsed = {x: i for i, x in enumerate(items)}

        # Mapped comparisons
        comparisons_parsed = []
        for arg1_id, arg2_id, tie in comparisons:
            comparisons_parsed.append([
                items_parsed[arg1_id],
                items_parsed[arg2_id],
                tie
            ])

        # Initialize zero-vector for merits
        merits = np.zeros(len(items))

        return (items_parsed, comparisons_parsed, merits)

    @staticmethod
    def __pfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of non-ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param s: annotation quality score
        :param t: difference threshold
        :return: propability of item i beating item j
        """
        p = np.exp(i) / (np.exp(i) + np.exp(j) * np.exp(t))
        return np.log10(p)

    @staticmethod
    def __tfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param t: difference threshold
        :return: propability of item i beating item j
        """
        f1 = np.exp(i) * np.exp(j) * (np.square(np.exp(t)) - 1)
        f2 = (np.exp(i) + np.exp(j) * np.exp(t)) * (np.exp(i) * np.exp(t) + np.exp(j))
        p = f1 / f2
        return np.log10(p)

    def __rfunc__(self, i: float, l: float) -> float:
        """
        Function to compute regularized probability
        :param i: item merit
        :param l: regularization factor
        :return: value of __pfunc__ for matches with dummy item weighted by l
        """
        return l * (self.__pfunc__(i, 1, 0) + self.__pfunc__(1, i, 0))

    def __log_likelihood__(self, merits: np.ndarray) -> float:
        """
        Log-Likelihood Function
        :param merits: merit vector
        :return: log-likelihood value
        """
        k: float = 0  # Maximization sum

        # Summing Edge Probabilities
        for arg1, arg2, tie in self.comparisons:
            if tie:
                k += self.__tfunc__(merits[arg1], merits[arg2], self.threshold)
            else:
                k += self.__pfunc__(merits[arg1], merits[arg2], self.threshold)

        # Regularization
        for x in range(len(self.items)):
            k += self.__rfunc__(merits[x], self.regularization)

        return -1 * k

    def fit(self, regularization: float = 0, threshold: float = 0) -> None:
        """
        Optimize the model for merits
        :param regularization: regularization parameter
        :param threshold: difference threshold
        """
        self.merits = np.ones(len(self.items))
        self.threshold = threshold
        self.regularization = regularization

        res = minimize(self.__log_likelihood__, self.merits, method='BFGS', options={"maxiter": 100})
        self.merits = res.x

    def get_merits(self, normalize=False) -> list:
        """
        Returns the merits mapped to items
        :param normalize: if true, returns normalized merit vector to 0-1 range instead of original scores
        :return: dict in the form of {argument_id: merit} sorted by merits
        :exception: Exception if model was not fitted
        """
        if not self.merits.any():
            raise Exception('Model has to be fitted first!')
        else:
            d = {argument_id: self.merits[index] for argument_id, index in self.items.items()}
            if normalize:
                mi = min(d.values())
                ma = max(d.values())
                def normalize(mi, ma, v): return (v-mi)/(ma-mi)
                d.update({k: normalize(mi, ma, v) for k, v in d.items()})
            return sorted(d.items(), key=lambda kv: kv[1])


class PairwiseAggregator:
    def __init__(self, threshold, margin, log_scores=True, logit_scores=False):
        self.threshold = threshold
        self.margin = margin
        self.log_scores = log_scores
        self.logit_scores = logit_scores

    def _infer_tie(self, p):
        if not 0 <= p <= 1:
            raise ValueError('Got invalid p of ' + str(p) + '. Expected p in Interval [0, 1]')

        if not 0 <= (self.threshold + self.margin) <= 1:
            raise ValueError('Got invalid threshold and margin of ' + str(self.threshold + self.margin) +
                             '. Expected p in Interval [0, 1]')

        if not 0 <= (self.threshold - self.margin) <= 1:
            raise ValueError('Got invalid threshold and margin of ' + str(self.threshold - self.margin) +
                             '. Expected p in Interval [0, 1]')

        if p > self.threshold + self.margin:
            return False
        elif p < self.threshold - self.margin:
            return False
        else:
            return True

    def _order_pair(self, id_a, id_b, p):
        if not 0 <= p <= 1:
            raise ValueError('Got invalid p of ' + str(p) + '. Expected p in Interval [0, 1]')
        if p >= self.threshold:
            return id_a, id_b, p
        else:
            return id_b, id_a, 1 - p

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        if self.log_scores:
           pairwise_scores["score"] = pairwise_scores["score"].apply(np.exp)
        if self.logit_scores:
            pairwise_scores["score"] = pairwise_scores["score"].apply(lambda x: np.exp(x)/(1+np.exp(x)))
        return pairwise_scores

        """"
        data = []
        if self.log_scores:
            for _, (id_a, id_b, p) in pairwise_scores.iterrows():
                data.append(self._order_pair(id_a, id_b, np.exp(p)))
        else:
            for _, (id_a, id_b, p) in pairwise_scores.iterrows():
                data.append(self._order_pair(id_a, id_b,p))
        return pd.DataFrame(data, columns=pairwise_scores.columns)
        """


class BradleyTerryAggregator(PairwiseAggregator):
    def __init__(self, tie_margin: float = 0.05, tie_threshold: float = 0.05, regularization: float = 0.2,
                 max_iter: int = 100, log_scores=False, logit_scores=False, normalize_scores=False, cython=False):
        """
        Constructor
        :param tie_margin: score margin to declare ties
        :param regularization: regularization parameter
        :param tie_threshold: difference threshold
        :param max_iter: maximum iterations for the LL optimizer
        """
        super().__init__(0.5, tie_margin, log_scores=log_scores, logit_scores=logit_scores)
        self.threshold = tie_threshold
        self.regularization = regularization
        self.max_iter = max_iter
        self.normalize = normalize_scores

        if not cython:
            self.optimize = self._optimize_python
        else:
            self.optimize = self._optimize_cython

    def __str__(self):
        return "bradleyterry"

    def _optimize_cython(self, comparisons, n_samples, regularization, threshold):
        from ._bradleyterry import __log_likelihood__
        # Transform comparisons into fast iterable matrix
        comparison_matrix = np.zeros(shape=(len(comparisons), 3), dtype=np.intc)
        for i, (id_a, id_b, tie) in enumerate(comparisons):
            comparison_matrix[i, 0] = int(id_a)
            comparison_matrix[i, 1] = int(id_b)
            comparison_matrix[i, 2] = int(tie)
        # Initialize merit vector
        merits = np.ones(shape=(n_samples,), dtype=np.double)
        # Optimize using BFGS
        res = minimize(__log_likelihood__, merits, (comparison_matrix, regularization, threshold), method="BFGS")
        return res.x

    def _optimize_python(self, comparisons, n_samples, regularization, threshold):
        # Initialize merit vector
        merits = np.ones(shape=(n_samples,), dtype=np.double)
        # Optimize using BFGS
        res = minimize(self.__log_likelihood__, merits, (comparisons, regularization, threshold), method="BFGS")
        return res.x

    @staticmethod
    def __pfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of non-ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param t: difference threshold
        :return: probability of item i beating item j
        """
        p = np.exp(i) / (np.exp(i) + np.exp(j) * np.exp(t))
        return np.log10(p)

    @staticmethod
    def __tfunc__(i: float, j: float, t: float) -> float:
        """
        Function to compute pairwise comparison probabilities of ties
        :param i: merit of the winning item
        :param j: merit of the loosing item
        :param t: difference threshold
        :return: probability of item i beating item j
        """
        f1 = np.exp(i) * np.exp(j) * (np.square(np.exp(t)) - 1)
        f2 = (np.exp(i) + np.exp(j) * np.exp(t)) * (np.exp(i) * np.exp(t) + np.exp(j))
        p = f1 / f2
        return np.log10(p)

    def __rfunc__(self, i: float, l: float) -> float:
        """
        Function to compute regularized probability
        :param i: item merit
        :param l: regularization factor
        :return: value of __pfunc__ for matches with dummy item weighted by l
        """
        return l * (self.__pfunc__(i, 1, 0) + self.__pfunc__(1, i, 0))

    def __log_likelihood__(self, merits: np.ndarray, comparisons: np.ndarray, regularization: float, threshold: float) -> float:
        """
        Log-Likelihood Function
        :param merits: merit vector
        :return: log-likelihood value
        """
        k: float = 0  # Maximization sum
        # Summing Edge Probabilities
        for arg1, arg2, tie in comparisons:
            if tie:
                k += self.__tfunc__(merits[arg1], merits[arg2], threshold)
            else:
                k += self.__pfunc__(merits[arg1], merits[arg2], threshold)
        # Regularization
        for i in range(merits.shape[0]):
            k += self.__rfunc__(merits[i], regularization)
        return -1 * k

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the aggregation and return the calculated merits.
        :param pairwise_scores: pairwise score data
        """

        pairwise_scores = super().__call__(pairwise_scores)
        self.items = list(set(pairwise_scores["id_a"].unique().tolist() + pairwise_scores["id_b"].unique().tolist()))
        item_mapping = {x: i for i, x in enumerate(self.items)}

        # Mapped comparisons
        self.comparisons = []
        for _, (id_a, id_b, p) in pairwise_scores.iterrows():
            tie = super()._infer_tie(p)
            self.comparisons.append([
                item_mapping[id_a],
                item_mapping[id_b],
                tie
            ])
        res = self.optimize(comparisons=self.comparisons, n_samples=len(self.items), regularization=self.regularization,
                            threshold=self.threshold)

        scores = {doc_id: res[index] for doc_id, index in item_mapping.items()}
        df = pd.DataFrame(scores.items(), columns=["docno", "score"])
        if self.normalize:
            df["score"] = (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())
            return df
        else:
            return df


class GreedyAggregator(PairwiseAggregator):
    """

    References
    ----------
    William W. Cohen, Robert E. Schapire, Yoram Singer Learning to Order Things. J. Artif. Intell. Res. 10: 243-270 (1999)

    """

    def __init__(self, log_scores: bool = True, logit_scores: bool = False):
        super().__init__(0.5, 0, log_scores=log_scores, logit_scores=logit_scores)

    def __str__(self):
        return "greedy"

    def __call__(self, pairwise_scores: pd.DataFrame) -> pd.DataFrame:
        pairwise_scores = super().__call__(pairwise_scores)

        items = list(set(pairwise_scores["id_a"].unique().tolist() + pairwise_scores["id_b"].unique().tolist()))
        item_mapping = {x: i for i, x in enumerate(items)}

        # Construct score lookup
        scores = np.zeros(shape=(len(items), len(items)), dtype=np.float32)
        for _, (id_a, id_b, p) in pairwise_scores.iterrows():
            scores[item_mapping[id_a], item_mapping[id_b]] = p

        # Calculate initial score for each item
        pi_v = np.zeros(shape=len(items), dtype=np.float32)
        for v in range(pi_v.shape[0]):
            pi_v[v] = np.sum(scores[v, :]) - np.sum(scores[:, v])

        # Initialize ranks
        ranks = np.zeros(shape=len(items), dtype=np.int32)

        for i in range(ranks.shape[0]):
            # Choose remaining item with the highest potential
            t = np.argmax(np.where(ranks == 0, pi_v, -np.inf))
            # Assign rank to t (inverted ranks to qualify as descendingly sortable scores)
            ranks[t] = len(ranks) - i
            # Adjust remaining scores
            for v in np.where(ranks == 0)[0]:
                pi_v[v] = pi_v[v] + scores[t, v] - scores[v, t]

        return pd.DataFrame(zip(item_mapping.keys(), ranks), columns=["docno", "score"])


VALID_MERIT = None
def compute_ranking(df, ids):
    ranking = {}
    nan_count = 0
    global VALID_MERIT
    for id_source in ids:
        df_source = df[df['id_source'] == id_source]
        #print(df_source) 
        comparisons = []
        #for i, row in df_source.iterrows():
        #    if row['slider_score'] == 0:
        #        comparisons.append((row['id_model1'], row['id_model2'], 1))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0))
        #    elif row['slider_score'] == 1:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.875))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.125))
        #    elif row['slider_score'] == 2:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.75))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.25))
        #    elif row['slider_score'] == 3:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.6125))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.3875))
        #    elif row['slider_score'] == 4:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.5))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.5))
        #    elif row['slider_score'] == 5:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.375))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.625))
        #    elif row['slider_score'] == 6:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.25))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.75))
        #    elif row['slider_score'] == 7:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0.125))
        #        comparisons.append((row['id_model2'], row['id_model1'], 0.875))
        #    elif row['slider_score'] == 8:
        #        comparisons.append((row['id_model1'], row['id_model2'], 0))
        #        comparisons.append((row['id_model2'], row['id_model1'], 1))
        #    else:
        #        print('ERROR IN SLIDER SCORE')



        for i, row in df_source.iterrows():
            if row['slider_score'] == 4:
                comparisons.append((row['id_model1'], row['id_model2'], True))
            elif row['slider_score'] < 4:
                comparisons.append((row['id_model1'], row['id_model2'], False))
            elif row['slider_score'] > 4:
                comparisons.append((row['id_model2'], row['id_model1'], False))
            else:
                print('ERROR IN SLIDER SCORE')

        # avearge scores of same comparisons
        #comparisons = pd.DataFrame(comparisons, columns=['id_a', 'id_b', 'score'])
        #comparisons = comparisons.groupby(['id_a', 'id_b']).mean().reset_index().values.tolist()

        # Initialize the model with given comparisons
        bt = BradleyTerry(comparisons)

        #comparison_df = pd.DataFrame(comparisons, columns=['id_a', 'id_b', 'score'])
        #print(comparison_df)
        #bt2 = GreedyAggregator()(comparison_df)
        #print(bt2)
        # to list of tuples
        #merits = [(x[0], x[1]) for x in bt2.values.tolist()]
        #print(merits)

        # Fit the model using supplied hyperparameters
        bt.fit(regularization=0.3, threshold=0.01)

        merits = bt.get_merits(normalize=True)
        #print(merits)
        if np.isnan(merits[0][1]):
            nan_count += 1
            merits = VALID_MERIT
            # shuffle second values in merits
            second_values = [x[1] for x in merits]
            random.shuffle(second_values)
            merits = [(merits[i][0], second_values[i]) for i in range(len(merits))]
        for i, merit in enumerate(merits):
            if merit[0] not in ranking:
                ranking[merit[0]] = []
            ranking[merit[0]].append(merit[1])
        VALID_MERIT = merits

    if nan_count > 0:
        print('% nan count: ', nan_count/len(ids))
    return ranking


def compare_sampling(df, mode=None, annotators=[3,4,5,6,7], num_random=None):
    tmp_df = df[df['user_id'].isin(annotators)]
    ids = tmp_df.id_source.unique()
    full_ranking = compute_ranking(tmp_df, ids)

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
        print(circle_ids)

    models = list(set(tmp_df.id_model1.unique().tolist()+tmp_df.id_model2.unique().tolist()))
    random.shuffle(models)
    model_pos_in_circle = {model: i for i, model in enumerate(models)}
    # only keep rows in tmp_df where both models are in next to each other in the circle
    circle_tmp_df = tmp_df[tmp_df.apply(lambda x: (model_pos_in_circle[x['id_model1']],
                            model_pos_in_circle[x['id_model2']]) in circle_ids or (model_pos_in_circle[x['id_model2']], model_pos_in_circle[x['id_model1']]) in circle_ids, axis=1)]
    circle_ranking = compute_ranking(circle_tmp_df, ids)

    pearson_rs = []
    ndcg_1s = []
    for i in range(len(ids)):
        x = []
        y = []
        for key, value in full_ranking.items():
            x.append(value[i])
            y.append(circle_ranking[key][i])

        res = stats.pearsonr(x, y)
        ndcg_1 = ndcg_score([x], [y], k=1)
        pearson_rs.append(res.statistic)
        ndcg_1s.append(ndcg_1)
    rng = np.random.default_rng()
    res = stats.bootstrap((pearson_rs, ), np.mean, confidence_level=0.95, random_state=1, method='percentile', n_resamples=10000).confidence_interval
    return np.mean(pearson_rs), res.low, res.high, len(circle_tmp_df), np.mean(ndcg_1s)


def get_final_ranking(df):
    ranking = {}
    for _ in range(10):
        tmp_ranking = compute_ranking(df, df.id_source.unique())
        for k, v in tmp_ranking.items():
            if k not in ranking:
                ranking[k] = []
            ranking[k].append(v)
    mean_ranking = {}
    for k, v in ranking.items():
        mean_ranking[k] = np.mean(v)


    abs_counts = {}
    for id_source in df.id_source.unique():
        tmp_ranking = compute_ranking(df[df['id_source'] == id_source], [id_source])
        #unpack all values
        tmp_ranking = {k: v[0] for k, v in tmp_ranking.items()}
        # replace values with ranking
        tmp_ranking = {k: sorted(tmp_ranking, key=lambda x: tmp_ranking[x]).index(k)+1 for k in tmp_ranking}
        for k, v in tmp_ranking.items():
            if k not in abs_counts:
                abs_counts[k] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
            abs_counts[k][7-v] += 1
    print(abs_counts)
    # print mean abs_count for each model
    for k, v in abs_counts.items():
        print(k, np.sum([k2*v2 for k2, v2 in v.items()])/len(df.id_source.unique()))

    for i, x in enumerate(sorted(mean_ranking.items(), key=lambda kv: kv[1])[::-1]):
        print('Rank {}: {} (mean merit: {})'.format(i+1, x[0], x[1]))

    # print absolute counts each model won a comparisons
    counts = {}
    #df = df[(df['id_model1'].isin(['10a-00ss','60a-40ss'])) & (df['id_model2'].isin(['10a-00ss','60a-40ss']))]
    for i, row in df.iterrows():
        if row['slider_score'] == 4:
            if 'tie' not in counts:
                counts['tie'] = 0
            counts['tie'] += 1
        elif row['slider_score'] < 4:
            if row['id_model1'] not in counts:
                counts[row['id_model1']] = 0
            counts[row['id_model1']] += 1
        elif row['slider_score'] > 4:
            if row['id_model2'] not in counts:
                counts[row['id_model2']] = 0
            counts[row['id_model2']] += 1
        else:
            print('ERROR IN SLIDER SCORE')
    print('Absolute counts:')
    print(sorted(counts.items(), key=lambda kv: kv[1])[::-1])


def get_agreement(df):
    pearsons = []
    for _ in range(10):
        for user_id1 in df.user_id.unique():
            for user_id2 in df.user_id.unique():
                if user_id1 < user_id2:
                    tmp_pearsons = []
                    rankings1 = compute_ranking(df[df['user_id'] == user_id1], df.id_source.unique())
                    rankings2 = compute_ranking(df[df['user_id'] == user_id2], df.id_source.unique())
                    for i in range(len(list(rankings1.values())[0])):
                        x = []
                        y = []
                        for k in list(set(list(rankings1.keys())+list(rankings2.keys()))):
                            x.append(rankings1[k][i])
                            y.append(rankings2[k][i])
                        res = stats.pearsonr(x, y)
                        pearsons.append(res.statistic)
                        tmp_pearsons.append(res.statistic)
                    #print('User {} vs User {}: {}'.format(user_id1, user_id2, np.mean(tmp_pearsons)))
    pearsons = sum(pearsons)/len(pearsons)
    print('Pearson\'s R: {}'.format(pearsons))


def analyze_prestudy():
    columns = ['a_id', 'user_id', 'post_id', 'post_text', 'annotation_date', 'result', 'issue', 'comments']
    df_pre = pd.read_csv('../../data/style-transfer/prestudy_pairs_results.csv')
    df = pd.read_csv('../../data/style-transfer/study_pairs_results.csv')
    print(len(df))
    df = pd.concat([df_pre, df])
    df = df[~df['post_id'].str.contains('_45a-55ss')]
    df['result'] = df['result'].apply(literal_eval)
    df['slider_score'] = df['result'].apply(lambda x: int(x['rangeslider1']))
    df['id_source'] = df['post_id'].apply(lambda x: x.split('_')[0])
    df['id_model1'] = df['post_id'].apply(lambda x: x.split('_')[1])
    df['id_model2'] = df['post_id'].apply(lambda x: x.split('_')[-1])

    # only keep where id_source is in df_pre
    df_pre['id_source'] = df_pre['post_id'].apply(lambda x: x.split('_')[0])
    df = df[df['id_source'].isin(df_pre['id_source'].unique())]

    # keep only users that are in [3,4,5,6,7]
    df = df[df['user_id'].isin([3,4,5,6,7])]
    print(len(df))

    # only keep post_ids that appear 5 times in the df
    ids_to_keep = df['post_id'].value_counts()[df['post_id'].value_counts() == 5].index.tolist()
    df = df[df['post_id'].isin(ids_to_keep)]

    #print(df.columns)
    final_sampling_comparison_data = {
        'num_annotators': [],
        'mode': [],
        'num_comparisons': [],
        'percent_comparisons': [],
        'pearson_r': [],
        'ci_low': [],
        'ci_high': [],
        'ndcg_1': []
    }
    print('Calculating agreement')
    get_agreement(df)
    print('Calculating final ranking')
    get_final_ranking(df)
    print('-'*50)
    print('Comparing sampling methods (annotators)')
    for m in ['circle', 'extended', 'reduced']:#, 'random_5', 'random_6', 'random_9']:
        print('Sampling method: {}'.format(m))
        for num_annotators in reversed(range(5)):
            pearson_rs = []
            cis_low = []
            cis_high = []
            num_comparisons = []
            ndcg_1s = []
            for annotator_combination in combinations(df.user_id.unique(), num_annotators+1):
                for _ in range(10):
                    if 'random' in m:
                        out = compare_sampling(df, 'random', list(annotator_combination), num_random=int(m.split('_')[-1]))
                    else:
                        out = compare_sampling(df, m, list(annotator_combination))
                    pearson_rs.append(out[0])
                    cis_low.append(out[1])
                    cis_high.append(out[2])
                    num_comparisons.append(out[3])
                    ndcg_1s.append(out[4])
            final_sampling_comparison_data['num_annotators'].append(num_annotators+1)
            final_sampling_comparison_data['mode'].append(m)
            final_sampling_comparison_data['num_comparisons'].append(np.mean(num_comparisons))
            final_sampling_comparison_data['percent_comparisons'].append(np.mean(num_comparisons)/len(df)*100)
            final_sampling_comparison_data['pearson_r'].append(np.mean(pearson_rs))
            final_sampling_comparison_data['ci_low'].append(np.mean(cis_low))
            final_sampling_comparison_data['ci_high'].append(np.mean(cis_high))
            final_sampling_comparison_data['ndcg_1'].append(np.mean(ndcg_1s))

    final_sampling_comparison_df = pd.DataFrame(final_sampling_comparison_data)
    print(final_sampling_comparison_df)
    final_sampling_comparison_df.to_csv('../../data/style-transfer/sampling_comparison.csv', index=False)


def analyze_study():
    columns = ['a_id', 'user_id', 'post_id', 'post_text', 'annotation_date', 'result', 'issue', 'comments']
    df_pre = pd.read_csv('../../data/style-transfer/prestudy_pairs_results.csv')
    df = pd.read_csv('../../data/style-transfer/study_pairs_results.csv')
    print(len(df))
    df = pd.concat([df_pre, df])
    df = df[~df['post_id'].str.contains('_45a-55ss')]
    df['result'] = df['result'].apply(literal_eval)
    df['slider_score'] = df['result'].apply(lambda x: int(x['rangeslider1']))
    df['id_source'] = df['post_id'].apply(lambda x: x.split('_')[0])
    df['id_model1'] = df['post_id'].apply(lambda x: x.split('_')[1])
    df['id_model2'] = df['post_id'].apply(lambda x: x.split('_')[-1])

    # keep only users that are in [3,4,5,6,7]
    df = df[df['user_id'].isin([3,4,5,6,7])]
    print(len(df))

    # only keep post_ids that appear 5 times in the df
    ids_to_keep = df['post_id'].value_counts()[df['post_id'].value_counts() == 5].index.tolist()
    df = df[df['post_id'].isin(ids_to_keep)]

    print('Calculating agreement')
    get_agreement(df)
    print('Calculating final ranking')
    get_final_ranking(df)


def analyze_abs_study():
    columns = ['a_id', 'user_id', 'post_id', 'post_text', 'annotation_date', 'result', 'issue', 'comments']
    df = pd.read_csv('../../data/style-transfer/study_pairs_abs_results.csv')
    df['result'] = df['result'].apply(literal_eval)
    df['app'] = df['result'].apply(lambda x: int(x['otherErrorQuestion1'][-1]))
    df['sim'] = df['result'].apply(lambda x: int(x['otherErrorQuestion2'][-1])-5 if x['otherErrorQuestion2'][-2:] != '10' else int(x['otherErrorQuestion2'][-2:])-5)
    df['fluency'] = df['result'].apply(lambda x: int(x['otherErrorQuestion3'][-2:])-10)
    df['id_source'] = df['post_id'].apply(lambda x: x.split('_')[0])
    df['id_model'] = df['post_id'].apply(lambda x: x.split('_')[1])

    # keep only users that are in [3,4,5,6,7]
    df = df[df['user_id'].isin([2,3,4,5,6,7,8,9,10,11])]
    print(len(df))
    
    # print # of annotations per user
    print(df['user_id'].value_counts())
    # only keep post_ids that appear 5 times in the df
    ids_to_keep = df['post_id'].value_counts()[df['post_id'].value_counts() == 5].index.tolist()
    df = df[df['post_id'].isin(ids_to_keep)]
    print(len(df))

    # calc mean of  app, sim, fluency for each model
    df_mean = df.groupby(['id_model']).mean().reset_index()
    df_mean['app'] = df_mean['app'].apply(lambda x: round(x, 2))
    df_mean['sim'] = df_mean['sim'].apply(lambda x: round(x, 2))
    df_mean['fluency'] = df_mean['fluency'].apply(lambda x: round(x, 2))
    print(df_mean[['id_model', 'app', 'sim', 'fluency']])

    # calc krippendorff's alpha between annotators [2,3,4,5,6] and [7,8,9,10,11]
    df_1 = df[df['user_id'].isin([2,3,4,5,6])]
    df_2 = df[df['user_id'].isin([7,8,9,10,11])]
    print('Calculating agreement')
    rd_df1_app = [l+[np.nan for _ in [df_2[df_2['user_id']==x].sort_values('post_id')['app'].tolist() for x in df_2.user_id.unique()][0]] for l in [df_1[df_1['user_id']==x].sort_values('post_id')['app'].tolist() for x in df_1.user_id.unique()]]
    rd_df2_app = [[np.nan for _ in [df_1[df_1['user_id']==x].sort_values('post_id')['app'].tolist() for x in df_1.user_id.unique()][0]]+l for l in [df_2[df_2['user_id']==x].sort_values('post_id')['app'].tolist() for x in df_2.user_id.unique()]]
    rd_df1_sim = [l+[np.nan for _ in [df_2[df_2['user_id']==x].sort_values('post_id')['sim'].tolist() for x in df_2.user_id.unique()][0]] for l in [df_1[df_1['user_id']==x].sort_values('post_id')['sim'].tolist() for x in df_1.user_id.unique()]]
    rd_df2_sim = [[np.nan for _ in [df_1[df_1['user_id']==x].sort_values('post_id')['sim'].tolist() for x in df_1.user_id.unique()][0]]+l for l in [df_2[df_2['user_id']==x].sort_values('post_id')['sim'].tolist() for x in df_2.user_id.unique()]]
    rd_df1_fluency = [l+[np.nan for _ in [df_2[df_2['user_id']==x].sort_values('post_id')['fluency'].tolist() for x in df_2.user_id.unique()][0]] for l in [df_1[df_1['user_id']==x].sort_values('post_id')['fluency'].tolist() for x in df_1.user_id.unique()]]
    rd_df2_fluency = [[np.nan for _ in [df_1[df_1['user_id']==x].sort_values('post_id')['fluency'].tolist() for x in df_1.user_id.unique()][0]]+l for l in [df_2[df_2['user_id']==x].sort_values('post_id')['fluency'].tolist() for x in df_2.user_id.unique()]]


    print(df_2[df_2['user_id']==6].sort_values('post_id'))
    print(df_2[df_2['user_id']==8].sort_values('post_id'))
    print(df_2[df_2['user_id']==9].sort_values('post_id'))
    print(df_2[df_2['user_id']==10].sort_values('post_id'))
    print(df_2[df_2['user_id']==11].sort_values('post_id'))

    ka_app = krippendorff.alpha(reliability_data=rd_df1_app+rd_df2_app, level_of_measurement='ordinal')
    ka_sim = krippendorff.alpha(reliability_data=rd_df1_sim+rd_df2_sim, level_of_measurement='ordinal')
    ka_fluency = krippendorff.alpha(reliability_data=rd_df1_fluency+rd_df2_fluency, level_of_measurement='ordinal')

    print('App Krippendorff\'s alpha: {}'.format(ka_app))
    print('Sim Krippendorff\'s alpha: {}'.format(ka_sim))
    print('Fluency Krippendorff\'s alpha: {}'.format(ka_fluency))


if __name__ == '__main__':
    #analyze_prestudy()
    #analyze_study()
    analyze_abs_study()
