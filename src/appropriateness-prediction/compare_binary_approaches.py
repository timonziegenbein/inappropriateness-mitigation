import json
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from scipy.stats import ttest_ind

RESULTDIMS = [
    'eval_Inappropriateness_macroF1',
]

model_dir = '../../data/models/'

approaches = [
    'binary-debertav3-conservative',
    'binary-debertav3-conservative-no-issue'
]


test_dict = {x: [] for x in RESULTDIMS}
for approach in approaches:
    tmp_results = []
    for repeat in range(5):
        for k in range(5):
            with open(model_dir+approach+'/fold{}/fold{}.{}/test_results.json'.format(repeat,k,repeat), 'r') as f:
                tmp_result = json.load(f)
            tmp_results.append(tmp_result)
    d = {}
    for k, _ in tmp_results[0].items():
        d[k] = np.mean([d[k] for d in tmp_results])
    for dim in RESULTDIMS:
        test_dict[dim].append(d[dim])

test_dict['approach'] = approaches

df = pd.DataFrame(data=test_dict)

### Print F1-scores (table 4 in the paper)
print(df.sort_values('eval_Appropriateness_macroF1', ascending=False).round(4))

test_dict = {x: [] for x in RESULTDIMS}
for approach in approaches:
    tmp_results = []
    for repeat in range(5):
        for k in range(5):
            with open(model_dir+approach+'/fold{}/fold{}.{}/test_results.json'.format(repeat,k,repeat), 'r') as f:
                tmp_result = json.load(f)
            tmp_results.append(tmp_result)
    for dim in RESULTDIMS:
        test_dict[dim].append([x[dim] for x in tmp_results])

### Check significance of all approaches
for dim in RESULTDIMS:
    if 'F1' in dim:
        for i, approach1 in enumerate(approaches):
            for j, approach2 in enumerate(approaches):
                if i<j:
                    w, p = wilcoxon(test_dict[dim][i], test_dict[dim][j], mode='exact')
                    print((dim, approach1, approach2))
                    print(p<=0.5)
