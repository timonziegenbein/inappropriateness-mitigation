import torch
import json
import os
import sys
import pandas as pd
import numpy as np
import os
import argparse
from calculate_metrics import MetricsCalculator
from trlx.trainer.accelerate_base_trainer import *
from trlx.utils.logging import *
from datasets import Dataset
from peft import LoraConfig
from typing import List
from peft.utils.config import TaskType
import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

os.environ['WANDB_MODE'] = 'offline'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append('../soft-labeling/')

GEN_ARGS = {
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 0,
    "temperature": 1.0,
    "num_return_sequences": 5,
}

PROMPT_PATTERN = '''Here is some text: {{{}}}. Here is a rewrite of the text that is more appropriate and makes only minimal changes: {{{}}}.'''

INSTRUCT_PRE = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'''
INSTRUCT_PATTERN = '''### Instruction:\nRewrite the following argument to be more appropriate and make only minimal changes to the original argument.\n\n### Input:\n{}\n\n### Response:\n{}\n\n'''

FEW_SHOT_EXAMPLES_SUB = [
    (''''Towed three times and impounded for 30 days each time? Man, you're just not getting the message, are you? If you are in California, you bet the police can forfeit your vehicle and it doesn't take three times to make it a charm. Technically, your vehicle could be subject to forfeiture proceedings after your first suspended license beef. Someone like you is exactly the reason the legislature designed that law, because your privilege to drive has been taken away from you and yet you obviously continue to drive. People like you are involved in an exponentially higher than average number of traffic accidents so the legislature figured maybe people like you should have your vehicles forfeited to the state if you just didn't go along with the game plan. Voila - I give you California Vehicle Code section 14607.6...and a link to it below. It would also be worth your time to review 14607.4, whether or not you live in California. You really need to stop driving. Really.''',
     '''If you are in California, the police can forfeit your vehicle after fewer than three incidents. Technically, your vehicle could be subject to forfeiture proceedings after your first suspended license. The legislature designed that law with people in mind that don't take suspended licenses seriously and continue to drive anyway. Such people tend to be involved in an exponentially higher-than-average number of traffic accidents, so the legislature figured that they should have their vehicles taken into custody if they don't take the suspension seriously. Voila - I give you California Vehicle Code section 14607.6 and a link to it below. You might also want to take time to review 14607.4, whether or not you live in California. After three suspensions, you should consider stopping driving.'''),
    (''''Yes i am completely for it. People are arguing that it is barbaric and inhumane but who can stand up and say that some perv who has raped and killed a child still has human rights and the right to live. We would put down a dangerous dog why not do the same to some of the scum that lives in our country. The justice system in britain at the moment is hopeless. Far to many people are gettin away with all sorts and something needs to be done!!''',
     '''Yes, I am for it. People argue that it is barbaric and inhumane, but who can stand up and say that a person conducting child abuse still has the right to live? We would put down a dangerous dog; why not act in a similar way with criminals living in our country? I don't think very highly of the justice system in Britain at the moment. Far too many people are getting away with all sorts of things, and something needs to be done!'''),
    (''''Porn is Wrong. mainly because they are Not Doing it Right. it should be Hi Def. in three years, it will be in 3-D.''',
     '''Porn is wrong because they are not doing it right. But jokes aside, it should be in higher resolution. In three years, it will be in 3D.'''),
    (''''There should be no argument in this really...whatever way yu see a fetus...its still a living form that has been created in a very intimate way... you shouldn't be changing what mothernature or God or fate or whatever has decided for you...and if you didn;t wannna get preggo in the first place...don't have sex or use protection. Yeh there are some women that get raped and it's very unfortunate but they should give the child up for adoption. It's not the child's fault that it was created. So why should the goring being have to pay the ultimate price of it's life?''',
     '''In point of view, whatever way you see a fetus it's still a living form that has been created in a very intimate way. You shouldn't be changing what mother nature or God or fate or whatever has decided for you and if you didn't wanna get pregnant in the first place, don't have sex or use protection. I see the issue that there there are some women that get raped, and it's very unfortunate, but I think they should give the child up for adoption in such cases. It's not the child's fault that it was created. So why should the goring being have to pay the ultimate price of its life?'''),
    (''''Believing "Evolution" as in Darwinism and the like, is like believing the puzzle can be solved by pouring the pieces out because two pieces kind of stuck together.''',
     '''Believing in "Evolution," as in Darwinism and the like, is like believing the puzzle can be solved by pouring the pieces out because two pieces kind of stuck together. The probability of such things happening tends to be very low; thus, the "evolution" theory seems to be very unlikely, in my opinion.'''),
    (''''The conversation can not be used as evidence in a court of law. I don't know what the lady hoped to gain from recording the conversation other than to create more drama. Some people are hooked on drama and they actually do what they can to create it. Run as far away and as fast as you can from these types. They will suck you dry.''',
     '''The conversation cannot be used as evidence in a court of law. I don't know what the lady hoped to achieve by recording the conversation except that she wanted to create even more drama. Some people like drama and try to create it. I would suggest that you do not surround yourself with such people because they only cost you energy.'''),
    (''''i would turn in my wife because its wrong to kill someone. it could have been an accident but it was still wrong and besides the police are going to find out who killed that person but i don't want her to leave me for a long period of time so i would tell but then again i wouldn't.''',
     '''On the one hand, I consider it to be the right thing to turn in my wife because it's wrong to kill someone. It could have been an accident, but it was still wrong, and besides, the police are going to find out who killed that person. On the other hand, I don't want her to leave me for a long period of time, so I'm a bit torn in this regard.'''),
    (''''it dose not show kids expressions and unforms dose not show is it''',
     '''School uniforms do not let kids express themselves, and it doesn't let them show who they are.'''),
    (''''Firebug, WebDeveloper, TabMix, FaviconizeTab, GreaseMonkey, IETab (to use when you visit microsot.com). Just some reason why i prefer Firefox''',
     '''Firefox has many great tools and plugins, such as Firebug, WebDeveloper, TabMix, FaviconizeTab, GreaseMonkey, and IETab (to use when you visit microsot.com). Those tools are just some of the reasons why I prefer Firefox.''')
]

FEW_SHOT_EXAMPLES_CORE = [
    ('''Hitler invaded Poland in 1932 and the world turned against Germany. In fact, there are dozens if cases in the last 100 years where countries have invaded other nations and the world has caused uproar and rose up against it. Yet some dumb Texan does it and gets away with it. Try him for war crimes, along with Tony Blair and have them both executed or imprisoned.''',
     '''Hitler invaded Poland in 1932, and the world turned against Germany. In fact, there are dozens of cases in the last 100 years where countries have invaded other nations, and the world has caused uproar and rose up against it. Yet an American does it and gets away with it. He should be prosecuted for war crimes, along with Tony Blair.'''),
    (''''There should be no argument in this really...whatever way yu see a fetus...its still a living form that has been created in a very intimate way... you shouldn't be changing what mothernature or God or fate or whatever has decided for you...and if you didn;t wannna get preggo in the first place...don't have sex or use protection. Yeh there are some women that get raped and it's very unfortunate but they should give the child up for adoption. It's not the child's fault that it was created. So why should the goring being have to pay the ultimate price of it's life?''',
     '''In point of view, whatever way you see a fetus it's still a living form that has been created in a very intimate way. You shouldn't be changing what mother nature or God or fate or whatever has decided for you and if you didn't wanna get pregnant in the first place, don't have sex or use protection. I see the issue that there there are some women that get raped, and it's very unfortunate, but I think they should give the child up for adoption in such cases. It's not the child's fault that it was created. So why should the goring being have to pay the ultimate price of its life?'''),
    ('''We will be able to ban water bottles until we get out of this recession!''',
     '''Banning water bottles is a costly or unpopular policy that can only be implemented when the economy is doing well.'''),
    ('''tv because only tv can bring u live news at books u can't find also it's educational''',
     '''TV is better than books because TV can bring you live news that you cannot find in books. Also, it's educational.''')
]

FEW_SHOT_EXAMPLE_ROOT = [
    ('''Fair trade capitalist. Extreme right, I hate any form of marxism. I believe a man should be judge by who he is not his hide. I believe that the solution for the problems of this world is found in the bible. Public education is a waste of money, due to the literacy rate and no basic understanding of economics. Having this as a basis, the world will end one day, not worried about it, because of the laws of nature, namely entropy. The hatred for Israel and Christianity will continue to grow. Nothing will change we have always had storms and always will. We will always have wars, because man is basically selfish''',
     '''I advocate for fair trade and free markets. I strongly reject any form of Marxism. I value individual character over group identity. I find guidance and hope in the bible. I question the effectiveness and efficiency of public education, given the low literacy rate and lack of economic literacy. Based on these views, I accept that the world will end one day, according to the laws of nature, such as entropy. I also expect that Israel and Christianity will face more hatred. I do not think that the world will change much. There have always been storms and wars, and there always will be, because human nature is flawed and selfish.''')
]

FEW_SHOT_EXAMPLES = [
    ('''Coming from a casual internet user, I prefer IE because it has a sleeker design, and I find the bookmark/history thing easier. I'm using firefox though, cuz I don't want to transfer my bookmarks again after my brother made it my default browser.''',
     '''Although I use Firefox as my default browser because I don't want to transfer my bookmarks, as a casual Internet user I prefer IE because it has a sleeker design and an easier bookmarking function.'''),
    ('''Porn is wrong when it is not done in moderation. Porn addicts turn out to have intimacy issues in their relationships and mistreat and view women in a negative manner. Also people who are addicted to porn, expect all women to look like porn stars and act like porn stars, when in reality that is not what sex is about... But it tends to screw up their reality.''',
     '''I believe porn is wrong if not consumed in moderation, as many porn addicts tend to have intimacy issues in their relationships and are prone to mistreat and view women in a negative manner. In addition, people who are addicted to porn have a screwed up view of sex and expect all women to look and act like porn stars, when in reality that is not the case.I believe porn is wrong if not consumed in moderation, as many porn addicts tend to have intimacy issues in their relationships and are prone to mistreat and view women in a negative manner. In addition, people who are addicted to porn have a screwed up view of sex and expect all women to look and act like porn stars, when in reality that is not the case.'''),
    ('''THE SCHOOL UNIFORM IS A VERY GOOOOOOOOOD IDEA , WHY ?? becouse the school uniform makes pupils concentrated on their education than on their clothes and I believe that school uniform instills discipline among pupils it makes pupils with diferent material statuses more equal :)''', '''School uniforms are a good idea because they make students focus more on their education than on their clothes, which I think leads to more discipline and makes students of different material status more equal.''')
]


def process_issue(x):
    x = x.replace('-', ' ').strip().capitalize()
    if x[-1] != ['.', '!', '?', ':']:
        x = x+':'
    return x


def build_dataset(tokenizer, df):
    ds = Dataset.from_pandas(df, split='train')

    def tokenize(sample):
        prompt, post_text = sample["prompt"], sample["post_text"]
        prompt_input_ids = tokenizer.encode(prompt)
        post_text_input_ids = tokenizer.encode(post_text)
        #i = len(FEW_SHOT_EXAMPLES_SUB) - 1
        #while len(prompt_input_ids) + int(len(post_text_input_ids) * 2) > 2048:
        #    prompt = prompt[:-len(INSTRUCT_PATTERN[:-4].format(post_text))]
        #    prompt = prompt[:-len(INSTRUCT_PATTERN.format(FEW_SHOT_EXAMPLES_SUB[i][0], FEW_SHOT_EXAMPLES_SUB[i][1]))]
        #    prompt = prompt + INSTRUCT_PATTERN[:-4].format(post_text)
        #    prompt_input_ids = tokenizer.encode(prompt)
        #    i -= 1
        sample["input_ids"] = prompt_input_ids
        sample["query"] = sample['issue']+' ISSUE_END '+tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def prepare_labeled_data():
    df_out = pd.read_csv('/bigwork/nhwpziet/arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv')
    df_out = df_out[df_out['Inappropriateness'] == 1]
    df_out = df_out[['post_id', 'post_text', 'fold0.0', 'issue']]
    df_out['prompt'] = df_out['post_text'].apply(lambda x: create_prompt(x))
    df_out = df_out.reset_index(drop=True)

    df_out['post_id'] = [i for i in range(len(df_out))]
    return df_out


def prepare_with_unlabeled_data():
    df_labeled = prepare_labeled_data()
    df_unlabeled = prepare_unlabeled_data()
    df_out = pd.concat([df_labeled, df_unlabeled])
    df_out['post_id'] = [i for i in range(len(df_out))]

    return df_out


def add_prediction(df, predictions):
    # set column names of predictions to post_id and prediction
    predictions.columns = ['post_id', 'prediction']
    df['prediction'] = predictions['prediction'].values.tolist()
    df['prediction'] = df['prediction'].apply(lambda x: [x.split(",")[0][1:], x.split(",")[1][1:-1]])
    df['appropriate'] = df['prediction'].apply(lambda x: float(x[0]))
    df['inappropriate'] = df['prediction'].apply(lambda x: float(x[1]))
    return df


def prepare_unlabeled_data():
    df_createdebate = pd.read_csv('/bigwork/nhwpziet/appropriateness-style-transfer/data/iac2/createdebate.csv', sep='\t')
    df_convinceme = pd.read_csv('/bigwork/nhwpziet/appropriateness-style-transfer/data/iac2/convinceme.csv', sep='\t')
    df_gaq = pd.read_csv('/bigwork/nhwpziet/appropriateness-style-transfer/data/GAQCorpus_split/GAQ.csv', sep='\t')

    df_createdebate = add_prediction(df_createdebate, pd.read_csv(
        "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_createdebate.txt", sep="\t", header=None))
    df_convinceme = add_prediction(df_convinceme, pd.read_csv(
        "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_convinceme.txt", sep="\t", header=None))
    df_gaq = add_prediction(df_gaq, pd.read_csv(
        "/bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue/fold0/ensemble_predictions_GAQ.txt", sep="\t", header=None))

    df_createdebate = df_createdebate[df_createdebate['inappropriate'] >= 0.5]
    df_convinceme = df_convinceme[df_convinceme['inappropriate'] >= 0.5]
    df_gaq = df_gaq[df_gaq['inappropriate'] >= 0.5]

    df_createdebate['fold0.0'] = 'TRAIN'
    df_convinceme['fold0.0'] = 'TRAIN'
    df_gaq['fold0.0'] = 'TRAIN'

    #df_createdebate = df_createdebate[df_createdebate['text'].str.len() > 50]
    #df_convinceme = df_convinceme[df_convinceme['text'].str.len() > 50]
    #df_gaq = df_gaq[df_gaq['text'].str.len() > 50]

    df_createdebate = df_createdebate[['text_id', 'text', 'fold0.0', 'discussion_title']]
    df_convinceme = df_convinceme[['text_id', 'text', 'fold0.0', 'discussion_title']]
    df_gaq = df_gaq[['id', 'text', 'fold0.0', 'title']]

    df_createdebate = df_createdebate.rename(columns={'text_id': 'post_id', 'text': 'post_text', 'fold0.0': 'fold0.0', 'discussion_title': 'issue'})
    df_convinceme = df_convinceme.rename(columns={'text_id': 'post_id', 'text': 'post_text', 'fold0.0': 'fold0.0', 'discussion_title': 'issue'})
    df_gaq = df_gaq.rename(columns={'id': 'post_id', 'text': 'post_text', 'fold0.0': 'fold0.0', 'title': 'issue'})

    df_out = pd.concat([df_createdebate, df_convinceme, df_gaq], ignore_index=True)
    print(df_out.head())
    df_out['prompt'] = df_out['post_text'].apply(lambda x: create_prompt(x))
    df_out = df_out.reset_index(drop=True)

    df_out['post_id'] = [i for i in range(len(df_out))]

    return df_out


def create_prompt(argument):
    #tmp_prompt_pattern = ''.join([INSTRUCT_PATTERN.format(x[0], x[1]) for x in FEW_SHOT_EXAMPLES_SUB])
    tmp_prompt_pattern = INSTRUCT_PRE
    prompt = tmp_prompt_pattern + INSTRUCT_PATTERN[:-4].format(argument)
    return prompt


def main(dir_name, app_weight=0.5, similarity_weight=0.5):
    valid_df = prepare_labeled_data()
    valid_df = valid_df[valid_df['fold0.0'] == 'VALID']
    train_df = prepare_unlabeled_data()
    # keep only instances that of post_text that are shorter than the max length in tmp_df
    # df = prepare_with_unlabeled_data()
    #if len(df) > 25600:
    #    df = df.sample(25600, random_state=42)
    print(len(train_df))

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    config = TRLConfig(
        train=TrainConfig(
            seq_length=384,
            epochs=(len(train_df) // 4) * 3,
            total_steps=(len(train_df) // 4) * 3,
            batch_size=4,
            checkpoint_interval=1000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            checkpoint_dir=dir_name,
            seed=42
        ),
        model=ModelConfig(
            model_path="/bigwork/nhwpziet/appropriateness-style-transfer/data/models/instruction-finetuning/llama-7b-instruct",
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path="/bigwork/nhwpziet/appropriateness-style-transfer/data/models/instruction-finetuning/llama-7b-instruct",
            truncation_side="right",
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs={
                "lr": 5e-6,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
            },
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing",
            kwargs={
                "T_max": int(len(train_df)) * 3,
                "eta_min": 1.508e-6,
            },
        ),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=16,
            ppo_epochs=4,
            init_kl_coef=0.001857,
            target=None,
            horizon=(len(train_df) // 4) * 3,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.2,
            scale_reward=None,
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs={
                "max_new_tokens": 384,
                "min_new_tokens": 10,
            },
        ),
    )

    hparams = {}
    config = TRLConfig.update(config.to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    print(f"Using device: {device}")

    metrics_calculator = MetricsCalculator(
            semantic_similarity=True,
            token_edit_distance=True,
            perplexity=False,
            classifier_prediction=True,
            device='cuda:'+str(device),
            )

    config.model.peft_config = LoraConfig(
        peft_type="LORA",
        r=8,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        out_ = [x.split('\n### Response:')[1].replace('</s>', '') for x in samples]
        in_ = [x.split('\n### Input:')[1].split('\n### Response:')[0] for x in samples]
        metrics = metrics_calculator.calculate_metrics(out_, in_)
        app_scores = metrics['classifier_predictions_app']
        app_scores_binary = metrics['classifier_predictions']
        token_edit_distance = metrics['token_edit_distances']
        semantic_similarities_ = list(metrics['semantic_similarities'])
        rewards, app_rewards, app_binary_rewards, ted_rewards, ss_rewards = [], [], [], [], []
        for a, a_binary, t, s in zip(app_scores, app_scores_binary, token_edit_distance, semantic_similarities_):
            #rewards.append(2*(a*s)/(a+s))
            # weighted harmonic mean
            #rewards.append(1/((app_weight/a)+(similarity_weight/s)))
            rewards.append(app_weight*a + similarity_weight*s)
            app_rewards.append(a)
            app_binary_rewards.append(a_binary)
            ted_rewards.append(t)
            ss_rewards.append(s)
        return rewards, app_rewards, app_binary_rewards, ted_rewards, ss_rewards

    # sort train_df by length of prompts
    #train_df['prompt_len'] = train_df['prompt'].str.len()
    #train_df = train_df.sort_values(by=['prompt_len'])
    #train_df = train_df.reset_index(drop=True)
    #train_df = train_df[-4:]
    #print(train_df['prompt_len'])
    train_prompts = train_df['prompt'].tolist()
    valid_prompts = valid_df['prompt'].tolist()

    trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=valid_prompts,
        config=config,
    )


def clean_checkpoints(dir_name):
    # go into each folder in dir_name
    for folder in os.listdir(dir_name):
        # remove pytorch_model dir in each folder
        if os.path.isdir('{}/pytorch_model'.format(os.path.join(dir_name, folder))):
            os.system('rm -r {}/pytorch_model'.format(os.path.join(dir_name, folder)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str)
    parser.add_argument('--app_weight', type=float)
    parser.add_argument('--similarity_weight', type=float)
    args = parser.parse_args()
    main(args.dir_name, args.app_weight, args.similarity_weight)
    clean_checkpoints(args.dir_name)
