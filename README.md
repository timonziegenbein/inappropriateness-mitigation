This repository contains the source code used to obtain the results reported in the paper [LLM-based Rewriting of Inappropriate Argumentation using Reinforcement Learning from Machine Feedback
](https://arxiv.org/abs/2406.03363) published at the ACL2024.
--
We most notably published the code to train the **Rewriting Approaches for Inappropriateness Mitigation**. 

## What does Inappropriateness mean?
An argument “has an appropriate style if the used language supports the creation of credibility and emotions as well as if it is proportional to the issue.” Their annotation guidelines further suggest that “the choice of words and the grammatical complexity should [...] appear suitable for the topic discussed within the given setting [...], matching the way credibility and emotions are created [...]”. 
> [Wachsmuth et al. (2017)](https://aclanthology.org/E17-1017/)

If an argument does **not** follow this definition, we consider it to be inappropriate.

## What makes an Argument (In)appropriate?
There are many reasons for Inappropriateness; if you are interested in them, we refer to our previous work on [Modeling Appropriate Language in Argumentation](https://aclanthology.org/2023.acl-long.238)

## Reproducibility of the Results
All the code used to obtain the results reported in the paper is in the folder `src`. This includes the following:
- `src/annotation-interface`: The code for the annotation interfaces used to collect the annotations for the evaluation of our approaches
- `src/appropriateness-prediction`: The code for the training of a binary (in)appropriateness classifier
- `src/inappropriateness-mitigation`: The code for training our models to mitigate inappropriateness through rewriting
- `src/instruction-finetuning`: The code for training the baseline models to mitigate inappropriateness through rewriting
- `src/soft_labeling`: The code for soft-labeling the extended argument dataset with (in)appropriateness labels

## Using our Work 
If you are interested in using the models or the corpus, please cite the following paper:

[LLM-based Rewriting of Inappropriate Argumentation using Reinforcement Learning from Machine Feedback
](https://arxiv.org/abs/2406.03363) (Ziegenbein et al., ACL 2024)
