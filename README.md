
# Transformers on Bounded Dyck Languages
Code for ACL 2021 paper [Self-Attention Networks Can Process Bounded Hierarchical Languages](https://arxiv.org/abs/2105.11115)

## Getting started

* Install the required packages.

```
pip install -r requirements.txt
```

* Evaluate different positional encoding schemes (Figure 4 (a)):

```bash
for d in acl2021/experiments_embedding/*; do
	python src/run_lm.py ${d}
done
```

* Compare Transformer and LSTM with different memory dims (Figure 4 (b, c)):

```bash
for d in acl2021/experiments_memory/*; do
	python src/run_lm.py ${d}
done
```

## The config file for specifying experiments
This repository exclusively uses `yaml` configuration files for specifying each experiment.
Here's an explanation of what each part of the `yaml` configs means:

The first portions specify the datasets' locations and properties of the specific Dyck-(k,m) language.
For generating data with `rnns-stacks/generate_mbounded_dyck.py`, only this portion is needed.
 - `corpus`:
    - `train_corpus_loc`: The filepath for the training corpus
    - `dev_corpus_loc`: The filepath for the development corpus
    - `test_corpus_loc`: The filepath for the test corpus
- `language`:
    - `train_bracket_types`: The number of unique bracket types, also _k_ in Dyck-(k,m) for the training set.
    - `train_max_length`: The maximum length of any training example 
    - `train_min_length`: The minimum length of any training example 
    - `train_max_stack_depth`: The maximum number of unclosed open brackets at any step of a training example
    - `train_sample_count`: Number of samples in tokens (!!) not lines, for the training set.
    - `dev_bracket_types`: The number of unique bracket types in the development set, also _k_ in Dyck-(k,m).
    - `dev_max_length`: The maximum length of any development example 
    - `dev_min_length`: The minimum length of any development example 
    - `dev_max_stack_depth`: The maximum number of unclosed open brackets at any step of a development example
    - `dev_sample_count`: Number of samples in tokens (!!) not lines, for the development set.
    - `test_bracket_types`: The number of unique bracket types, also _k_ in Dyck-(k,m) for the test set.
    - `test_max_length`: The maximum length of any test example 
    - `test_min_length`: The minimum length of any test example 
    - `test_max_stack_depth`: The maximum number of unclosed open brackets at any step of a test example
    - `test_sample_count`: Number of samples in tokens (!!) not lines, for the test set.

Note that running an experiment training an LM with a specific `corpus` and `language` configuration doesn't generate the corresponding dataset; instead, you should first run  `rnns-stacks/generate_mbounded_dyck.py` to generate the dataset, and then use `rnns-stacks/run_lm.py` to train and evaluate the LM.

The next portions of the `yaml` configuration files is for specifying properties of the LSTM LMs.

- `lm`: 
     - `embedding_dim`: The dimensionality of the word embeddings.
     - `hidden_dim`: The dimensionality of the LSTM hidden states.
     - `lm_type`: Chooses RNN type; pick from RNN, GRU, LSTM.
     - `num_layers`: Chooses number of stacked RNN layers
     - `save_path`: Filepath (relative to reporting directory) where model parameters are saved.
 - `reporting`: 
     - `reporting_loc`: Path specifying where to (optionally construct a folder, if it doesn't exist) to hold the output metrics and model parameters.
     - `reporting_methods`: Determines how to evaluate trained LMs. `constraints` provides an evaluation metric determining whether models know which bracket should be closed, whether the sequence can end, and whether an open bracket can be seen at each timestep.
 - `training`: 
     - `batch_size`: Minibatch size for training. Graduate student descent has found that smaller batches seems to be better in general. (100: too big. 1: maybe the best? But very slow. 10: good)
     - `dropout`: Dropout to apply between the LSTM and the linear (softmax matrix) layer constructing logits over the vocabulary.
     - `learning_rate`: Learning rate to initialize Adam to. Note that a 0.5-factor-on-plateau decay is implemented; each time the learning rate is decayed, Adam is restarted.
     - `max_epochs`: Number of epochs after which to halt training if it has not already early-stopped.
     - `seed`: Doesn't actually specify random seed; used to distinguish multiple runs in summarizing results. Maybe should have specified random seeds, but wouldn't replicate across different GPUs anyway...

## Code layout
- `generate_mbounded_dyck.py`: Code for generating samples from distributions over Dyck-(k,m).
- `run_lm.py`: Code for running experiments with `yaml` configs.
- `rnn.py`: Classes for specifying RNN models.
- `transformer.py`: Classes for specifying Transformer models.
- `lm.py`: Classes for specifying probability distributions given an encoding of the sequence.
- `dataset.py`: Classes for loading and serving examples from disk.
- `training_regimen.py`: Script for training an LM on samples
- `reporter.py`: Classes for specifying how results should be reported on a given experiment.
- `utils.py`: Provides some constants (like the Dyck-(k,m) vocabulary) as well as paths to corpora and results.

## Citation

```
@inproceedings{yao2021dyck,
    title={Self-Attention Networks Can Process Bounded Hierarchical Languages},
    author={Yao, Shunyu and Peng, Binghui and Papadimitriou, Christos and Narasimhan, Karthik},
    booktitle={Association for Computational Linguistics (ACL)},
    year={2021}
}
```

## Acknowledgements

The code heavily borrows from [dyckkm-learning](https://github.com/john-hewitt/dyckkm-learning). Thanks John!

For any questions please contact Shunyu Yao `<shunyuyao.cs@gmail.com>`.