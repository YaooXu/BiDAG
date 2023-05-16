# KGReasoning-BiDAG

This repo is forked from [KGReasoning](https://github.com/snap-stanford/KGReasoning),
including  the official Pytorch implementation of _Prediction and Calibration: Complex Reasoning over Knowledge Graph with Bi-directional Directed Acyclic Graph Neural Network_.

**models**
- BetaE
- Query2Box
- GQE
- BiDAG _(Ours)_



**KG Data**

The KG data (FB15k, FB15k-237, NELL995) mentioned in the BetaE paper and the Query2box paper can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip). Note the two use the same training queries, but the difference is that the valid/test queries in BetaE paper have a maximum number of answers, making it more realistic.

Each folder in the data represents a KG, including the following files.
- `train.txt/valid.txt/test.txt`: KG edges
- `id2rel/rel2id/ent2id/id2ent.pkl`: KG entity relation dicts
- `train-queries/valid-queries/test-queries.pkl`: `defaultdict(set)`, each key represents a query structure, and the value represents the instantiated queries
- `train-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`)
- `valid-easy-answers/test-easy-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`) / valid graph (edges in `train.txt`+`valid.txt`)
- `valid-hard-answers/test-hard-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the **additional** answers obtained in the validation graph (edges in `train.txt`+`valid.txt`) / test graph (edges in `train.txt`+`valid.txt`+`test.txt`)

We represent the query structures using a tuple in case we run out of names :), (credits to @michiyasunaga). For example, 1p queries: (e, (r,)) and 2i queries: ((e, (r,)),(e, (r,))). Check the code for more details.

**Examples**

Please refer to the `train.sh` for the scripts of model on all 3 datasets.
