# CoRR

Here is the code for Cooperative Retriever and Ranker in Deep Recommenders based on recommendation library [RecStudio](https://github.com/ustcml/RecStudio)


### Dataset

With RecStudio, the dataset can be downloaded automatically by specifying dataset name.


### Run

To run CoRR algorithm, you should run:

```bash
python run.py -m CoRR -d amazon-electronics --batch_size 512
```

If you just want to have a try, a tiny dataset is recommended: ml-100k.

For general recommendation, you should run:

```bash
python run.py -m CoRRMF -d amazon-electronics --batch_size 1024
```

The default retriever and ranker are [SASRec, DIN] for sequential recommendation and [MF+DeepFM] for general recommendation. If you want to specify retriever and ranker, run like this:

```bash
python run.py -m CoRR -d amazon-electronics --retriever Caser --ranker BST
```

The default number of negatives is 20, you can specify it with arguments `--num_neg`, i.e. `--num_neg 100`.