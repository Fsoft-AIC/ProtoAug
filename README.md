# "[ICML 2025] Provably Improving Generalization of Few-shot models with synthetic data"

This is the implementation of full version of paper Provably Improving Generalization of Few-shot models with synthetic data. 

This code is heavily adapted from DataDream codebase: https://github.com/ExplainableML/DataDream. Thanks DataDream authors for their great work!

The codebase for lightweight version can be found at: https://github.com/cuonglannguyen/ProtoAug_lightweight

![](figures\SynSup_pipeline-Page-1-1.pdf)

## Preliminary Setup

Before we start, few-shot real data should be formed in the following way. Assuming we use 16-shot, each data files should be located in the path `data/$DATASET/real_train_fewshot/seed$SEED_NUMBER/$CLASS_NAME/$FILE`. The list of `CLASS_NAME` For each `DATASET` can be found in `DataDream/util_data.py` file. 

For instance, the ImageNet 16-shot data should be stored as follows:

```bash
📂 data
|_📂 imagenet
  |_📂 real_train_fewshot
    |_📂 seed0
      |_📂 abacus
        |_📄 n02666196_17944.JPEG
        |_📄 n02666196_10754.JPEG
        |_📄 n02666196_10341.JPEG
        |_📄 n02666196_26262.JPEG
        |_📄 n02666196_16203.JPEG
        |_📄 n02666196_15765.JPEG
        |_📄 n02666196_16339.JPEG
        |_📄 n02666196_7225.JPEG
        |_📄 n02666196_13227.JPEG
        |_📄 n02666196_19345.JPEG
        |_📄 n02666196_19170.JPEG
        |_📄 n02666196_9008.JPEG
        |_📄 n02666196_20311.JPEG
        |_📄 n02666196_17676.JPEG
        |_📄 n02666196_16649.JPEG
      |_📂 clothes iron
      |_📂 great white shark
      |_📂 goldfish
      |_📂 tench
      ...
```


## Step

1. Install the necessary dependencies in `requirements.txt`.
2. **DataDream**: Follow the instructions in the `DataDream` folder.
3. **Dataset Generation**: Follow the instructions in the `generate` folder.
4. **Train Classifier**: Follow the instructions in the `classify` folder.
