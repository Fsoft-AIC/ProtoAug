#### We assume a synthetic dataset is generated. In this step, we cluster the real and synthetic images with Faiss, train the classifier, using both synthetic and few-shot real data.

## Setup

Fill in the paths in `local.yaml` file or folder yaml_file. User add dataset names and their paths in keys 'clip_download_dir', `real_train_data_dir`, `real_test_data_dir`, and `synth_train_data_dir`.

## Running
Please follow the example in the bash folder for running

```python
bash bash_run.sh
```
