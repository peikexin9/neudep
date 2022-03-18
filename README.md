## Introduction

NeuDep is a tool to detect binary memory dependencies. 

## Installation
First, create the conda environment,

`conda create -n neudep python=3.9 scipy scikit-learn requests`

and activate the conda environment:

`conda activate neudep`

Then, install the latest PyTorch (assume you have GPU):

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

Enter the neudep root directory: e.g., `path/to/neudep`, and install neudep:

`pip install --editable .`

Install PyArrow for large datasets: 

`pip install pyarrow`

For faster training install NVIDIA's apex library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## File Structure

`checkpoints` - store the pretrained and finetuned checkpoints. We provide the pretrained model [here](https://drive.google.com/drive/folders/1MdI_0q1hOgTfa0QKioWSrhVmsFWLYyZe?usp=sharing): 

`command` - scripts to pretrain and finetune the model. Hyperparameters are included in the training scripts too, e.g., `command/finetune/table3/finetune_table3_all.sh` we have `MAX_SENTENCES=32` indicating batch size is 32.

`fairseq` - implementation of model architecture, preprocessing pipeline, and training task loss

`data-src` - preprocessed dataset for model to binarize. We put the sample data for obtaining result in Table 4 at [here](https://drive.google.com/drive/folders/1xZt-SYyC0neSl7cKsP97d03AaLNjKZ72?usp=sharing)

`data-bin` - stores binarized dataset for actual training