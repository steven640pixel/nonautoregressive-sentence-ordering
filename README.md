# Non-Autoregressive Sentence Ordering
This is the implementation of our paper, entitled "Non-Autoregressive Sentence Ordering", which was accepted at Findings of EMNLP 2023.

## Prerequisites
You may need a machine with GPUs and Pytorch v1.6.0 Python 3

1. Install Pytorch with CUDA and Python 3.6
2. Install Transformers 

Our implementation uses the pre-trained huggingface model for sentences encoding. 
The pre-trained model may be downloaded to a directory or invoked by huggingface.

## Data

The data scripts can be used to load data. 
All data should be downloaded to a dataset/ directory. And each line is a paragraph. We provide SIND dataset sample. Other datasets are easy to access and process. 
For the AAN, NIPS and NSF data, we obtained from the authors of Sentence Ordering and Coherence Modeling using Recurrent Neural Networks. The arXiv data is able to obtain from the authors of Neural Sentence Ordering. 
The SIND dataset can be downloaded from the Visual Storytelling website.  The ROCStory dataset can be downloaded from the ROCStories website.

## Training and test
The main.py script should be used to train and test the model. The modeling_bert.py, encoder.py and utils are scripts that construct the model. 

* Train the models using the following command and set args.do_train to True.

+ Test the models setting args.do_train to False and the trained sample model can be [downloaded](https://drive.google.com/file/d/1cdlkbZe-oNHKnGNFGDUeynItv2Sq-fHk/view?usp=sharing).

## Citation
#
If this repo is useful to your work, please cite:

```
@inproceedings{bin2023non,
  title={Non-Autoregressive Sentence Ordering},
  author={Bin, Yi and Shi, Wenhao and Ji, Bin and Zhang, Jipeng and Ding, Yujuan and Yang, Yang},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={4198--4214},
  year={2023}
}
```
