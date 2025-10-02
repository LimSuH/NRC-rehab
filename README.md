### NRC-rehab
They can be expanded with Post-Stroke rehabilitation action Recognition [__MMeViT__](https://github.com/ye-Kim/MMeViT),  
for end-to-end Domiciliary Stroke rehabilitation System.
<br>
<br>

# RAST-G@
This is an official implementation of arXiv paper: [AI-Based Stroke Rehabilitation Domiciliaty Assessment System with ST-GCN Attention](https://arxiv.org/abs/2510.00049)
<br>

## Overview
<img width="3684" height="1755" alt="fig3" src="https://github.com/user-attachments/assets/c6b82aac-a2df-41d4-bf7d-587c4e06685e" />
We evaluate in-home rehabilitation movements using XYZ skeleton sequences extracted from RGB-D cameras and annotated with expert therapist scores. RAST-G@ predicts a 50 scale score for each input action using an ST-GCN with attention backbone.
<br>
<br>


## Requirements
To install libraries, run pip install -r <code>requirements.txt</code>  
<br>


## Files
Please refer only to the RAST-G@ directory. This repository is positioned one level above the core model to enable extension and integration with other models and codebases.
- <code>train.py</code> : Python script to launch model training
- <code>data_processing.py</code> : Script that constructs the training set and returns a __torch.utils.data.Dataset__ object.
- <code>RAST-process.py</code> : Utility functions for per-frame preprocessing
- <code>caseRecord</code> : In every epoch metrics and training configuration are written to __description.txt__
- <code>net</code> : Directory containing the model implementation.
<br>
  
## Train
please run this:
<pre><code>python RAST-G@/train.py -m this is example log from RAST-G@ training.\
--ex_train [your_train_data.csv]\
--ex_test [your_test_data.csv]\
... #other args ...
</code></pre>
<br>

### Citation
If you found this repo useful, please consider citing our paper.
<pre><code>@article{lim2025aibasedstrokerehabilitationdomiciliary,
      title={AI-Based Stroke Rehabilitation Domiciliary Assessment System with ST_GCN Attention}, 
      author={Suhyeon Lim and Ye-eun Kim and Andrew J. Choi},
      year={2025},
      eprint={2510.00049},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2510.00049}, 
}
</code></pre>
