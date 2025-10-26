# **A Video Action Recognition Model Guided by Temporal Action Semantics**
[![ADMA 2025 Accepted](https://img.shields.io/badge/ADMA-2025%20Accepted-red)](https://adma2025.github.io/accepted_papers.html)
## Prepare in advance

Download the project and install the library in requirement.txt.

```bash
$ pip install -r requirements.txt
```



## Experimental Datasets

HMDB51 and UCF101 can be downloaded by following this [link](https://github.com/open-mmlab/mmaction2). Animal Kingdom can be downloaded from [here](https://sutdcv.github.io/Animal-Kingdom/). Charades can be downloaded from [here](https://prior.allenai.org/projects/charades).



## Running

First you need to generate auxiliary statements corresponding to the dataset

```bash
$ python action_text_prompt.py --dataset [DATASET NAME] --top_k [ACTION WORD NUMBER]
```

Find more implementation details in the script generate_CLIP.py.

Then run the following 

```bash
$ python main.py --dataset [DATASET NAME] --models [MODEL NAME]
```

Find more implementation details in the script main.py.

## Citation
If our work is helpful to your research, please cite our paper:

### Text Format (Plain text, suitable for Word, etc.)
Ji, Z., Zhang, J. (2026). A Video Action Recognition Model Guided by Temporal Action Semantics. In: Yoshikawa, M., Meng, X., Cao, Y., Xiao, C., Chen, W., Wang, Y. (eds) Advanced Data Mining and Applications. ADMA 2025. Lecture Notes in Computer Science(), vol 16199. Springer, Singapore. https://doi.org/10.1007/978-981-95-3459-3_27


### BibTeX Format (Suitable for LaTeX)
@InProceedings{
    author    = {Ji, Zeyong and Zhang, Jinqu},
    editor    = {Yoshikawa, Masatoshi and Meng, Xiaofeng and Cao, Yang and Xiao, Chuan and Chen, Weitong and Wang, Yanda},
    title     = {A Video Action Recognition Model Guided by Temporal Action Semantics},
    booktitle = {Advanced Data Mining and Applications},
    year      = {2026},
    publisher = {Springer Nature Singapore},
    address   = {Singapore},
    pages     = {321--335},
    doi       = {10.1007/978-981-95-3459-3_27}
}