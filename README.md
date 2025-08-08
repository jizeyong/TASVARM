# **A Video Action Recognition Model Guided by Temporal Action Semantics**

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
