# Aligning open educational resources to new taxonomies: How AI technologies can help and in which scenarios
![model](https://user-images.githubusercontent.com/39921289/227646739-aa7baa2a-6a18-4e0b-a4f9-3ab7399819d7.png)

This repo contains the code for the paper:

Zhi Li, Zachary A Pardos, and Cheng Ren. 2024. [Aligning Open Educational Resources to New Taxonomies: How AI technologies can help and in which scenarios](https://doi.org/10.1016/j.compedu.2024.105027). Computers & Education (2024).

## Dependencies
The code is written and run with the following packages:
* Python 3.6.8
* PyTorch 1.7.1+cu101
* NumPy 1.18.5
* pandas 1.1.5
* scikit-learn 0.24.2
* sentence-transformers 1.2.1
* efficientnet-pytorch 0.7.0


## Data
* `data/` contains the Common Core Mathematics skills list and text embeddings of the skill descriptions.
* `data/examples` contains two example problems for demostration purpose.


## Instructions
### Data Preparation
Put the problem texts and images in a folder similar to `data/examples`. Put training labels if any in a csv file with two columns `source` and `target`.

### Training
Prepare the problems data and labels as described above. Use `TextEncoder`, `ImageEncoder`, and `Fusion` in `encoder/` to encode problem data into embeddings, then use `ClassificationModelTrainer` and `SimilarityMatchingModelTrainer` in `training.py` to train the models.

### Evaluation
Prepare the problems data as described above. Use `TextEncoder`, `ImageEncoder`, and `Fusion` in `encoder/` to encode problem data into embeddings, then use `ClassificationModelTester` and `SimilarityMatchingModelTester` in `inference.py` to run the models.

We also release two pre-trained models that map problems to Common Core skills in `models/`, and provide a `demo.ipynb` to show how to use the pre-trained models to predict the skills for the given problems.


## References
We referenced the following repos for the code:
* [pytorch_compact_bilinear_pooling](https://github.com/gdlg/pytorch_compact_bilinear_pooling)
