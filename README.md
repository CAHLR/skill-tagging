This repo contains the code for the papers:

Zhi Li, Zachary A Pardos, and Cheng Ren. (In press). [Aligning Open Educational Resources to New Taxonomies: How AI technologies can help and in which scenarios](https://doi.org/10.1016/j.compedu.2024.105027). Computers & Education.

Yerin Kwak, and Zachary A Pardos. 2024. [Bridging large language model disparities: Skill tagging of multilingual educational content](https://doi.org/10.1111/bjet.13465). British Journal of Educational Technology.

# Aligning open educational resources to new taxonomies: How AI technologies can help and in which scenarios

![model](https://user-images.githubusercontent.com/39921289/227646739-aa7baa2a-6a18-4e0b-a4f9-3ab7399819d7.png)

## Dependencies

The code is written and run with the following packages:

- Python 3.6.8
- PyTorch 1.7.1+cu101
- NumPy 1.18.5
- pandas 1.1.5
- scikit-learn 0.24.2
- sentence-transformers 1.2.1
- efficientnet-pytorch 0.7.0

## Data

- `data/` contains the Common Core Mathematics skills list and text embeddings of the skill descriptions.
- `data/examples` contains two example problems for demonstration purpose.

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

- [pytorch_compact_bilinear_pooling](https://github.com/gdlg/pytorch_compact_bilinear_pooling)

# Bridging large language model disparities: Skill tagging of multilingual educational content

## Dependencies

The code is written and run with the following packages:

- Python 3.8.8
- NumPy 1.23.4
- pandas 2.0.3
- Openai 1.3.3
- accelerate 0.21.0
- peft 0.4.0
- bitsandbytes 0.40.2
- transformers 4.31.0
- trl 0.4.7

## Data

`data/examples` contains example files for fine-tuning LLMs for demonstration purposes.

- `gpt_finetuning_train.jsonl`: training file for fine-tuning GPT
- `gpt_finetuning_val.jsonl`: validation file for fine-tuning GPT
- `llama_finetuning_train.csv`: training file for fine-tuning Llama2
- `llama_finetuning_val.csv`: validation file for fine-tuning Llama2

## Instructions

`llm-GPT.ipynb` and `llm-Llama2.ipynb` contain code implementing two performance improvement strategies discussed in the paper - Fine-tuning (Section 4.3.2) and Standards in the prompt (Section 4.3.3)

### Fine-tuning

- GPT: Prepare the training and validation JSONL files following the format of `data/examples/gpt_finetuning_train.jsonl` and `data/examples/gpt_finetuning_val.jsonl`, and upload the files. Next, create a fine-tuned model. Once your job is completed, use the model for inference.
- Llama2: To minimize VRAM usage during the fine-tuning process, we used the 4-bit precision technique with parameter-efficient fine-tuning (PEFT) - QLoRA. Prepare the training and validation CSV files following the format of `data/examples/llama_finetuning_train.jsonl` and `data/examples/llama_finetuning_val.jsonl`, and load the files. Start the fine-tuning process, and once your job is finished, store your model.

### Standards in the prompt

Import a CSV file containing the problems that you intend to tag with skills, and include a list of standards into the prompt.

## References

We referenced the following repos for the code:

- [openai-python](https://github.com/openai/openai-python)
- [openai-Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- [meta-Fine-tuning](https://llama.meta.com/docs/how-to-guides/fine-tuning/)
- [Fine-Tune-Llama 2](https://github.com/MuhammadMoinFaisal/LargeLanguageModelsProjects/tree/main/Fine-Tune%20Llama%202)
