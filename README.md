# Classification of noisy texts

Various models evaluation on noisy texts

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
usage: run_experiment.py [-h] [--model-name MODEL_NAME]
                         [--dataset-name DATASET_NAME] [--comment COMMENT]
                         [--datapath DATAPATH] [--noise-level NOISE_LEVEL]
                         [--embeddings-path EMBEDDINGS_PATH] [-y]
                         [--original-train] [--sample-data SAMPLE_DATA]

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
  --dataset-name DATASET_NAME
  --comment COMMENT
  --datapath DATAPATH
  --noise-level NOISE_LEVEL
  --embeddings-path EMBEDDINGS_PATH
  -y                    yes to all
  --original-train      train_on_original_dataset
  --sample-data SAMPLE_DATA
```

Where model name in (CharCNN, FastText, YoonKim, AttentionedYoonKim)
