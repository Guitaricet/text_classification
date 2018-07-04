# Classification of noisy texts

Various models evaluation on noisy texts

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en
```

## Run

```bash
python experiment_imdb.py --model-name CharCNN
```

Where model name in (CharCNN, FastText, YoonKim, AttentionedYoonKim)
