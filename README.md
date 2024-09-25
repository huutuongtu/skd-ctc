# Guiding Frame-Level CTC Alignments Using Self-knowledge Distillation
A PyTorch Implementation of Guiding Frame-Level CTC Alignments Using Self-knowledge Distillation

## Preparing :
You should prepare the training, development, and test datasets by following the structure provided [here](https://github.com/huutuongtu/skd-ctc/tree/main/dataset). Each file should contain two columns:

1. Path – The path to the audio file.
2. Transcript – The corresponding transcript for the audio (the transcript should be normalized, such as removing all punctuation, converting to lowercase, etc., or you may need to modify the [vocabulary](https://github.com/huutuongtu/skd-ctc/blob/main/vocab.json)).

## Training :
```
pip install -r requirements.txt
python3 train.py
```

## Citations :
```
@article{SKD-CTC,
  title={Guiding Frame-Level CTC Alignments Using Self-knowledge Distillation},
  author={Eungbeom Kim, Hantae Kim, Kyogu Lee},
  journal={INTERSPEECH 2024},
  year={2024},
}
```
