# Vision Transformer (ViT)

This repository is an __unofficial implementation__ of Vision Transformer proposed in [here](https://arxiv.org/pdf/2010.11929.pdf).


## Requirements
1. torch
2. torchvision
3. tqdm


## Training
To run the training code, run the following code

```
python -m src.ViT.train
```
To counter overfitting, L2 penalty is applied with Adam optimizer with decay value of 0.003, also dropout with a probability of 0.3 is applied.

## Results
| Dataset | Accuracy | #Epochs |
|---------|----------|---------|
| MNIST   |   95%    |   30    |

## References
1. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
