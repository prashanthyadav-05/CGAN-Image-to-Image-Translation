# Task-04: Image-to-Image Translation with cGAN (pix2pix)

## Objective
Implement an image-to-image translation model using a Conditional Generative Adversarial Network (cGAN) called pix2pix.

This project uses the paired **CMP Facades** dataset. The model learns:

**Architectural label/map → realistic facade photograph**

## Model
The implementation contains:

- U-Net based Generator
- PatchGAN Discriminator
- Conditional adversarial loss
- L1 reconstruction loss
- Adam optimizer
- PyTorch
- Automatic dataset download
- Generated sample images after every epoch

The pix2pix approach was introduced by Isola et al. for paired image-to-image translation.

## Project structure

```text
Task-04-Pix2Pix/
│
├── train.py
├── translate.py
├── requirements.txt
├── README.md
├── .gitignore
├── checkpoints/
├── results/
└── data/
```

## 1. Create virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

If `py -3.11` is not available, install Python 3.11 first.

## 2. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train the model

For a quick internship demonstration:

```powershell
python train.py --epochs 5
```

The first run automatically downloads and extracts the CMP Facades dataset into `data/facades`.

For better results, increase the number of epochs:

```powershell
python train.py --epochs 20
```

CPU training can take a long time. A CUDA-enabled NVIDIA GPU is much faster.

## 4. Check the output

After each epoch, a comparison image is saved in:

```text
results/
```

Example:

```text
epoch_001.png
epoch_002.png
...
```

Each image contains:

1. Input label/map
2. Real target facade
3. Generated facade

The trained weights are saved in:

```text
checkpoints/generator_latest.pth
checkpoints/discriminator_latest.pth
```

## 5. Translate a test image

After training:

```powershell
python translate.py --input data/facades/test/100.jpg
```

The generated image is saved as:

```text
results/translated.png
```

The script accepts either a normal input image or an original pix2pix paired image. For a paired image, the left half is automatically used as the input.

## Important note about the dataset

Pix2pix requires paired/aligned training data. The CMP Facades dataset contains paired images stored side-by-side. The code splits each image into an input and target image before training.

## Loss function

The generator is trained with:

```text
Generator Loss = GAN Loss + 100 × L1 Loss
```

The discriminator learns to distinguish:

```text
(real input, real target)
```

from

```text
(real input, generated target)
```

The conditional input is important because the discriminator judges the generated image together with its input.

## Expected result

After sufficient training, the generator should learn to convert architectural label maps into facade-like photographs while preserving the structure of the input.

Results improve as training continues. Five epochs is mainly for a quick demonstration; more epochs generally produce better visual quality.

## GitHub upload

Do NOT upload:

- `venv/`
- `data/`
- large generated files
- model checkpoints if repository size is a concern

The included `.gitignore` handles the common generated folders.

## Suggested repository title

`Task-04-Pix2Pix-Image-to-Image-Translation`

## References

1. Isola, P., Zhu, J., Zhou, T., & Efros, A. A. "Image-to-Image Translation with Conditional Adversarial Networks."
2. Berkeley pix2pix dataset repository.
