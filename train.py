import argparse
import tarfile
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

DATA_URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "facades"
CHECKPOINT_DIR = Path("checkpoints")
RESULT_DIR = Path("results")

IMG_SIZE = 256
BATCH_SIZE = 1
LR = 0.0002
BETA1 = 0.5
LAMBDA_L1 = 100.0


def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    archive = DATA_DIR / "facades.tar.gz"

    if DATASET_DIR.exists():
        return

    print("Downloading the CMP Facades pix2pix dataset...")
    urllib.request.urlretrieve(DATA_URL, archive)

    print("Extracting dataset...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    archive.unlink(missing_ok=True)
    print("Dataset ready.")


class FacadesDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.jpg"))

        if not self.files:
            raise RuntimeError(f"No .jpg images found in {self.root}")

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        width, height = image.size

        # Pix2pix paired images are stored side-by-side:
        # left = input/label map, right = target/photo.
        input_image = image.crop((0, 0, width // 2, height))
        target_image = image.crop((width // 2, 0, width, height))

        return self.transform(input_image), self.transform(target_image)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """U-Net generator used in the pix2pix-style implementation."""

    def __init__(self):
        super().__init__()

        self.d1 = DownBlock(3, 64, normalize=False)
        self.d2 = DownBlock(64, 128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.d6 = DownBlock(512, 512)
        self.d7 = DownBlock(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.u1 = UpBlock(512, 512, dropout=True)
        self.u2 = UpBlock(1024, 512, dropout=True)
        self.u3 = UpBlock(1024, 512, dropout=True)
        self.u4 = UpBlock(1024, 512)
        self.u5 = UpBlock(1024, 256)
        self.u6 = UpBlock(512, 128)
        self.u7 = UpBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)

        b = self.bottleneck(d7)

        u1 = self.u1(b)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))

        return self.final(torch.cat([u7, d1], dim=1))


class PatchDiscriminator(nn.Module):
    """70x70-style PatchGAN discriminator."""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            DownBlock(64, 128),
            DownBlock(128, 256),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, input_image, target_image):
        pair = torch.cat([input_image, target_image], dim=1)
        return self.model(pair)


def denormalize(tensor):
    return (tensor.detach().cpu() * 0.5 + 0.5).clamp(0, 1)


def save_sample(input_image, target_image, generated_image, epoch):
    import matplotlib.pyplot as plt

    RESULT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [
        denormalize(input_image[0]).permute(1, 2, 0),
        denormalize(target_image[0]).permute(1, 2, 0),
        denormalize(generated_image[0]).permute(1, 2, 0),
    ]
    titles = ["Input", "Target", "Generated"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"epoch_{epoch:03d}.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a pix2pix cGAN on CMP Facades.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    download_dataset()

    train_dataset = FacadesDataset(DATASET_DIR, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LR,
        betas=(BETA1, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LR,
        betas=(BETA1, 0.999),
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        generator_total = 0.0
        discriminator_total = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        )

        for real_input, real_target in progress:
            real_input = real_input.to(device)
            real_target = real_target.to(device)

            # -------------------------
            # Train discriminator
            # -------------------------
            fake_target = generator(real_input)

            pred_real = discriminator(real_input, real_target)
            pred_fake = discriminator(real_input, fake_target.detach())

            loss_d_real = adversarial_loss(
                pred_real,
                torch.ones_like(pred_real),
            )
            loss_d_fake = adversarial_loss(
                pred_fake,
                torch.zeros_like(pred_fake),
            )
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # -------------------------
            # Train generator
            # -------------------------
            fake_target = generator(real_input)
            pred_fake = discriminator(real_input, fake_target)

            loss_g_gan = adversarial_loss(
                pred_fake,
                torch.ones_like(pred_fake),
            )
            loss_g_l1 = reconstruction_loss(fake_target, real_target)
            loss_g = loss_g_gan + LAMBDA_L1 * loss_g_l1

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            generator_total += loss_g.item()
            discriminator_total += loss_d.item()

            progress.set_postfix(
                G=f"{loss_g.item():.3f}",
                D=f"{loss_d.item():.3f}",
            )

        avg_g = generator_total / len(train_loader)
        avg_d = discriminator_total / len(train_loader)

        print(
            f"Epoch {epoch}: "
            f"Generator Loss={avg_g:.4f}, "
            f"Discriminator Loss={avg_d:.4f}"
        )

        torch.save(
            generator.state_dict(),
            CHECKPOINT_DIR / "generator_latest.pth",
        )
        torch.save(
            discriminator.state_dict(),
            CHECKPOINT_DIR / "discriminator_latest.pth",
        )

        save_sample(real_input, real_target, fake_target, epoch)

    print("\nTraining complete.")
    print("Generator:", CHECKPOINT_DIR / "generator_latest.pth")
    print("Discriminator:", CHECKPOINT_DIR / "discriminator_latest.pth")
    print("Sample outputs:", RESULT_DIR)


if __name__ == "__main__":
    main()
