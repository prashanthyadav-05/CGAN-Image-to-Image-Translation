import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from train import UNetGenerator, IMG_SIZE


def load_input(path):
    image = Image.open(path).convert("RGB")

    # If the user supplies an original pix2pix paired image,
    # use its left half as the input/label map.
    if image.width >= 2 * image.height:
        image = image.crop((0, 0, image.width // 2, image.height))

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return transform(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Translate an image using a trained pix2pix generator.")
    parser.add_argument("--input", required=True, help="Path to an input image.")
    parser.add_argument(
        "--model",
        default="checkpoints/generator_latest.pth",
        help="Path to generator checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="results/translated.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    input_tensor = load_input(args.input).to(device)

    with torch.no_grad():
        generated = model(input_tensor)[0].cpu()

    generated = (generated * 0.5 + 0.5).clamp(0, 1)
    output = generated.permute(1, 2, 0).numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(output_path, output)
    print("Saved translated image to:", output_path)


if __name__ == "__main__":
    main()
