"""Load only a trained checkpoint; never serve random initial weights."""

from io import BytesIO
from pathlib import Path
import os
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from helper_lib.model import CLASSES
from helper_lib.data_loader import image_transform
from helper_lib.utils import load_model

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "cifar10.pt"
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class ImageClassifier:
    def __init__(self, checkpoint):
        torch.set_num_threads(int(os.getenv("CNN_NUM_THREADS", "4")))
        self.model = load_model(checkpoint)
        self.transform = image_transform()

    def predict(self, data):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        raise ValueError("Upload a PNG or JPEG image.")
                    if image.width * image.height > 16_000_000:
                        raise ValueError("Image must contain at most 16 million pixels.")
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    tensor = self.transform(image).unsqueeze(0)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError("Invalid or excessively large image.") from exc
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
        index = int(probabilities.argmax())
        return {
            "class": CLASSES[index],
            "class_id": index,
            "confidence": float(probabilities[index]),
            "probabilities": {label: float(probabilities[i]) for i, label in enumerate(CLASSES)},
        }
