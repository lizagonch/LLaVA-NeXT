import json
import os
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from llava.mm_utils import process_images, tokenizer_image_token


class VLMCompressionDataset(Dataset):
    """Simple dataset of image-text pairs for VLM compression."""

    def __init__(self, annotations: str, image_folder: str):
        with open(annotations, "r") as f:
            self.data = json.load(f)
        self.image_folder = image_folder

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:  # type: ignore[override]
        item = self.data[idx]
        image = Image.open(os.path.join(self.image_folder, item["image"])).convert("RGB")
        text = item["text"]
        return image, text


class CompressionCollator:
    def __init__(self, tokenizer, image_processor, model_cfg):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_cfg = model_cfg
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch: List[Tuple[Image.Image, str]]):
        images, texts = zip(*batch)
        image_tensor = process_images(list(images), self.image_processor, self.model_cfg)
        ids = [tokenizer_image_token(t, self.tokenizer, return_tensors="pt") for t in texts]
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {"input_ids": input_ids, "images": image_tensor, "attention_mask": attention_mask}


def build_compression_dataloader(
    annotations: str,
    image_folder: str,
    tokenizer,
    image_processor,
    model_cfg,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = VLMCompressionDataset(annotations, image_folder)
    collator = CompressionCollator(tokenizer, image_processor, model_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collator)
