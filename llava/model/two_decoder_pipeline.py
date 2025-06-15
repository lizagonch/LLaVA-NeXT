import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)

from .linguistic_refiner import LinguisticRefiner


class TwoDecoderPipeline(nn.Module):
    """Stack a frozen vision-language decoder with a linguistic refiner and a solver decoder."""

    def __init__(self, vlm_model: str, solver_model: str, refiner_layers: int = 2):
        super().__init__()

        model_name = get_model_name_from_path(vlm_model)
        self.vlm_tokenizer, self.vlm, self.vlm_processor, _ = load_pretrained_model(vlm_model, None, model_name)
        for p in self.vlm.parameters():
            p.requires_grad = False
        self.vlm.eval()

        hidden = self.vlm.config.hidden_size
        self.refiner = LinguisticRefiner(hidden, num_layers=refiner_layers)

        self.solver_tokenizer = AutoTokenizer.from_pretrained(solver_model)
        self.solver = AutoModelForCausalLM.from_pretrained(solver_model)

    @torch.no_grad()
    def compress(self, images, prompts, **kwargs):
        if not isinstance(images, list):
            images = [images]
            prompts = [prompts]

        image_tensor = process_images(images, self.vlm_processor, self.vlm.config)
        image_tensor = image_tensor.to(device=self.vlm.device, dtype=self.vlm.dtype)

        input_ids = [tokenizer_image_token(p, self.vlm_tokenizer, return_tensors="pt") for p in prompts]
        pad_id = self.vlm_tokenizer.pad_token_id or self.vlm_tokenizer.eos_token_id or 0
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id).to(self.vlm.device)

        outputs = self.vlm(input_ids=input_ids, images=image_tensor, output_hidden_states=True, **kwargs)
        return outputs.hidden_states[-1]

    def forward(self, images, prompts, **generate_kwargs):
        hidden = self.compress(images, prompts)
        refined = self.refiner(hidden)
        solver_inputs = self.solver_tokenizer("", return_tensors="pt").to(self.solver.device)
        solver_inputs["inputs_embeds"] = refined
        out = self.solver.generate(**solver_inputs, **generate_kwargs)
        return self.solver_tokenizer.decode(out[0], skip_special_tokens=True)
