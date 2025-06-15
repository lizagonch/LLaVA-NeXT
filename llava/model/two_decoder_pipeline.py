import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .linguistic_refiner import LinguisticRefiner


class TwoDecoderPipeline(nn.Module):
    """Stack a frozen decoder-only VLM with a linguistic refiner and a solver decoder."""

    def __init__(self, vlm_model: str, solver_model: str, refiner_layers: int = 2):
        super().__init__()
        self.vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_model)
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_model)
        for p in self.vlm.parameters():
            p.requires_grad = False

        hidden = self.vlm.config.hidden_size
        self.refiner = LinguisticRefiner(hidden, num_layers=refiner_layers)

        self.solver_tokenizer = AutoTokenizer.from_pretrained(solver_model)
        self.solver = AutoModelForCausalLM.from_pretrained(solver_model)

    @torch.no_grad()
    def compress(self, inputs, **kwargs):
        tokens = self.vlm_tokenizer(inputs, return_tensors="pt").to(self.vlm.device)
        outputs = self.vlm(**tokens, output_hidden_states=True, **kwargs)
        hidden = outputs.hidden_states[-1]
        return hidden

    def forward(self, inputs, **generate_kwargs):
        hidden = self.compress(inputs)
        refined = self.refiner(hidden)
        solver_inputs = self.solver_tokenizer("", return_tensors="pt").to(self.solver.device)
        solver_inputs["inputs_embeds"] = refined
        out = self.solver.generate(**solver_inputs, **generate_kwargs)
        return self.solver_tokenizer.decode(out[0], skip_special_tokens=True)
