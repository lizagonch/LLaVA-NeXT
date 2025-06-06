import argparse
import time
from io import BytesIO

import torch
from PIL import Image
import requests

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None


def load_image(image_path: str) -> Image.Image:
    """Load an image from a local path or URL."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image


def build_prompt(question: str, model, roles) -> str:
    """Construct the prompt with image token."""
    if getattr(model.config, "mm_use_im_start_end", False):
        prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{question}"
    else:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt()


ENCODER_NAMES = {
    0: "internvit",
    1: "texify",
    2: "convnext",
    3: "pix2struct",
}


def main(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        "cuda:0",
    )

    roles = conv_templates["llava_v1"].roles
    image = load_image(args.image)
    image_tensors = process_images([image], image_processor, model.config)
    image_tensors = image_tensors.to(dtype=torch.float16, device="cuda", non_blocking=True)

    # Warm up
    prompt = build_prompt(args.question, model, roles)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    with torch.inference_mode():
        model.generate(input_ids, images=image_tensors, image_sizes=[image.size], max_new_tokens=1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    torch.cuda.synchronize()
    end = time.time()

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    inference_time = end - start

    flops = None
    if FlopCountAnalysis is not None:
        try:
            flop_counter = FlopCountAnalysis(
                model,
                (
                    input_ids,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    image_tensors,
                    [image.size],
                ),
            )
            flops = flop_counter.total()
        except Exception:
            flops = None

    answer = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print("Answer:", answer)
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Peak memory: {peak_memory:.2f} MB")
    if flops is not None:
        print(f"FLOPs: {flops:.2f}")
    else:
        print("FLOPs: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile LLaVA-NeXT inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image", required=True, help="Path or URL to the image")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)

