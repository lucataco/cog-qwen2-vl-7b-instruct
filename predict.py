# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/qwen/Qwen2-VL-7B-Instruct/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_CACHE)

    def predict(
        self,
        media: Path = Input(description="Input image or video file"),
        prompt: str = Input(
            description="Custom prompt to guide the description",
            default="Describe this in detail."
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=128,
            ge=1,
            le=512
        )
    ) -> str:
        """Run inference on a single image or video"""
        # Determine if input is video based on extension
        is_video = str(media).lower().endswith(('.mp4', '.avi', '.mov', '.wmv'))
        # Prepare messages format
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video" if is_video else "image",
                    "image" if not is_video else "video": f"file://{media}",
                },
                {"type": "text", "text": prompt},
            ],
        }]
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Generate response
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text