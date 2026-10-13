import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        params = json.loads(args["model_config"])["parameters"]
        path = params["model_path"]["string_value"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()

    def execute(self, requests):
        prompts, limits, counts = [], [], []
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy().reshape(-1)
            budget = pb_utils.get_input_tensor_by_name(request, "max_tokens").as_numpy().reshape(-1)
            prompts += [t.decode() for t in texts]
            limits += [int(b) for b in budget]
            counts.append(len(texts))

        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **batch, max_new_tokens=max(limits), do_sample=False, pad_token_id=self.tokenizer.pad_token_id
            )
        generated = output[:, batch["input_ids"].shape[1]:]

        responses, cursor = [], 0
        for count in counts:
            rows = generated[cursor:cursor + count]
            cursor += count
            texts = np.array([self.tokenizer.decode(r, skip_special_tokens=True) for r in rows], dtype=object).reshape(-1, 1)
            lengths = np.array([[int((r != self.tokenizer.pad_token_id).sum())] for r in rows], dtype=np.int32)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("text", texts),
                        pb_utils.Tensor("num_tokens", lengths),
                    ]
                )
            )
        return responses
