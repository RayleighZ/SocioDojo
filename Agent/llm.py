import torch
import transformers
from impl.top_k import top_k
from sympy.physics.units import temperature

from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import PartialState

DEBUG = True

class BaseLLMAget:
    def __init__(self, temperature: float, top_p: int):
        self.temperature = temperature
        self.top_p = top_p

    # message: only pure prompt, chatgpt roles should not be put here
    def pipeline(self, message, device):
        raise NotImplementedError('LLMAgent pipeline not implemented')


class Llama27bAgent(BaseLLMAget):
    def __init__(self, temperature: float, top_p: int, model_path: str):
        super().__init__(temperature, top_p)
        model = LlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True)
        distributed_state = PartialState()
        model.to(distributed_state.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, load_in_4bit=True)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            device_map='auto',
        )

    def pipeline(self, message: str, device: str) -> str:
        if DEBUG:
            return {'function_call': 'read'}
        resp = self.pipeline(
            message,
            do_sample=True,
            top_k=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=self.temperature,
            device=device
        )
        return resp