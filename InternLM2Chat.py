from BaseModel import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class InternLM2Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self):
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda().eval()
        print('================ Model loaded ================')

    def chat(self, prompt: str, history: List[dict], meta_instruction: str = '') -> str:
        response, history = self.model.chat(self.tokenizer, prompt, history, temperature=0.1,
                                            meta_instruction=meta_instruction)
        return response, history
