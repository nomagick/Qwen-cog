# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
from transformers import AutoModel, AutoTokenizer

import patch_qwen

MODEL_PATH = './Qwen-14B-Chat'

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = patch_qwen.load_model_on_gpus(MODEL_PATH)
        model = patch_qwen.patch(model)

        self.model = model.eval()

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for completion, in chatml format",
            default='''<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
请使用英文重复这段话："为了使模型生成最优输出，当使用 Qwen 时需要使用特定的输入格式(chatml)，请按照示例格式组织输入。"<|im_end|>
<|im_start|>assistant
''',
        ),
        max_tokens: int = Input(
            description="Max new tokens to generate", default=2048, ge=1, le=8192
        ),
        temperature: float = Input(description="Temperature", default=0.75, ge=0, le=5),
        top_p: float = Input(description="Top_p", default=0.8, ge=0, le=1),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        
        yield from self.model.chat_stream_raw(
            self.tokenizer, prompt, 
            max_new_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p,
            max_window_size=8192
        )
