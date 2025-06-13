

import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLMWrapper:
    def __init__(self, model_name="gpt-3.5", api=False):
        self.api = api
        if api:
            openai.api_key = "YOUR_OPENAI_API_KEY"
            self.model_name = model_name
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def query(self, prompt):
        if self.api:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512
            )
            return response["choices"][0]["message"]["content"]
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
