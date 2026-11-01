import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(
    model_id
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)


def generate_answer(
        query,
        context
):

    prompt = f"""
You are a trusted assistant.

Answer ONLY from context.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.2,
        do_sample=True
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )