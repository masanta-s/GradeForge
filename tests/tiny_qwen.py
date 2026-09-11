"""A tiny model with Qwen3.5's exact architecture plus a small word-level tokenizer with a chat
template, saved like a HuggingFace repo, so training code can run end to end in seconds."""
import re
from pathlib import Path

import torch

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
VOCAB = 512
SPECIALS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
_WORDS = re.compile(r"\w+|[^\w\s]")


def make_tiny_qwen(path: Path, texts: list[str] = ()) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForImageTextToText, PreTrainedTokenizerFast
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    path = Path(path)
    config = Qwen3_5Config(
        text_config=dict(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=64, vocab_size=VOCAB, linear_num_value_heads=2,
                         linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
                         layer_types=LAYER_TYPES, pad_token_id=0),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128))
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    with torch.no_grad():   # a trained head can be confident; the default init caps logits near 2.6
        model.lm_head.weight.normal_(0, 0.3)
    model.save_pretrained(path, safe_serialization=True)

    words = []
    for text in [*texts, TEMPLATE, "system user assistant quality feedback 0 1 2 3 4 5 6 7 8 9 { } \" : , ."]:
        for word in _WORDS.findall(text):
            if word not in words and word not in SPECIALS:
                words.append(word)
    vocab = {token: i for i, token in enumerate(SPECIALS + words[:VOCAB - len(SPECIALS)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(r"\w+|[^\w\s]", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
                                        eos_token="<|im_end|>", additional_special_tokens=SPECIALS[2:])
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(path)
    return path
