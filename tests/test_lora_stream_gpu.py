"""Layer-streamed LoRA training on a tiny model with Qwen3.5's exact architecture (hybrid Gated
DeltaNet + attention, vision tower): same gradients as normal training, learns, resumes, merges."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from transformers import AutoModelForImageTextToText
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    config = Qwen3_5Config(
        text_config=dict(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=64, vocab_size=512, linear_num_value_heads=2,
                         linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
                         layer_types=LAYER_TYPES, pad_token_id=0),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128))
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    # A random lm_head at the usual init (std 0.02) caps every logit near 2.6 after the final norm,
    # so even normal training can't get below ~3.7 loss. A trained model's head isn't that timid.
    with torch.no_grad():
        model.lm_head.weight.normal_(0, 0.3)
    path = tmp_path_factory.mktemp("tiny_qwen35")
    model.save_pretrained(path, safe_serialization=True)
    return path


def examples(n=6, seed=0, pattern=(7, 8, 9, 10)):
    """Prompts of random tokens; the answer is always the same short pattern (learnable)."""
    from src.learning.lora_stream_trainer import Tokenized

    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        prompt = torch.randint(20, 500, (12 + i,), generator=g).tolist()
        out.append(Tokenized(prompt + list(pattern), [-100] * len(prompt) + list(pattern)))
    return out


def streamed(tiny_dir, **kwargs):
    from src.learning.lora_stream_trainer import LoraSettings, StreamedLoRAModel

    return StreamedLoRAModel(tiny_dir, LoraSettings(rank=4, alpha=8), device="cuda", dtype=torch.float32,
                             stream_min_numel=0, **kwargs)


def test_streamed_gradients_match_normal_training(tiny_dir):
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForImageTextToText

    from src.learning.lora_stream_trainer import TARGET_MODULES, collate

    model = streamed(tiny_dir)
    initial = model.lora_state()
    batch = examples(3)
    normaliser = sum(e.targets for e in batch)
    model.loss_and_backward(batch, normaliser)
    got = {n: p.grad.detach().cpu() for n, p in model.peft.named_parameters() if "lora_" in n}
    assert all(p.device.type == "meta" for n, p in model.peft.named_parameters()
               if any(n == name for names in model.streamed.values() for name, _ in names))  # nothing left on the GPU

    # the same loss on a normally loaded model: everything on the GPU, ordinary autograd
    ref = get_peft_model(AutoModelForImageTextToText.from_pretrained(tiny_dir, dtype=torch.float32),
                         LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGET_MODULES,
                                    task_type="CAUSAL_LM")).cuda()
    set_peft_model_state_dict(ref, {k: v.cuda() for k, v in initial.items()})
    ids, mask, labels = collate(batch, 0)
    text = ref.base_model.model.model.language_model
    hidden = text(inputs_embeds=text.embed_tokens(ids.cuda()), attention_mask=mask.cuda()).last_hidden_state
    shifted = labels[:, 1:].cuda()
    keep = shifted != -100
    logits = ref.base_model.model.lm_head(hidden[:, :-1][keep])
    (F.cross_entropy(logits.float(), shifted[keep], reduction="sum") / normaliser).backward()
    want = {n: p.grad.detach().cpu() for n, p in ref.named_parameters() if "lora_" in n}

    # 3 Gated DeltaNet layers x 3 targets + 1 attention layer x 4 + 4 MLPs x 3 = 25 modules, A and B each
    assert got.keys() == want.keys() and len(got) == 50
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol=1e-4, atol=1e-6, msg=name)
    assert any(float(g.abs().sum()) > 0 for g in got.values())


def test_training_learns_stops_early_and_saves(tiny_dir, tmp_path):
    from src.learning.lora_stream_trainer import TrainSettings, train_lora

    model = streamed(tiny_dir)
    train, validation = examples(16, seed=1), examples(4, seed=2)
    before = model.evaluate(validation, 4096)
    result = train_lora(model, train, validation, TrainSettings(learning_rate=2e-2, epochs=12, examples_per_step=4,
                                                                patience=4), tmp_path / "run")
    after = model.evaluate(validation, 4096)
    assert after < before * 0.5, (before, after)       # it learned the answer pattern
    assert result.best_validation_loss == pytest.approx(min(result.validation_loss))
    assert not (tmp_path / "run" / "checkpoint.pt").exists()

    adapter = model.save_adapter(tmp_path / "adapter")
    reloaded = streamed(tiny_dir, init_adapter=adapter)
    assert reloaded.evaluate(validation, 4096) == pytest.approx(after, rel=1e-5)


def test_an_interrupted_run_resumes(tiny_dir, tmp_path):
    from src.learning.lora_stream_trainer import TrainSettings, train_lora

    settings = TrainSettings(learning_rate=5e-3, epochs=2, examples_per_step=4, checkpoint_every=2)
    train = examples(16, seed=3)
    calls = {"n": 0}

    def stop_after_three():
        calls["n"] += 1
        return calls["n"] == 3

    with pytest.raises(InterruptedError):
        train_lora(streamed(tiny_dir), train, [], settings, tmp_path / "run", should_stop=stop_after_three)
    assert (tmp_path / "run" / "checkpoint.pt").exists()
    result = train_lora(streamed(tiny_dir), train, [], settings, tmp_path / "run")
    assert result.resumed and result.steps == 8 and len(result.train_loss) == 8


def test_merged_weights_equal_the_adapter_applied(tiny_dir, tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    from src.learning.lora_merge import merge_adapter
    from src.learning.lora_stream_trainer import TrainSettings, train_lora

    model = streamed(tiny_dir)
    train_lora(model, examples(8, seed=4), [], TrainSettings(learning_rate=5e-3, epochs=2, examples_per_step=4),
               tmp_path / "run")
    adapter = model.save_adapter(tmp_path / "adapter")
    merged_dir = merge_adapter(tiny_dir, adapter, tmp_path / "merged")
    assert (merged_dir / "config.json").exists()

    ids = torch.tensor([examples(1, seed=5)[0].input_ids])
    base = AutoModelForImageTextToText.from_pretrained(tiny_dir, dtype=torch.float32)
    with_adapter = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged = AutoModelForImageTextToText.from_pretrained(merged_dir, dtype=torch.float32)
    with torch.no_grad():
        want = with_adapter(input_ids=ids).logits
        got = merged(input_ids=ids).logits
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
