"""导出成 LLaMA 的权重布局：改个名字，现成的推理框架就能加载它"""
import json
import shutil
from pathlib import Path

import torch

from chat import IM_END, IM_START, ROLE, load_tokenizer
from model import GPT, GPTConfig

out = Path("minisanguo")
out.mkdir(exist_ok=True)
sft = torch.load("sft.pt", weights_only=False)
cfg = GPTConfig(**sft["model_config"])
src, dst = sft["model"], {}
head_dim, hidden = cfg.d_model // cfg.n_head, None

for key, w in src.items():
    if key == "embed.weight":
        dst["model.embed_tokens.weight"] = w
    elif key == "head.weight":
        dst["lm_head.weight"] = w                              # 共享词嵌入时 transformers 会自己绑定，写出来也无妨
    elif key == "norm.weight":
        dst["model.norm.weight"] = w
    else:
        i, rest = key.split(".")[1], key.split(".", 2)[2]
        p = f"model.layers.{i}."
        if rest == "norm1.weight":
            dst[p + "input_layernorm.weight"] = w
        elif rest == "norm2.weight":
            dst[p + "post_attention_layernorm.weight"] = w
        elif rest == "attn.qkv.weight":                        # 我们把 q、k、v 合成了一个矩阵，拆回三个
            q, k, v = w.chunk(3, dim=0)
            dst[p + "self_attn.q_proj.weight"], dst[p + "self_attn.k_proj.weight"] = q, k
            dst[p + "self_attn.v_proj.weight"] = v
        elif rest == "attn.proj.weight":
            dst[p + "self_attn.o_proj.weight"] = w
        elif rest == "mlp.gate_up.weight":                     # SwiGLU 的 gate 和 up 也是合在一起的
            gate, up = w.chunk(2, dim=0)
            dst[p + "mlp.gate_proj.weight"], dst[p + "mlp.up_proj.weight"] = gate, up
            hidden = gate.shape[0]
        elif rest == "mlp.down.weight":
            dst[p + "mlp.down_proj.weight"] = w

tok = load_tokenizer()
TEMPLATE = ("{% for m in messages %}{% if m['role'] == 'assistant' %}{{ m['content'] + '" + IM_END + "\\n' }}"
            "{% else %}{{ '" + IM_START + "' + {'system': '系统', 'user': '用户'}[m['role']] + '\\n' + m['content']"
            " + '" + IM_END + "\\n' }}{% if m['role'] == 'user' %}{{ '" + IM_START + "助手\\n' }}{% endif %}"
            "{% endif %}{% endfor %}")
config = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": cfg.d_model,
    "intermediate_size": hidden, "num_hidden_layers": cfg.n_layer, "num_attention_heads": cfg.n_head,
    "num_key_value_heads": cfg.n_head, "head_dim": head_dim, "vocab_size": cfg.vocab_size,
    "max_position_embeddings": cfg.seq_len, "rope_theta": cfg.rope_theta, "hidden_act": "silu",
    "rms_norm_eps": torch.finfo(torch.float32).eps,            # nn.RMSNorm 不传 eps 时就用它
    "tie_word_embeddings": True, "attention_bias": False, "mlp_bias": False, "torch_dtype": "float32",
    "bos_token_id": tok.token_to_id(IM_START), "eos_token_id": tok.token_to_id(IM_END),
}
(out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
(out / "generation_config.json").write_text(json.dumps({"eos_token_id": config["eos_token_id"]}, indent=1), encoding="utf-8")
(out / "tokenizer_config.json").write_text(json.dumps({
    "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": IM_START, "eos_token": IM_END, "pad_token": IM_END,
    "unk_token": "<unk>", "model_max_length": cfg.seq_len, "chat_template": TEMPLATE}, ensure_ascii=False, indent=1),
    encoding="utf-8")
shutil.copy("tokenizer_chat.json", out / "tokenizer.json")
torch.save(dst, out / "pytorch_model.bin")

print(f"{len(src)} 个张量 → {len(dst)} 个（qkv 拆成 3 份、gate_up 拆成 2 份）")
for name in list(dst)[:3] + ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight"]:
    print(f"  {name:48s} {tuple(dst[name].shape)}")
print("目录内容：", "、".join(sorted(p.name for p in out.iterdir())),
      f"（共 {sum(p.stat().st_size for p in out.iterdir()) / 1e6:.1f} MB）")

model = GPT(cfg)                                               # 反过来装回我们自己的结构，确认一个张量都没丢
back = {"embed.weight": dst["model.embed_tokens.weight"], "head.weight": dst["lm_head.weight"],
        "norm.weight": dst["model.norm.weight"]}
for i in range(cfg.n_layer):
    p = f"model.layers.{i}."
    back[f"blocks.{i}.norm1.weight"] = dst[p + "input_layernorm.weight"]
    back[f"blocks.{i}.norm2.weight"] = dst[p + "post_attention_layernorm.weight"]
    back[f"blocks.{i}.attn.qkv.weight"] = torch.cat([dst[p + f"self_attn.{n}_proj.weight"] for n in "qkv"])
    back[f"blocks.{i}.attn.proj.weight"] = dst[p + "self_attn.o_proj.weight"]
    back[f"blocks.{i}.mlp.gate_up.weight"] = torch.cat([dst[p + f"mlp.{n}_proj.weight"] for n in ("gate", "up")])
    back[f"blocks.{i}.mlp.down.weight"] = dst[p + "mlp.down_proj.weight"]
model.load_state_dict(back)
x = torch.randint(0, cfg.vocab_size, (2, 32))
with torch.no_grad():
    ref = GPT(cfg)
    ref.load_state_dict(src)
    print(f"装回来再前向一遍，最大误差 {(model(x) - ref(x)).abs().max():.2e}")
