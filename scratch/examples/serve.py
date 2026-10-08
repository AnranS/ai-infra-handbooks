from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("minisanguo")
model = AutoModelForCausalLM.from_pretrained("minisanguo")
text = tok.apply_chat_template([{"role": "user", "content": "这句话是谁说的：「吾自有计。」"}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=20)[0, ids.shape[1]:], skip_special_tokens=True))

# 或者起一个服务（需要 GPU）：
#   vllm serve ./minisanguo --max-model-len 128
#   python -m sglang.launch_server --model-path ./minisanguo --context-length 128
