import torch

V, T, G, PROMPTS = 16, 6, 8, 32          # 词表大小、生成长度、每个 prompt 采样几条（GRPO 的组）、每步几个 prompt


class Policy(torch.nn.Module):
    """最小的"语言模型"：下一个 token 只看上一个 token"""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, 32)
        self.head = torch.nn.Linear(32, V)

    def forward(self, tok):
        return self.head(torch.tanh(self.emb(tok)))


def reward(prompts, seq):
    """规则奖励：每个"等于上一个 token + 1"的位置得分，满分 1"""
    prev = torch.cat([prompts[:, None], seq[:, :-1]], 1)
    return (seq == (prev + 1) % V).float().mean(1)


class Engine:
    """推理引擎的替身：bf16 权重，只负责采样，并顺手返回每个 token 的 logprob"""

    def __init__(self):
        self.model = Policy().to(torch.bfloat16)

    def load_weights(self, state):
        self.model.load_state_dict(state)                 # 权重同步：fp32 主权重 → bf16 推理权重

    @torch.no_grad()
    def generate(self, prompts, gen):
        tok, logps = prompts[:, None], []
        for _ in range(T):
            lp = self.model(tok[:, -1]).float().log_softmax(-1)
            nxt = torch.multinomial(lp.exp(), 1, generator=gen)
            logps.append(lp.gather(1, nxt))
            tok = torch.cat([tok, nxt], 1)
        return tok[:, 1:], torch.cat(logps, 1)


def token_logp(model, prompts, seq):
    inp = torch.cat([prompts[:, None], seq[:, :-1]], 1)
    return model(inp).log_softmax(-1).gather(2, seq[..., None]).squeeze(-1)


torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
policy, engine = Policy(), Engine()
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
for step in range(1, 41):
    engine.load_weights(policy.state_dict())                              # ① 同步权重
    prompts = torch.randint(0, V, (PROMPTS,), generator=gen).repeat_interleave(G)
    seq, engine_logp = engine.generate(prompts, gen)                      # ② rollout
    r = reward(prompts, seq).view(PROMPTS, G)                             # ③ 打分
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).view(-1, 1)   # ④ 组内相对优势
    with torch.no_grad():
        old_logp = token_logp(policy, prompts, seq)                       # ⑤ 训练端重算 old logprob
    for _ in range(2):                                                    # ⑥ PPO 式的裁剪更新，每批数据用两遍
        ratio = (token_logp(policy, prompts, seq) - old_logp).exp()
        loss = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if step % 8 == 0 or step == 1:
        gap = (engine_logp - old_logp).abs()
        print(f"step {step:>2}：平均奖励 {r.mean():.2f}，推理引擎与训练端的 logprob 差：平均 {gap.mean():.4f}，最大 {gap.max():.4f}")
