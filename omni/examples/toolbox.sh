# 1. 在固定提交上查东西：不受工作区影响
git grep -n 'class OmniScheduler' 921ea2c8 -- sglang_omni/
git show 921ea2c8:sglang_omni/pipeline/stage/input.py | sed -n '44,60p'

# 2. 一个文件 / 一个函数是怎么变成今天这样的
git log --oneline --follow -- sglang_omni/scheduling/simple_scheduler.py | head
git log -S 'batch_wait_when_idle' --oneline -- sglang_omni/

# 3. 不启动服务、不需要 GPU，看一个模型的流水线最终长什么样
sgl-omni config resolve --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --show diff --vocoder.factory.max_batch_size 16

# 4. 打开通信层的追踪日志：每条边第一次选了什么传输方式
SGLANG_OMNI_COMM_TRACE=1 sgl-omni serve ...
