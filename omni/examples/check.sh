python3 tools/check_code.py                          # 全部页面，约 10 分钟（第一次会构建 router）
python3 tools/check_code.py docs/journey/stage.md    # 只核对一章
REF=main python3 tools/check_code.py                 # 换到最新的 main 上看哪些地方变了
