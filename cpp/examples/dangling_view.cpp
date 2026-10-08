#include <cstdio>
#include <string>
#include <string_view>

std::string load_prompt() { return std::string(64, 'x'); }   // 64 个字符：超过短字符串优化的长度，内容在堆上

int main() {
  std::string_view v = load_prompt();   // 临时的 string 在这一行结束时就析构了
  std::printf("%c\n", v[0]);            // 读已经释放的内存
}
