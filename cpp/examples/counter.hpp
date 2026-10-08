#pragma once

static int static_counter = 0;   // 内部链接：每个翻译单元一份
inline int inline_counter = 0;   // C++17 inline 变量：整个程序一份
