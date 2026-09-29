// C++ 练习题的测试工具：测试程序按行输出 CASE <名字> PASS | FAIL <说明> | ERROR <说明>，由 practice/cppjudge.py 解析。
// 用法：pj::run("名字", [] { pj::require(条件, "说明"); pj::require_eq(实际, 期望, "说明"); });
#pragma once
#include <cstdio>
#include <exception>
#include <functional>
#include <sstream>
#include <string>

namespace pj {

struct Failure {
  std::string msg;
};

inline void require(bool ok, const std::string& what) {
  if (!ok) throw Failure{what};
}

template <class A, class B>
void require_eq(const A& got, const B& want, const std::string& what) {
  if (!(got == want)) {
    std::ostringstream os;
    os << what << "：期望 " << want << "，实际 " << got;
    throw Failure{os.str()};
  }
}

// 断言 f() 抛出 E 类型的异常
template <class E, class F>
void require_throws(F&& f, const std::string& what) {
  try {
    f();
  } catch (const E&) {
    return;
  } catch (const std::exception& e) {
    throw Failure{what + "：应该抛出指定类型的异常，实际抛出了 " + e.what()};
  }
  throw Failure{what + "：应该抛出异常，但没有抛出"};
}

inline void run(const char* name, const std::function<void()>& f) {
  try {
    f();
    std::printf("CASE %s PASS\n", name);
  } catch (const Failure& e) {
    std::printf("CASE %s FAIL %s\n", name, e.msg.c_str());
  } catch (const std::exception& e) {
    std::printf("CASE %s ERROR 抛出了异常：%s\n", name, e.what());
  }
  std::fflush(stdout);
}

}  // namespace pj
