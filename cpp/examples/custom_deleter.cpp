#include <cstdio>
#include <memory>

// C 风格的句柄 API（形状和 cudaStreamCreate / cudaStreamDestroy、ncclCommInitRank / ncclCommDestroy 一样）
struct handle_t {
  int id;
};
handle_t* handle_create(int id) {
  std::printf("create %d\n", id);
  return new handle_t{id};
}
void handle_destroy(handle_t* h) {
  std::printf("destroy %d\n", h->id);
  delete h;
}

struct HandleDeleter {
  void operator()(handle_t* h) const { handle_destroy(h); }
};
using Handle = std::unique_ptr<handle_t, HandleDeleter>;

int main() {
  Handle a(handle_create(1));
  std::unique_ptr<handle_t, void (*)(handle_t*)> b(handle_create(2), handle_destroy);
  std::printf("sizeof(unique_ptr<int>)=%zu\n", sizeof(std::unique_ptr<int>));
  std::printf("sizeof(Handle)=%zu\n", sizeof(Handle));
  std::printf("sizeof(函数指针当删除器)=%zu\n", sizeof(b));
}
