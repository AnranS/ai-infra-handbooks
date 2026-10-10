PY="${PYTHON:-python3}"
EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
if [ "$(uname)" = Darwin ]; then LD_FLAGS="-undefined dynamic_lookup"; else LD_FLAGS=""; fi   # macOS：Python 的符号要到加载时才有，链接器得显式允许未定义符号
"${CXX:-g++}" -O2 -std=c++20 -shared -fPIC $LD_FLAGS $("$PY" -m pybind11 --includes) kvpool_py.cpp -o "kvpool_py$EXT"
"$PY" test_kvpool.py
