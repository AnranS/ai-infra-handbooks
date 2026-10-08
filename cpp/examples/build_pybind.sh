PY="${PYTHON:-python3}"
EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O2 -std=c++20 -shared -fPIC $("$PY" -m pybind11 --includes) kvpool_py.cpp -o "kvpool_py$EXT"
"$PY" test_kvpool.py
