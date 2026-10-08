cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DKVPOOL_SANITIZE=address,undefined > /dev/null
cmake --build build -j 8 > /dev/null
./build/test_block_pool
ctest --test-dir build -Q && echo "ctest：全部通过"
