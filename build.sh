
rm -rf build/
mkdir build && cd build
uv run cmake -DCMAKE_BUILD_TYPE=Debug ..
uv run cmake --build . --config Release -j$(nproc)
cp mcts_cpp.*.so ../src/python/
cd ..
