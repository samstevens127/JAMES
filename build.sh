# Go to project root
rm -rf build/
mkdir build && cd build
uv run cmake -DCMAKE_BUILD_TYPE=Debug ..
uv run cmake --build . --config Release -j$(nproc)
# Move the NEWLY built file to your python folder
cp mcts_cpp.*.so ../src/python/
cd ..
