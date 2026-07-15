#!/bin/bash
set -e
xcrun -sdk macosx metal -c shader.metal -o shader.air
xcrun -sdk macosx metallib shader.air -o shader.metallib
clang++ -std=c++17 -O2 main.cpp -o raytracer -I./metal-cpp \
  -framework Metal -framework Foundation -framework QuartzCore
echo "built"