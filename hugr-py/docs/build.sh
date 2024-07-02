#! /bin/bash

mkdir build

touch build/.nojekyll  # Disable jekyll to keep files starting with underscores

sphinx-build -b html ./api-docs ./build/api-docs
