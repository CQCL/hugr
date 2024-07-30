#! /bin/bash

mkdir build

touch build/.nojekyll  # Disable jekyll to keep files starting with underscores
# copy redirect file
cp ./_static/_redirect.html ./build/index.html
poetry run sphinx-build -b html ./api-docs ./build
