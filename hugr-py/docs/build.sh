#! /bin/bash

mkdir build

touch build/.nojekyll  # Disable jekyll to keep files starting with underscores

sphinx-multiversion ./api-docs ./build
