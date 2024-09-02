#!/bin/sh

# Installs a fixed UV version
#
# This reads the `UV_CACHE_DIR` environment variable if it is set,
# and stores the downloaded dependencies in that directory.

curl -LsSf https://astral.sh/uv/0.3.4/install.sh | sh
