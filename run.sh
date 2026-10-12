#!/usr/bin/env bash
set -euo pipefail
package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$package_dir"
export PYTHONPATH="$package_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" train.py "$@"
