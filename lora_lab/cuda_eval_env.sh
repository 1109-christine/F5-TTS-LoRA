# SOURCE this file in the same shell before starting faster-whisper evaluation.
# Finds installed libraries; does not install or upgrade packages.
_f5_libs="$(python - <<'PY'
import site
import sys
from pathlib import Path
paths = []
for base in site.getsitepackages() + [site.getusersitepackages()]:
    for name in ('cudnn', 'cublas', 'cuda_runtime'):
        path = Path(base) / 'nvidia' / name / 'lib'
        if path.is_dir() and str(path) not in paths:
            paths.append(str(path))
path = Path(sys.prefix) / 'lib'
if path.is_dir():
    paths.append(str(path))
print(':'.join(paths))
PY
)" || return 1
if [[ -n "$_f5_libs" ]]; then
  export LD_LIBRARY_PATH="${_f5_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _f5_libs
