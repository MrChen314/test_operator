#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = [
    ("ds_tmem", "test_ds_tmem.py"),
    ("dkv_mma", "test_dkv_mma.py"),
    ("dq_mma", "test_dq_mma.py"),
]


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[run] cwd={cwd} :: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    base_env = os.environ.copy()
    base_env.setdefault("MAX_JOBS", "192")

    for module, test_script in MODULES:
        mod_dir = ROOT / module
        try:
            print("=" * 72)
            print(f"Module: {module}")
            print("=" * 72)
            run_cmd([sys.executable, "setup.py", "build_ext", "--inplace"], mod_dir, base_env)
            run_cmd([sys.executable, test_script], mod_dir, base_env)
        except subprocess.CalledProcessError as exc:
            print(f"[failed] module={module}, returncode={exc.returncode}")
            return exc.returncode

    print("All ds_unit_tests modules passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
