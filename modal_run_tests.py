from __future__ import annotations

import subprocess

import modal


REPO_REMOTE_PATH = "/root/flash-kmeans"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio",
        "python -m pip install triton numpy tqdm ninja",
    )
    .add_local_dir(
        ".",
        remote_path=REPO_REMOTE_PATH,
        copy=True,
        ignore=[".git", "__pycache__", "*.pyc", ".pytest_cache"],
    )
    .workdir(REPO_REMOTE_PATH)
    .run_commands("python -m pip install -e .")
)

app = modal.App("flash-kmeans-tests")


@app.function(image=image, gpu="L4", timeout=60 * 60)
def run_tests(command: str = "python examples/testapi.py") -> str:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=REPO_REMOTE_PATH,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}\n{completed.stdout}")
    return completed.stdout


@app.local_entrypoint()
def main(command: str = "python examples/testapi.py"):
    print(run_tests.remote(command=command))
