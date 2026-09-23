"""
kaggle_client.py

Thin wrapper around the `kaggle` CLI for:
  1. pushing kplc-kaggle-notebook/ as a new run (push_kernel)
  2. keeping serve.py's config (BACKEND_URL, SESSION_WEBHOOK_SECRET,
     CLOUDFLARE_TUNNEL_TOKEN) in sync via a private Kaggle Dataset instead
     of the Kaggle Secrets UI add-on.

WHY NOT KAGGLE SECRETS (Add-ons -> Secrets)?
Kaggle Secrets are UI-managed and their binding to a kernel is NOT
preserved across pushes made via the CLI/API — each `kaggle kernels push`
can silently drop it, which is why BACKEND_URL kept disappearing. Dataset
*attachments*, by contrast, are declared in kernel-metadata.json itself
(dataset_sources), so they DO survive every CLI push. This module writes
BACKEND_URL / SESSION_WEBHOOK_SECRET / CLOUDFLARE_TUNNEL_TOKEN into a
private dataset on every /session/start call and makes sure the kernel's
metadata references it, so nobody has to open the Kaggle UI again after
the very first run.

Trade-off: a private Kaggle Dataset is cleartext (only visible to the
dataset owner, i.e. you), whereas Kaggle Secrets are encrypted at rest.
For a single-tenant private kernel authenticated with your own API key,
that's a reasonable trade for something that actually works when pushed
by a script.
"""

import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

SECRETS_DATASET_SLUG = "kplc-chatbot-secrets"
SECRETS_DATASET_TITLE = "KPLC chatbot session secrets (private)"
SECRETS_FILE_NAME = "secrets.json"


class KagglePushError(RuntimeError):
    pass


def _ensure_kaggle_credentials(username: str, key: str):
    config_dir = Path.home() / ".kaggle"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "kaggle.json"
    config_path.write_text(json.dumps({"username": username, "key": key}))
    # kaggle CLI refuses to run if this file is group/world readable
    config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def secrets_dataset_ref(kaggle_username: str) -> str:
    return f"{kaggle_username}/{SECRETS_DATASET_SLUG}"


def _dataset_exists(dataset_ref: str) -> bool:
    result = subprocess.run(
        ["kaggle", "datasets", "status", dataset_ref],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def push_secrets_dataset(
    kaggle_username: str,
    kaggle_key: str,
    secrets: dict,
    timeout: int = 120,
) -> str:
    """Creates (first run) or updates (every run after) a PRIVATE Kaggle
    Dataset containing `secrets` as a JSON file. Returns the dataset's
    "username/slug" reference to attach as a dataset_source.

    Called automatically from /session/start — no manual Kaggle UI step.
    """
    if not kaggle_username or not kaggle_key:
        raise KagglePushError("KAGGLE_USERNAME / KAGGLE_KEY are not set.")

    _ensure_kaggle_credentials(kaggle_username, kaggle_key)
    dataset_ref = secrets_dataset_ref(kaggle_username)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / SECRETS_FILE_NAME).write_text(json.dumps(secrets))
        (tmp_path / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": SECRETS_DATASET_TITLE,
                    "id": dataset_ref,
                    "licenses": [{"name": "CC0-1.0"}],
                }
            )
        )

        exists = _dataset_exists(dataset_ref)
        cmd = (
            ["kaggle", "datasets", "version", "-p", str(tmp_path), "-m", "sync session secrets", "-r", "zip"]
            if exists
            else ["kaggle", "datasets", "create", "-p", str(tmp_path), "-r", "zip"]
        )
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        except subprocess.CalledProcessError as e:
            raise KagglePushError(
                f"Failed to push secrets dataset (exit {e.returncode}): {e.stdout}\n{e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise KagglePushError(f"Pushing secrets dataset timed out after {timeout}s") from e

    return dataset_ref


def push_kernel(
    kernel_path: str,
    kaggle_username: str,
    kaggle_key: str,
    extra_dataset_sources: list[str] | None = None,
    timeout: int = 120,
) -> str:
    """Runs `kaggle kernels push -p <kernel_path>`.

    kernel_path must contain kernel-metadata.json and the code file it
    references (serve.py). If extra_dataset_sources is given, those
    dataset refs are merged into kernel-metadata.json's dataset_sources
    (via a temp copy, so the repo checkout itself isn't mutated) before
    pushing — this is how the secrets dataset gets attached automatically
    on every run.

    Raises KagglePushError on any failure. Returns the CLI's stdout on
    success.

    NOTE: a successful push means Kaggle *accepted* the run and queued it —
    it does NOT mean the model is loaded or serving yet. That confirmation
    comes later via serve.py's POST to /session/ready.
    """
    if not kaggle_username or not kaggle_key:
        raise KagglePushError("KAGGLE_USERNAME / KAGGLE_KEY are not set.")

    if not Path(kernel_path, "kernel-metadata.json").exists():
        raise KagglePushError(f"No kernel-metadata.json found under {kernel_path}")

    _ensure_kaggle_credentials(kaggle_username, kaggle_key)

    push_path = kernel_path
    tmp_dir = None
    if extra_dataset_sources:
        tmp_dir = tempfile.mkdtemp()
        push_path = str(Path(tmp_dir) / "kernel")
        shutil.copytree(kernel_path, push_path)
        meta_path = Path(push_path, "kernel-metadata.json")
        meta = json.loads(meta_path.read_text())
        sources = set(meta.get("dataset_sources", []))
        sources.update(extra_dataset_sources)
        meta["dataset_sources"] = sorted(sources)
        meta_path.write_text(json.dumps(meta, indent=2))

    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", push_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise KagglePushError(
            f"kaggle kernels push failed (exit {e.returncode}): {e.stdout}\n{e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise KagglePushError(f"kaggle kernels push timed out after {timeout}s") from e
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result.stdout
