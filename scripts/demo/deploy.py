"""Publish only the final inference payload and a free ZeroGPU Space."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi


FINAL_SHA256 = "0f8a3f2572ec2caf4fafbcdd074b5283bd8c610602502763b3c04487ce1f123d"
ROOT = Path(__file__).resolve().parents[2]


def copy_runtime(destination: Path) -> None:
    for directory in ("recurrent_qwen", "dataset"):
        shutil.copytree(ROOT / "scripts" / directory, destination / "scripts" / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "scripts/demo").mkdir()
    for name in ("__init__.py", "__main__.py", "inference.py", "gradio_app.py", "proxy.py"):
        shutil.copy2(ROOT / "scripts/demo" / name, destination / "scripts/demo" / name)
    (destination / "scripts" / "__init__.py").write_text("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--max-steps", type=int, choices=range(1, 257), default=256, metavar="1..256")
    args = parser.parse_args()
    required = ("adapter_model.pt", "recurrent_config.json", "tokenizer.json", "tokenizer_config.json")
    for name in required:
        if not (args.checkpoint / name).is_file():
            parser.error(f"Missing inference file: {args.checkpoint / name}")
    with (args.checkpoint / "adapter_model.pt").open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != FINAL_SHA256:
        parser.error("Weights do not match the frozen final checkpoint")
    api = HfApi()
    account = api.whoami()["name"]
    if any(repo.split("/")[0] != account for repo in (args.model_id, args.space_id)):
        parser.error(f"Both destinations must belong to the authenticated account {account}")
    with TemporaryDirectory(prefix="loopformer-hf-") as temp:
        model_dir, space_dir = Path(temp) / "model", Path(temp) / "space"
        model_dir.mkdir(); space_dir.mkdir()
        for name in (*required, "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt"):
            if (args.checkpoint / name).is_file():
                shutil.copy2(args.checkpoint / name, model_dir / name)
        copy_runtime(model_dir)
        shutil.copy2(ROOT / "demo/requirements-inference.txt", model_dir / "requirements.txt")
        card = (ROOT / "docs/demo_model_card.md").read_text().replace("MODEL_REPO_ID", args.model_id)
        (model_dir / "README.md").write_text(card)
        api.create_repo(args.model_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(repo_id=args.model_id, folder_path=model_dir,
                                   commit_message="Publish frozen LoopFormer pointer checkpoint and inference code")
        copy_runtime(space_dir)
        for name in ("app.py", "requirements.txt"):
            shutil.copy2(ROOT / "demo/hf_space" / name, space_dir / name)
        shutil.copy2(ROOT / "demo/requirements-inference.txt", space_dir / "requirements-inference.txt")
        (space_dir / "README.md").write_text(
            "---\ntitle: LoopFormer\nemoji: 🔁\ncolorFrom: red\ncolorTo: yellow\nsdk: gradio\n"
            "sdk_version: 6.30.0\npython_version: '3.12'\napp_file: app.py\npinned: false\n---\n"
            "Live pointer execution with streamed intermediate predictions.\n")
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", private=False,
                        space_hardware="zero-a10g", exist_ok=True)
        # Never select a billable fallback if ZeroGPU is unavailable.
        api.request_space_hardware(args.space_id, hardware="zero-a10g")
        for key, value in {"LOOPFORMER_MODEL_ID": args.model_id,
                           "LOOPFORMER_MODEL_REVISION": commit.oid,
                           "LOOPFORMER_MAX_STEPS": str(args.max_steps)}.items():
            api.add_space_variable(args.space_id, key=key, value=value)
        api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=space_dir,
                          commit_message="Deploy native streaming pointer demo")
        print(json.dumps({"model": args.model_id, "revision": commit.oid,
                          "space": args.space_id, "maxSteps": args.max_steps}, indent=2))


if __name__ == "__main__":
    main()
