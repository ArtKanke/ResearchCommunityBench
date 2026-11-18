"""Deploy the lab server to Modal (UNTESTED: no Modal account was available overnight).

  modal secret create republic-lab LAB_ADMIN_KEY=... ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
  modal deploy scripts/modal_app.py
  # -> https://<workspace>--republic-lab-serve.modal.run ; use it as LAB_URL and LAB_PUBLIC_URL,
  #    and its hostname as LAB_HOST for the run-stage egress allowlist.

One container only (SQLite on a Modal Volume is single-writer), kept warm for the event.
"""

import os

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "uvicorn", "pydantic", "numpy", "scipy", "scikit-learn", "pandas", "rdkit", "litellm", "httpx")
    .add_local_dir(f"{ROOT}/lab", "/root/lab")
    .add_local_dir(f"{ROOT}/data", "/root/data")
)
vol = modal.Volume.from_name("republic-lab-db", create_if_missing=True)
app = modal.App("republic-lab", image=image)


@app.function(secrets=[modal.Secret.from_name("republic-lab")], volumes={"/db": vol},
              min_containers=1, max_containers=1, timeout=24 * 3600, cpu=8, memory=16384)
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def serve():
    os.environ.setdefault("LAB_DB", "/db/lab.sqlite")
    os.environ.setdefault("LAB_PUBLIC_URL", os.environ.get("LAB_PUBLIC_URL", ""))
    from lab.server import app as fastapi_app

    return fastapi_app
