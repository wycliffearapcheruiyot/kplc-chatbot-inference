# KPLC Chatbot: Kaggle model server

`serve.py` runs **on a Kaggle GPU**. It loads Qwen3-4B + an embedding model,
serves `/generate` and `/embed`, exposes them through a Cloudflare Tunnel,
tells the backend it's ready, and shuts itself down after 15 idle minutes.

This repo is only *stored* on GitHub. It is never deployed to Render or run
anywhere but Kaggle.

## How it gets started

```
Recruiter clicks "Start demo"
  -> Render backend  POST /session/start
  -> backend downloads THIS repo from GitHub (tarball, latest of KAGGLE_KERNEL_REF)
  -> backend runs `kaggle kernels push`   (starts the run on Kaggle)
  -> Kaggle GPU runs serve.py  ...  POSTs /session/ready to the backend
```

Push a change to `serve.py` on GitHub and the **next** session automatically
uses it. No redeploy of the backend needed.

## Files

| File | Purpose |
|---|---|
| `serve.py` | The script that runs on Kaggle each session |
| `kernel-metadata.json` | Kaggle kernel settings: GPU on, internet on, private, which Dataset to attach |

## One-time setup

1. **Model Dataset.** Create it with the `kaggle-dataset-setup` repo. Its slug
   (`wycliffecheruiyot/qwen3-4b`) is already in `dataset_sources` here.
2. **Cloudflare named tunnel.** Zero Trust dashboard -> Networks -> Tunnels ->
   Create (Cloudflared). Copy the **token**. Add a Public Hostname
   (e.g. `model.yourdomain.com`) -> HTTP -> `localhost:8000`. That hostname is
   the backend's `MODEL_TUNNEL_URL`.
3. **First push, by hand,** from a clone of this repo (needs `pip install kaggle`
   and your Kaggle API token):
   ```bash
   kaggle kernels push -p .
   ```
   The run will fail immediately with a missing-secret message. Expected, because
   secrets can only be attached once the kernel exists.
4. **Config is automatic — no Kaggle Secrets step.** The Render backend
   pushes `BACKEND_URL` / `SESSION_WEBHOOK_SECRET` / `CLOUDFLARE_TUNNEL_TOKEN`
   into a private Kaggle Dataset (`<you>/kplc-chatbot-secrets`) and attaches
   it to this kernel on every `/session/start` call — see the backend's
   `kaggle_client.py`. `serve.py` reads them from that mounted dataset file
   first, falling back to the old Kaggle Secrets UI only if that dataset
   isn't found (e.g. testing this kernel standalone). You can still set the
   Kaggle Secrets by hand as that fallback if you want, but it's optional.
5. **Backend env vars on Render:** see the backend repo's `.env.example`.
   Set `SESSION_WEBHOOK_SECRET`, `CLOUDFLARE_TUNNEL_TOKEN`, and
   `MODEL_TUNNEL_URL` there — `BACKEND_URL` is auto-filled from Render's
   own `RENDER_EXTERNAL_URL`.

After that, everything is triggered from the dashboard.

## Notes

- The kernel `id` in `kernel-metadata.json` must belong to the Kaggle account
  whose `KAGGLE_USERNAME` / `KAGGLE_KEY` the backend uses.
- If a session fails to load, `serve.py` reports `/session/ended` so the
  dashboard returns to idle. Check the run's logs on Kaggle for the reason.
- Session cap is ~9–12 h on Kaggle regardless; the idle shutdown normally
  ends sessions long before that.
