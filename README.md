> **Part of the [KPLC Chatbot System](https://github.com/wycliffearapcheruiyot/kplc-chatbot-system).**
> This repo: Starts/manages the Kaggle notebook that runs the model
> Sibling repos: [kplc-chatbot-web](https://github.com/wycliffearapcheruiyot/kplc-chatbot-web), [kplc-chatbot-admin](https://github.com/wycliffearapcheruiyot/kplc-chatbot-admin), [kplc-chatbot-gateway](https://github.com/wycliffearapcheruiyot/kplc-chatbot-gateway), [kplc-chatbot-dataset-sync](https://github.com/wycliffearapcheruiyot/kplc-chatbot-dataset-sync), [kplc-chatbot-kb-builder](https://github.com/wycliffearapcheruiyot/kplc-chatbot-kb-builder), [kplc-chatbot-db-infra](https://github.com/wycliffearapcheruiyot/kplc-chatbot-db-infra)

# KPLC Chatbot: Render backend

Sits on Render, waiting. Triggered via `POST /session/start`, it pushes
`kplc-kaggle-notebook/` (bundled in this same repo) to Kaggle, which spins
up a GPU, loads the model, and calls back over a Cloudflare Tunnel. This
backend then proxies `/generate` and `/embed` to that tunnel until Kaggle's
own idle-watchdog shuts the session down.

**Fully automated config handoff — no Kaggle UI steps per deploy.** Every
`/session/start` call also pushes `BACKEND_URL` / `SESSION_WEBHOOK_SECRET`
/ `CLOUDFLARE_TUNNEL_TOKEN` into a private Kaggle Dataset and attaches it
to the kernel push. This exists because Kaggle's UI-managed Secrets
(Add-ons -> Secrets) don't reliably stay bound to a kernel across CLI/API
pushes — dataset attachments, declared in `kernel-metadata.json`, do. You
still create your Cloudflare Tunnel once by hand (that's an external
service, not something `kaggle kernels push` touches), but you never have
to open the Kaggle Secrets UI again after your very first manual test.

```
Recruiter clicks "Start demo"
  -> POST /session/start          (this backend)
  -> kaggle kernels push -p kplc-kaggle-notebook/
  -> Kaggle GPU runs serve.py
  -> serve.py POSTs /session/ready (this backend)      <- session usable now
  -> frontend polls GET /session/status until "ready"
  -> frontend calls POST /generate / POST /embed        (proxied to Kaggle)
  -> after 15 idle min, serve.py POSTs /session/ended
```

## Endpoints

| Method | Path              | Auth               | Purpose |
|---|---|---|---|
| POST | `/session/start`  | none (add your own if public) | Push kernel to Kaggle, start a run. Idempotent while starting/ready. |
| POST | `/session/ready`  | `X-Webhook-Secret` | Called by `serve.py`. Marks session ready. |
| POST | `/session/ended`  | `X-Webhook-Secret` | Called by `serve.py`. Marks session ended. |
| GET  | `/session/status` | none | `{state, started_at, ready_at, ended_at, error_detail}` |
| POST | `/generate`       | none (add your own if public) | Proxied to the Kaggle tunnel's `/generate`. 503 if not ready. |
| POST | `/embed`          | none (add your own if public) | Proxied to the Kaggle tunnel's `/embed`. 503 if not ready. |
| GET  | `/health`         | none | Render health check. |

`/session/start` and `/generate`/`/embed` have no auth of their own here —
add an API key check or rate limiting in front if this is public-facing;
the webhook secret only protects the two Kaggle callbacks.

## Deploying on Render

1. Push this whole folder (including `kplc-kaggle-notebook/`) as one repo.
2. New Web Service on Render, pointing at that repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Must stay at 1 worker.** Session state lives in memory in this
     process; multiple workers would each think a session was idle
     independently. If you need more than one worker, move
     `session_manager.py`'s state into Redis or a database row first.
5. Set the environment variables from `.env.example` in Render's dashboard.
6. Do the one-time setup pieces that are genuinely external services:
   the model dataset (`kplc-kaggle-notebook/README.md` step 1) and the
   Cloudflare Tunnel + its token (same doc, step 2). Skip the "Attach
   Kaggle Secrets" step entirely — that's now automatic, handled by
   `push_secrets_dataset()` on every `/session/start`.

## Always-on mode: 8-hour Kaggle restarts

Set `AUTO_KEEP_ALIVE=true` and this backend keeps a Kaggle session running
without anyone clicking "Start demo":

- `supervisor.py` runs as a background thread. When the session state is
  idle / ended / error it pushes a new kernel (5 min back-off after a failure,
  so a quota or 409 problem doesn't hammer Kaggle).
- `serve.py` exits cleanly after `RESTART_EVERY_HOURS` (default 8): it waits up
  to 60s for in-flight requests, stops the tunnel, calls `/session/ended`, and
  the supervisor relaunches within ~30s. The model reloads, so expect a few
  minutes where `/generate` answers 503 ("no model session") once per cycle.
- Idle shutdown is off in this mode, so a quiet night doesn't end the session.
- While `ready`, the supervisor probes the tunnel's `/health`; three misses in a
  row mark the session errored and trigger a relaunch.
- If this Render service restarts while Kaggle keeps running, the supervisor
  finds the live model server through the tunnel and adopts it instead of
  pushing a duplicate. `serve.py`'s `/health` reports `uptime_seconds` for this.
- `GET /session/status` also returns `keep_alive` and `restart_in_seconds`.

**Quota.** Kaggle's free GPU allowance is about 30 hours a week, and an
always-on session needs 168. When the quota runs out, pushes fail and the
supervisor keeps retrying every 5 minutes until it resets. Use
`KEEP_ALIVE_WINDOW_UTC` to confine launches to the hours you actually need.

### Rotating several Kaggle accounts

Set `KAGGLE_ACCOUNTS` (see `.env.example`) to a JSON object of
`username: key` pairs. Launch number `n` (0, 1, 2, ...) runs on account
`(n // SESSIONS_PER_ACCOUNT) % number_of_accounts`, so with the default of 3
each account serves three 8-hour sessions in a row, then the next account takes
over, and after the last one it wraps to the first. `KAGGLE_ACCOUNTS` replaces
`KAGGLE_USERNAME` / `KAGGLE_KEY` when it is set.

- **Where the counter lives.** Nowhere on Render. Each run is told its
  `SESSION_NUMBER` and reports it back through `/health` and its `/session/ready`
  and `/session/ended` webhooks, so a Render service that slept or restarted
  picks the rotation up again. If every one of those is lost, the count restarts
  at the first account.
- **Per-account calls.** Every `kaggle` CLI call gets that account's credentials
  explicitly (the CLI would otherwise let a leftover `KAGGLE_USERNAME` env var or
  `KAGGLE_API_TOKEN` override them), and the kernel id is rewritten to
  `<that account>/kplc-chatbot-serving`. Each account gets its own private
  secrets dataset and kernel, created on its first launch.
- **A broken account is skipped.** After `ACCOUNT_SKIP_AFTER_FAILURES` (3)
  failed launches in a row the rotation moves to the next account. A 409 is not
  counted, because launching on another account while a run is still going
  would start a second GPU session.
- `GET /session/status` shows `rotation` (accounts, launches so far, which
  account is next) but never usernames or keys.

**Every account needs** phone verification with GPU enabled, and access to the
model dataset named in `kernel-metadata.json` (currently owned by the first
account). Make that dataset public, or share it with each account, or the
launch fails on the other accounts.

**Quota arithmetic.** Running 24/7, each account is busy 24 hours out of every
N days (N accounts), i.e. `24 * 7 / N` hours a week. Staying under about 30
hours a week takes at least 6 accounts; 7 gives exactly 24 hours each.

**Terms.** Kaggle's terms require every person to have their own User ID and
don't allow one ID to be shared by several people. Use accounts that belong to
real people who agreed to this, and check kaggle.com/terms before relying on
the rotation.

**Render sleep.** This setup does not keep the Render service awake. On the
free tier it sleeps after 15 idle minutes and the supervisor pauses with it.
The Kaggle run still ends itself at the 8h mark, and its `/session/ended`
webhook (which retries through a cold start) wakes Render; the supervisor then
starts with the app and relaunches Kaggle.

## Troubleshooting: build fails on `pydantic-core` / maturin / Rust

If the build log shows `maturin failed`, `Read-only file system`, or
`Cargo metadata failed`, Render is building with a newer Python than the
pinned dependency versions have prebuilt wheels for, so pip falls back to
compiling `pydantic-core` from Rust source — which fails because Render's
build sandbox has a read-only cargo cache.

Fix: set `PYTHON_VERSION` (see `.env.example`) as an environment variable
in the Render dashboard, e.g. `3.12.7`, and redeploy. `requirements.txt`
already uses `>=` pins so pip picks a version with a prebuilt wheel for
that Python version.

## Local testing

```bash
cp .env.example .env   # fill in real values
pip install -r requirements.txt
uvicorn main:app --reload
```

`POST /session/start` will genuinely try to push to Kaggle if credentials
are set — use a throwaway kernel id in `kplc-kaggle-notebook/kernel-metadata.json`
if you just want to test the plumbing.
