# RL bootcamp — student guide

Dentate is the training module of the bootcamp. It fine-tunes a small model on a verifiable task family, measures
whether it can already produce checkable answers, and runs GRPO only when that measurement says a reward exists. The
pipeline is architecture-agnostic; the demo trains the bundled Spiral reference model (a ~0.1M-parameter looped core)
on CPU through every stage:

```
project (.dentate) → SFT → measured support (pass@k sweep) → support gate → GRPO → frozen evaluation
```

If the support sweep finds no harvestable gap, the gate fails and GRPO does not run. That is a measured result
(post-GRPO accuracy is *absent*, not zero). Reward and evaluation share one semantic verifier; the frozen evaluation
set is content-disjoint from both training pools. Each run ends with a held-out score and the verdict behind it, in
the same result package (`result.dentate`) whichever way you run it.

Three ways to run it, same package, same site:

| | Where it runs | Setup | Best for |
|---|---|---|---|
| **Local** | your laptop, CPU | `pip install "dentate[demo]"` | iterating on projects, keeping data (`~/.dentate`) |
| **Colab** | Google's CPU runtime | one click | no local Python |
| **Hosted** | dentate.cortex.a2olabs.com workers | sign in | sharing results and papers publicly |

## 1. Local (`pip`)

Python 3.11+ on macOS, Linux or Windows.

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "dentate[demo]"                          # torch CPU wheels: pip install torch --index-url https://download.pytorch.org/whl/cpu   first, if you want the small build
dentate demo doctor                                  # torch / transformers / spiral / tokenizer / SPA — exit 1 on any failure
dentate demo init                                    # copies the bundled starter + pinned SmolLM tokenizer to ~/.dentate/demo
dentate demo run                                     # trains the starter → ~/.dentate/experiments/<timestamp>/
dentate serve                                        # http://127.0.0.1:8793 — the site + your local lab
```

`demo run` prints one JSON line per stage and ends with the output directory. It contains:

| file | what |
|---|---|
| `results.json` | `status` (`done` / `gate_failed` / `insufficient_heldout`), `sft`, `support`, `gate`, `grpo`, `frozen_before`, `frozen_after` |
| `metrics.json` | per-step training rows |
| `provenance.json` | project, tokenizer file hashes, seeds, verifier version, frozen tasks |
| `result.dentate` | the package: project + the three JSON documents + `checkpoints/{sft,grpo}.pt` — import it in the lab, share it, reproduce it |
| `runs/` | native Store artifacts (checkpoint lineage, event log) |

Flags: `--project P` (any `.dentate` file), `--out DIR` (must not exist — results are never overwritten),
`--tokenizer DIR`, `--quiet`. Precedence for the tokenizer: `--tokenizer` → `$DENTATE_TOKENIZER_DIR` →
`~/.dentate/demo/tokenizer` → the copy inside the installed package; every candidate is checked against the SHA-256
digests in `dentate/demo/TOKENIZER.json` (SmolLM-135M-Instruct, one pinned Hugging Face revision), so a corrupted or
swapped tokenizer fails loudly. `DENTATE_HOME` moves the whole data root (default `~/.dentate`).

**Edit the project.** `~/.dentate/demo/starter.dentate` is plain JSON:

```json
{"version": 1, "name": "Bootcamp: support before reinforcement",
 "architecture": {"name": "spiral-reason", "width": 64, "embedding": 32, "loops": 2},
 "training": {"kind": "arith", "seed": 42, "sft_steps": 100, "grpo_steps": 20, "batch_size": 2,
              "learning_rate": 0.001, "rl_learning_rate": 0.00001, "samples": 4, "eval_tasks": 8,
              "max_new_tokens": 64, "temperature": 0.8, "kl_beta": 0.04}}
```

Bounds are enforced (`width ∈ {32,64,128}`, `loops 1..8`, `sft_steps 1..500`, `grpo_steps 1..100`, `kind ∈
{arith, modular, count, parity, compare}`, …) — the same schema the hosted site accepts. Good first experiments:
more `sft_steps` (does support appear?), more `loops` at the same width (depth without parameters), `kl_beta` 0 vs
0.1 during GRPO, a different `kind`.

**The site locally.** `dentate serve` runs exactly the hosted application on loopback with an implicit local session:
public pages, the Lab (project editor, experiments launched as real local CPU jobs, publish toggle, downloads,
papers) and the owner research dashboard, all in one process. Data root `--data` (default `~/.dentate`): research
store at its root, lab state under `.bootcamp/`. It binds loopback only by design — for a network-facing site use
`python -m dentate.bootcamp serve` behind OIDC (see the README). `dentate research-serve` is the old dashboard alone.

**Terminal.** `dentate serve` also embeds a terminal dock at the bottom of the lab and research pages (toggle with
the ⌃` / ⌘` shortcut or the strip's arrow; drag its top edge to resize). It is locked until you open the URL the
server prints at start — `open http://127.0.0.1:8793/?token=…` — which carries a per-launch secret (like Jupyter's);
opening it once stores the secret as an HttpOnly cookie and drops it from the address bar. Pass your own with
`--terminal-token` or `DENTATE_TERMINAL_TOKEN`; the server also writes it 0600 to `<data>/.bootcamp/terminal-token`
(the Colab notebook reads it from there). Each tab is a real shell running as you, in the data root, with `dentate`
and `cortex` on `PATH` — the first line tells which of `cortex`, `codex`, `claude`, `omp` are installed and how to
install the missing ones (nothing is installed for you). With a RunPod or Lambda API key in Settings → Integrations
the new-tab dropdown also lists your instances (RunPod pods exposing SSH, active Lambda instances) and opens `ssh` to
them; an instance that runs one of your experiments is preselected and badged. Tabs survive navigation and page
reloads (the server keeps the shell and replays its scrollback) until you close them or stop the server. The dock
exists only in this local mode — never on the hosted site — and every terminal request needs the launch cookie, a
loopback or admitted `--proxy-host` Host, and the same Origin + CSRF checks as every other lab mutation. `--proxy-host`
is for a proxy that authenticates you itself (Colab's port proxy); never a public tunnel, which would expose a shell
as your user, and never a loopback name (refused). Windows needs WSL (`dentate serve` reports the terminal as
unsupported there).

**Downloadable CLI.** No Python? The release attaches self-contained one-folder bundles:
[macOS arm64](https://github.com/Xpitfire/dentate-bootcamp/releases/latest/download/dentate-macos-arm64.zip) ·
[Linux x64](https://github.com/Xpitfire/dentate-bootcamp/releases/latest/download/dentate-linux-x64.zip) ·
[Windows x64](https://github.com/Xpitfire/dentate-bootcamp/releases/latest/download/dentate-windows-x64.zip).
Unzip, then `./dentate/dentate demo doctor` (Windows: `dentate\dentate.exe`). Same commands as above.

## 2. Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xpitfire/dentate-bootcamp/blob/main/dentate_bootcamp.ipynb)

`examples/colab/dentate_bootcamp.ipynb`: install → `dentate demo init` → run the starter with live progress → start
`dentate serve` in the background → open it through Colab's port proxy. The notebook resolves the proxy hostname
first and starts the server with `--proxy-host <that host>`, because the browser cannot reach the VM's
`127.0.0.1:8793` and local mode answers only loopback plus explicitly admitted proxy hosts. Use the "open in a new
tab" link to launch and publish experiments; the inline frame is a read-only preview (Colab embeds it cross-site, which
blocks the session cookie). The CPU runtime is enough; expect a few minutes for the starter. Everything after the install is offline. The notebook also executes
headless (`jupyter nbconvert --execute`) on plain Linux: the Colab calls are guarded, so it doubles as a smoke test.

## 3. Hosted — https://dentate.cortex.a2olabs.com

Browse without an account: landing, **Papers** (built-in and public user papers), **Docs**, **Results** (public
experiments with their `results.json` / `provenance.json` and downloadable `result.dentate`).

Sign in (OIDC) to get a **Lab**: upload or edit a project, launch it on the bootcamp workers, watch progress, download
the package. Every experiment and paper is **private by default**; the publish toggle makes it public (public
experiments are pinned and never expire; private ones are retained 24 h). Papers are Markdown (GFM + KaTeX math) and
may link one of your experiments so the results page and the paper reference each other. Limits: one running job per
person, four retained, ~30 min per job, bounded uploads (48 MiB) — see the README's hosted section.

Operators: workers run `python -m dentate.bootcamp worker --persistent --token-env DENTATE_WORKER_TOKEN`; the
frontdoor takes `DENTATE_WORKER_URLS` (comma list of worker origins, max 64) and the same `DENTATE_WORKER_TOKEN`
(≥ 32 chars). `GET /` on a worker reports `{"busy", "persistent"}`; `POST /release` stops and wipes it.

## Reproducing and comparing

* `result.dentate` is a ZIP with fixed members; `dentate.bootcamp.project.load(path)` validates it and returns the
  project + members without unpickling anything. `checkpoints/sft.pt` / `grpo.pt` are native checkpoints:
  `python -m dentate.cli reason-bench --checkpoint <extracted>.pt --teacher "$DENTATE_TOKENIZER_DIR" --device cpu --n-iters <loops>`
  runs a *new* benchmark (not the frozen bootcamp comparison).
* Imported packages: metrics are historical; a run always cold-starts a new model, never resumes imported weights.
* Keep the tokenizer with the package: `provenance.json` records its file hashes and the pinned revision.

## Troubleshooting

| symptom | fix |
|---|---|
| `dentate demo doctor` → `FAIL torch` | `pip install "dentate[demo]"` (CPU wheel: add `--index-url https://download.pytorch.org/whl/cpu` for torch first) |
| `FAIL spiral-reason` | `pip install spiral-lm` |
| `untrusted tokenizer … sha256 mismatch` | `dentate demo init` rewrites the pinned copy; unset a stale `DENTATE_TOKENIZER_DIR` |
| `output directory already exists` | pass a new `--out`; results are never overwritten |
| `status: insufficient_heldout` | the task kind's finite space is exhausted by the training pools — lower `eval_tasks` or change `kind` |
| `status: gate_failed` | expected for cold small models: raise `sft_steps`, then look at `results.json → support` |
| `dentate serve` refuses `--host 0.0.0.0` | by design (implicit local session); use `python -m dentate.bootcamp serve` for a network-facing site |
| `http://127.0.0.1:8793` does not open from Colab / a remote notebook | the browser is not on the VM: run the notebook's serve cell (Colab port proxy), or start `dentate serve --proxy-host <proxy hostname>` and open the proxied URL; other hosts get `403 Local mode accepts loopback hostnames only` |
