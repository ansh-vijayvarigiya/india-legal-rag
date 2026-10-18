# Troubleshooting

Environment problems hit while building this, and what fixes them. None of these are project bugs;
they are all properties of this machine or of a dependency.

| Symptom | Cause | Fix |
|---|---|---|
| Repeated `Windows fatal exception: access violation` with an identical PyTorch stack, while tests still pass | `faulthandler` reporting a handled structured exception inside torch's CPU kernels on a hybrid-core CPU | Run `pytest -p no:faulthandler`. A standalone script needs no flag and exits 0. |
| `UnicodeEncodeError: 'charmap' codec can't encode character` printing Devanagari, an en dash or a curly quote | PowerShell's console code page is not UTF-8 | `$env:PYTHONIOENCODING = "utf-8"`, or write output to a file |
| `ImportError: Modest backend is deprecated` from `selectolax` | `selectolax` 1.x removed its old backend | Import `from selectolax.lexbor import LexborHTMLParser` |
| `bash` fails with a WSL distribution error | `bash` on `PATH` resolves to the WSL launcher, and no distribution is installed | Use Git's shell: `C:\Program Files\Git\bin\bash.exe` |
| `Exception ignored in: QdrantClient.__del__` and `ModuleNotFoundError: msvcrt` at exit | `portalocker` cannot import `msvcrt` while the interpreter is shutting down | Harmless, and does not affect the exit code. Call `ChunkIndex.close()` to release the lock deliberately. |
| `recreate()` leaves old points in the index | Qdrant local mode does not persist a collection deletion | Handled inside `ChunkIndex.recreate`, which drops the storage directory. See `decisions.md` D-015. |
| Embedding runs on CPU although the machine has a GPU | The installed PyTorch is the CPU-only build (`2.14.1+cpu`) | Install PyTorch from the CUDA 12.8 wheel index. CPU is acceptable at this corpus size; see `decisions.md` D-006. |
| `warning: LF will be replaced by CRLF` on commit | Git's `core.autocrlf` on Windows | Cosmetic only |
| `python -m legalrag.inspector` prints `missing ScriptRunContext` warnings and serves nothing | A bare module run has no Streamlit runtime; only `streamlit run` sets one up | Handled inside the module, which starts the server itself when it is not already under one. `streamlit run src/legalrag/inspector.py` also works. |
| `python -m legalrag.inspector` appears to hang: no error, nothing served, the last output is `Email:` | Streamlit prompts for an email on first run unless the server is headless, and with no TTY the prompt waits forever | Handled inside the module, which passes `--server.headless=true` when it starts the server. Writing `~/.streamlit/credentials.toml` with an empty `email` also clears it. |
| `LEGALRAG_DATA_DIR` is set, but the index is written to `./data/qdrant` | `Settings.qdrant_path` is its own field and does not follow `data_dir` | Set `LEGALRAG_QDRANT_PATH` as well. Keeping them independent is deliberate (D-001, Task 1), but the two must be moved together. |
| The Answer tab says no AI key is configured | `LEGALRAG_LLM_API_KEY` is unset, so there is no model to ask | Put `LEGALRAG_LLM_API_KEY=...` in a `.env` file in the project root and restart. Search, evaluation and the Passages tab all work without a key. |
| Asking two or three questions in quick succession returns a rate-limit message | The free plan meters roughly 8,000 tokens a minute, and a request carrying five passages is about a third of that | Wait twenty seconds, as the message says. Nothing is broken. A paid plan or a different provider removes the wait. |
| An answer stops mid-sentence | The reply hit its token cap, because a reasoning model spends part of the cap on hidden reasoning | Already handled: `finish_reason == "length"` raises instead of displaying the fragment. Raise `MAX_REPLY_TOKENS` in `answer.py` if it recurs. |
| A citation like `[1]` in an answer points at a passage that does not use it | Models mark citations differently, including full-width `【1】` and `[1†L2-L4]` | `_normalise_markers` rewrites every form to `[1]` before parsing. If a new form appears, add it there. |
