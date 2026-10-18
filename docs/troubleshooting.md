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
| `LEGALRAG_DATA_DIR` is set, but the index is written to `./data/qdrant` | `Settings.qdrant_path` is its own field and does not follow `data_dir` | Set `LEGALRAG_QDRANT_PATH` as well. Keeping them independent is deliberate (D-001, Task 1), but the two must be moved together. |
