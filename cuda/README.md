# CUDA 进阶手册

A Chinese-language MkDocs Material site for learning CUDA to the level needed for AI Infra / LLM inference / GPU kernel jobs.

## Layout

- `docs/`: pages (`basics/`, `kernels/`, `advanced/`, `tools/`, `career/`); code blocks with `title="x.cu"` / `title="x.py"` are complete programs
- `tools/check_code.py`: extracts every titled block, compiles each `.cu` with nvcc 12.9 and 13.4 (flags come from the `nvcc` comment line), runs Triton scripts with `TRITON_INTERPRET=1`, and packs `docs/assets/cuda-examples.tar.gz` with a Makefile
- `tools/check_torch.py`: runs the other Python scripts (the framework chapters' PyTorch examples, the CuTe layout-algebra chapter) and compares their output with the `text title="输出"` block that follows (CPU PyTorch from `../cpp/.venv-py`)
- `tools/emu_run.py` + `tools/emu/include/`: a CUDA-on-CPU emulator (one coroutine per CUDA thread, real barriers for `__syncthreads` and warp primitives) that executes the kernels' self-checks without a GPU
- `.toolkit/`, `.toolkit12/`: nvcc 13.4 / 12.9 assembled from NVIDIA redistributable archives (plus cuBLAS, NCCL, NVTX); `.cutlass/`: CUTLASS headers; `.venv-triton/`: CPU torch + triton

## Commands

```bash
.venv/bin/python tools/check_code.py          # compile everything with both toolkits, run Triton, rebuild the pack
.venv/bin/python tools/emu_run.py             # run the kernels on the CPU emulator
../cpp/.venv-py/bin/python tools/check_torch.py   # run the PyTorch / CuTe-layout scripts, compare outputs
.venv/bin/mkdocs build --strict               # site -> site/
```

`.toolkit*/`, `.cutlass/` and the virtual environments are not in the repository; recreate them locally before running the checks. To build all three handbooks into one site, run `./build.sh` at the repository root.
