# Build for DGX Spark from x86 Linux

The cross-build container uses Ubuntu 24.04, GCC 13's AArch64 cross compiler,
CUDA 13.0's x86 compiler, ARM64 SBSA CUDA libraries, and ARM64 libibverbs.
It requires Docker but no local NVIDIA driver, GPU, QEMU, or privileged container.
The output targets Linux/aarch64 and GB10 (`sm_121a`). The target needs compatible
DGX OS drivers and system libraries; Ubuntu 24.04 is the repository's tested ABI.

From the repository root:

```bash
scripts/spark-cross image             # install/build the development image
scripts/spark-cross build             # configure and build dgpp-serve
file build-spark-cross/dgpp-serve
```

The image is named `dgpp-spark-cross:cuda13`; override with `DGPP_CROSS_IMAGE`.
Build artifacts stay in `build-spark-cross/`, owned by your host user. The source
is mounted at its original absolute path so compile_commands.json paths remain
useful. Compilers and headers referenced there live inside the container.
Compilation defaults to two jobs; use `DGPP_BUILD_JOBS=4 scripts/spark-cross build`
if memory permits. Re-run `image` after changing the Dockerfile. Package updates
are not locked; use `docker build --no-cache` with the same Dockerfile to refresh
cached installation layers deliberately.

For an interactive development shell or individual tools:

```bash
scripts/spark-cross shell
scripts/spark-cross command nvcc --version
scripts/spark-cross command cmake --build --preset spark-cross --target all --parallel 2
```

The `spark-cross` CMake preset selects `cmake/spark-cross.cmake`. Python and CMake
run on x86, while C++ code and CUDA host code compile for ARM64. CUDA device code
compiles for GB10. The default build includes the server and its dependencies;
other targets can be built explicitly. Do not run the cross-built CTest suite
on the x86 host: executable and GPU validation belong on the Spark.

To stage the server, ARM64 CUDA shared libraries, and existing deployment files:

```bash
scripts/spark-cross command cmake --install build-spark-cross --prefix "$PWD/install/spark"
```

Transfer that directory to the Spark and follow `docs/release-install.md` for
runtime dependencies and deployment. The existing `scripts/release.sh` performs
native execution and `ldd` checks, so it is not a cross-packaging entry point.
The container does not mount SSH keys or deploy anything automatically.
