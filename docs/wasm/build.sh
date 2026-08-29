#!/bin/sh
# Reproducible build for engine.wasm. The checked-in binary is the
# artifact of exactly this command (treat it like vendored code:
# rebuild whole, never patch). Toolchain: Ubuntu clang 18.1.3
# (--target=wasm32 with the builtin wasm_simd128.h header) and its
# bundled wasm-ld. No fast-math, no relaxed SIMD - determinism
# depends on plain IEEE-754 instructions only; see engine.c.
set -e
cd "$(dirname "$0")"
clang --target=wasm32 -O2 -msimd128 -nostdlib -ffreestanding \
  -Wall -Wextra \
  -Wl,--no-entry \
  -Wl,--export=__heap_base \
  -Wl,-z,stack-size=131072 \
  -o engine.wasm engine.c
ls -l engine.wasm
