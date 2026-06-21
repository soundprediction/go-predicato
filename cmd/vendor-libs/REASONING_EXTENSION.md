# Bundled reasoning extension

The `reasoning` ladybug extension (`REASON_ENTAILS` / `REASON_DERIVE` / `REASON_CONTRADICTS`,
source in `pensiero/extension/reasoning`) is vendored here alongside `fts`/`vector` and seeded
into lbug's home extension dir by `seedBundledExtension("reasoning", …)` so `LOAD reasoning`
resolves offline. `cmd/extract-vendor-libs.sh` decompresses the per-platform `.gz` into
`lib/extensions/libreasoning.lbug_extension`.

## Compression
All vendored native binaries are gzip-compressed in this dir (the loader/extract step gunzips):
- `liblbug-linux-x86_64.so.gz` (the ladybug shared lib, 28MB → ~9MB)
- `libreasoning-linux-x86_64.lbug_extension.gz` (142KB → ~47KB)
- `lib{fts,vector}-<platform>.lbug_extension.gz`

## Platform coverage (TODO)
Currently vendored for **linux-x86_64 only**. The reasoning extension must be built per platform
against **ladybug v0.17.0** (ABI-gated by the loader) and vendored as:
- `libreasoning-darwin-arm64.lbug_extension.gz`
- `libreasoning-darwin-x86_64.lbug_extension.gz`
- `libreasoning-linux-aarch64.lbug_extension.gz`

Build per platform: clone `github.com/ladybugdb/ladybug` at tag `v0.17.0`, copy
`pensiero/extension/reasoning` into `extension/reasoning`, register it in
`extension/extension_config.cmake` + `extension/CMakeLists.txt`, and
`cmake -B build -G Ninja -DBUILD_EXTENSIONS=reasoning -DLBUG_API_USE_PRECOMPILED_LIB=ON
-DLBUG_API_PRECOMPILED_LIB_PATH=<platform liblbug> -DCMAKE_BUILD_TYPE=Release`, then
`ninja -C build lbug_reasoning_extension` and `gzip` the artifact into this dir with the
matching `<os>-<arch>` name. (The corresponding `liblbug-<platform>.{so,dylib}.gz` come from
the LadybugDB release per `download-liblbug.sh`.)
