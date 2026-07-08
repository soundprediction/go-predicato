# Predicato knowledge-graph gRPC server.
#
# Self-contained: predicato vendors its Ladybug native lib under cmd/vendor-libs/
# (extracted by cmd/extract-vendor-libs.sh at build time), so this image builds
# from the repo alone — no external lib fetch, no sibling checkouts.
#
#   docker build -t predicato .
#   docker run --rm predicato serve-grpc --addr 0.0.0.0:50072 --db-driver ladybug
#
# Published by .github/workflows/docker-publish.yml to
# ghcr.io/soundprediction/predicato.

# ---- build stage ---------------------------------------------------------
FROM golang:1.26-bookworm AS build

WORKDIR /src

# Dependency layer first for caching.
COPY go.mod go.sum ./
RUN go mod download

# Full source.
COPY . .

# Extract the vendored Ladybug native lib (liblbug.so + lbug.h [+ extensions])
# for this platform into cmd/lib-ladybug/. Mirrors `make generate`.
RUN bash cmd/extract-vendor-libs.sh

# Build the CLI with CGO + the system_ladybug tag (mirrors the Makefile's
# build-cli target), pointing the runtime rpath at the image's lib dir.
ENV CGO_ENABLED=1
RUN CGO_CFLAGS="-I$(pwd)/cmd/lib-ladybug" \
    CGO_LDFLAGS="-L$(pwd)/cmd/lib-ladybug -Wl,-rpath,/opt/predicato/lib" \
    go build -tags system_ladybug -o /out/predicato ./cmd/main.go

# Collect the runtime native libs (liblbug.so, its .so.0 SONAME symlink, and any
# bundled ladybug extensions).
RUN mkdir -p /out/lib \
    && cp -a cmd/lib-ladybug/liblbug.so* /out/lib/ \
    && if [ -d cmd/lib-ladybug/extensions ]; then cp -a cmd/lib-ladybug/extensions /out/lib/extensions; fi

# ---- runtime stage -------------------------------------------------------
# trixie (GLIBC 2.40): predicato embeds a go-candle libcandle_binding.so it
# dlopens at startup which needs GLIBC >= 2.39; bookworm (2.36) makes it log a
# "GLIBC_2.39 not found" warning on every start. liblbug.so is the "compat"
# build and runs fine on the newer glibc.
FROM debian:trixie-slim

# liblbug.so is a C++ shared library → needs libstdc++ and libgomp at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/predicato /usr/local/bin/predicato
COPY --from=build /out/lib/ /opt/predicato/lib/

ENV LD_LIBRARY_PATH=/opt/predicato/lib

EXPOSE 50072

ENTRYPOINT ["predicato"]
CMD ["serve-grpc", "--addr", "0.0.0.0:50072", "--db-driver", "ladybug"]
