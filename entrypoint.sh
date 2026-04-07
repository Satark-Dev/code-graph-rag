#!/bin/sh
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  LIBDIR="/lib/x86_64-linux-gnu" ;;
    aarch64) LIBDIR="/lib/aarch64-linux-gnu" ;;
    *)       LIBDIR="/lib" ;;
esac
export LD_PRELOAD="$LIBDIR/libz.so.1:$LIBDIR/libzstd.so.1"
# exec python -m code-graph-rag "$@"
# exec /app/.venv/bin/python -m codebase_rag.cli "$@"
exec /app/.venv/bin/newrelic-admin run-program /app/.venv/bin/python -m codebase_rag.cli "$@"
