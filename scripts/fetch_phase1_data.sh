#!/usr/bin/env bash
# fetch_phase1_data.sh
#
# Download the 5 Phase 1 primary data sources listed in
# docs/design/phase1_data_pipeline.md §1 into a shared raw directory.
#
# Usage:
#   scripts/fetch_phase1_data.sh [OUT_DIR]
#     OUT_DIR defaults to <repo>/data/raw.
#
# Design:
# - Each source is fetched by a separate function; a failure is logged as a
#   warning and other sources continue (fail-tolerant, per Phase 1 design).
# - Snapshot versions (UniDic, Wikipedia dump date) are pinned as shell
#   variables at the top of this file. Update them intentionally.
# - Existing outputs are left untouched — re-running is idempotent.
# - After every attempted fetch a content-addressed manifest is written to
#   $OUT_DIR/manifest.sha256 (deterministic ordering, no timestamps).
#
# The Wikipedia dump is huge (~30 GB); the script fetches it optimistically
# but expects the operator to have sufficient bandwidth and disk. Set
# SKIP_WIKIPEDIA=1 in the environment to skip S4 entirely.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-$REPO/data/raw}"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Pinned snapshot versions. Update deliberately.
# ---------------------------------------------------------------------------
UNIDIC_VERSION="3.1.1"
WIKIPEDIA_DUMP_DATE="20260601"

log()  { printf '[fetch_phase1_data] %s\n' "$*" >&2; }
warn() { printf '[fetch_phase1_data][warn] %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# S1: pyopenjtalk-plus (BSD-3-Clause, ~800K entries)
# ---------------------------------------------------------------------------
fetch_pyopenjtalk_plus() {
    local dst="$OUT_DIR/pyopenjtalk_plus"
    if [ -d "$dst/.git" ]; then
        warn "S1 pyopenjtalk-plus already present at $dst — skipping"
        return 0
    fi
    log "S1: cloning pyopenjtalk-plus"
    git clone --depth 1 https://github.com/tsukumijima/pyopenjtalk-plus "$dst"
}

# ---------------------------------------------------------------------------
# S2: UniDic-cwj (CC-BY-4.0)
# ---------------------------------------------------------------------------
fetch_unidic() {
    local dst="$OUT_DIR/unidic-cwj-$UNIDIC_VERSION"
    if [ -f "$dst/lex.csv" ]; then
        warn "S2 UniDic-cwj lex.csv already present at $dst — skipping"
        return 0
    fi
    log "S2: fetching UniDic-cwj $UNIDIC_VERSION"
    mkdir -p "$dst"
    local url="https://clrd.ninjal.ac.jp/unidic_archive/cwj/${UNIDIC_VERSION}/unidic-cwj-${UNIDIC_VERSION}.zip"
    local zip="$OUT_DIR/unidic-cwj-${UNIDIC_VERSION}.zip"
    curl -fL --retry 3 "$url" -o "$zip"
    unzip -q -o "$zip" -d "$dst"
    rm -f "$zip"
}

# ---------------------------------------------------------------------------
# S3: JMDict (EDRDG)
# ---------------------------------------------------------------------------
fetch_jmdict() {
    local dst="$OUT_DIR/jmdict"
    if [ -f "$dst/JMdict_e.xml" ]; then
        warn "S3 JMDict already present at $dst — skipping"
        return 0
    fi
    log "S3: fetching JMDict"
    mkdir -p "$dst"
    curl -fL --retry 3 http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz \
        -o "$dst/JMdict_e.gz"
    gunzip -f "$dst/JMdict_e.gz"
}

# ---------------------------------------------------------------------------
# S4: Wikipedia JA enterprise HTML dump (CC-BY-SA-4.0 + GFDL, very large)
# ---------------------------------------------------------------------------
fetch_wikipedia() {
    if [ "${SKIP_WIKIPEDIA:-0}" = "1" ]; then
        warn "S4 Wikipedia skipped (SKIP_WIKIPEDIA=1)"
        return 0
    fi
    local dst="$OUT_DIR/wikipedia"
    local archive="jawiki-NS0-${WIKIPEDIA_DUMP_DATE}-ENTERPRISE-HTML.json.tar.gz"
    if [ -d "$dst/extracted" ]; then
        warn "S4 Wikipedia already extracted at $dst — skipping"
        return 0
    fi
    log "S4: fetching Wikipedia HTML dump (${WIKIPEDIA_DUMP_DATE})"
    mkdir -p "$dst"
    local url="https://dumps.wikimedia.org/other/enterprise_html/runs/${WIKIPEDIA_DUMP_DATE}/${archive}"
    curl -fL --retry 3 "$url" -o "$dst/$archive"
    mkdir -p "$dst/extracted"
    tar -xzf "$dst/$archive" -C "$dst/extracted"
}

# ---------------------------------------------------------------------------
# S5: 青空文庫 mirror (作品ごとに license 分岐)
# ---------------------------------------------------------------------------
fetch_aozora() {
    local dst="$OUT_DIR/aozora"
    if [ -d "$dst/.git" ]; then
        warn "S5 Aozora mirror already present at $dst — skipping"
        return 0
    fi
    log "S5: cloning aozorabunko mirror (large!)"
    git clone --depth 1 https://github.com/aozorabunko/aozorabunko "$dst"
}

# ---------------------------------------------------------------------------
# Driver — each fetcher is fail-tolerant.
# ---------------------------------------------------------------------------
overall_status=0
for fn in fetch_pyopenjtalk_plus fetch_unidic fetch_jmdict fetch_wikipedia fetch_aozora; do
    if $fn; then
        log "$fn: ok"
    else
        rc=$?
        warn "$fn: failed (rc=$rc), continuing"
        overall_status=1
    fi
done

# ---------------------------------------------------------------------------
# Content-addressed manifest — sorted so re-runs on identical bytes are
# byte-identical (no timestamps).
# ---------------------------------------------------------------------------
if command -v shasum >/dev/null 2>&1; then
    HASHER="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
    HASHER="sha256sum"
else
    warn "no shasum/sha256sum available; skipping manifest"
    exit $overall_status
fi

manifest="$OUT_DIR/manifest.sha256"
tmp="$OUT_DIR/manifest.sha256.tmp"
: > "$tmp"
# Directory-sorted, portable, deterministic.
( cd "$OUT_DIR" && \
  find . -type f ! -name 'manifest.sha256*' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 $HASHER ) > "$tmp"
mv "$tmp" "$manifest"
log "manifest written to $manifest"

exit $overall_status
