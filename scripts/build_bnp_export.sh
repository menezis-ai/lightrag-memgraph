#!/usr/bin/env bash
# Build the BNP delivery tarball.
#
# Reads .bnp-export-ignore at the repo root, runs rsync to a staging
# directory excluding every listed pattern, then tars the result.
# The source tree is untouched.
#
# Usage:
#   scripts/build_bnp_export.sh             # → ./dist/twin-bnp-export-<sha>.tar.gz
#   OUT=/tmp/twin.tar.gz scripts/build_bnp_export.sh   # custom path
#
# The script is intentionally stdlib-only (bash + rsync + tar). No
# Python, no npm — BNP transit cannot assume anything.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

IGNORE_FILE="${REPO_ROOT}/.bnp-export-ignore"
if [[ ! -f "${IGNORE_FILE}" ]]; then
    echo "error: ${IGNORE_FILE} not found" >&2
    exit 1
fi

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")"
OUT="${OUT:-${REPO_ROOT}/dist/twin-bnp-export-${SHA}.tar.gz}"
STAGING="$(mktemp -d -t twin-bnp-export.XXXXXX)"
trap 'rm -rf "${STAGING}"' EXIT

mkdir -p "$(dirname "${OUT}")"

echo ">>> staging tree at ${STAGING}/twin"
rsync -a \
    --exclude-from="${IGNORE_FILE}" \
    "${REPO_ROOT}/" \
    "${STAGING}/twin/"

# Sanity sweep: scream loudly if any obvious internal marker survived.
# Patterns extracted from project_bnp_export_cleanup memory.
FORBIDDEN_RE='192\.168\.1\.(49|61|212)|sigilum\.fr|maquette\.sig|twin-real|37\.59\.104\.111|julien\.dabert|jdabert@'
if grep -rEI "${FORBIDDEN_RE}" "${STAGING}/twin/" >/tmp/bnp-export-leak.log 2>/dev/null; then
    echo "ERROR: forbidden marker detected in staging tree" >&2
    echo "see /tmp/bnp-export-leak.log for the leak list" >&2
    exit 2
fi

# Equivalent sweep for stakeholder names. Word-boundary anchored so
# substrings inside English words ('Manual', 'Salah'... wait that's a
# name) don't trip. Names are intentionally Twin-team-specific.
NAMES_RE='\b(HORVAT|Salah|Fabrice|Vihn|Anas|Geoffrey|Cassandre|Alberto|Chaki|Yazid|Chafi)\b'
if grep -rEI "${NAMES_RE}" "${STAGING}/twin/" >/tmp/bnp-export-names.log 2>/dev/null; then
    echo "ERROR: stakeholder name detected in staging tree" >&2
    echo "see /tmp/bnp-export-names.log for the name list" >&2
    exit 3
fi

echo ">>> packing ${OUT}"
tar -czf "${OUT}" -C "${STAGING}" twin

SIZE="$(du -h "${OUT}" | cut -f1)"
COUNT="$(tar -tzf "${OUT}" | wc -l | tr -d ' ')"
echo ""
echo "OK  ${OUT}"
echo "    sha=${SHA}  size=${SIZE}  files=${COUNT}"
