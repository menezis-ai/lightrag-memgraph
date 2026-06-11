#!/usr/bin/env bash
# Build the delivery zip.
#
# Reads .export-ignore at the repo root, runs rsync to a staging
# directory excluding every listed pattern, then zips the result.
# The source tree is untouched.
#
# Usage:
#   scripts/build_export.sh             # → ./dist/twin-export-<sha>.zip
#   OUT=/tmp/twin.zip scripts/build_export.sh   # custom path
#
# The script is intentionally stdlib-only (bash + rsync + zip). No
# Python, no npm — restricted transit cannot assume anything.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

IGNORE_FILE="${REPO_ROOT}/.export-ignore"
if [[ ! -f "${IGNORE_FILE}" ]]; then
    echo "error: ${IGNORE_FILE} not found" >&2
    exit 1
fi

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")"
OUT="${OUT:-${REPO_ROOT}/dist/twin-export-${SHA}.zip}"
STAGING="$(mktemp -d -t twin-export.XXXXXX)"
trap 'rm -rf "${STAGING}"' EXIT

mkdir -p "$(dirname "${OUT}")"

echo ">>> staging tree at ${STAGING}/twin"
rsync -a \
    --exclude-from="${IGNORE_FILE}" \
    "${REPO_ROOT}/" \
    "${STAGING}/twin/"

# Sanity sweep: scream loudly if any obvious internal marker survived.
FORBIDDEN_RE='192\.168\.1\.(49|61|212)|sigilum\.fr|maquette\.sig|twin-real|37\.59\.104\.111|julien\.dabert|jdabert@|\bOVH\b|ovh-twin|/Users/julien|\b[Bb][Nn][Pp]\b|[Bb][Nn][Pp][_.-]|[_.-][Bb][Nn][Pp]\b|[Pp]aribas'
if grep -rEI "${FORBIDDEN_RE}" "${STAGING}/twin/" >/tmp/twin-export-leak.log 2>/dev/null; then
    echo "ERROR: forbidden marker detected in staging tree" >&2
    echo "see /tmp/twin-export-leak.log for the leak list" >&2
    exit 2
fi

# Equivalent sweep for stakeholder names. Word-boundary anchored so
# substrings inside English words don't trip. Names are intentionally
# Twin-team-specific.
NAMES_RE='\b(HORVAT|Salah|Fabrice|Vihn|Anas|Geoffrey|Cassandre|Alberto|Chaki|Yazid|Chafi|Louis|Eric|Fay[cç]al|Manu|Timoth[eé]e?)\b'
if grep -rEI "${NAMES_RE}" "${STAGING}/twin/" >/tmp/twin-export-names.log 2>/dev/null; then
    echo "ERROR: stakeholder name detected in staging tree" >&2
    echo "see /tmp/twin-export-names.log for the name list" >&2
    exit 3
fi

echo ">>> packing ${OUT}"
rm -f "${OUT}"
# -X drops extra Unix attributes that macOS Archive Utility chokes
# on. -r recursive, -q quiet.
(cd "${STAGING}" && zip -qrX "${OUT}" twin)

SIZE="$(du -h "${OUT}" | cut -f1)"
COUNT="$(unzip -l "${OUT}" | tail -1 | awk '{print $2}')"
echo ""
echo "OK  ${OUT}"
echo "    sha=${SHA}  size=${SIZE}  files=${COUNT}"
