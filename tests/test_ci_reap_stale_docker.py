"""Regression tests for scripts/ci_reap_stale_docker.sh (fake docker + date).

The reaper is a DESTRUCTIVE mechanism whose CI runs are usually no-ops (a
healthy runner has nothing stale), so its dangerous branches would otherwise
ship untested — the PR #422 review caught two real guard bugs that way. These
tests drive the exact shipped script through PATH shims:

- ``docker`` — a stub over flat state files that reproduces the semantics the
  guards depend on: ``--filter name=`` is a SUBSTRING match (that is why the
  script must re-check the prefix itself), ``rm -f`` detaches the container
  from its network, ``network rm`` fails while containers are attached, and a
  network's bare ``{{.Created}}`` renders as Go's unparseable default string
  (pinning the ``.Format`` requirement).
- ``date`` — fixed ``now`` via ``$FAKE_NOW`` and RFC3339 parsing via python,
  so age scenarios are deterministic and the suite runs on macOS (BSD date)
  exactly as on the Linux runners.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from tests._repo_only import require_repo_path

# Drives the shipped CI reaper script, which the export excludes.
require_repo_path("scripts/ci_reap_stale_docker.sh")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ci_reap_stale_docker.sh"

NOW = 1_800_000_000
HOUR = 3600

FAKE_DOCKER = r"""#!/usr/bin/env bash
# Fake docker over flat state files. Container lines:
#   id|name|created|running|network
# Network lines:
#   name|created|attached
set -eu
state="$FAKE_STATE_DIR"
echo "docker $*" >> "$state/calls.log"

filter_value() { printf '%s' "${1#name=}"; }

case "$1 ${2:-}" in
  "ps -a")
    # --filter name=X is a SUBSTRING match, exactly like the real CLI.
    pat="$(filter_value "${4:?}")"
    while IFS='|' read -r id name created running network; do
      case "$name" in *"$pat"*) echo "$id $name" ;; esac
    done < "$state/containers.txt"
    ;;
  "inspect -f")
    fmt="$3"; target="$4"
    while IFS='|' read -r id name created running network; do
      if [ "$id" = "$target" ] || [ "$name" = "$target" ]; then
        case "$fmt" in
          '{{.Created}}') echo "$created"; exit 0 ;;
          '{{.State.Running}}') echo "$running"; exit 0 ;;
        esac
      fi
    done < "$state/containers.txt"
    echo "no such container: $target" >&2; exit 1
    ;;
  "rm -f")
    target="$3"
    found=0
    : > "$state/containers.new"
    while IFS='|' read -r id name created running network; do
      if [ "$id" = "$target" ] || [ "$name" = "$target" ]; then
        found=1
        # Detach from its network, mirroring the real daemon.
        : > "$state/networks.new"
        while IFS='|' read -r nname ncreated nattached; do
          if [ "$nname" = "$network" ]; then
            echo "$nname|$ncreated|$((nattached - 1))" >> "$state/networks.new"
          else
            echo "$nname|$ncreated|$nattached" >> "$state/networks.new"
          fi
        done < "$state/networks.txt"
        mv "$state/networks.new" "$state/networks.txt"
      else
        echo "$id|$name|$created|$running|$network" >> "$state/containers.new"
      fi
    done < "$state/containers.txt"
    mv "$state/containers.new" "$state/containers.txt"
    [ "$found" = 1 ] || { echo "no such container: $target" >&2; exit 1; }
    ;;
  "network ls")
    pat="$(filter_value "${4:?}")"
    while IFS='|' read -r name created attached; do
      case "$name" in *"$pat"*) echo "$name" ;; esac
    done < "$state/networks.txt"
    ;;
  "network inspect")
    fmt="$4"; target="$5"
    while IFS='|' read -r name created attached; do
      if [ "$name" = "$target" ]; then
        case "$fmt" in
          '{{len .Containers}}') echo "$attached"; exit 0 ;;
          '{{.Created.Format "2006-01-02T15:04:05Z07:00"}}') echo "$created"; exit 0 ;;
          '{{.Created}}')
            # Go time.Time default rendering — GNU date cannot parse this.
            # A script regressing to the bare template gets garbage and can
            # never reap a network, which the reap assertions then catch.
            echo "2026-07-29 15:20:16.969681853 +0200 +0200"; exit 0 ;;
        esac
      fi
    done < "$state/networks.txt"
    echo "no such network: $target" >&2; exit 1
    ;;
  "network rm")
    target="$3"
    while IFS='|' read -r name created attached; do
      if [ "$name" = "$target" ] && [ "$attached" != "0" ]; then
        echo "error: network $target has active endpoints" >&2; exit 1
      fi
    done < "$state/networks.txt"
    found=0
    : > "$state/networks.new"
    while IFS='|' read -r name created attached; do
      if [ "$name" = "$target" ]; then found=1; else
        echo "$name|$created|$attached" >> "$state/networks.new"
      fi
    done < "$state/networks.txt"
    mv "$state/networks.new" "$state/networks.txt"
    [ "$found" = 1 ] || { echo "no such network: $target" >&2; exit 1; }
    ;;
  *)
    echo "fake docker: unhandled invocation: $*" >&2; exit 64
    ;;
esac
"""

FAKE_DATE = r"""#!/usr/bin/env bash
# Deterministic date: `date +%s` -> $FAKE_NOW, `date -d TS +%s` -> parsed.
set -eu
if [ "${1:-}" = "+%s" ]; then
  echo "$FAKE_NOW"
  exit 0
fi
if [ "${1:-}" = "-d" ]; then
  python3 - "$2" <<'PY'
import re, sys
from datetime import datetime

ts = sys.argv[1]
# Trim nanosecond fractions to the 6 digits fromisoformat accepts.
ts = re.sub(r"\.(\d{6})\d+", r".\1", ts)
ts = ts.replace("Z", "+00:00")
try:
    print(int(datetime.fromisoformat(ts).timestamp()))
except ValueError:
    sys.exit(1)
PY
  exit $?
fi
echo "fake date: unhandled invocation: $*" >&2
exit 64
"""


def _rfc3339(epoch: int, *, container: bool) -> str:
    """Container .Created = RFC3339 string with Z + nanoseconds (as the real
    daemon returns); network .Created.Format = RFC3339 with numeric offset."""
    from datetime import datetime, timezone

    base = datetime.fromtimestamp(epoch, tz=timezone.utc)
    if container:
        return base.strftime("%Y-%m-%dT%H:%M:%S") + ".450296277Z"
    return base.strftime("%Y-%m-%dT%H:%M:%S") + "+00:00"


class Rig:
    def __init__(self, tmp_path: Path):
        self.state = tmp_path / "state"
        self.state.mkdir()
        (self.state / "calls.log").write_text("")
        self.containers: list[str] = []
        self.networks: list[str] = []
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, body in (("docker", FAKE_DOCKER), ("date", FAKE_DATE)):
            shim = bin_dir / name
            shim.write_text(body)
            shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
        self.bin_dir = bin_dir

    def container(
        self, cid: str, name: str, age: int, *, running: bool | str, network: str = ""
    ) -> None:
        created = _rfc3339(NOW - age, container=True)
        if isinstance(running, bool):
            state = "true" if running else "false"
        else:
            state = running
        self.containers.append(f"{cid}|{name}|{created}|{state}|{network}")

    def network(self, name: str, age: int, *, attached: int) -> None:
        created = _rfc3339(NOW - age, container=False)
        self.networks.append(f"{name}|{created}|{attached}")

    def run(
        self,
        *,
        run_id: str = "999",
        attempt: str = "2",
        env: dict[str, str] | None = None,
    ):
        (self.state / "containers.txt").write_text(
            "".join(line + "\n" for line in self.containers)
        )
        (self.state / "networks.txt").write_text(
            "".join(line + "\n" for line in self.networks)
        )
        full_env = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "FAKE_STATE_DIR": str(self.state),
            "FAKE_NOW": str(NOW),
            "GITHUB_RUN_ID": run_id,
            "GITHUB_RUN_ATTEMPT": attempt,
            **(env or {}),
        }
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env=full_env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def remaining_containers(self) -> list[str]:
        return [
            line.split("|")[1]
            for line in (self.state / "containers.txt").read_text().splitlines()
        ]

    def remaining_networks(self) -> list[str]:
        return [
            line.split("|")[0]
            for line in (self.state / "networks.txt").read_text().splitlines()
        ]

    def calls(self) -> str:
        return (self.state / "calls.log").read_text()


@pytest.fixture()
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def test_guards_and_destructive_pass_end_to_end(rig: Rig) -> None:
    """One realistic runner state; every guard exercised in a single pass."""
    # 1. Substring-but-not-prefix: real docker's --filter WILL surface it;
    #    only the script's own prefix re-check protects it.
    rig.container("c1", "prod-twin-ci-cache", age=9 * HOUR, running=False)
    # 2. Current attempt (999/2) protected even when old; previous attempt
    #    (999/1) of the SAME run is eligible.
    rig.container(
        "c2",
        "twin-ci-999-2-integration-a",
        age=2 * HOUR,
        running=False,
        network="twin-ci-999-2-net",
    )
    rig.container("c3", "twin-ci-memgraph-999-1-a", age=2 * HOUR, running=False)
    # 3. Foreign run under the stopped cutoff: protected by age.
    rig.container("c4", "twin-ci-777-1-fresh", age=10 * 60, running=False)
    # 4. Two-tier running bound: 2h-old RUNNING container of a foreign run is
    #    protected (a slow job may still own it — act_runner may ignore
    #    timeout-minutes); a 7h-old RUNNING one is a dead run's leak.
    rig.container(
        "c5",
        "twin-ci-666-1-slowmg",
        age=2 * HOUR,
        running=True,
        network="twin-ci-666-1-net",
    )
    rig.container(
        "c6",
        "twin-ci-555-1-deadmg",
        age=7 * HOUR,
        running=True,
        network="twin-ci-555-1-net",
    )
    # 5. Stale stopped container pinning its network: both must go, container
    #    first (the network only empties once the container is removed).
    rig.container(
        "c7",
        "twin-ci-444-1-mg",
        age=2 * HOUR,
        running=False,
        network="twin-ci-444-1-net",
    )
    # 7. Unreadable state must fail SAFE: never classified into the short
    #    tier, even when ancient — same posture as an unparseable timestamp.
    rig.container("c8", "twin-ci-888-1-unknownstate", age=9 * HOUR, running="unknown")

    rig.network("twin-ci-999-2-net", age=2 * HOUR, attached=1)
    rig.network("twin-ci-666-1-net", age=2 * HOUR, attached=1)
    rig.network("twin-ci-555-1-net", age=7 * HOUR, attached=1)
    rig.network("twin-ci-444-1-net", age=2 * HOUR, attached=1)
    # 6. Old, already-empty foreign network: reaped. Same but younger: kept.
    rig.network("twin-ci-333-1-net", age=2 * HOUR, attached=0)
    rig.network("twin-ci-222-1-net", age=10 * 60, attached=0)
    # Substring-not-prefix network survives.
    rig.network("prod-twin-ci-net", age=9 * HOUR, attached=0)

    result = rig.run(run_id="999", attempt="2")
    assert result.returncode == 0, result.stderr

    assert sorted(rig.remaining_containers()) == [
        "prod-twin-ci-cache",
        "twin-ci-666-1-slowmg",
        "twin-ci-777-1-fresh",
        "twin-ci-888-1-unknownstate",
        "twin-ci-999-2-integration-a",
    ]
    assert sorted(rig.remaining_networks()) == [
        "prod-twin-ci-net",
        "twin-ci-222-1-net",
        "twin-ci-666-1-net",
        "twin-ci-999-2-net",
    ]

    calls = rig.calls()
    # Containers are reaped before any network removal (the pinned network
    # only becomes empty because its container went first).
    assert calls.index("docker rm -f c7") < calls.index(
        "docker network rm twin-ci-444-1-net"
    )
    # The network age is read through the explicit RFC3339 .Format template —
    # the bare {{.Created}} yields Go's unparseable rendering.
    assert '{{.Created.Format "2006-01-02T15:04:05Z07:00"}}' in calls
    # The 7h dead running memgraph went, and its network with it.
    assert "docker rm -f c6" in calls
    assert "docker network rm twin-ci-555-1-net" in calls


def test_previous_attempt_is_eligible_current_is_not(rig: Rig) -> None:
    rig.container("a1", "twin-ci-422-2-integration-x", age=3 * HOUR, running=False)
    rig.container("a2", "twin-ci-422-1-integration-x", age=3 * HOUR, running=False)
    result = rig.run(run_id="422", attempt="2")
    assert result.returncode == 0, result.stderr
    assert rig.remaining_containers() == ["twin-ci-422-2-integration-x"]


@pytest.mark.parametrize(
    ("env", "expected_message"),
    [
        ({"TWIN_CI_REAP_PREFIX": "prod-"}, "unsafe reap prefix"),
        ({"TWIN_CI_REAP_PREFIX": "twin-ci"}, "unsafe reap prefix"),
        ({"TWIN_CI_REAP_AGE_SECONDS": "abc"}, "TWIN_CI_REAP_AGE_SECONDS"),
        ({"TWIN_CI_REAP_AGE_SECONDS": "-5"}, "TWIN_CI_REAP_AGE_SECONDS"),
        # Syntactically valid but unsafe: below the hard floors, or a running
        # bound undercutting the stopped one. The knobs may only RAISE.
        ({"TWIN_CI_REAP_AGE_SECONDS": "0"}, "unsafe reap cutoffs"),
        ({"TWIN_CI_REAP_RUNNING_AGE_SECONDS": "0"}, "unsafe reap cutoffs"),
        ({"TWIN_CI_REAP_AGE_SECONDS": "30000"}, "unsafe reap cutoffs"),
        (
            {"TWIN_CI_REAP_RUNNING_AGE_SECONDS": "6h"},
            "TWIN_CI_REAP_RUNNING_AGE_SECONDS",
        ),
    ],
)
def test_invalid_configuration_is_refused_without_touching_docker(
    rig: Rig, env: dict[str, str], expected_message: str
) -> None:
    rig.container("z1", "twin-ci-111-1-old", age=9 * HOUR, running=False)
    result = rig.run(env=env)
    assert result.returncode == 2
    assert expected_message in result.stderr
    # Refusal must happen before ANY docker invocation.
    assert rig.calls() == ""
    assert rig.remaining_containers() == ["twin-ci-111-1-old"]


def test_empty_cutoff_env_behaves_like_unset(rig: Rig) -> None:
    """`${VAR:-default}` substitutes on empty too: an empty env var means
    "use the default", it must not trip the integer validation."""
    rig.container("z1", "twin-ci-111-1-old", age=9 * HOUR, running=False)
    result = rig.run(env={"TWIN_CI_REAP_AGE_SECONDS": ""})
    assert result.returncode == 0, result.stderr
    # Default cutoff (60 min) applies: the 9h-old stopped container is reaped.
    assert rig.remaining_containers() == []
