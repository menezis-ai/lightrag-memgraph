"""Retry for Memgraph write/write transaction conflicts.

Memgraph resolves a write/write conflict by aborting one side with
``Cannot resolve conflicting transactions``. In this package that conflict
is **designed in**, not anomalous: every folder-membership writer bumps
``__membership_epoch`` on the DocStatus node specifically so two workers
racing from the same snapshot collide, which is what makes the delete-claim
in ``docstatus_impl.claim_last_membership_delete`` safe. The mechanism
firing is the mechanism working — but the loser of the race still surfaces
as an exception, and before this module only the buffered graph flush
retried it. The other 16 write paths propagated it to the caller, which is
how an unrelated ingestion test could fail on a green commit and pass on
re-run.

Two invariants hold this together. Both are load-bearing; read them before
widening anything here.

**1. Retry ONLY on the conflict predicate — never on transport or timeout
errors.** Memgraph *aborts* the conflicting transaction, so nothing from
the losing attempt committed and re-running it is safe. A lost response is
a different failure: the write may well have committed unobserved, and
re-running a compare-and-set then reads its own effect as someone else's.
``claim_last_membership_delete`` is exactly that shape —
``WHERE n.__delete_claim IS NULL ... SET n.__delete_claim = $claim`` — so a
retry after a committed-but-unseen write finds the claim already set,
returns no record, and reports "claim failed" for a claim that actually
succeeded, stranding the document. The narrow predicate is what keeps that
unreachable. Widening this to ``except Exception`` reintroduces it.

**2. The retried callable must re-acquire its own write slot and session.**
Callers pass a thunk that opens ``acquire_write_slot()`` and
``get_session()`` inside, so the semaphore is released across the backoff
sleep (a scarce write slot must not be held while sleeping) and each
attempt runs on a fresh session rather than one whose transaction was just
aborted.

Detection is a lowercase substring match, not an exception type: Memgraph
surfaces this as a plain message rather than neo4j's typed
``TransientError`` (``grep -rn TransientError src/`` returns nothing).

Whole-operation re-run is safe for every wrapped path because they are all
MERGE/SET or ``DETACH DELETE`` shaped. The one construct that is not
naturally idempotent, ``SET n.__membership_epoch = coalesce(..., 0) + 1``
(7 sites in ``docstatus_impl``), is write-only: nothing in ``src/`` or
``tests/`` ever reads, compares, or orders on that value. It is a conflict
tripwire, so a double-increment from a retry is inert. If a future change
ever gives the epoch a *reader*, this assumption dies with it and the
multi-query paths need real transactions instead.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger("twindb_lightrag_memgraph")

_T = TypeVar("_T")

# Total attempts, not retries: 3 == the original try plus 2 more. Kept at the
# value _buffered_graph has used since the buffered flush was introduced.
MAX_WRITE_ATTEMPTS = 3

# Linear backoff: attempt N sleeps N * this. Deliberately short — the loser of
# a Memgraph write conflict just needs the winner to commit, not a real
# congestion backoff.
BACKOFF_STEP_SECONDS = 0.05


def is_conflicting_transaction(exc: BaseException) -> bool:
    """True for Memgraph's write/write conflict abort.

    Substring match by necessity — see the module docstring on why there is
    no exception type to catch here.
    """
    return "cannot resolve conflicting transactions" in str(exc).lower()


async def with_conflict_retry(
    op_name: str,
    fn: Callable[[], Awaitable[_T]],
    *,
    max_attempts: int = MAX_WRITE_ATTEMPTS,
) -> _T:
    """Run ``fn``, retrying only on a Memgraph transaction conflict.

    ``fn`` must be re-runnable and must acquire its own write slot and
    session (invariants 1 and 2 in the module docstring). Any exception that
    is not a conflict propagates on the first occurrence, and a conflict on
    the final attempt propagates too — this never converts a failure into a
    silent success.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt < max_attempts and is_conflicting_transaction(exc):
                logger.warning(
                    "%s hit a Memgraph transaction conflict "
                    "(attempt %d/%d); retrying",
                    op_name,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(BACKOFF_STEP_SECONDS * attempt)
                continue
            raise
    # Unreachable: the loop either returns or raises on the last attempt.
    raise AssertionError(f"{op_name}: retry loop exited without a result")
