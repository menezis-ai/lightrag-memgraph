"""Post-indexation hook registry.

Allows registering async callbacks that fire after each
``LightRAG._insert_done`` completes (i.e., after a document batch
is fully indexed and all storage callbacks have run).
"""

import logging

logger = logging.getLogger("lightrag_memgraph")

_post_index_hooks: list = []


def register_post_index_hook(callback) -> None:
    """Register an async callback to run after each ``_insert_done``.

    Callback signature::

        async def my_hook(lightrag_instance: LightRAG) -> None:
            workspace = lightrag_instance.workspace
            ...

    Hooks run sequentially in registration order.
    Exceptions are logged but do **not** propagate — a failing hook
    must never break the indexation pipeline.
    """
    _post_index_hooks.append(callback)


def clear_post_index_hooks() -> None:
    """Remove all registered hooks (useful in tests)."""
    _post_index_hooks.clear()


def _hook_name(hook) -> str:
    """Return a log label without trusting callback metadata or stringification."""
    try:
        name = getattr(hook, "__name__", None)
        if type(name) is not str or not name:
            name = type(hook).__name__
        return name if type(name) is str else "<unknown>"
    except Exception:
        return "<unknown>"


async def _run_post_index_hooks(lightrag_instance) -> None:
    """Execute every registered hook. Errors are logged, not raised."""
    for hook in _post_index_hooks:
        try:
            await hook(lightrag_instance)
        except Exception:
            logger.exception("Post-index hook %s failed", _hook_name(hook))
