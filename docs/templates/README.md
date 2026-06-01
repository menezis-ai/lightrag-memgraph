# Twin Tag Categories — Config-as-Code template

Quick template for any Twin agent owner (Saad / CFT, etc.) who needs
to ship a tag-category taxonomy without touching the Twin source code.

## Files

| File | Purpose |
|---|---|
| `twin-categories.schema.json` | JSON Schema (draft 2020-12) describing the expected shape. Validate your file against it before shipping (`ajv validate`, `jsonschema -i …`). |
| `twin-categories.template.json` | Working example with 6 categories. Copy it, rename, tweak. |

## What is a "category" (a.k.a. "domain") in Twin?

The top-level bucket a tag lives under. In the WebUI sidebar it shows
up as **All domains → Network / Infrastructure / Compliance / …**. Tags
themselves live *inside* a category. Categories are decided by
Knowledge governance; tags are created by stewards in the UI.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | Stable machine id. Lowercase, no spaces (`^[a-z0-9_-]+$`). Tags reference it. **Never rename in place** — removing it without migrating the tags pointing at it leaves them orphan. |
| `label` | `string` | Display name in the UI. 1–80 chars. |
| `color` | `string` | `#RRGGBB` hex. Picks the chip color + sidebar accent. |

## Lifecycle — replace-on-boot

Twin **mirrors** this file on every server reboot. Editing the file +
restarting Twin = publishing a taxonomy change. There is no API to
mutate categories — by design (the doctrine is "categories are
governance, not user-generated"; bypass at your own risk).

```
# 1. edit your category file
$ vim /etc/twin/categories.json

# 2. restart the Twin server (orchestrator does this; locally:)
$ kill -HUP $(pgrep -f lightrag-server) || lightrag-server
```

## How Twin reads it

In the server bootstrap (typically `register()`):

```python
from twindb_lightrag_memgraph import register

register(
    mount_server=True,
    webui_stores="memgraph",
    webui_categories_config="/etc/twin/categories.json",   # ← here
    # …other flags…
)
```

If the file path is unset, Twin falls back to its internal seed
(Oracle / Infrastructure / Network / Payment / Lifecycle / Governance).

## Operational notes

- **Removing a category** leaves any tag that referenced its `id`
  pointing at an orphan id. Twin logs them as `WARN` on next boot but
  never auto-deletes user data. Decide whether to migrate (PATCH the
  tag's `category` to a new id) or accept the orphan state.
- **Adding a category** is free — restart Twin, the new id is
  available immediately for tag creation.
- **Renaming a label or recoloring** is also free; it doesn't break
  tag references because tags pin to `id`, not `label`.
- **Renaming an id** = remove + add = breaks tag references. Don't.
  If you really need to: ship a migration that PATCHes existing tags
  in `WebuiTag_{workspace}` before changing the id in the JSON.
