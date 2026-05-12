/**
 * TagChip — small label rendered next to documents/entities.
 *
 * Ported from Desktop/UI/icons.jsx (where TagChip was bundled with Icon).
 * Split out here because TagChip needs its own typing and tag-semantic
 * resolution. Semantics drive the visual variant (critical / warning).
 *
 * The semantics lookup can be injected (default: empty map). In the real
 * app we'll wire a `TagSemanticsContext` or pass it from the parent that
 * already has the loaded vocabulary; this stub keeps Icons.tsx pure.
 */

import { Icon } from './Icon';

export type TagSemantic = 'critical' | 'warning';

export type TagSemanticsMap = Readonly<Record<string, TagSemantic>>;

export interface TagChipProps {
  tag: string;
  removable?: boolean;
  onRemove?: (tag: string) => void;
  /** Pre-resolved semantic, overrides any lookup in `semanticsMap`. */
  semantics?: TagSemantic | null;
  /** Map used when `semantics` is not given. Defaults to {}. */
  semanticsMap?: TagSemanticsMap;
}

export function TagChip({
  tag,
  removable = false,
  onRemove,
  semantics,
  semanticsMap = {},
}: TagChipProps) {
  const sem = semantics ?? semanticsMap[tag] ?? null;
  const cls = sem ? `tag-chip ${sem}` : 'tag-chip';
  const className = removable ? `${cls} removable` : cls;

  return (
    <span className={className}>
      {tag}
      {removable && (
        <button
          type="button"
          className="x"
          onClick={(e) => {
            e.stopPropagation();
            onRemove?.(tag);
          }}
          aria-label={`Remove ${tag}`}
        >
          <Icon name="x" size={10} />
        </button>
      )}
    </span>
  );
}
