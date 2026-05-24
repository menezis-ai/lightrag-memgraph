// A11y helpers for modals — focus trap, Esc, initial focus, ARIA wiring.
// Usage:
//   const ref = useRef(null);
//   useModalA11y({ open: true, onClose, ref, titleId: "addsource-title" });
//   return <div role="dialog" aria-modal="true" aria-labelledby="addsource-title" ref={ref}>...</div>

const { useEffect: __useEffect, useRef: __useRef } = React;

window.useModalA11y = function useModalA11y({ open, onClose, ref }) {
  __useEffect(() => {
    if (!open || !ref.current) return;
    const node = ref.current;
    const previouslyFocused = document.activeElement;

    // Initial focus: first focusable, preferring inputs/textareas, falling back to the container
    const focusable = () => Array.from(node.querySelectorAll(
      'input:not([disabled]):not([type=hidden]),textarea:not([disabled]),select:not([disabled]),button:not([disabled]),[href],[tabindex]:not([tabindex="-1"])'
    )).filter(el => !el.hasAttribute("aria-hidden") && el.offsetParent !== null);

    const first = focusable().find(el => ["INPUT","TEXTAREA","SELECT"].includes(el.tagName))
                || focusable()[0]
                || node;
    if (first && typeof first.focus === "function") {
      // Defer to let layout settle (e.g. modal mount animation)
      setTimeout(() => first.focus(), 30);
    }

    const onKey = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose && onClose();
        return;
      }
      if (e.key === "Tab") {
        const items = focusable();
        if (items.length === 0) {
          e.preventDefault();
          return;
        }
        const idx = items.indexOf(document.activeElement);
        if (e.shiftKey) {
          if (idx <= 0) {
            e.preventDefault();
            items[items.length - 1].focus();
          }
        } else {
          if (idx === items.length - 1 || idx === -1) {
            e.preventDefault();
            items[0].focus();
          }
        }
      }
    };

    node.addEventListener("keydown", onKey);
    return () => {
      node.removeEventListener("keydown", onKey);
      if (previouslyFocused && typeof previouslyFocused.focus === "function") {
        try { previouslyFocused.focus(); } catch (_) {}
      }
    };
  }, [open]);
};
