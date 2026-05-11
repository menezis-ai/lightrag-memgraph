/**
 * Tabler icon SVGs as React components — outline only.
 *
 * Ported from the design proto at Desktop/UI/icons.jsx (originally
 * a window-globals JSX file consumed via Babel standalone). Now a typed
 * ESM React component.
 *
 * Adding a new icon: drop a JSX path/group expression in `paths` keyed by
 * its Tabler name. The same name doubles as a discriminated union for
 * `name` so callers get IntelliSense and a compile error on typos.
 */

import type { ReactNode } from 'react';

const paths: Record<string, ReactNode> = {
  'file-text': (
    <>
      <path d="M14 3v4a1 1 0 0 0 1 1h4" />
      <path d="M17 21H7a2 2 0 0 1 -2 -2V5a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2z" />
      <path d="M9 9h1M9 13h6M9 17h6" />
    </>
  ),
  'brand-confluence': (
    <>
      <path d="M3.5 17.5c2 -3.5 4 -4 6 -1s4 3.5 7 1" />
      <path d="M20.5 6.5c-2 3.5 -4 4 -6 1s-4 -3.5 -7 -1" />
    </>
  ),
  cloud: (
    <path d="M6.657 18C4.085 18 2 15.993 2 13.517c0 -2.475 2.085 -4.482 4.657 -4.482c.529 -1.811 2.114 -3.151 4.06 -3.422c1.945 -.271 3.868 .567 4.925 2.148c1.486 -.62 3.236 -.354 4.434 .674c1.198 1.029 1.616 2.633 1.058 4.065" />
  ),
  link: (
    <>
      <path d="M9 15l6 -6" />
      <path d="M11 6l.463 -.536a5 5 0 0 1 7.071 7.072l-.534 .464" />
      <path d="M13 18l-.397 .534a5.068 5.068 0 0 1 -7.127 0a4.972 4.972 0 0 1 0 -7.071l.524 -.463" />
    </>
  ),
  'cloud-upload': (
    <>
      <path d="M7 18a4.6 4.4 0 0 1 0 -9a5 4.5 0 0 1 11 2h1a3.5 3.5 0 0 1 0 7h-1" />
      <path d="M9 15l3 -3l3 3M12 12v9" />
    </>
  ),
  tags: (
    <>
      <path d="M3 8v4.172a2 2 0 0 0 .586 1.414l5.71 5.71a2.41 2.41 0 0 0 3.408 0l3.592 -3.592a2.41 2.41 0 0 0 0 -3.408l-5.71 -5.71a2 2 0 0 0 -1.414 -.586H5a2 2 0 0 0 -2 2z" />
      <path d="M18 19l1.592 -1.592a4.82 4.82 0 0 0 0 -6.816L13 4" />
      <circle cx="7" cy="9" r="1" />
    </>
  ),
  folder: (
    <path d="M5 4h4l3 3h7a2 2 0 0 1 2 2v8a2 2 0 0 1 -2 2H5a2 2 0 0 1 -2 -2V6a2 2 0 0 1 2 -2" />
  ),
  settings: (
    <>
      <path d="M10.325 4.317c.426 -1.756 2.924 -1.756 3.35 0a1.724 1.724 0 0 0 2.573 1.066c1.543 -.94 3.31 .826 2.37 2.37a1.724 1.724 0 0 0 1.065 2.572c1.756 .426 1.756 2.924 0 3.35a1.724 1.724 0 0 0 -1.066 2.573c.94 1.543 -.826 3.31 -2.37 2.37a1.724 1.724 0 0 0 -2.572 1.065c-.426 1.756 -2.924 1.756 -3.35 0a1.724 1.724 0 0 0 -2.573 -1.066c-1.543 .94 -3.31 -.826 -2.37 -2.37a1.724 1.724 0 0 0 -1.065 -2.572c-1.756 -.426 -1.756 -2.924 0 -3.35a1.724 1.724 0 0 0 1.066 -2.573c-.94 -1.543 .826 -3.31 2.37 -2.37c1 .608 2.296 .07 2.572 -1.065z" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  'info-circle': (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8h.01M11 12h1v4h1" />
    </>
  ),
  lock: (
    <>
      <rect x="5" y="11" width="14" height="10" rx="2" />
      <path d="M12 16v.01M8 11V7a4 4 0 0 1 8 0v4" />
    </>
  ),
  'lock-open': (
    <>
      <rect x="5" y="11" width="14" height="10" rx="2" />
      <path d="M12 16v.01M8 11V7a4 4 0 0 1 8 -2" />
    </>
  ),
  world: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M3.6 9h16.8M3.6 15h16.8M11.5 3a17 17 0 0 0 0 18M12.5 3a17 17 0 0 1 0 18" />
    </>
  ),
  'circle-check': (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M9 12l2 2l4 -4" />
    </>
  ),
  'alert-triangle': (
    <>
      <path d="M12 9v4" />
      <path d="M10.363 3.591l-8.106 13.534a1.914 1.914 0 0 0 1.636 2.871h16.214a1.914 1.914 0 0 0 1.636 -2.87l-8.106 -13.536a1.914 1.914 0 0 0 -3.274 0z" />
      <path d="M12 16h.01" />
    </>
  ),
  'loader-2': <path d="M12 3a9 9 0 1 0 9 9" />,
  bell: (
    <>
      <path d="M10 5a2 2 0 0 1 4 0a7 7 0 0 1 4 6v3a4 4 0 0 0 2 3h-16a4 4 0 0 0 2 -3v-3a7 7 0 0 1 4 -6" />
      <path d="M9 17v1a3 3 0 0 0 6 0v-1" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M3 12h1M12 3v1M20 12h1M12 20v1M5.6 5.6l.7 .7M18.4 5.6l-.7 .7M17.7 17.7l.7 .7M6.3 17.7l-.7 .7" />
    </>
  ),
  moon: (
    <path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 0 0 7.92 12.446a9 9 0 1 1 -8.313 -12.454z" />
  ),
  'external-link': (
    <>
      <path d="M12 6H6a2 2 0 0 0 -2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2 -2v-6" />
      <path d="M11 13l9 -9" />
      <path d="M15 4h5v5" />
    </>
  ),
  send: (
    <>
      <path d="M10 14l11 -11M21 3l-6.5 18a.55 .55 0 0 1 -1 0L10 14L3 10.5a.55 .55 0 0 1 0 -1L21 3" />
    </>
  ),
  refresh: (
    <>
      <path d="M20 11A8.1 8.1 0 0 0 4.5 9M4 5v4h4" />
      <path d="M4 13a8.1 8.1 0 0 0 15.5 2M20 19v-4h-4" />
    </>
  ),
  trash: (
    <>
      <path d="M4 7l16 0" />
      <path d="M10 11l0 6" />
      <path d="M14 11l0 6" />
      <path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2 -2l1 -12" />
      <path d="M9 7v-3a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3" />
    </>
  ),
  activity: <path d="M3 12h4l3 8l4 -16l3 8h4" />,
  plus: <path d="M12 5v14M5 12h14" />,
  x: <path d="M18 6L6 18M6 6l12 12" />,
  search: (
    <>
      <circle cx="10" cy="10" r="7" />
      <path d="M21 21l-6 -6" />
    </>
  ),
  eye: (
    <>
      <circle cx="12" cy="12" r="2" />
      <path d="M22 12c-2.667 4.667 -6 7 -10 7s-7.333 -2.333 -10 -7c2.667 -4.667 6 -7 10 -7s7.333 2.333 10 7" />
    </>
  ),
  'arrow-right': <path d="M5 12h14M13 18l6 -6M13 6l6 6" />,
  'chevron-down': <path d="M6 9l6 6l6 -6" />,
  'chevron-right': <path d="M9 6l6 6l-6 6" />,
  minus: <path d="M5 12h14" />,
  'circle-dot': (
    <>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="3" fill="currentColor" />
    </>
  ),
};

export type IconName = keyof typeof paths;

export interface IconProps {
  name: IconName;
  size?: number;
  color?: string;
  strokeWidth?: number;
  className?: string;
}

export function Icon({
  name,
  size = 16,
  color = 'currentColor',
  strokeWidth = 1.5,
  className = '',
}: IconProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      data-icon={name}
    >
      {paths[name]}
    </svg>
  );
}

/** Document source types displayed in the Documents tab. */
export type SourceType = 'file' | 'confluence' | 'sharepoint' | 'url';

const SOURCE_ICON_MAP: Record<SourceType, IconName> = {
  file: 'file-text',
  confluence: 'brand-confluence',
  sharepoint: 'cloud',
  url: 'link',
};

export interface SourceIconProps {
  type: SourceType | string;
  size?: number;
}

/** Picks the right icon for a document source type, defaulting to file-text. */
export function SourceIcon({ type, size = 15 }: SourceIconProps) {
  const iconName = SOURCE_ICON_MAP[type as SourceType] ?? 'file-text';
  return <Icon name={iconName} size={size} />;
}
