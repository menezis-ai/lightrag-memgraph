import type { FormatCategory } from '../types/format';

/** Static floor — what every deployment accepts natively. Image support is
 *  deployment-dependent (vision tier): use `buildFormatCategories` with the
 *  runtime config's `extraUploadExtensions` so the tooltip never contradicts
 *  what the modal actually accepts. */
export const FORMAT_CATEGORIES: readonly FormatCategory[] = [
  { cat: 'Documents', fmts: 'TXT MD MDX DOCX PDF PPTX XLSX RTF ODT EPUB' },
  { cat: 'Markup', fmts: 'HTML HTM TEX' },
  { cat: 'Data', fmts: 'JSON XML YAML YML CSV LOG CONF INI PROPERTIES SQL' },
  { cat: 'Code', fmts: 'SH BAT C H CPP HPP PY JAVA JS TS SWIFT GO RB PHP CSS SCSS LESS' },
  { cat: 'Images', fmts: 'PNG JPG SVG · coming soon', future: true },
];

const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg'] as const;

/** Derive the tooltip categories from the backend-advertised extra upload
 *  extensions: when the vision tier serves images, the Images row switches
 *  from "coming soon" to the actually-accepted formats. */
export function buildFormatCategories(
  extraUploadExtensions?: readonly string[],
): readonly FormatCategory[] {
  const extras = new Set(
    (extraUploadExtensions ?? []).map((ext) =>
      ext.toLowerCase().replace(/^\./, ''),
    ),
  );
  const liveImages = IMAGE_EXTENSIONS.filter((ext) => extras.has(ext));
  if (liveImages.length === 0) return FORMAT_CATEGORIES;
  return FORMAT_CATEGORIES.map((category) =>
    category.cat === 'Images'
      ? {
          cat: 'Images',
          fmts: liveImages.map((ext) => ext.toUpperCase()).join(' '),
        }
      : category,
  );
}
