import type { FormatCategory } from '../types/format';

export const FORMAT_CATEGORIES: readonly FormatCategory[] = [
  { cat: 'Documents', fmts: 'TXT MD MDX DOCX PDF PPTX XLSX RTF ODT EPUB' },
  { cat: 'Markup', fmts: 'HTML HTM TEX' },
  { cat: 'Data', fmts: 'JSON XML YAML YML CSV LOG CONF INI PROPERTIES SQL' },
  { cat: 'Code', fmts: 'SH BAT C H CPP HPP PY JAVA JS TS SWIFT GO RB PHP CSS SCSS LESS' },
  { cat: 'Images', fmts: 'PNG JPG SVG · coming soon', future: true },
];
