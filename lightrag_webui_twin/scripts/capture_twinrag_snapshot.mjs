import { chromium } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

const APP_URL = process.env.TWINRAG_URL || 'http://127.0.0.1:5177/';
const OUT = process.env.TWINRAG_SNAPSHOT_OUT || 'tmp_showcase/twinrag-confluence-snapshot.html';

const screens = [
  { id: 'documents', label: 'Documents', tab: 'Documents' },
  { id: 'tags', label: 'Tags', tab: 'Tags' },
  { id: 'retrieval', label: 'Retrieval', tab: 'Retrieval' },
  { id: 'graph', label: 'Graph', tab: 'Graph' },
  { id: 'activity', label: 'Activity', tab: 'Activity' },
  {
    id: 'settings-api-keys',
    label: 'Settings / API keys',
    tab: 'Settings',
    afterTab: async (page) => {
      await page.getByTestId('settings-rail-api-keys').click();
      await page.getByTestId('settings-api-keys').waitFor({ timeout: 5000 });
    },
  },
];

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

async function inlineSnapshot(page) {
  return page.evaluate(() => {
    const keepProps = [
      'align-items',
      'background',
      'background-color',
      'border',
      'border-bottom',
      'border-left',
      'border-radius',
      'border-right',
      'border-top',
      'box-shadow',
      'box-sizing',
      'color',
      'display',
      'flex',
      'flex-basis',
      'flex-direction',
      'flex-grow',
      'flex-shrink',
      'flex-wrap',
      'font',
      'font-family',
      'font-size',
      'font-weight',
      'gap',
      'grid-template-columns',
      'height',
      'justify-content',
      'left',
      'letter-spacing',
      'line-height',
      'margin',
      'margin-bottom',
      'margin-left',
      'margin-right',
      'margin-top',
      'max-height',
      'max-width',
      'min-height',
      'min-width',
      'opacity',
      'overflow',
      'overflow-x',
      'overflow-y',
      'padding',
      'padding-bottom',
      'padding-left',
      'padding-right',
      'padding-top',
      'position',
      'right',
      'text-align',
      'text-decoration',
      'text-overflow',
      'text-transform',
      'top',
      'transform',
      'vertical-align',
      'white-space',
      'width',
      'z-index',
    ];

    const source = document.querySelector('.app');
    if (!source) throw new Error('TwinRAG app root not found');

    const clone = source.cloneNode(true);
    const sourceNodes = [source, ...source.querySelectorAll('*')];
    const cloneNodes = [clone, ...clone.querySelectorAll('*')];

    for (let i = 0; i < sourceNodes.length; i += 1) {
      const sourceNode = sourceNodes[i];
      const cloneNode = cloneNodes[i];
      if (!(sourceNode instanceof HTMLElement) || !(cloneNode instanceof HTMLElement)) {
        continue;
      }

      const rect = sourceNode.getBoundingClientRect();
      const computed = window.getComputedStyle(sourceNode);
      const style = [];

      for (const prop of keepProps) {
        const value = computed.getPropertyValue(prop);
        if (value) style.push(`${prop}:${value}`);
      }

      if (sourceNode === source) {
        style.push('width:1366px');
        style.push('height:768px');
        style.push('min-height:768px');
        style.push('max-height:768px');
      }

      if (computed.position === 'fixed') {
        style.push(`left:${rect.left}px`);
        style.push(`top:${rect.top}px`);
      }

      cloneNode.setAttribute('style', style.join(';'));
      cloneNode.removeAttribute('class');
      cloneNode.removeAttribute('id');
      cloneNode.removeAttribute('data-testid');
      cloneNode.removeAttribute('aria-live');
      cloneNode.removeAttribute('tabindex');
    }

    for (const node of clone.querySelectorAll('script,style,link,svg symbol')) {
      node.remove();
    }

    for (const input of clone.querySelectorAll('input, textarea')) {
      input.setAttribute('value', input.value || input.getAttribute('placeholder') || '');
      input.setAttribute('readonly', 'readonly');
    }

    for (const button of clone.querySelectorAll('button')) {
      button.setAttribute('type', 'button');
      button.removeAttribute('onclick');
    }

    return clone.outerHTML;
  });
}

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1366, height: 768 },
    deviceScaleFactor: 1,
  });

  await page.goto(APP_URL, { waitUntil: 'networkidle' });
  await page.locator('.app').waitFor({ timeout: 15000 });

  const captures = [];

  for (const screen of screens) {
    await page
      .locator('.tabs .tab')
      .filter({ hasText: new RegExp(`^${screen.tab}$`, 'i') })
      .click();
    await page.waitForTimeout(600);
    if (screen.afterTab) await screen.afterTab(page);
    await page.waitForTimeout(600);
    captures.push({
      ...screen,
      html: await inlineSnapshot(page),
    });
  }

  await browser.close();

  const nav = screens
    .map(
      (screen) =>
        `<a href="#${screen.id}" style="display:inline-block;margin:0 6px 6px 0;padding:7px 10px;border:1px solid #C4CFD9;border-radius:6px;color:#2D5A8E;text-decoration:none;font:600 12px -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;background:#FFFFFF">${escapeHtml(screen.label)}</a>`,
    )
    .join('');

  const body = captures
    .map(
      (capture) => `
<div id="${capture.id}" style="margin:0 0 28px 0;padding:0;background:#FFFFFF;border:1px solid #C4CFD9;border-radius:8px;overflow:hidden">
  <div style="padding:9px 12px;background:#F4F8FC;border-bottom:1px solid #C4CFD9;color:#2C3E50;font:650 13px -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif">${escapeHtml(capture.label)}</div>
  <div style="width:100%;overflow:hidden;background:#FFFFFF">
    <div style="width:1366px;height:768px;transform:scale(0.86);transform-origin:top left;margin-right:-191px;margin-bottom:-108px">
      ${capture.html}
    </div>
  </div>
</div>`,
    )
    .join('\n');

  const html = `
<div style="max-width:1180px;margin:0 auto;background:#FFFFFF;color:#2C3E50;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif">
  <div style="margin:0 0 12px 0;padding:12px 14px;border:1px solid #C4CFD9;border-radius:8px;background:#FAFCFE">
    <div style="font-size:18px;font-weight:650;margin:0 0 4px 0;color:#2C3E50">Twin Graph - tutoriel interface TwinRAG</div>
    <div style="font-size:13px;color:#5A6878;margin:0 0 10px 0">Snapshot HTML statique de l'interface actuelle. Aucun script, aucune image externe, aucun appel backend.</div>
    <div>${nav}</div>
  </div>
  ${body}
</div>`;

  await writeFile(OUT, html.trim(), 'utf8');
  console.log(`Wrote ${OUT}`);
}

await main();
