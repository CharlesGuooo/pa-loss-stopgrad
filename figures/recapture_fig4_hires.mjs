/**
 * Re-capture the ghost-braking BEV pair for the T-IV manuscript (Fig. 4).
 *
 * Frame: token af67f465f5994ac7bab19825336db644 -- boston-seaport, command
 * "straight", high-conflict, perceived delta-L2(sg-full) = 0.66 m. This is the
 * scene tagged [HC * paper] in the scene list.
 *
 * State: perceived domain, planner layer (2.8), compare on, A=full,
 * B=stopgrad, decoder level 3, timestep at max (11).
 *
 * WHY deviceScaleFactor rather than a wider viewport: a wider viewport reflows
 * the app and changes what the figure actually shows. deviceScaleFactor leaves
 * the CSS layout byte-identical and only doubles pixel density.
 *
 * WHY ?lang=en&figure=1: the paper needs English, and figure mode hides the
 * header, toolbar, score bars and footer so what lands in the PDF is the two
 * BEV panels rather than a screenshot of an interactive tool. The capture is
 * scoped to #panels for the same reason.
 *
 * This re-renders cached scene data. It does not re-run any model.
 *
 * Prereq:  python -m http.server 8000     (cwd = Visualization/)
 * Usage :  node export/recapture_fig4_hires.mjs
 */
import { chromium } from "playwright";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, "..", "..", "Publication", "T-IV_manuscript", "figures");
const OUT = path.join(OUT_DIR, "guo4.png");
const TOKEN = "af67f465f5994ac7bab19825336db644";
const SCALE = 2;

async function waitForScene(page) {
  await page.waitForFunction(() => {
    const info = document.getElementById("scene-info");
    return info && info.textContent && info.textContent.includes("token");
  }, { timeout: 30000 });
  await page.waitForTimeout(400);
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1400, height: 880 },
    deviceScaleFactor: SCALE,
  });

  // NOTE: figure mode is switched on *after* the controls are driven -- it
  // hides the toolbar, and playwright cannot click a display:none control.
  const qs = new URLSearchParams({
    token: TOKEN, includeAnchors: "1", lang: "en", bboxFocus: "action",
  });
  await page.goto(`http://localhost:8000/app/?${qs.toString()}`, { waitUntil: "networkidle" });
  await waitForScene(page);

  await page.click('#domain-seg button[data-domain="perceived"]');
  await page.click('#layer-seg button[data-layer="planner"]');
  const compare = page.locator("#compare-chk");
  if (!(await compare.isChecked())) await compare.check();
  await page.selectOption("#modelA", "full");
  await page.selectOption("#modelB", "stopgrad");
  for (let i = 0; i < 3; i++) {
    await page.click("#level-up");
    await page.waitForTimeout(80);
  }
  const maxT = await page.locator("#t-range").getAttribute("max");
  await page.locator("#t-range").evaluate((el, v) => {
    el.value = v; el.dispatchEvent(new Event("input"));
  }, maxT);
  await page.waitForTimeout(600);

  // Now strip the chrome and force a redraw so the canvas re-fits the taller
  // panel that freeing the toolbar/footer space opens up.
  await page.evaluate(() => {
    document.body.classList.add("figure-mode");
    window.dispatchEvent(new Event("resize"));
    if (typeof sizeCanvases === "function") sizeCanvases();
  });
  await page.waitForTimeout(300);
  await page.locator("#t-range").evaluate((el) => {
    el.dispatchEvent(new Event("input"));
  });
  await page.waitForTimeout(600);

  // Sanity check: the controls are hidden in figure mode, so confirm the state
  // actually took before capturing.
  const state = await page.evaluate(() => ({
    a: document.getElementById("modelA").value,
    b: document.getElementById("modelB").value,
    level: document.getElementById("level-val").textContent.trim(),
    t: document.getElementById("t-val").textContent.trim(),
    domain: document.querySelector('#domain-seg button.active')?.dataset.domain,
    layer: document.querySelector('#layer-seg button.active')?.dataset.layer,
    heads: [...document.querySelectorAll(".panel .phead")].map(h => h.textContent.trim()),
  }));
  console.log("[recapture] state:", JSON.stringify(state, null, 2));
  if (state.a !== "full" || state.b !== "stopgrad" ||
      state.domain !== "perceived" || state.layer !== "planner") {
    throw new Error("capture state did not take; refusing to write the figure");
  }

  await page.locator("#panels").screenshot({ path: OUT });
  console.log(`[recapture] wrote ${OUT}  (lang=en, figure mode, scale=${SCALE})`);

  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
