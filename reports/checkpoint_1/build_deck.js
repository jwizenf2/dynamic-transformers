const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "Jeremy Wizenfeld";
pres.title = "Dynamic Transformers with Reinforcement Learning";

// ---------------------------------------------------------------- palette
// Restrained, print-like: charcoal text, one brick accent for "executed
// compute" and emphasis, a slate secondary, and cool greys for everything else.
const INK = "1F2933"; // charcoal - headings and primary text
const ACCENT = "9A3B3B"; // brick - executed heads, sparse emphasis
const SECOND = "3D5A73"; // slate blue - "our work" vs the paper
const GRAY = "E4E7EB"; // skipped layers, rules
const DARKGRAY = "A6ADB6"; // pruned heads
const WHITE = "FFFFFF";
const PANEL = "EEF1F4"; // highlight panel fill
const CARD = "F8F9FA"; // ordinary card fill
const RULE = "D5DAE0"; // card outlines
const CODEBG = "F5F6F8";
const CODETXT = "1F2933";
const CODECMT = "7A8390";
const BODY = "3A4350";
const MUTED = "6B7480";

const HFONT = "Cambria";
const BFONT = "Calibri";
const MFONT = "Courier New";

const M = 0.62; // left margin
const CW = 13.333 - 2 * M; // content width

// ---------------------------------------------------------------- helpers
function titleBar(slide, kicker, title, dark) {
  slide.addText(kicker.toUpperCase(), {
    x: M,
    y: 0.36,
    w: CW,
    h: 0.24,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 11,
    bold: true,
    charSpacing: 2,
    color: MUTED,
  });
  slide.addText(title, {
    x: M,
    y: 0.62,
    w: CW,
    h: 0.62,
    isTextBox: true,
    margin: 0,
    valign: "top",
    fontFace: HFONT,
    fontSize: 32,
    bold: true,
    color: INK,
  });
}

// One sample's architecture: rows = layers (top->bottom), cols = heads.
// value 1 = executed head, 0 = pruned head, -1 = layer never runs
function headGrid(slide, x, y, cell, gap, rows, opts) {
  const o = opts || {};
  rows.forEach((row, r) => {
    row.forEach((v, c) => {
      let col = o.skipColor || GRAY;
      if (v === 1) col = o.onColor || ACCENT;
      else if (v === 0) col = o.offColor || DARKGRAY;
      slide.addShape(pres.ShapeType.rect, {
        x: x + c * (cell + gap),
        y: y + r * (cell + gap),
        w: cell,
        h: cell,
        fill: { color: col },
        line: { width: 0 },
      });
    });
  });
}

// The running example used throughout: depth 6, heads 2 and 7 dropped in 3-5
function exampleSpec() {
  const rows = [];
  for (let l = 0; l < 12; l++) {
    const row = [];
    for (let h = 0; h < 12; h++) {
      if (l < 3) row.push(1); // full-head prefix
      else if (l < 6) row.push(h === 2 || h === 7 ? 0 : 1); // chosen subset
      else row.push(-1); // layer never executed
    }
    rows.push(row);
  }
  return rows;
}

function fullSpec() {
  return Array.from({ length: 12 }, () => Array(12).fill(1));
}

function codeBlock(slide, x, y, w, h, lines, fontSize) {
  slide.addShape(pres.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.03,
    fill: { color: CODEBG },
    line: { color: RULE, width: 0.75 },
  });
  const runs = [];
  lines.forEach((ln, i) => {
    const isLast = i === lines.length - 1;
    if (typeof ln === "string") {
      const comment = ln.trimStart().startsWith("#");
      runs.push({
        text: ln === "" ? " " : ln,
        options: {
          fontFace: MFONT,
          fontSize,
          color: comment ? CODECMT : CODETXT,
          breakLine: !isLast,
        },
      });
    } else {
      runs.push({
        text: ln.t === "" ? " " : ln.t,
        options: {
          fontFace: MFONT,
          fontSize,
          color: ln.c || CODETXT,
          bold: !!ln.b,
          breakLine: !isLast,
        },
      });
    }
  });
  slide.addText(runs, {
    x: x + 0.22,
    y: y + 0.14,
    w: w - 0.44,
    h: h - 0.28,
    isTextBox: true,
    margin: 0,
    valign: "top",
    lineSpacingMultiple: 1.08,
  });
}

function card(slide, x, y, w, h, tint) {
  // Outlined, faintly filled boxes. Former dark "panels" become a slightly
  // deeper grey so the visual hierarchy survives without dark fills.
  const dark = tint === INK || tint === "1F2B4D";
  slide.addShape(pres.ShapeType.rect, {
    x,
    y,
    w,
    h,
    fill: { color: dark ? PANEL : CARD },
    line: { color: RULE, width: 0.75 },
  });
}

function iconDot(slide, x, y, d, label, color) {
  // Thin outlined circle with a numeral; no solid pill.
  slide.addShape(pres.ShapeType.ellipse, {
    x,
    y,
    w: d,
    h: d,
    fill: { color: WHITE },
    line: { color: color || INK, width: 1 },
  });
  slide.addText(label, {
    x,
    y,
    w: d,
    h: d,
    isTextBox: true,
    margin: 0,
    align: "center",
    valign: "middle",
    fontFace: BFONT,
    fontSize: 12,
    bold: true,
    color: color || INK,
  });
}

function footer(slide, text) {
  slide.addText(text, {
    x: M,
    y: 7.02,
    w: CW,
    h: 0.24,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 9.5,
    color: MUTED,
    italic: true,
  });
}

// ================================================================ 1. TITLE
{
  const s = pres.addSlide();

  headGrid(s, 8.55, 1.55, 0.26, 0.075, exampleSpec());
  s.addText("layer 0", {
    x: 7.15,
    y: 1.5,
    w: 1.3,
    h: 0.25,
    isTextBox: true,
    margin: 0,
    align: "right",
    fontFace: BFONT,
    fontSize: 9,
    color: MUTED,
  });
  s.addText("layer 11", {
    x: 7.15,
    y: 5.42,
    w: 1.3,
    h: 0.25,
    isTextBox: true,
    margin: 0,
    align: "right",
    fontFace: BFONT,
    fontSize: 9,
    color: MUTED,
  });
  s.addText("12 attention heads", {
    x: 8.55,
    y: 5.78,
    w: 4.05,
    h: 0.25,
    isTextBox: true,
    margin: 0,
    align: "center",
    fontFace: BFONT,
    fontSize: 9,
    color: MUTED,
  });

  s.addText("Dynamic Transformers with\nReinforcement Learning", {
    x: M,
    y: 2.05,
    w: 6.5,
    h: 1.5,
    isTextBox: true,
    margin: 0,
    fontFace: HFONT,
    fontSize: 32,
    bold: true,
    color: INK,
    lineSpacingMultiple: 0.98,
  });
  s.addText("Improving sample-based architecture selection", {
    x: M,
    y: 3.62,
    w: 6.3,
    h: 0.36,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 16,
    italic: true,
    color: MUTED,
  });

  s.addText(
    [
      { text: "Meng, Zhang, Chen & Wang (2024). ", options: { bold: true } },
      {
        text: "Sample-Based Dynamic Hierarchical Transformer with Layer and Head Flexibility via Contextual Bandit. ",
        options: { italic: true },
      },
      { text: "Proceedings on Engineering Sciences 6(2), 439–452." },
    ],
    {
      x: M,
      y: 4.42,
      w: 6.3,
      h: 0.95,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 11.5,
      color: BODY,
      lineSpacingMultiple: 1.25,
    },
  );

  s.addText("Jeremy Wizenfeld", {
    x: M,
    y: 5.55,
    w: 6.3,
    h: 0.3,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 12.5,
    color: INK,
  });

  s.addNotes(
    [
      "Open with the picture on the right: that grid is one real routing decision from my run.",
      "Rows are the 12 layers of BERT, columns are the 12 attention heads.",
      "Red = actually computed. Grey = never computed.",
      "This sample used 6 of 12 layers, and inside layers 3-5 it dropped heads 2 and 7.",
      "The next sample gets a different grid. That is the whole paper in one image.",
      "I reproduced the core algorithm from scratch - no author code exists publicly, and none was used.",
    ].join(" "),
  );
}

// ================================================================ 2. PROBLEM
{
  const s = pres.addSlide();
  titleBar(s, "The problem", "Transformers spend the same compute on every input");

  s.addText(
    [
      {
        text: "A trained Transformer is a fixed budget. ",
        options: { bold: true, color: INK, breakLine: false },
      },
      {
        text: "Every input passes through every layer and every attention head, easy or hard. The cost is set once, by the architecture — never by the input.",
        options: { color: BODY },
      },
    ],
    {
      x: M,
      y: 1.6,
      w: 6.6,
      h: 1.04,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 15,
      lineSpacingMultiple: 1.2,
    },
  );

  card(s, M, 2.72, 6.6, 1.6, CARD);
  s.addText('"Apple unveils the new iPhone at fall event"', {
    x: M + 0.28,
    y: 2.98,
    w: 6.05,
    h: 0.36,
    isTextBox: true,
    margin: 0,
    fontFace: HFONT,
    fontSize: 17,
    bold: true,
    color: INK,
  });
  s.addText(
    "Sci/Tech — obvious from the first two words. Still pays for the full network.",
    {
      x: M + 0.28,
      y: 3.42,
      w: 6.05,
      h: 0.66,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 13,
      color: BODY,
    },
  );

  card(s, M, 4.52, 6.6, 1.6, CARD);
  s.addText('"Apple ordered to pay €13bn in EU tax ruling"', {
    x: M + 0.28,
    y: 4.78,
    w: 6.05,
    h: 0.36,
    isTextBox: true,
    margin: 0,
    fontFace: HFONT,
    fontSize: 17,
    bold: true,
    color: INK,
  });
  s.addText("Business? World? Sci/Tech? Same first word — but you have to read the whole sentence. Same budget as above.", {
    x: M + 0.28,
    y: 5.22,
    w: 6.05,
    h: 0.66,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 13,
    color: BODY,
  });

  s.addText("L × H", {
    x: 7.65,
    y: 1.66,
    w: 5.1,
    h: 1.28,
    isTextBox: true,
    margin: 0,
    align: "center",
    fontFace: HFONT,
    fontSize: 64,
    bold: true,
    color: ACCENT,
  });
  s.addText("layers × heads, computed for every input,\nregardless of how hard it is", {
    x: 7.65,
    y: 2.98,
    w: 5.1,
    h: 0.7,
    isTextBox: true,
    margin: 0,
    align: "center",
    fontFace: BFONT,
    fontSize: 13,
    color: MUTED,
    lineSpacingMultiple: 1.15,
  });

  headGrid(s, 8.79, 3.78, 0.185, 0.05, fullSpec(), {});
  s.addText("e.g. BERT-base: 12 layers × 12 heads — all of it, every time", {
    x: 7.65,
    y: 6.66,
    w: 5.1,
    h: 0.26,
    isTextBox: true,
    margin: 0,
    align: "center",
    fontFace: BFONT,
    fontSize: 11,
    italic: true,
    color: MUTED,
  });

  s.addNotes(
    [
      "Keep this general: a Transformer of any size runs L layers and H heads per layer on every input.",
      "The cost is decided once, at the architecture level. But difficulty varies per sample, at inference time.",
      "The grid is just an example - BERT-base happens to be 12 by 12, which is why the paper uses it - the argument does not depend on those numbers.",
      "The two headlines make it concrete: both start with the word Apple. The iPhone launch is Sci/Tech from its first two words; the tax ruling could be Business, World or Sci/Tech and needs the whole sentence - and both get the identical budget. That mismatch is the entire motivation.",
      "Note the framing: this is not about making a smaller model. It is about spending a variable amount on a fixed model.",
    ].join(" "),
  );
}

// ================================================================ 3. PRIOR WORK
{
  const s = pres.addSlide();
  titleBar(s, "Where this sits", "Everyone else picks one architecture, once");

  const rows = [
    [
      "1",
      "Knowledge distillation",
      "DistilBERT trains one smaller student model. Cheaper for everything — including inputs that needed the capacity.",
      INK,
    ],
    [
      "2",
      "Head / structured pruning",
      "Rank heads by importance and delete them globally. A head useless on average may be the one that matters here.",
      INK,
    ],
    [
      "3",
      "Early exit",
      "FastBERT stops when a classifier is confident. Genuinely per-sample — but adapts depth only, never head identity.",
      SECOND,
    ],
    [
      "4",
      "Neural architecture search",
      "Searches the space properly, but trains an extra controller network and still emits one architecture for all inputs.",
      INK,
    ],
  ];

  let y = 1.7;
  rows.forEach(([n, head, body, col]) => {
    iconDot(s, M, y + 0.04, 0.4, n, col);
    s.addText(head, {
      x: M + 0.6,
      y,
      w: 7.05,
      h: 0.3,
      isTextBox: true,
      margin: 0,
      fontFace: HFONT,
      fontSize: 16,
      bold: true,
      color: INK,
    });
    s.addText(body, {
      x: M + 0.6,
      y: y + 0.31,
      w: 7.05,
      h: 0.62,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12.5,
      color: BODY,
      lineSpacingMultiple: 1.1,
    });
    y += 1.24;
  });

  card(s, 8.55, 1.7, 4.16, 4.59, INK);
  s.addText("THE GAP", {
    x: 8.85,
    y: 2.0,
    w: 3.56,
    h: 0.26,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 11,
    bold: true,
    charSpacing: 2,
    color: ACCENT,
  });
  s.addText(
    [
      {
        text: "Nothing chooses both how deep and which heads, per sample, without training a second network to do it.",
        options: { color: INK, breakLine: true },
      },
      { text: " ", options: { breakLine: true, fontSize: 8 } },
      {
        text: "DHT's answer: a contextual bandit. No auxiliary controller, no differentiable relaxation — just an online statistical model that scores discrete choices.",
        options: { color: BODY },
      },
    ],
    {
      x: 8.85,
      y: 2.36,
      w: 3.56,
      h: 3.7,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 14,
      lineSpacingMultiple: 1.22,
    },
  );

  footer(
    s,
    "Baselines the paper compares against (Tables 1–2, p. 447): 12-layer vanilla BERT, 1/3/6-layer DistilBERT, FastBERT.",
  );

  s.addNotes(
    [
      "Walk down the four rows quickly - thirty seconds total.",
      "The key contrast is row 3 versus the gap box: FastBERT is the closest prior work because it is genuinely per-sample,",
      "but it only decides when to stop. It never decides which heads to keep.",
      "DHT decides depth AND head identity, and it does it with a bandit rather than a learned controller network.",
      "That is the specific claim I am reproducing.",
    ].join(" "),
  );
}

// ================================================================ 4. THE IDEA
{
  const s = pres.addSlide();
  titleBar(s, "The method", "Three bandits, arranged as a hierarchy");

  // small helpers for the flow diagram
  const box = (x, y, w, text, accent) =>
    (s.addShape(pres.ShapeType.rect, {
      x,
      y,
      w,
      h: 0.55,
      fill: { color: accent ? WHITE : CARD },
      line: { color: accent ? ACCENT : RULE, width: accent ? 1.25 : 0.75 },
    }),
    s.addText(text, {
      x,
      y,
      w,
      h: 0.55,
      isTextBox: true,
      margin: 0.04,
      align: "center",
      valign: "middle",
      fontFace: BFONT,
      fontSize: accent ? 12.5 : 11,
      bold: !!accent,
      color: accent ? ACCENT : INK,
    }));
  const arrow = (x, y, w) =>
    s.addText("→", {
      x,
      y,
      w: w || 0.25,
      h: 0.55,
      isTextBox: true,
      margin: 0,
      align: "center",
      valign: "middle",
      fontFace: BFONT,
      fontSize: 14,
      color: MUTED,
    });
  const under = (x, y, w, text) =>
    s.addText(text, {
      x,
      y: y + 0.6,
      w,
      h: 0.42,
      isTextBox: true,
      margin: 0,
      align: "center",
      fontFace: BFONT,
      fontSize: 10,
      italic: true,
      color: MUTED,
      lineSpacingMultiple: 1.05,
    });
  const stage = (y, text) =>
    s.addText(text, {
      x: M,
      y,
      w: 8,
      h: 0.24,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 10.5,
      bold: true,
      charSpacing: 1.4,
      color: MUTED,
    });

  // ---- stage 1: decide the depth
  stage(1.5, "STAGE 1 · DECIDE HOW DEEP");
  let x = M;
  const y1 = 1.8;
  box(x, y1, 1.0, "sample x");
  x += 1.0;
  arrow(x, y1);
  x += 0.25;
  box(x, y1, 2.35, "b = 3 prefix layers, no gradients");
  x += 2.35;
  arrow(x, y1);
  x += 0.25;
  box(x, y1, 1.15, "UCB_L", true);
  under(x, y1, 1.15, "bandit 1");
  x += 1.15;
  arrow(x, y1);
  x += 0.25;
  box(x, y1, 2.3, "depth L: 3, 6, 9 or 12");

  // ---- stage 2: build and train
  stage(2.98, "STAGE 2 · BUILD THE NETWORK, GROUP BY GROUP, THEN TRAIN");
  x = M;
  const y2 = 3.28;
  box(x, y2, 1.0, "sample x");
  x += 1.0;
  arrow(x, y2);
  x += 0.25;
  box(x, y2, 2.05, "b prefix layers, with gradients");
  x += 2.05;
  arrow(x, y2);
  x += 0.25;
  const gx = x;
  box(x, y2, 1.05, "UCB_H", true);
  under(x, y2, 1.05, "bandit 2 · how many");
  x += 1.05;
  arrow(x, y2);
  x += 0.25;
  box(x, y2, 0.95, "TSB", true);
  under(x, y2, 0.95, "bandit 3 · which");
  x += 0.95;
  arrow(x, y2);
  x += 0.25;
  box(x, y2, 2.0, "c = 3 layers, those H heads");
  const gw = x + 2.0 - gx;
  x += 2.0;
  arrow(x, y2);
  x += 0.25;
  box(x, y2, 1.55, "repeat until depth L");
  x += 1.55;
  arrow(x, y2);
  x += 0.25;
  box(x, y2, 1.2, "classifier");
  // bracket text under the repeated group
  s.addText("one group — repeated (L − b) / c times, each time from the current hidden state", {
    x: gx,
    y: y2 + 1.02,
    w: gw + 1.8,
    h: 0.24,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 10,
    italic: true,
    color: MUTED,
  });

  // ---- who decides what, from what
  const hdr = { fill: { color: PANEL }, color: INK, bold: true };
  const c = (t, o) => ({ text: t, options: Object.assign({ color: BODY }, o || {}) });
  const rows = [
    [c("", hdr), c("Decides", hdr), c("What it looks at (the context)", hdr), c("How often", hdr)],
    [
      c("UCB_L", { bold: true, color: ACCENT }),
      c("how many layers, L"),
      c("prefix output of x, joined to the average prefix output seen at depth L during calibration"),
      c("once per sample"),
    ],
    [
      c("UCB_H", { bold: true, color: ACCENT }),
      c("how many heads, H, for the next c layers"),
      c("current hidden state, joined to the average state seen with H heads, plus a marker for which group this is"),
      c("once per group"),
    ],
    [
      c("TSB", { bold: true, color: ACCENT }),
      c("which H heads to keep"),
      c("current hidden state, joined to one feature per head (that head's slice of the embedding)"),
      c("once per group"),
    ],
  ];
  s.addTable(rows, {
    x: M,
    y: 4.72,
    w: CW,
    colW: [1.0, 3.0, 6.49, 1.6],
    rowH: [0.3, 0.42, 0.5, 0.42],
    fontFace: BFONT,
    fontSize: 10.5,
    valign: "middle",
    border: { type: "solid", color: RULE, pt: 0.75 },
    fill: { color: WHITE },
    margin: [0.03, 0.1, 0.03, 0.1],
  });

  s.addText(
    [
      { text: "Integrated mode: ", options: { bold: true, color: INK } },
      { text: "one UCB (UCB_cp) picks L and H together at stage 1; TSB still picks which heads, group by group. The prefix runs twice on purpose — once to decide, once to learn — and is never frozen.", options: { color: BODY } },
    ],
    {
      x: M,
      y: 6.44,
      w: CW,
      h: 0.48,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 11,
      lineSpacingMultiple: 1.12,
    },
  );

  s.addNotes(
    [
      "This is the slide the talk is built around: three bandits, arranged as a hierarchy - that is the word 'hierarchical' in the paper's title.",
      "Stage 1. The sample goes through the first three layers with no gradients. That gives a context vector. Bandit one, UCB_L, reads it and commits to a depth: 3, 6, 9 or 12.",
      "Stage 2. The sample goes through the same three layers again, this time with gradients so they can learn. Now we build the rest of the network one group of three layers at a time.",
      "For each group: bandit two, UCB_H, looks at the current hidden state and decides how many heads to keep - 8, 10 or 12. Then bandit three, TSB, decides which specific heads. Those three layers run with exactly those heads. Repeat until we reach the depth UCB_L chose, then classify.",
      "So UCB_H and TSB are called once per group, and they see the hidden state as it is at that point - a deeper group sees a more processed representation.",
      "The table is the thing to have people write down: who decides what, and what information it uses. Notice the contexts are not one-hot IDs; the arm features are averaged activations collected in a calibration pass, so 'depth 9' is described by what depth-9 outputs actually look like.",
      "Integrated mode is the paper's second variant: fuse UCB_L and UCB_H into one bandit that picks the pair. TSB is unchanged. I ran tree mode.",
    ].join(" "),
  );
}

// ================================================================ 6. UCB
{
  const s = pres.addSlide();
  titleBar(s, "Bandits 1 and 2", "Linear UCB, used twice: for depth, then for head count");

  s.addText(
    [
      { text: "score", options: { color: INK } },
      { text: "a", options: { color: INK, subscript: true } },
      { text: "  =  w", options: { color: INK } },
      { text: "T", options: { color: INK, superscript: true } },
      { text: "θ", options: { color: INK } },
      { text: "a", options: { color: INK, subscript: true } },
      { text: "   +   α √( w", options: { color: ACCENT } },
      { text: "T", options: { color: ACCENT, superscript: true } },
      { text: " A", options: { color: ACCENT } },
      { text: "a", options: { color: ACCENT, subscript: true } },
      { text: "-1", options: { color: ACCENT, superscript: true } },
      { text: " w )", options: { color: ACCENT } },
    ],
    {
      x: M,
      y: 1.64,
      w: 6.5,
      h: 0.46,
      isTextBox: true,
      margin: 0,
      fontFace: MFONT,
      fontSize: 19,
      bold: true,
    },
  );
  s.addText(
    [
      { text: "expected reward", options: { color: MUTED } },
      { text: "                    exploration bonus", options: { color: ACCENT } },
    ],
    {
      x: M,
      y: 2.1,
      w: 6.5,
      h: 0.26,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 11.5,
      italic: true,
    },
  );

  s.addText(
    [
      { text: "The vocabulary. ", options: { bold: true, color: INK } },
      {
        text: "An arm is one option on a menu. The context w is a list of numbers describing this sample (pooled activations), joined to a list describing the option. Each arm keeps its own small linear model θ",
        options: { color: BODY },
      },
      { text: "a", options: { color: BODY, subscript: true } },
      { text: " that predicts how good that option will be for inputs like this one.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 7 } },
      { text: "Choosing. ", options: { bold: true, color: INK } },
      {
        text: "Score every arm and take the highest. The first term is the model's best guess; the second is a bonus that is large when this arm has rarely been tried on inputs like this, so the bandit keeps checking options it is unsure about — and the bonus shrinks as evidence comes in.",
        options: { color: BODY, breakLine: true },
      },
      { text: " ", options: { breakLine: true, fontSize: 7 } },
      { text: "Learning. ", options: { bold: true, color: INK } },
      {
        text: "Once the reward is known, only the chosen arm is updated — ridge regression, one observation at a time. At evaluation the bonus is off, so inference is deterministic.",
        options: { color: BODY },
      },
    ],
    {
      x: M,
      y: 2.5,
      w: 6.5,
      h: 2.7,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12,
      lineSpacingMultiple: 1.14,
    },
  );

  card(s, M, 5.3, 6.5, 1.2, CARD);
  s.addText(
    [
      { text: "Same algorithm, two menus.  ", options: { bold: true, color: INK } },
      { text: "UCB_L", options: { bold: true, color: ACCENT } },
      { text: ": arms are the depths {3, 6, 9, 12}; context is the prefix output; chosen once per sample.  ", options: { color: BODY } },
      { text: "UCB_H", options: { bold: true, color: ACCENT } },
      { text: ": arms are the head counts {8, 10, 12}; context is the hidden state at the start of the group plus a marker for which group; chosen once per group. They keep separate statistics.", options: { color: BODY } },
    ],
    {
      x: M + 0.28,
      y: 5.42,
      w: 5.94,
      h: 1.0,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 11,
      lineSpacingMultiple: 1.12,
    },
  );

  codeBlock(
    s,
    7.35,
    1.62,
    5.36,
    4.5,
    [
      "# src/dht/bandits/linear_ucb.py",
      "",
      "def scores(self, features, *, explore=True):",
      "    # A = lambda*I + sum(w wT)    b = sum(reward*w)",
      "    theta = solve(self.A, self.b)",
      "    mean = (w * theta).sum(-1)",
      "",
      "    if not explore:          # evaluation: mean only",
      "        return mean",
      "",
      "    inv_w = solve(self.A, w)",
      "    variance = (w * inv_w).sum(-1)     # w' Ainv w",
      "    return mean + self.exploration * variance.sqrt()",
      "",
      "",
      "def update(self, arm, features, reward):",
      "    # ONLY the arm that actually executed",
      "    self.A[arm] += torch.outer(w, w)",
      "    self.b[arm] += reward * w",
      "    self.counts[arm] += 1",
    ],
    11,
  );

  footer(s, "Paper: §3.1 and Algorithm 1, p. 441.");

  s.addNotes(
    [
      "Start with the vocabulary, because 'arm' and 'context' are the two words everything else hangs on.",
      "An arm is an option on a menu. The context is a feature vector describing the sample we are looking at, joined to a feature vector describing the option.",
      "The same algorithm is used twice with two different menus: UCB_L chooses among depths, once per sample; UCB_H chooses among head counts, once per group of three layers, and its context includes which group it is deciding for.",
      "Each arm has its own little linear model. Score is prediction plus a bonus. The bonus is the whole point of UCB:",
      "it is large for arms we have rarely tried on inputs like this one, so we keep testing our uncertain options, and it fades as we learn.",
      "In the code: theta comes from solving A theta = b - that is ridge regression. The bonus is the square root of w-transpose A-inverse w.",
      "Point at update: only the arm that actually ran changes. We never find out what the other depths would have done.",
    ].join(" "),
  );
}

// ================================================================ 7. TS
{
  const s = pres.addSlide();
  titleBar(s, "Bandit 3", "Thompson sampling picks which of the H heads to keep");

  codeBlock(
    s,
    M,
    1.62,
    6.28,
    2.9,
    [
      "# src/dht/bandits/logistic_ts.py",
      "",
      "def sample_parameters(self):",
      "    # theta_h ~ N(mean_h, diag(1 / precision_h))",
      "    noise = torch.randn(self.mean.shape, ...)",
      "    return self.mean + noise / self.precision.sqrt()",
      "",
      "def select(self, features, count, *, explore=True):",
      "    theta = self.sample_parameters() if explore \\",
      "            else self.mean          # frozen at eval",
      "    prob = (w * theta).sum(-1).sigmoid()   # sigmoid",
      "    top = argsort(prob, descending=True)[:count]",
      "    return tuple(sorted(top))   # ORIGINAL head positions",
    ],
    11,
  );

  card(s, M, 4.72, 6.28, 1.72, CARD);
  s.addText(
    [
      { text: "Why not use UCB here too?  ", options: { bold: true, color: INK } },
      {
        text: "UCB_H has just said 'keep 8 heads'. Choosing 8 of 12 means 495 possible subsets — far too many to treat each as an arm. Instead every head is its own arm, each gets a score, and we keep the K = H highest, where H is whatever UCB_H just chose for this group.",
        options: { color: BODY },
      },
    ],
    {
      x: M + 0.28,
      y: 4.94,
      w: 5.72,
      h: 1.3,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12.5,
      lineSpacingMultiple: 1.12,
    },
  );

  s.addText("What Thompson sampling is", {
    x: 7.28,
    y: 1.62,
    w: 5.43,
    h: 0.32,
    isTextBox: true,
    margin: 0,
    fontFace: HFONT,
    fontSize: 18,
    bold: true,
    color: INK,
  });
  s.addText(
    "UCB adds an explicit bonus for uncertainty. Thompson sampling gets exploration for free from randomness: keep a belief about each option, draw one random guess from that belief, and act as if the guess were true.",
    {
      x: 7.28,
      y: 2.0,
      w: 5.43,
      h: 1.0,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 13,
      color: BODY,
      lineSpacingMultiple: 1.18,
    },
  );

  card(s, 7.28, 3.1, 5.43, 3.34, CARD);
  s.addText("Step by step, for heads", {
    x: 7.56,
    y: 3.28,
    w: 4.87,
    h: 0.3,
    isTextBox: true,
    margin: 0,
    fontFace: HFONT,
    fontSize: 15,
    bold: true,
    color: INK,
  });
  s.addText(
    [
      { text: "1  Belief. ", options: { bold: true, color: INK } },
      { text: "For each head, a Gaussian over parameters that say how useful this head is for inputs like the current one.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "2  Draw. ", options: { bold: true, color: INK } },
      { text: "Sample one parameter vector per head from its belief, score each head with a sigmoid, keep the K best.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "3  Explore by accident. ", options: { bold: true, color: INK } },
      { text: "A head with a wide, uncertain belief sometimes draws a high score and gets picked — that is how it earns evidence.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "4  Update. ", options: { bold: true, color: INK } },
      { text: "The chosen subset gets one shared yes/no label from the reward. Only the selected heads update their beliefs (a small logistic-regression step); the rest are untouched.", options: { color: BODY } },
    ],
    {
      x: 7.56,
      y: 3.62,
      w: 4.87,
      h: 2.7,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12,
      lineSpacingMultiple: 1.12,
    },
  );

  footer(s, "Paper: §3.2, p. 441–442.");

  s.addNotes(
    [
      "First contrast it with UCB in one sentence: UCB explores with an explicit bonus; Thompson sampling explores by being random in a principled way.",
      "The idea: for every head keep a belief - a Gaussian - about how useful it is. Draw one guess from each belief, act on the guesses, keep the K best heads.",
      "Because the draw is random, an uncertain head sometimes wins. That is the exploration, and there is no knob to tune.",
      "Code: sample_parameters is the draw, select is the sigmoid-and-top-K. Point at the last line: we return the ORIGINAL head positions.",
      "Head 7 must keep using head 7's slice of the weights. If you renumbered the survivors you would silently rewire the pretrained model.",
      "Bottom-left card ties it to the hierarchy: UCB_H has just chosen H; TSB now picks which H. Why not UCB here? Because 8-of-12 is 495 subsets. Give each head its own arm instead - a semi-bandit.",
      "Update: the subset produces one loss, so one shared yes/no label goes to every selected head. Unselected heads learn nothing.",
    ].join(" "),
  );
}

// ================================================================ 8. REWARD
{
  const s = pres.addSlide();
  titleBar(s, "The signal", "Reward = did I get that accuracy cheaply?");

  const step = (y, h, n, head, formula, body) => {
    card(s, M, y, 6.1, h, CARD);
    iconDot(s, M + 0.22, y + 0.2, 0.36, n, ACCENT);
    s.addText(head, {
      x: M + 0.7,
      y: y + 0.18,
      w: 5.2,
      h: 0.4,
      isTextBox: true,
      margin: 0,
      valign: "middle",
      fontFace: HFONT,
      fontSize: 15,
      bold: true,
      color: INK,
    });
    if (formula) {
      s.addText(formula, {
        x: M + 0.7,
        y: y + 0.62,
        w: 5.2,
        h: 0.34,
        isTextBox: true,
        margin: 0,
        fontFace: BFONT,
        fontSize: 14,
        bold: true,
        color: ACCENT,
      });
    }
    s.addText(body, {
      x: M + 0.7,
      y: y + (formula ? 1.0 : 0.62),
      w: 5.2,
      h: h - (formula ? 1.1 : 0.72),
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12,
      color: BODY,
      lineSpacingMultiple: 1.12,
    });
  };

  step(
    1.58,
    1.62,
    "1",
    "Price the outcome",
    "δ  =  loss  ×  compute spent",
    "Multiply the sample's loss by the compute the decision used. Small δ = accurate and cheap; large δ = inaccurate, expensive, or both.",
  );
  step(
    3.3,
    1.8,
    "2",
    "Compare with the previous batch",
    "reward  =  average δ of the previous batch  ÷  δ of this sample",
    "Above 1: this decision did better than the batch before it. Below 1: worse. Comparing with the recent past keeps the number meaningful as losses fall over training.",
  );
  step(
    5.2,
    1.46,
    "3",
    "Send it to the bandit that made the decision",
    null,
    "UCB_L: compute = layers run. UCB_H: compute = heads kept in its group. UCB_cp (integrated): layers × heads. TSB learns yes / no only, so UCB_H's reward becomes a label: did it cross 1?",
  );

  // ---- worked example
  s.addText("Worked example — the depth decision", {
    x: 7.0,
    y: 1.58,
    w: 5.7,
    h: 0.32,
    isTextBox: true,
    margin: 0,
    fontFace: HFONT,
    fontSize: 15,
    bold: true,
    color: INK,
  });
  const hdr = { fill: { color: PANEL }, color: INK, bold: true, align: "center" };
  const c = (t, o) => ({ text: t, options: Object.assign({ color: BODY, align: "center" }, o || {}) });
  const rows = [
    [c("", hdr), c("loss", hdr), c("layers", hdr), c("δ = loss × layers", hdr), c("reward", hdr)],
    [c("previous batch, average", { align: "left", color: MUTED }), c("—"), c("—"), c("2.4", { bold: true }), c("—")],
    [c("sample A, ran 3 layers", { align: "left", bold: true, color: INK }), c("0.60"), c("3"), c("1.8"), c("2.4 ÷ 1.8 = 1.33", { bold: true, color: SECOND })],
    [c("sample B, ran 12 layers", { align: "left", bold: true, color: INK }), c("0.50"), c("12"), c("6.0"), c("2.4 ÷ 6.0 = 0.40", { bold: true, color: ACCENT })],
  ];
  s.addTable(rows, {
    x: 7.0,
    y: 1.98,
    w: 5.7,
    colW: [1.95, 0.7, 0.75, 1.2, 1.1],
    rowH: [0.34, 0.4, 0.4, 0.4],
    fontFace: BFONT,
    fontSize: 11,
    valign: "middle",
    border: { type: "solid", color: RULE, pt: 0.75 },
    fill: { color: WHITE },
    margin: [0.03, 0.08, 0.03, 0.08],
  });

  card(s, 7.0, 3.72, 5.7, 2.82, CARD);
  s.addText(
    [
      { text: "Reading it. ", options: { bold: true, color: INK } },
      { text: "A did better than the previous batch (1.33 > 1): its accuracy was worth the compute. ", options: { color: BODY } },
      { text: "B was more accurate than A, yet scores worse (0.40 < 1)", options: { bold: true, color: INK } },
      { text: " — it spent four times the layers for a small gain in loss. The bandit learns that, for inputs like B, the extra depth was not worth it.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 7 } },
      { text: "Same idea for heads. ", options: { bold: true, color: INK } },
      { text: "For UCB_H, replace 'layers' with 'heads kept in this group' and compare with the previous batch's average for that same group.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 7 } },
      { text: "Then the label for TSB. ", options: { bold: true, color: INK } },
      { text: "TSB only takes yes / no, so its label is simply whether UCB_H's reward crossed 1 for that group.", options: { color: BODY } },
    ],
    {
      x: 7.28,
      y: 3.9,
      w: 5.14,
      h: 2.5,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12,
      lineSpacingMultiple: 1.12,
    },
  );

  footer(s, "Paper: §4.3, p. 443–444. The paper also multiplies by a constant scale α; it is 1 in these runs.");

  s.addNotes(
    [
      "One question drives every reward: did this decision get its accuracy cheaply?",
      "Step one: price the outcome. Loss times compute. Small is good.",
      "Step two: compare with the previous batch. Reward is their average divided by ours, so above one means we did better than recently. Dividing by the recent past is what keeps the number meaningful as losses fall over training.",
      "Step three: hand it to whichever bandit made that decision - UCB_L uses layers as the compute, UCB_H uses heads, and TSB gets a yes/no version of UCB_H's number.",
      "Now the example, slowly. Previous batch averaged 2.4. Sample A used 3 layers and got loss 0.6: delta 1.8, reward 1.33 - better than before.",
      "Sample B used all 12 layers and got a lower loss, 0.5. But delta is 6.0 and reward is 0.4 - worse. It paid four times the compute for a small gain.",
      "That is the whole design in one contrast: the reward is not about being accurate, it is about being accurate per unit of compute.",
      "If asked about alpha: it is a constant scale factor in the paper; it is 1 in my runs, so I left it off the slide.",
    ].join(" "),
  );
}

// ================================================================ 9. CODE: CONTROLLER
{
  const s = pres.addSlide();
  titleBar(s, "Code walkthrough", "Two functions: route(), then execute()");

  s.addText("STAGE 1 — route(): pick the depth. No learning happens here.", {
    x: M,
    y: 1.6,
    w: 6.05,
    h: 0.26,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 11,
    bold: true,
    charSpacing: 1.2,
    color: ACCENT,
  });
  codeBlock(
    s,
    M,
    1.92,
    6.05,
    1.85,
    [
      "@torch.no_grad()                    # no learning here",
      "def route(self, model, tokens, sample_id):",
      "    model.eval()                    # dropout off",
      "    hidden, mask = model.prefix(tokens)   # first 3 layers",
      "    w = self.features.aggregate(hidden, mask, 'depth')",
      "    arm = self.route_ucb.select(w)  # UCB_L: pick a depth",
      "    return RouteDecision(sample_id, self.depths[arm],",
      "                         arm, w[arm])  # keep the context",
    ],
    10.5,
  );

  s.addText(
    [
      { text: "What comes back: ", options: { bold: true, color: INK } },
      {
        text: "the chosen depth, plus a copy of the numbers UCB_L looked at when it chose. The sample waits with other samples of the same depth; when its reward arrives later, UCB_L is updated using exactly those saved numbers.",
        options: { color: BODY },
      },
    ],
    {
      x: M,
      y: 3.96,
      w: 6.05,
      h: 1.05,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12,
      lineSpacingMultiple: 1.14,
    },
  );

  card(s, M, 5.14, 6.05, 1.44, CARD);
  s.addText(
    [
      { text: "Why one sample at a time?  ", options: { bold: true, color: INK } },
      {
        text: "Two samples in the same batch can keep different heads, so each runs on its own and their average loss makes one update. Simple and correct — but it is also why my timings are slow. Batching identical head choices is future work.",
        options: { color: BODY },
      },
    ],
    {
      x: M + 0.28,
      y: 5.32,
      w: 5.49,
      h: 1.16,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12,
      lineSpacingMultiple: 1.12,
    },
  );

  s.addText("STAGE 2 — execute(): pick heads, run the layers, then learn.", {
    x: 6.98,
    y: 1.6,
    w: 5.73,
    h: 0.26,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 11,
    bold: true,
    charSpacing: 1.2,
    color: ACCENT,
  });
  codeBlock(
    s,
    6.98,
    1.92,
    5.73,
    3.98,
    [
      "def execute(self, model, tokens, route):",
      "    hidden, mask = model.prefix(tokens)  # same 3 layers,",
      "                                         # now learning",
      "",
      "    for gi, start in enumerate(self.groups):  # 3 at a time",
      "        if start >= route.total_depth:",
      "            break                    # chosen depth reached",
      "",
      "        wc = self.feat(hidden, mask, 'head', gi)",
      "        count = self.counts[self.head_ucb.select(wc)]",
      "        # ^ UCB_H: how many heads for this group",
      "",
      "        wt = self.feat(hidden, mask, 'ts', gi)",
      "        heads = self.ts.select(wt, count)   # TSB: which",
      "",
      "        for l in range(start, start + 3):    # run group",
      "            hidden = model.layers[l](hidden, mask, heads)",
      "",
      "    return Execution(model.classify(hidden), ...)",
    ],
    10.5,
  );

  s.addText(
    [
      { text: "Read it top to bottom: ", options: { bold: true, color: INK } },
      { text: "run the prefix → for each group of 3 layers: ask UCB_H how many heads, ask TSB which ones, run those layers with those heads → stop at the chosen depth → classify. The loss from classify() is what trains the model and scores the bandits.", options: { color: BODY } },
    ],
    {
      x: 6.98,
      y: 5.98,
      w: 5.73,
      h: 0.98,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 11,
      lineSpacingMultiple: 1.12,
    },
  );

  footer(s, "src/dht/policies/controller.py — the paper's Algorithm 2 (p. 444) in about forty lines. The training loop calls route(), then execute(), then updates.");

  s.addNotes(
    [
      "Two functions, and that is the algorithm.",
      "Left: route. No gradients, dropout off, run the first three layers, build the context, ask UCB_L for a depth. It returns the depth and a copy of the context it decided on.",
      "Why keep the copy? The sample waits for others of the same depth before it trains. By the time the reward comes back the model has changed, so we update UCB_L with the numbers it actually saw, not fresh ones.",
      "Right: execute. Run the same three layers again, this time learning. Then loop over groups of three layers.",
      "Each group: ask UCB_H how many heads, ask TSB which heads, run the three layers with exactly those heads. Stop when we reach the chosen depth, then classify.",
      "The loss from classify does two jobs: it trains the model, and it becomes the reward that scores all three bandits.",
      "If someone asks why per-sample: samples in one batch can keep different heads, so they run one at a time. Correct first; fast later.",
    ].join(" "),
  );
}

// ================================================================ 10. PHYSICAL
{
  const s = pres.addSlide();
  titleBar(s, "Beyond the paper", "Masking a head is not the same as skipping it", true);

  s.addText(
    [
      { text: "The paper describes head selection as a mask:  ", options: { color: BODY } },
      { text: "out = Σ ξ", options: { color: INK, bold: true } },
      { text: "h", options: { color: INK, bold: true, subscript: true } },
      { text: " · Att", options: { color: INK, bold: true } },
      { text: "h", options: { color: INK, bold: true, subscript: true } },
      { text: "(x),   ξ = 0 or 1", options: { color: INK, bold: true } },
      {
        text: ".  A mask on its own saves nothing — every head is still computed, then some are multiplied by zero. So the implementation slices the weight matrices instead, and an operator profiler measures the actual saving.",
        options: { color: BODY },
      },
    ],
    {
      x: M,
      y: 1.58,
      w: CW,
      h: 0.85,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 13.5,
      lineSpacingMultiple: 1.18,
    },
  );

  codeBlock(
    s,
    M,
    2.56,
    6.55,
    2.42,
    [
      "# src/dht/models/attention.py",
      "def project(projection):",
      "    # slice the ROWS for selected heads BEFORE the matmul",
      "    return F.linear(",
      "        hidden,",
      "        index_select(projection.weight, 0, columns),",
      "        index_select(projection.bias,   0, columns),",
      "    ).view(batch, length, len(executed), self.head_dim)",
      "",
      "# output projection keeps the ORIGINAL head columns",
      "out_w = index_select(self.output.weight, 1, columns)",
    ],
    10.5,
  );

  s.addText(
    [
      { text: "Measured, not inferred.  ", options: { color: ACCENT, bold: true } },
      {
        text: "A CPU profiler counts real mm / addmm / bmm shapes on real tokens. Sliced and dense-masked outputs match exactly, so the saving costs nothing in accuracy.",
        options: { color: BODY, breakLine: true },
      },
      { text: " ", options: { breakLine: true, fontSize: 7 } },
      { text: "What it is not:  ", options: { color: INK, bold: true } },
      {
        text: "the FFNs, the dense AdamW state and the routing prefix are all still full cost. This is a forward-attention saving, not a total training speedup.",
        options: { color: BODY },
      },
    ],
    {
      x: M,
      y: 5.12,
      w: 6.55,
      h: 1.5,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12.5,
      lineSpacingMultiple: 1.15,
    },
  );

  const nums = [
    ["22.35", "GFLOP · full 12 layers, 12 heads", MUTED],
    ["16.76", "GFLOP · 9 layers, heads masked to zero", MUTED],
    ["15.45", "GFLOP · 9 layers, heads physically sliced", ACCENT],
  ];
  nums.forEach(([n, label, col], i) => {
    const y = 2.56 + i * 0.92;
    s.addText(n, {
      x: 7.35,
      y,
      w: 1.68,
      h: 0.6,
      isTextBox: true,
      margin: 0,
      align: "right",
      fontFace: HFONT,
      fontSize: 34,
      bold: true,
      color: col,
    });
    s.addText(label, {
      x: 9.15,
      y: y + 0.18,
      w: 3.56,
      h: 0.4,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12,
      color: col === ACCENT ? INK : MUTED,
    });
  });

  card(s, 7.35, 5.32, 5.36, 1.42, "1F2B4D");
  s.addText("One real trace, validation record 4", {
    x: 7.63,
    y: 5.48,
    w: 4.8,
    h: 0.26,
    isTextBox: true,
    margin: 0,
    fontFace: BFONT,
    fontSize: 11,
    bold: true,
    charSpacing: 1.2,
    color: ACCENT,
  });
  s.addText(
    [
      { text: "layers visited: [0,1,2, 0,1,2, 3,4,5]", options: { color: INK, breakLine: true } },
      { text: "Q/K/V width: 640, not 768   ·   layers 6–11 never called", options: { color: BODY } },
    ],
    {
      x: 7.63,
      y: 5.8,
      w: 4.8,
      h: 0.8,
      isTextBox: true,
      margin: 0,
      fontFace: MFONT,
      fontSize: 11,
      lineSpacingMultiple: 1.2,
    },
  );

  s.addNotes(
    [
      "This is the part of the work that is mine rather than the paper's, so give it a full minute.",
      "The paper describes head selection as a zero-one mask on each head's output. That is the right definition, but a mask by itself does not save compute:",
      "every head is still computed and some are then multiplied by zero.",
      "So I slice the projection weight rows for the selected heads before the matmul, and slice the matching output-projection columns.",
      "Emphasise the word ORIGINAL in the comment: head 7 must keep writing into head 7's output columns.",
      "The numbers on the right are from a real operator profiler on real tokens, not an analytic estimate:",
      "twenty-two point three billion for the full model, sixteen point eight if you mask, fifteen point five if you actually slice.",
      "And the trace at the bottom is a real routing decision - you can see the prefix run twice, and layers six through eleven never appear.",
      "Honesty caveat if asked: the FFNs, the dense AdamW state and the routing pass are all still full cost. This is not a total training speedup.",
    ].join(" "),
  );
}

// ================================================================ 12. RESULTS
{
  const s = pres.addSlide();
  titleBar(s, "Results", "Within 1 percentage point, at 39% of the compute");

  s.addText(
    "Matched protocol: same pinned BERT-base-uncased encoder, same seed, same 114,000 / 6,000 AG News split, 3 epochs, batch 8, lr 2e-5 with linear warmup-decay, Apple M2 Max / MPS. Checkpoints were chosen on validation; the 7,600-record public test set was then evaluated once. Macro F1 throughout.",
    {
      x: M,
      y: 1.6,
      w: CW,
      h: 0.66,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 12.5,
      color: BODY,
      lineSpacingMultiple: 1.16,
    },
  );

  const hdr = { fill: { color: PANEL }, color: INK, bold: true, align: "center" };
  const rows = [
    [
      { text: "", options: { fill: { color: PANEL } } },
      { text: "Paper, test F1", options: hdr },
      { text: "Ours, test F1", options: hdr },
      { text: "Ours, validation F1", options: hdr },
      { text: "Task FLOPs on test", options: hdr },
    ],
    [
      { text: "Full 12-layer BERT", options: { bold: true, color: INK } },
      { text: ".944", options: { align: "center", color: BODY } },
      { text: ".9490", options: { align: "center", bold: true, color: SECOND } },
      { text: ".9480", options: { align: "center", color: BODY } },
      { text: "1.00×", options: { align: "center", color: BODY } },
    ],
    [
      { text: "T-DHT (tree mode)", options: { bold: true, color: INK } },
      { text: ".945", options: { align: "center", color: BODY } },
      { text: ".9400", options: { align: "center", bold: true, color: SECOND } },
      { text: ".9438", options: { align: "center", color: BODY } },
      { text: ".39×", options: { align: "center", bold: true, color: SECOND } },
    ],
  ];
  s.addTable(rows, {
    x: M,
    y: 2.4,
    w: CW,
    colW: [3.0, 2.0, 2.0, 2.4, 2.69],
    rowH: [0.38, 0.44, 0.44],
    fontFace: BFONT,
    fontSize: 13,
    valign: "middle",
    border: { type: "solid", color: RULE, pt: 0.75 },
    fill: { color: WHITE },
    margin: [0.06, 0.12, 0.06, 0.12],
  });

  s.addText(
    "Task FLOPs = forward attention + FFN matmuls on the path each sample actually took, as a fraction of the full model. The paper reports .42× for T-DHT in training and .81× at inference.",
    {
      x: M,
      y: 4.02,
      w: CW,
      h: 0.28,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 11,
      italic: true,
      color: MUTED,
    },
  );

  card(s, M, 4.42, 6.05, 2.34, CARD);
  s.addText("What we can say", {
    x: M + 0.28,
    y: 4.58,
    w: 5.49,
    h: 0.3,
    isTextBox: true,
    margin: 0,
    fontFace: HFONT,
    fontSize: 16,
    bold: true,
    color: INK,
  });
  s.addText(
    [
      { text: "The baseline reproduces. ", options: { bold: true, color: SECOND } },
      { text: "Test F1 .9490 against the paper's .944.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 5 } },
      { text: "T-DHT lands 0.0090 below our baseline (0.90 percentage points) and 0.0050 below the paper's T-DHT. ", options: { bold: true, color: INK } },
      { text: "Close, but it does not reproduce their slight gain over the full model.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 5 } },
      { text: "Compute matches the paper's training figure: ", options: { bold: true, color: INK } },
      { text: "our training-path ratio is .40× against their .42×.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 5 } },
      { text: "Caveats: one seed; FLOP counts cover forward matmuls only.", options: { color: MUTED, italic: true } },
    ],
    {
      x: M + 0.28,
      y: 4.9,
      w: 5.49,
      h: 1.8,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 11,
      lineSpacingMultiple: 1.1,
    },
  );

  card(s, 6.98, 4.42, 5.73, 2.34, CARD);
  s.addText("What the policy actually learned", {
    x: 7.26,
    y: 4.58,
    w: 5.17,
    h: 0.3,
    isTextBox: true,
    margin: 0,
    fontFace: HFONT,
    fontSize: 16,
    bold: true,
    color: INK,
  });
  s.addText(
    [
      { text: "It exits early, aggressively. ", options: { bold: true, color: INK } },
      { text: "62.6% of test samples stop after 3 layers; mean depth 4.75 of 12. Plausible on an easy dataset like AG News, but far more aggressive than the paper's .81× inference ratio.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 5 } },
      { text: "No collapse. ", options: { bold: true, color: INK } },
      { text: "All four depths and all three head counts are used; UCB_H picks 10 heads most often.", options: { color: BODY, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 5 } },
      { text: "Fewer FLOPs, slower clock. ", options: { bold: true, color: INK } },
      { text: "Matched benchmark: 174 samples/s fixed vs 68 for T-DHT, 2.55× slower, because samples run one at a time plus an extra routing pass. Training took 1.6× longer.", options: { color: BODY } },
    ],
    {
      x: 7.26,
      y: 4.9,
      w: 5.17,
      h: 1.8,
      isTextBox: true,
      margin: 0,
      fontFace: BFONT,
      fontSize: 11,
      lineSpacingMultiple: 1.1,
    },
  );

  footer(s, "Paper values: Tables 1–2, printed p. 447. Ours: artifacts/runs/ag-news-*-full-seed7/{final_test,benchmark,summary}.json.");

  s.addNotes(
    [
      "This is the slide the whole talk builds to, so slow down. Everything here is now test against test.",
      "First the baseline: point nine four nine against their point nine four four. That says the data pipeline, the pretrained weights and the training loop are right. If the baseline were off, nothing else here would mean anything.",
      "Now T-DHT. Test F1 point nine four zero: nine thousandths below our own full model - under one percentage point - and five thousandths below the paper's T-DHT number. Close, but I do not reproduce their claim that T-DHT slightly beats the full model.",
      "Compute: on the training path I measure point four zero of the full model's matmul FLOPs; the paper reports point four two. That agreement is the strongest evidence that I built their mechanism and not something that merely resembles it.",
      "Right-hand box. The policy exits early a lot: sixty-three percent of test samples stop at three layers. That is why my inference ratio is point three nine when the paper reports point eight one. AG News is an easy dataset, so this is plausible, but it is a real difference in behaviour.",
      "It has not collapsed: every depth and every head count still gets used.",
      "Last line is the honest engineering caveat: fewer FLOPs did not make it faster on a stopwatch. Two and a half times slower at inference and 1.6 times longer to train, because each sample runs on its own with an extra routing pass. That is where the next phase of the project starts - then take questions.",
      "If asked about the gap size: say it as nine thousandths, or point nine of a percentage point - not 'point nine points'.",
    ].join(" "),
  );
}

pres.writeFile({ fileName: process.argv[2] }).then((f) => console.log("wrote", f));
