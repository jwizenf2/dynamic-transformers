// September 23 progress update. Same visual system as reports/checkpoint_1/build_deck.js.
// Usage: node build_deck.js <output.pptx>   (needs pptxgenjs on NODE_PATH)
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "Jeremy Wizenfeld";
pres.title = "September 23 Progress Update";

// ---------------------------------------------------------------- palette
const INK = "1F2933";
const ACCENT = "9A3B3B";
const SECOND = "3D5A73";
const GRAY = "E4E7EB";
const DARKGRAY = "A6ADB6";
const WHITE = "FFFFFF";
const PANEL = "EEF1F4";
const CARD = "F8F9FA";
const RULE = "D5DAE0";
const CODEBG = "F5F6F8";
const CODETXT = "1F2933";
const CODECMT = "7A8390";
const BODY = "3A4350";
const MUTED = "6B7480";
// depth shades for the routing strips on the advice slide
const SHALLOW = "E6CFCF";
const MID = "C48B8B";

const HFONT = "Cambria";
const BFONT = "Calibri";
const MFONT = "Courier New";

const M = 0.62;
const CW = 13.333 - 2 * M;

// ---------------------------------------------------------------- helpers
function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontFace: BFONT, color: BODY, ...o });
}

function titleBar(slide, kicker, title) {
  text(slide, kicker.toUpperCase(), {
    x: M, y: 0.36, w: CW, h: 0.24, fontSize: 11, bold: true, charSpacing: 2, color: MUTED,
  });
  text(slide, title, {
    x: M, y: 0.62, w: CW, h: 0.62, valign: "top", fontFace: HFONT, fontSize: 30, bold: true, color: INK,
  });
}

function headGrid(slide, x, y, cell, gap, rows) {
  rows.forEach((row, r) => {
    row.forEach((v, c) => {
      const col = v === 1 ? ACCENT : v === 0 ? DARKGRAY : GRAY;
      slide.addShape(pres.ShapeType.rect, {
        x: x + c * (cell + gap), y: y + r * (cell + gap), w: cell, h: cell,
        fill: { color: col }, line: { color: col, width: 0.5 },
      });
    });
  });
}

// Real T-DHT test sample #0: depth 6, heads 1, 3, 8, 9 dropped in layers 3-5
function exampleSpec() {
  const rows = [];
  for (let l = 0; l < 12; l++) {
    const row = [];
    for (let h = 0; h < 12; h++) {
      if (l < 3) row.push(1);
      else if (l < 6) row.push([1, 3, 8, 9].includes(h) ? 0 : 1);
      else row.push(-1);
    }
    rows.push(row);
  }
  return rows;
}

function codeBlock(slide, x, y, w, h, lines, fontSize) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.03, fill: { color: CODEBG }, line: { color: RULE, width: 0.75 },
  });
  const runs = lines.map((ln, i) => {
    const l = typeof ln === "string" ? { t: ln, c: ln.trimStart().startsWith("#") ? CODECMT : CODETXT } : ln;
    return {
      text: l.t === "" ? " " : l.t,
      options: { fontFace: MFONT, fontSize, color: l.c || CODETXT, bold: !!l.b, breakLine: i < lines.length - 1 },
    };
  });
  slide.addText(runs, {
    x: x + 0.2, y: y + 0.12, w: w - 0.4, h: h - 0.24, isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.05,
  });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: fill || CARD }, line: { color: RULE, width: 0.75 } });
}

function cardTitle(slide, x, y, w, t) {
  text(slide, t, { x, y, w, h: 0.32, fontFace: HFONT, fontSize: 16, bold: true, color: INK });
}

function footer(slide, t) {
  text(slide, t, { x: M, y: 7.02, w: CW, h: 0.24, fontSize: 9.5, color: MUTED, italic: true });
}

const gapLine = { text: " ", options: { breakLine: true, fontSize: 4 } };
function bullets(items) {
  // items: array of [boldLead, rest]
  const runs = [];
  items.forEach(([lead, rest], i) => {
    const last = i === items.length - 1;
    runs.push({ text: lead, options: { bold: true, color: INK } });
    runs.push({ text: rest, options: { color: BODY, breakLine: !last } });
    if (!last) runs.push(gapLine);
  });
  return runs;
}

function smallTable(slide, x, y, colW, rows) {
  const hdr = { fill: { color: PANEL }, color: INK, bold: true, align: "center" };
  const body = rows.slice(1).map((r) =>
    r.map((c, j) =>
      j === 0
        ? { text: c, options: { bold: true, color: INK } }
        : { text: c, options: { align: "center", color: BODY } },
    ),
  );
  slide.addTable([rows[0].map((c) => ({ text: c, options: hdr })), ...body], {
    x, y, colW, rowH: 0.27, fontFace: BFONT, fontSize: 10.5, valign: "middle",
    border: { type: "solid", color: RULE, pt: 0.75 }, fill: { color: WHITE }, margin: [0.02, 0.08, 0.02, 0.08],
  });
}

// ================================================================ 1. TITLE
{
  const s = pres.addSlide();
  headGrid(s, 8.55, 1.55, 0.26, 0.075, exampleSpec());
  text(s, "layer 0", { x: 7.15, y: 1.5, w: 1.3, h: 0.25, align: "right", fontSize: 9, color: MUTED });
  text(s, "layer 11", { x: 7.15, y: 5.42, w: 1.3, h: 0.25, align: "right", fontSize: 9, color: MUTED });
  text(s, "12 attention heads · one real routing decision", {
    x: 8.55, y: 5.78, w: 4.05, h: 0.25, align: "center", fontSize: 9, color: MUTED,
  });

  text(s, "SEPTEMBER 23 PROGRESS UPDATE", {
    x: M, y: 1.62, w: 6.5, h: 0.3, fontSize: 12, bold: true, charSpacing: 2, color: ACCENT,
  });
  text(s, "Dynamic Transformers with\nReinforcement Learning", {
    x: M, y: 2.05, w: 6.6, h: 1.5, fontFace: HFONT, fontSize: 32, bold: true, color: INK, lineSpacingMultiple: 0.98,
  });
  text(s, "Letting each input choose how much of the network to run: which layers and which attention heads, picked per sample by a bandit / RL policy. Next: carrying the idea from text classification to tool-calling generation.", {
    x: M, y: 3.66, w: 6.3, h: 1.1, fontSize: 15, italic: true, color: MUTED, lineSpacingMultiple: 1.12,
  });
  text(s, [
    { text: "Base paper: ", options: { bold: true } },
    { text: "Meng et al. (2024), Sample-Based Dynamic Hierarchical Transformer with Layer and Head Flexibility via Contextual Bandit (DHT).", options: { italic: true } },
  ], { x: M, y: 4.95, w: 6.3, h: 0.55, fontSize: 11, lineSpacingMultiple: 1.2 });
  text(s, "Jeremy Wizenfeld", { x: M, y: 5.72, w: 6.3, h: 0.3, fontSize: 13, color: INK });

  s.addNotes([
    "Quick recap of the project: a transformer that decides per input how many layers and which attention heads to run.",
    "The grid is a real routing decision from my trained model: rows are layers, columns heads, red computed, grey skipped.",
    "Today: what I finished since last time, where I want to take it next (tool calling), and where I'd like your advice.",
  ].join(" "));
}

// ================================================================ 2. DONE
{
  const s = pres.addSlide();
  titleBar(s, "Since last time", "EDA, integrated search, TS, and a real speedup");

  const colW = 5.95, gx = M + colW + 0.22, rowH = 2.62, y1 = 1.52, y2 = y1 + rowH + 0.2;
  const pad = 0.24;

  // --- EDA
  card(s, M, y1, colW, rowH);
  cardTitle(s, M + pad, y1 + 0.16, colW - 2 * pad, "Dataset EDA: all four paper datasets");
  text(s, bullets([
    ["AG News, DBpedia, Yelp Polarity, Yelp Full: ", "class balance, text length vs the 128 / 300-token limits, vocabulary, and a Naive Bayes difficulty baseline."],
    ["AG News is short: ", "median 51 tokens, so ~60% of every 128-token input is padding."],
    ["One sentence at a time: ", "each example shows the text, the depth and heads DHT picked, and full BERT's vs DHT's prediction."],
  ]), { x: M + pad, y: y1 + 0.58, w: colW - 2 * pad, h: rowH - 0.72, fontSize: 11.5, valign: "top", lineSpacingMultiple: 1.08 });

  // --- F-DHT
  card(s, gx, y1, colW, rowH);
  cardTitle(s, gx + pad, y1 + 0.16, colW - 2 * pad, "Integrated search (F-DHT) implemented");
  text(s, "Paper has two modes: tree (depth → head count → heads, step by step) and integrated (depth + head count chosen as one pair). Both trained on full AG News:", {
    x: gx + pad, y: y1 + 0.54, w: colW - 2 * pad, h: 0.62, fontSize: 11, valign: "top",
  });
  smallTable(s, gx + pad, y1 + 1.2, [2.1, 1.7, 1.67], [
    ["", "Test macro F1", "Compute vs BERT"],
    ["Full BERT", ".949", "1.00×"],
    ["Tree (T-DHT)", ".940", ".39×"],
    ["Integrated (F-DHT)", ".941", ".43×"],
  ]);
  text(s, "They agree on 97.7% of predictions but pick the same depth only 51% of the time.", {
    x: gx + pad, y: y1 + 2.3, w: colW - 2 * pad, h: 0.26, fontSize: 10.5, italic: true, color: MUTED,
  });

  // --- TS
  card(s, M, y2, colW, rowH);
  cardTitle(s, M + pad, y2 + 0.16, colW - 2 * pad, "Thompson sampling, explained");
  text(s, bullets([
    ["Flavor: ", "contextual, top-K Thompson sampling. Each head has its own Bayesian logistic regression (diagonal Gaussian posterior, Laplace updates; Chapelle & Li 2011)."],
    ["Each step: ", "draw a random weight vector per head, score all 12, keep the top K; after the batch, kept heads share one 0/1 reward."],
    ["Finding: ", "learned head scores sit between 0.34 and 0.37, so the random draw, not the model, mostly decides which heads are dropped."],
  ]), { x: M + pad, y: y2 + 0.58, w: colW - 2 * pad, h: rowH - 0.72, fontSize: 11.5, valign: "top", lineSpacingMultiple: 1.08 });

  // --- speed
  card(s, gx, y2, colW, rowH, PANEL);
  cardTitle(s, gx + pad, y2 + 0.16, colW - 2 * pad, "Parallel execution fix");
  text(s, "Samples used to run one at a time. Now the whole batch runs together (each sample's heads gathered and batched), with bandit features pooled on the GPU. Decisions verified identical.", {
    x: gx + pad, y: y2 + 0.54, w: colW - 2 * pad, h: 0.62, fontSize: 11, valign: "top",
  });
  smallTable(s, gx + pad, y2 + 1.2, [2.1, 1.7, 1.67], [
    ["T-DHT speed vs full BERT", "Before", "After"],
    ["Training", "0.59×", "1.60×"],
    ["Inference (batch 32)", "0.26×", "1.12×"],
  ]);
  text(s, "Fewer FLOPs now means less wall-clock time, in both training and inference.", {
    x: gx + pad, y: y2 + 2.03, w: colW - 2 * pad, h: 0.26, fontSize: 10.5, italic: true, color: MUTED,
  });

  footer(s, "Our runs: AG News, BERT-base, seed 7, one public-test evaluation per selected checkpoint. Speed: Apple M2 Max (MPS). Paper reports F-DHT .943 / T-DHT .945 F1.");

  s.addNotes([
    "Four things since last time.",
    "EDA: I profiled all four datasets from the paper. The notable one: AG News headlines are short, so most of each input is padding, and I built per-sentence walkthroughs comparing full BERT and DHT.",
    "Integrated search: the paper's second mode. Both modes land about 0.9 points under full BERT at roughly 40% of the compute. They almost always agree on the answer but often disagree on depth.",
    "Thompson sampling: it's contextual top-K Thompson sampling with a Bayesian logistic model per head. The interesting part is that the heads end up barely distinguishable, so head choice is mostly exploration.",
    "Speed: before, DHT was slower than full BERT despite fewer FLOPs because every sample ran alone. Batching fixed that: now 1.6x faster in training and 1.12x at inference, same decisions.",
  ].join(" "));
}

// ================================================================ 3. NEXT
{
  const s = pres.addSlide();
  titleBar(s, "Next: from classification to generation", "Route each tool-calling request to a smaller network");

  // data strip
  text(s, [
    { text: "Model  ", options: { bold: true, color: ACCENT } },
    { text: "Qwen2.5-0.5B-Instruct (24 layers)      ", options: { color: BODY } },
    { text: "Train  ", options: { bold: true, color: ACCENT } },
    { text: "xlam-function-calling-60k (60k verified calls, ~3,600 APIs)      ", options: { color: BODY } },
    { text: "Evaluate  ", options: { bold: true, color: ACCENT } },
    { text: "BFCL single-turn (simple → parallel-multiple, plus irrelevance)", options: { color: BODY } },
  ], { x: M, y: 1.44, w: CW, h: 0.5, fontSize: 11.5, valign: "top", lineSpacingMultiple: 1.1 });

  // left: examples
  const lx = M, lw = 6.35;
  text(s, "What the model sees: tools as JSON schemas in the prompt, then the user request", {
    x: lx, y: 2.02, w: lw, h: 0.26, fontSize: 11.5, bold: true, color: INK,
  });

  text(s, "EASY · BFCL simple_0", { x: lx, y: 2.34, w: lw, h: 0.22, fontSize: 9.5, bold: true, charSpacing: 1, color: MUTED });
  codeBlock(s, lx, 2.58, lw, 1.08, [
    { t: "tools:  calculate_triangle_area(base: int, height: int, unit?: str)", c: SECOND },
    { t: "user:   \"Find the area of a triangle with a base of 10 units" },
    { t: "         and height of 5 units.\"" },
    { t: "target: calculate_triangle_area(base=10, height=5)", c: ACCENT, b: true },
  ], 10);

  text(s, "HARDER · BFCL parallel_multiple_1 (one distractor tool, two calls)", { x: lx, y: 3.78, w: lw, h: 0.22, fontSize: 9.5, bold: true, charSpacing: 1, color: MUTED });
  codeBlock(s, lx, 4.02, lw, 1.62, [
    { t: "tools:  volume_cylinder.calculate(radius, height)", c: SECOND },
    { t: "        area_rectangle.calculate(length, breadth)", c: SECOND },
    { t: "        area_circle.calculate(radius)", c: SECOND },
    { t: "user:   \"Find the area of a rectangle with length 7 and breadth 3." },
    { t: "         Also, calculate the area of a circle with radius 5.\"" },
    { t: "target: area_rectangle.calculate(length=7.0, breadth=3.0)", c: ACCENT, b: true },
    { t: "        area_circle.calculate(radius=5.0)", c: ACCENT, b: true },
  ], 10);

  text(s, bullets([
    ["Trap case: ", "\"Solve the quadratic a=1, b=2, c=3\" with only math.sum available → correct answer is no call."],
    ["Graded objectively: ", "the call is parsed and checked for the right function, required arguments, and types/values (argument order ignored) → 1 or 0."],
  ]), { x: lx, y: 5.78, w: lw, h: 1.1, fontSize: 10.5, valign: "top", lineSpacingMultiple: 1.05 });

  // right: loss + RL
  const rx = M + lw + 0.3, rw = CW - lw - 0.3, pad = 0.24;
  card(s, rx, 2.02, rw, 1.98);
  cardTitle(s, rx + pad, 2.14, rw - 2 * pad, "Loss and reward");
  text(s, "Network loss: trains the model and its exit heads", { x: rx + pad, y: 2.5, w: rw - 2 * pad, h: 0.22, fontSize: 10.5, color: MUTED });
  text(s, "L = − Σₜ log p( yₜ | prompt, y<ₜ ; architecture a )", {
    x: rx + pad, y: 2.73, w: rw - 2 * pad, h: 0.3, fontFace: MFONT, fontSize: 11.5, color: INK, bold: true,
  });
  text(s, "Router reward: correct call minus a compute price", { x: rx + pad, y: 3.12, w: rw - 2 * pad, h: 0.22, fontSize: 10.5, color: MUTED });
  text(s, "r(a) = 𝟙[call correct] − λ · FLOPs(a) / FLOPs(full)", {
    x: rx + pad, y: 3.35, w: rw - 2 * pad, h: 0.3, fontFace: MFONT, fontSize: 11.5, color: ACCENT, bold: true,
  });
  text(s, "Sweeping λ traces a correctness-vs-compute curve.", { x: rx + pad, y: 3.68, w: rw - 2 * pad, h: 0.24, fontSize: 10.5, italic: true, color: MUTED });

  card(s, rx, 4.14, rw, 2.74, PANEL);
  cardTitle(s, rx + pad, 4.26, rw - 2 * pad, "RL ideas to compare");
  const ideas = [
    ["DHT bandits, ported", "LinUCB picks depth, Thompson sampling picks heads, from the first layers' read of the prompt. Baseline."],
    ["Budgeted bandit", "adjust λ online so average compute stays at a target (primal-dual / bandits with knapsacks)."],
    ["GRPO-style router", "sample several architectures per prompt, grade each call, reinforce the ones above the group average."],
  ];
  ideas.forEach(([h, d], i) => {
    const y = 4.66 + i * 0.66;
    s.addShape(pres.ShapeType.ellipse, { x: rx + pad, y: y + 0.02, w: 0.3, h: 0.3, fill: { color: WHITE }, line: { color: ACCENT, width: 1 } });
    text(s, String(i + 1), { x: rx + pad, y: y + 0.02, w: 0.3, h: 0.3, align: "center", valign: "middle", fontSize: 11, bold: true, color: ACCENT });
    text(s, [
      { text: h + ": ", options: { bold: true, color: INK } },
      { text: d, options: { color: BODY } },
    ], { x: rx + pad + 0.42, y, w: rw - 2 * pad - 0.42, h: 0.62, fontSize: 10.5, valign: "top", lineSpacingMultiple: 1.03 });
  });
  text(s, [
    { text: "Key control: ", options: { bold: true, color: INK } },
    { text: "a fixed smaller model at the same average compute.", options: { color: BODY } },
  ], { x: rx + pad, y: 6.6, w: rw - 2 * pad, h: 0.24, fontSize: 10.5 });

  footer(s, "Examples are real BFCL items (gorilla-llm/Berkeley-Function-Calling-Leaderboard). Training data: Salesforce APIGen / xlam-function-calling-60k.");

  s.addNotes([
    "Next step: move from classification to generation, specifically tool calling, because it's objectively graded.",
    "Each request comes with its own tool list. Easy ones need one obvious call; harder ones need picking the right tools and making several calls; trap cases need no call at all. That spread in difficulty is what gives a router something to learn.",
    "BFCL grades by parsing the call and checking function, arguments and values, so the reward is a clean 1 or 0, which is exactly the shape of DHT's bandit feedback.",
    "Two signals: cross-entropy on the target call trains the network; the router gets correct minus a compute price. Sweeping lambda gives a curve instead of a single point.",
    "Three router trainers to compare, from the paper's bandits up to a GRPO-style policy gradient. The control that matters is a static smaller model at the same compute: if routing can't beat that, it adds nothing.",
  ].join(" "));
}

// ================================================================ 4. ADVICE
{
  const s = pres.addSlide();
  titleBar(s, "Where I'd like advice", "Which approach is promising and still fast?");

  text(s, "When should the router decide how much of the network to use?", {
    x: M, y: 1.5, w: CW, h: 0.3, fontFace: HFONT, fontSize: 17, bold: true, color: INK,
  });

  const opts = [
    {
      title: "Once per request",
      desc: "Decided from the prompt; the whole response uses the same layers and heads.",
      view: "Essentially no overhead and batches cleanly.",
      strip: Array(14).fill(MID),
    },
    {
      title: "At a few break points",
      desc: "e.g. one setting while reasoning, another once the model starts writing the tool call.",
      view: "Adapts better; may cost more than it saves.",
      strip: [...Array(8).fill(SHALLOW), ...Array(6).fill(ACCENT)],
    },
    {
      title: "Every token",
      desc: "Each generated token picks its own depth.",
      view: "Most flexible; usually loses the real-world speedup.",
      strip: [MID, SHALLOW, ACCENT, SHALLOW, MID, ACCENT, SHALLOW, SHALLOW, ACCENT, MID, SHALLOW, ACCENT, MID, SHALLOW],
    },
  ];
  const cw = (CW - 2 * 0.25) / 3, cy = 1.95, ch = 2.55, pad = 0.24;
  opts.forEach((o, i) => {
    const x = M + i * (cw + 0.25);
    card(s, x, cy, cw, ch, i === 0 ? PANEL : CARD);
    cardTitle(s, x + pad, cy + 0.16, cw - 2 * pad, o.title);
    const cell = 0.2, gap = 0.06;
    o.strip.forEach((c, j) => {
      s.addShape(pres.ShapeType.rect, { x: x + pad + j * (cell + gap), y: cy + 0.62, w: cell, h: cell, fill: { color: c }, line: { color: c, width: 0.5 } });
    });
    text(s, o.desc, { x: x + pad, y: cy + 1.0, w: cw - 2 * pad, h: 0.72, fontSize: 12, valign: "top", lineSpacingMultiple: 1.08 });
    text(s, o.view, { x: x + pad, y: cy + 1.82, w: cw - 2 * pad, h: 0.6, fontSize: 11.5, italic: true, color: MUTED, valign: "top" });
  });
  text(s, "Each square is one generated token; darker = more layers run.", {
    x: M, y: cy + ch + 0.08, w: CW, h: 0.24, fontSize: 9.5, color: MUTED,
  });

  text(s, "How should the router be trained?", {
    x: M, y: 4.95, w: CW, h: 0.3, fontFace: HFONT, fontSize: 17, bold: true, color: INK,
  });
  const chips = ["Contextual bandit, as in DHT", "Bandit with a compute budget", "Policy-gradient RL (GRPO-style)"];
  chips.forEach((c, i) => {
    const x = M + i * (cw + 0.25);
    s.addShape(pres.ShapeType.roundRect, { x, y: 5.36, w: cw, h: 0.42, rectRadius: 0.06, fill: { color: WHITE }, line: { color: SECOND, width: 1 } });
    text(s, c, { x, y: 5.36, w: cw, h: 0.42, align: "center", valign: "middle", fontSize: 12.5, bold: true, color: SECOND });
  });

  card(s, M, 6.0, CW, 0.88, PANEL);
  text(s, [
    { text: "1.  ", options: { bold: true, color: ACCENT } },
    { text: "Are there papers or approaches you'd recommend that save real time, not just on paper?", options: { color: INK, breakLine: true } },
    { text: "2.  ", options: { bold: true, color: ACCENT } },
    { text: "Are there any you'd steer me away from?", options: { color: INK } },
  ], { x: M + 0.28, y: 6.08, w: CW - 0.56, h: 0.72, fontSize: 13.5, valign: "middle", lineSpacingMultiple: 1.15 });

  s.addNotes([
    "This is where I'd like your input.",
    "The main design choice is how often the router decides. Once per request is cheapest and keeps batching intact. Deciding at a few break points, like switching when the model starts writing the tool call, could adapt better but may cost more than it saves. Per-token routing is the most flexible, but what I've read says it usually loses the real speedup.",
    "Second, how to train the router: the paper's contextual bandit, a budgeted bandit, or a GRPO-style policy gradient.",
    "My two questions: any papers or approaches you'd recommend that give real wall-clock savings, and any you'd steer me away from?",
  ].join(" "));
}

pres.writeFile({ fileName: process.argv[2] }).then((f) => console.log("wrote", f));
