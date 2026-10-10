/*
 * Optimizer Playground — 画面と描画
 * MIT License
 */
(function () {
  'use strict';
  const { FUNCTIONS, METHODS, METHOD_ORDER, createRun, stepRun, makeProblem } = window.Optim;
  const Expr = window.Expr;

  const MAX_ITER = 5000;
  const STORE_KEY = 'optimizer-playground.problems.v1';
  const DEFAULT_ON = { gd: true, momentum: true, nesterov: false, rmsprop: false, adam: true, newton: true, damped: true };

  const $ = (id) => document.getElementById(id);
  const plot = $('plot');
  const pctx = plot.getContext('2d');
  const lossCanvas = $('loss');
  const lctx = lossCanvas.getContext('2d');

  // 問題：{ id, builtin, def（自作のみ。保存用）, fn（計算用）, start, lr }
  const app = {
    problems: [],
    cur: null,
    enabled: Object.assign({}, DEFAULT_ON),
    runs: {},
    playing: false,
    speed: 4,
    map: null,
    dpr: 1,
    editing: null, // 編集中の問題 id（新規は null）
  };
  const P = () => app.cur;
  const fn = () => app.cur.fn;

  // ---------- 保存 ----------
  function loadStore() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || '[]') || []; } catch (e) { return []; }
  }
  function saveStore() {
    const defs = app.problems.filter((p) => !p.builtin).map((p) => Object.assign({}, p.def, { start: p.start, lr: p.lr }));
    try { localStorage.setItem(STORE_KEY, JSON.stringify(defs)); } catch (e) { /* 保存できない環境では何もしない */ }
  }
  function uid() { return 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }

  function buildCustom(def) {
    const f = makeProblem(def, Expr);
    return { id: def.id, builtin: false, def, fn: f, start: (def.start || f.start).slice(), lr: Object.assign({}, f.lr) };
  }
  function initProblems() {
    for (const [k, F] of Object.entries(FUNCTIONS)) {
      app.problems.push({ id: k, builtin: true, fn: F, start: F.start.slice(), lr: Object.assign({}, F.lr) });
    }
    for (const def of loadStore()) {
      try { app.problems.push(buildCustom(def)); } catch (e) { console.warn('読み込めない自作関数を飛ばしました', def, e); }
    }
  }

  // ---------- 共有URL ----------
  function encodeDef(def) {
    const d = { n: def.name, e: def.expr, r: def.range, s: def.start };
    if (def.fmin != null && def.fmin !== '') d.f = def.fmin;
    if (def.minima && def.minima.length) d.m = def.minima;
    if (def.note) d.t = def.note;
    const bytes = new TextEncoder().encode(JSON.stringify(d));
    let bin = '';
    bytes.forEach((b) => (bin += String.fromCharCode(b)));
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function decodeDef(s) {
    const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/'));
    const d = JSON.parse(new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0))));
    const num4 = Array.isArray(d.r) && d.r.length === 4 && d.r.every((v) => isFinite(v));
    const num2 = Array.isArray(d.s) && d.s.length === 2 && d.s.every((v) => isFinite(v));
    if (typeof d.e !== 'string' || d.e.length > 500 || !num4 || !num2) throw new Error('bad');
    return {
      name: String(d.n || '共有された関数').slice(0, 40), expr: d.e, range: d.r.map(Number), start: d.s.map(Number),
      fmin: d.f != null ? Number(d.f) : null,
      minima: Array.isArray(d.m) ? d.m.filter((p) => Array.isArray(p) && p.length === 2).map((p) => p.map(Number)).slice(0, 20) : [],
      note: d.t ? String(d.t).slice(0, 160) : '',
    };
  }
  function importFromHash() {
    const m = /[#&]p=([A-Za-z0-9_-]+)/.exec(location.hash);
    if (!m) return null;
    try {
      const def = decodeDef(m[1]);
      const same = app.problems.find((p) => !p.builtin && p.def.expr === def.expr && JSON.stringify(p.def.range) === JSON.stringify(def.range));
      if (same) return same.id;
      def.id = uid();
      app.problems.push(buildCustom(def));
      saveStore();
      toast('共有された関数を追加しました');
      return def.id;
    } catch (e) {
      toast('URLの関数を読み込めませんでした');
      return null;
    }
  }

  let toastTimer = 0;
  function toast(msg) {
    $('toast').textContent = msg;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => ($('toast').textContent = ''), 4000);
  }

  // ---------- 色 ----------
  const colorProbe = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  function rgba(css) {
    colorProbe.clearRect(0, 0, 1, 1);
    colorProbe.fillStyle = '#000';
    colorProbe.fillStyle = css;
    colorProbe.fillRect(0, 0, 1, 1);
    const d = colorProbe.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2], d[3] / 255];
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  // ---------- 座標変換 ----------
  function toPx(x, y) {
    const [x0, x1, y0, y1] = fn().range;
    return [(x - x0) / (x1 - x0) * plot.width, (1 - (y - y0) / (y1 - y0)) * plot.height];
  }
  function toWorld(px, py) {
    const [x0, x1, y0, y1] = fn().range;
    return [x0 + px / plot.width * (x1 - x0), y0 + (1 - py / plot.height) * (y1 - y0)];
  }

  // ---------- 等高線（帯状に塗り分けて境界を線にする） ----------
  function buildMap() {
    const W = plot.width, H = plot.height;
    if (!W || !H || !app.cur) return;
    const F = fn();
    const [x0, x1, y0, y1] = F.range;
    const vals = new Float32Array(W * H);
    let vmin = Infinity, vmax = -Infinity;
    for (let j = 0; j < H; j++) {
      const y = y1 - (j + 0.5) / H * (y1 - y0);
      for (let i = 0; i < W; i++) {
        const x = x0 + (i + 0.5) / W * (x1 - x0);
        const fv = F.f(x, y);
        const v = isFinite(fv) ? Math.log10(Math.max(fv - F.fmin, 0) + 1e-2) : NaN;
        vals[j * W + i] = v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    const lo = rgba(cssVar('--map-lo')), hi = rgba(cssVar('--map-hi')), edge = rgba(cssVar('--map-edge'));
    const BANDS = 22;
    const span = vmax - vmin || 1;
    const band = new Int16Array(W * H);
    for (let k = 0; k < vals.length; k++) band[k] = isNaN(vals[k]) ? -1 : Math.min(BANDS - 1, Math.floor((vals[k] - vmin) / span * BANDS));
    const off = document.createElement('canvas');
    off.width = W; off.height = H;
    const octx = off.getContext('2d');
    const img = octx.createImageData(W, H);
    const d = img.data;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const k = j * W + i;
        const p = k * 4;
        if (band[k] < 0) { // 定義域の外（log の負など）は斜線で示す
          const on = ((i + j) >> 3) & 1;
          d[p] = edge[0]; d[p + 1] = edge[1]; d[p + 2] = edge[2]; d[p + 3] = on ? 70 : 25;
          continue;
        }
        const t = Math.pow((vals[k] - vmin) / span, 1.6);
        let r = lo[0] + (hi[0] - lo[0]) * t;
        let g = lo[1] + (hi[1] - lo[1]) * t;
        let b = lo[2] + (hi[2] - lo[2]) * t;
        const isEdge = (i + 1 < W && band[k + 1] !== band[k]) || (j + 1 < H && band[k + W] !== band[k]);
        if (isEdge) {
          const a = edge[3];
          r = r * (1 - a) + edge[0] * a;
          g = g * (1 - a) + edge[1] * a;
          b = b * (1 - a) + edge[2] * a;
        }
        d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    app.map = off;
  }

  // ---------- 実行の管理 ----------
  function resetRuns() {
    const F = fn();
    app.runs = {};
    for (const k of METHOD_ORDER) app.runs[k] = createRun(k, F, P().start, { lr: P().lr[k] });
    updateTable();
    draw();
  }
  function stepAll(n) {
    const F = fn();
    for (let i = 0; i < n; i++) {
      for (const k of METHOD_ORDER) {
        const r = app.runs[k];
        if (r.iter < MAX_ITER) stepRun(r, F);
      }
    }
  }
  // f* が推定値のとき、手法がそれより低い点を見つけたら f* を下げる
  function lowerEstimate(p, runs) {
    const F = p.fn;
    if (p.builtin || !F.fminEstimated) return false;
    let best = null;
    for (const r of runs) {
      if (r.status === 'diverged') continue;
      const v = r.vals[r.vals.length - 1];
      if (isFinite(v) && v < F.fmin - 1e-12 * (1 + Math.abs(F.fmin)) && (!best || v < best.v)) {
        best = { v, x: r.state.x, y: r.state.y };
      }
    }
    if (!best) return false;
    F.fmin = best.v;
    if (!(p.def.minima && p.def.minima.length)) F.minima = [[best.x, best.y]];
    updateNote();
    return true;
  }
  let mapTimer = 0;
  function afterSteps() {
    if (lowerEstimate(P(), Object.values(app.runs))) {
      clearTimeout(mapTimer);
      mapTimer = setTimeout(() => { buildMap(); draw(); }, 300);
    }
  }
  function allDone() {
    return METHOD_ORDER.every((k) => !app.enabled[k] || app.runs[k].status !== 'running' || app.runs[k].iter >= MAX_ITER);
  }

  function selectProblem(id) {
    const p = app.problems.find((q) => q.id === id) || app.problems[0];
    app.cur = p;
    updateNote();
    renderProblemList();
    renderFormula();
    buildTable();
    buildMap();
    resetRuns();
    markBatchCurrent();
  }
  function updateNote() {
    const p = app.cur, F = p.fn;
    let note = F.note || '';
    if (!p.builtin && F.fminEstimated) note += (note ? '　' : '') + `f* = ${fmtNum(F.fmin)} は推定値です（表示範囲の中で探し、手法がもっと低い点を見つけたら更新します）。`;
    $('funcNote').textContent = note;
  }
  function fmtNum(v) { return Math.abs(v) < 1e-12 ? '0' : (+v.toPrecision(6)).toString(); }

  // ---------- 問題の一覧 ----------
  function renderProblemList() {
    const box = $('problemList');
    box.innerHTML = '';
    for (const p of app.problems) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.setAttribute('role', 'option');
      b.setAttribute('aria-selected', p === app.cur ? 'true' : 'false');
      b.textContent = p.fn.label;
      if (!p.builtin) { const s = document.createElement('span'); s.className = 'mine'; s.textContent = '自作'; b.appendChild(s); }
      b.addEventListener('click', () => { if (p !== app.cur) { setPlaying(false); closeEditor(); selectProblem(p.id); } });
      box.appendChild(b);
    }
    const custom = !app.cur.builtin;
    $('editBtn').hidden = !custom;
    $('shareBtn').hidden = !custom;
    $('delBtn').hidden = !custom;
  }
  function renderFormula() {
    const F = fn();
    const text = F.exprText || F.expr || '';
    const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    $('formula').innerHTML = text ? `<b>f(x, y) = ${esc(text)}</b>` : '';
  }

  // ---------- 自作関数のエディタ ----------
  function openEditor(mode) {
    // mode: 'new' | 'edit' | 'dup'
    const ed = $('editor');
    const p = app.cur;
    let def;
    if (mode === 'new') {
      def = { name: '', expr: '', range: [-3, 3, -3, 3], start: [-2.5, 2], fmin: '', minima: [], note: '' };
    } else {
      const F = p.fn;
      def = {
        name: (mode === 'dup' ? `${F.label} のコピー` : F.label),
        expr: F.expr,
        range: F.range.slice(),
        start: p.start.slice(),
        fmin: (p.builtin || !F.fminEstimated) ? F.fmin : '',
        minima: (p.builtin || (p.def && p.def.minima && p.def.minima.length)) ? F.minima : [],
        note: p.builtin ? '' : (p.def.note || ''),
      };
    }
    app.editing = mode === 'edit' ? p.id : null;
    $('editorTitle').textContent = mode === 'edit' ? '自作の関数を編集' : '自作の関数を追加';
    $('edName').value = def.name;
    $('edExpr').value = def.expr;
    [$('edX0').value, $('edX1').value, $('edY0').value, $('edY1').value] = def.range.map(fmtNum);
    [$('edSX').value, $('edSY').value] = def.start.map((v) => fmtNum(+v.toFixed(4)));
    $('edFmin').value = def.fmin === '' || def.fmin == null ? '' : fmtNum(def.fmin);
    $('edMinima').value = def.minima.map((q) => `${fmtNum(+q[0].toFixed(6))}, ${fmtNum(+q[1].toFixed(6))}`).join(' ; ');
    $('edNote').value = def.note;
    ed.hidden = false;
    validateEditor();
    ed.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    $('edExpr').focus({ preventScroll: true });
  }
  function closeEditor() { $('editor').hidden = true; app.editing = null; }

  function readEditor() {
    const errs = [];
    const num = (id, label) => {
      const el = $(id);
      const v = parseFloat(el.value);
      const ok = el.value.trim() !== '' && isFinite(v);
      el.classList.toggle('invalid', !ok);
      if (!ok) errs.push(`${label}に数を入れてください`);
      return v;
    };
    const expr = $('edExpr').value.trim();
    const range = [num('edX0', 'x の最小'), num('edX1', 'x の最大'), num('edY0', 'y の最小'), num('edY1', 'y の最大')];
    if (!errs.length) {
      if (!(range[0] < range[1])) { errs.push('x の最小は最大より小さくしてください'); $('edX0').classList.add('invalid'); }
      if (!(range[2] < range[3])) { errs.push('y の最小は最大より小さくしてください'); $('edY0').classList.add('invalid'); }
    }
    const start = [num('edSX', '開始点 x'), num('edSY', '開始点 y')];
    let fmin = '';
    const fs = $('edFmin').value.trim();
    $('edFmin').classList.remove('invalid');
    if (fs !== '') {
      if (isFinite(parseFloat(fs))) fmin = parseFloat(fs);
      else { errs.push('f* は数か空欄にしてください'); $('edFmin').classList.add('invalid'); }
    }
    const minima = [];
    const ms = $('edMinima').value.trim();
    $('edMinima').classList.remove('invalid');
    if (ms) {
      for (const part of ms.split(/[;；]/)) {
        if (!part.trim()) continue;
        const xy = part.split(/[,，\s]+/).filter(Boolean).map(Number);
        if (xy.length === 2 && xy.every(isFinite)) minima.push(xy);
        else { errs.push('最小点は「x, y ; x, y」の形で書いてください'); $('edMinima').classList.add('invalid'); break; }
      }
    }
    return {
      def: { name: $('edName').value.trim(), expr, range, start, fmin, minima, note: $('edNote').value.trim() },
      errs,
    };
  }
  function validateEditor() {
    const out = $('edParsed');
    const src = $('edExpr').value;
    const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    if (!src.trim()) {
      out.innerHTML = '<span>式を入れると、ここに読み取った形と勾配が出ます</span>';
      $('edExpr').classList.remove('invalid');
      return null;
    }
    try {
      const c = Expr.compile(src);
      $('edExpr').classList.remove('invalid');
      out.innerHTML = `f = ${esc(c.text)}<br>∂f/∂x = ${esc(c.dxText)}<br>∂f/∂y = ${esc(c.dyText)}`;
      return c;
    } catch (e) {
      $('edExpr').classList.add('invalid');
      const pos = e.pos != null ? e.pos : 0;
      out.innerHTML = `<span class="err">${esc(e.message)}</span><br>${esc(src)}<br><span class="caret">${' '.repeat(Math.min(pos, src.length))}^</span>`;
      return null;
    }
  }
  function saveEditor(ev) {
    ev.preventDefault();
    const c = validateEditor();
    const { def, errs } = readEditor();
    if (!c) { toast('式を確認してください'); return; }
    if (errs.length) { toast(errs[0]); return; }
    if (!def.name) def.name = c.text.length > 24 ? c.text.slice(0, 22) + '…' : c.text;
    let prob;
    try {
      if (app.editing) {
        const old = app.problems.find((p) => p.id === app.editing);
        def.id = old.id;
        const sameExpr = old.def.expr === def.expr && JSON.stringify(old.def.range) === JSON.stringify(def.range);
        if (sameExpr) def.lr = old.lr; // 式と範囲が同じなら学習率の調整を引き継ぐ
        prob = buildCustom(def);
        app.problems[app.problems.indexOf(old)] = prob;
      } else {
        def.id = uid();
        prob = buildCustom(def);
        app.problems.push(prob);
      }
    } catch (e) {
      toast(e.message || '関数を作れませんでした');
      return;
    }
    saveStore();
    setPlaying(false);
    closeEditor();
    selectProblem(prob.id);
    toast('保存しました');
  }
  function deleteCurrent() {
    const p = app.cur;
    if (p.builtin) return;
    if (!window.confirm(`「${p.fn.label}」を削除しますか？`)) return;
    app.problems.splice(app.problems.indexOf(p), 1);
    saveStore();
    setPlaying(false);
    closeEditor();
    selectProblem(app.problems[0].id);
    toast('削除しました');
  }
  async function shareCurrent() {
    const p = app.cur;
    if (p.builtin) return;
    const def = Object.assign({}, p.def, { start: p.start });
    const url = `${location.origin}${location.pathname}#p=${encodeDef(def)}`;
    try {
      await navigator.clipboard.writeText(url);
      toast('この関数を開くURLをコピーしました');
    } catch (e) {
      window.prompt('このURLで関数を共有できます', url);
    }
  }

  // ---------- 描画 ----------
  function draw() {
    const W = plot.width, H = plot.height;
    if (!W || !app.cur) return;
    pctx.clearRect(0, 0, W, H);
    if (app.map) pctx.drawImage(app.map, 0, 0);
    const dpr = app.dpr;
    const halo = cssVar('--halo');
    const marker = cssVar('--marker');
    const F = fn();

    for (const [mx, my] of F.minima) {
      const [px, py] = toPx(mx, my);
      const s = 6 * dpr;
      pctx.strokeStyle = halo; pctx.lineWidth = 4 * dpr;
      cross(px, py, s);
      pctx.strokeStyle = marker; pctx.lineWidth = 1.6 * dpr;
      cross(px, py, s);
    }

    for (const k of METHOD_ORDER) {
      if (!app.enabled[k]) continue;
      const r = app.runs[k];
      const col = METHODS[k].color;
      const path = r.path;
      pctx.lineJoin = 'round'; pctx.lineCap = 'round';
      for (const pass of [0, 1]) {
        pctx.beginPath();
        for (let i = 0; i < path.length; i++) {
          const [px, py] = toPx(path[i][0], path[i][1]);
          if (i === 0) pctx.moveTo(px, py); else pctx.lineTo(px, py);
        }
        pctx.strokeStyle = pass === 0 ? halo : col;
        pctx.lineWidth = (pass === 0 ? 4.5 : 2) * dpr;
        pctx.stroke();
      }
      if (!METHODS[k].hasLr && path.length < 300) {
        pctx.fillStyle = col;
        for (const p of path) {
          const [px, py] = toPx(p[0], p[1]);
          pctx.beginPath(); pctx.arc(px, py, 2.4 * dpr, 0, Math.PI * 2); pctx.fill();
        }
      }
      const last = path[path.length - 1];
      const [lx, ly] = toPx(last[0], last[1]);
      pctx.beginPath();
      pctx.arc(lx, ly, 5.5 * dpr, 0, Math.PI * 2);
      pctx.fillStyle = col; pctx.fill();
      pctx.lineWidth = 2 * dpr; pctx.strokeStyle = halo; pctx.stroke();
    }

    const [sx, sy] = toPx(P().start[0], P().start[1]);
    pctx.beginPath();
    pctx.arc(sx, sy, 7 * dpr, 0, Math.PI * 2);
    pctx.lineWidth = 4.5 * dpr; pctx.strokeStyle = halo; pctx.stroke();
    pctx.lineWidth = 2 * dpr; pctx.strokeStyle = marker; pctx.stroke();

    drawLoss();
    $('startVal').textContent = `(${P().start[0].toFixed(3)}, ${P().start[1].toFixed(3)})`;
  }
  function cross(x, y, s) {
    pctx.beginPath();
    pctx.moveTo(x - s, y - s); pctx.lineTo(x + s, y + s);
    pctx.moveTo(x + s, y - s); pctx.lineTo(x - s, y + s);
    pctx.stroke();
  }

  function drawLoss() {
    const W = lossCanvas.width, H = lossCanvas.height;
    if (!W) return;
    const dpr = app.dpr;
    lctx.clearRect(0, 0, W, H);
    const muted = cssVar('--muted'), line = cssVar('--line');
    const padL = 44 * dpr, padR = 10 * dpr, padT = 8 * dpr, padB = 22 * dpr;
    const lg = (v) => Math.log10(Math.max(v, 1e-16));
    const fmin = fn().fmin;
    let ymin = Infinity, ymax = -Infinity, xmax = 10;
    for (const k of METHOD_ORDER) {
      if (!app.enabled[k]) continue;
      const r = app.runs[k];
      xmax = Math.max(xmax, r.vals.length - 1);
      for (const v of r.vals) { const y = lg(v - fmin); if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
    }
    if (!isFinite(ymin)) { ymin = -2; ymax = 2; }
    ymin = Math.floor(ymin); ymax = Math.ceil(ymax);
    if (ymax - ymin < 2) ymax = ymin + 2;
    const X = (i) => padL + i / xmax * (W - padL - padR);
    const Y = (v) => padT + (1 - (v - ymin) / (ymax - ymin)) * (H - padT - padB);

    lctx.font = `${10 * dpr}px ui-monospace, Menlo, monospace`;
    lctx.fillStyle = muted;
    lctx.strokeStyle = line;
    lctx.lineWidth = 1 * dpr;
    const stepY = Math.max(1, Math.ceil((ymax - ymin) / 6));
    lctx.textAlign = 'right'; lctx.textBaseline = 'middle';
    for (let e = ymin; e <= ymax; e += stepY) {
      const y = Y(e);
      lctx.beginPath(); lctx.moveTo(padL, y); lctx.lineTo(W - padR, y); lctx.stroke();
      lctx.fillText(`1e${e}`, padL - 4 * dpr, y);
    }
    lctx.textAlign = 'center'; lctx.textBaseline = 'top';
    lctx.fillText('0', X(0), H - padB + 5 * dpr);
    lctx.fillText(String(xmax), X(xmax) - 10 * dpr, H - padB + 5 * dpr);
    lctx.fillText('反復', (padL + W - padR) / 2, H - padB + 5 * dpr);

    for (const k of METHOD_ORDER) {
      if (!app.enabled[k]) continue;
      const r = app.runs[k];
      lctx.beginPath();
      r.vals.forEach((v, i) => {
        const x = X(i), y = Y(lg(v - fmin));
        if (i === 0) lctx.moveTo(x, y); else lctx.lineTo(x, y);
      });
      lctx.strokeStyle = METHODS[k].color;
      lctx.lineWidth = 1.8 * dpr;
      lctx.stroke();
    }
  }

  // ---------- 手法の表 ----------
  const STATUS_LABEL = { running: '計算中', converged: '収束', diverged: '発散', stuck: '停止' };
  function buildTable() {
    const tb = $('methodTable').querySelector('tbody');
    tb.innerHTML = '';
    for (const k of METHOD_ORDER) {
      const m = METHODS[k];
      const tr = document.createElement('tr');
      tr.dataset.key = k;
      const chk = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = !!app.enabled[k];
      chk.setAttribute('aria-label', `${m.label}を表示`);
      chk.addEventListener('change', () => { app.enabled[k] = chk.checked; tr.classList.toggle('off', !chk.checked); draw(); });
      const td0 = document.createElement('td'); td0.appendChild(chk);

      const td1 = document.createElement('td');
      td1.innerHTML = `<span class="mname"><span class="swatch" style="background:${m.color}"></span>${m.label}</span>`;

      const td2 = document.createElement('td');
      if (m.hasLr) {
        const inp = document.createElement('input');
        inp.type = 'number'; inp.step = 'any'; inp.min = '0';
        inp.value = P().lr[k];
        inp.setAttribute('aria-label', `${m.label}の学習率`);
        inp.addEventListener('change', () => {
          const v = parseFloat(inp.value);
          if (isFinite(v) && v > 0) {
            P().lr[k] = v;
            if (!P().builtin) saveStore();
            resetRuns();
          } else inp.value = P().lr[k];
        });
        td2.appendChild(inp);
      } else {
        td2.innerHTML = '<span class="dash">—</span>';
      }
      const td3 = document.createElement('td'); td3.className = 'num it';
      const td4 = document.createElement('td'); td4.className = 'num fv';
      const td5 = document.createElement('td'); td5.innerHTML = '<span class="st"></span>';
      tr.append(td0, td1, td2, td3, td4, td5);
      tr.classList.toggle('off', !app.enabled[k]);
      tb.appendChild(tr);
    }
  }
  function updateTable() {
    for (const tr of $('methodTable').querySelectorAll('tbody tr')) {
      const r = app.runs[tr.dataset.key];
      if (!r) continue;
      tr.querySelector('.it').textContent = r.iter;
      const v = r.vals[r.vals.length - 1] - fn().fmin;
      tr.querySelector('.fv').textContent = r.status === 'diverged' ? '∞' : v.toExponential(2);
      const st = tr.querySelector('.st');
      st.className = 'st ' + r.status;
      st.textContent = STATUS_LABEL[r.status] + (r.status === 'running' && r.iter >= MAX_ITER ? '（上限）' : '');
      if (r.reason) st.title = r.reason;
    }
  }

  // ---------- 全問題でまとめて比較 ----------
  let batchToken = 0;
  async function runBatch() {
    const token = ++batchToken;
    const btn = $('batchBtn');
    btn.disabled = true;
    const out = $('batchOut');
    const results = [];
    for (let i = 0; i < app.problems.length; i++) {
      if (token !== batchToken) return;
      const p = app.problems[i];
      btn.textContent = `計算中… ${i + 1}/${app.problems.length}`;
      await new Promise((r) => setTimeout(r, 0));
      const F = p.fn;
      const row = { p, cells: {} };
      const runs = {};
      for (const k of METHOD_ORDER) {
        const r = createRun(k, F, p.start, { lr: p.lr[k] });
        while (r.status === 'running' && r.iter < MAX_ITER) stepRun(r, F);
        runs[k] = r;
      }
      if (lowerEstimate(p, Object.values(runs)) && p === app.cur) { buildMap(); draw(); }
      const tol = 1e-6 * (1 + Math.abs(F.fmin));
      for (const k of METHOD_ORDER) {
        const r = runs[k];
        const gap = r.vals[r.vals.length - 1] - F.fmin;
        let kind;
        if (r.status === 'diverged') kind = 'bad';
        else if (r.status === 'stuck') kind = 'warn';
        else if (gap <= tol) kind = r.status === 'converged' ? 'ok' : 'slowok';
        else kind = r.status === 'converged' ? 'warn' : 'slow';
        row.cells[k] = { kind, iter: r.iter, gap, status: r.status };
      }
      results.push(row);
    }
    renderBatch(results);
    btn.disabled = false;
    btn.textContent = 'もう一度比較する';
  }
  function renderBatch(results) {
    const head = METHOD_ORDER.map((k) => `<th><span class="swatch" style="background:${METHODS[k].color};display:inline-block;vertical-align:-1px;margin-right:4px"></span>${METHODS[k].label}</th>`).join('');
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const rows = results.map(({ p, cells }) => {
      const okIters = METHOD_ORDER.filter((k) => cells[k].kind === 'ok').map((k) => cells[k].iter);
      const best = okIters.length ? Math.min(...okIters) : -1;
      const tds = METHOD_ORDER.map((k) => {
        const c = cells[k];
        switch (c.kind) {
          case 'ok': return `<td class="ok${c.iter === best ? ' best' : ''}">${c.iter}回</td>`;
          case 'slowok': return `<td class="slow">${MAX_ITER}回以上<span class="sub">ほぼ到達</span></td>`;
          case 'slow': return `<td class="slow">上限<span class="sub">差 ${c.gap.toExponential(1)}</span></td>`;
          case 'warn': return `<td class="warn">${c.status === 'stuck' ? '停止' : '別の点で停止'}<span class="sub">${c.iter}回・差 ${c.gap.toExponential(1)}</span></td>`;
          default: return `<td class="bad">発散<span class="sub">${c.iter}回目</span></td>`;
        }
      }).join('');
      return `<tr data-id="${esc(p.id)}"${p === app.cur ? ' class="current"' : ''}><td>${esc(p.fn.label)}${p.builtin ? '' : ' <span class="sub" style="display:inline">自作</span>'}</td>${tds}</tr>`;
    }).join('');
    $('batchOut').innerHTML = `<table class="bt"><thead><tr><th>問題</th>${head}</tr></thead><tbody>${rows}</tbody></table>
      <p class="small">緑＝最小値まで収束（下線は最速）、橙＝最小値ではない点（局所解・鞍点など）で止まった、灰＝${MAX_ITER}回で終わらなかった、赤＝発散。</p>`;
    for (const tr of $('batchOut').querySelectorAll('tbody tr')) {
      tr.addEventListener('click', () => {
        setPlaying(false); closeEditor(); selectProblem(tr.dataset.id);
        $('canvasWrap').scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
  }
  function markBatchCurrent() {
    for (const tr of $('batchOut').querySelectorAll('tbody tr')) tr.classList.toggle('current', app.cur && tr.dataset.id === app.cur.id);
  }

  // ---------- 再生 ----------
  let raf = 0;
  function loop() {
    stepAll(app.speed);
    afterSteps();
    updateTable();
    draw();
    if (allDone()) { setPlaying(false); return; }
    raf = requestAnimationFrame(loop);
  }
  function setPlaying(on) {
    app.playing = on;
    $('playBtn').textContent = on ? '⏸ 一時停止' : '▶ 再生';
    cancelAnimationFrame(raf);
    if (on) {
      if (allDone()) resetRuns();
      raf = requestAnimationFrame(loop);
    }
  }

  // ---------- サイズ ----------
  function resize() {
    app.dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const r = plot.getBoundingClientRect();
    const w = Math.round(r.width * app.dpr), h = Math.round(r.height * app.dpr);
    if (w !== plot.width || h !== plot.height) {
      plot.width = w; plot.height = h;
      buildMap();
    }
    const lr = lossCanvas.getBoundingClientRect();
    lossCanvas.width = Math.round(lr.width * app.dpr);
    lossCanvas.height = Math.round(lr.height * app.dpr);
    draw();
  }

  // ---------- PNG保存（凡例つき） ----------
  function savePng() {
    const out = document.createElement('canvas');
    out.width = plot.width; out.height = plot.height;
    const c = out.getContext('2d');
    c.drawImage(plot, 0, 0);
    const dpr = app.dpr;
    const items = METHOD_ORDER.filter((k) => app.enabled[k]);
    c.font = `${12 * dpr}px system-ui, sans-serif`;
    const lineH = 18 * dpr;
    const title = fn().label;
    const boxW = Math.max(c.measureText(title).width + 20 * dpr, ...items.map((k) => c.measureText(METHODS[k].label).width + 34 * dpr));
    const boxH = items.length * lineH + 30 * dpr;
    c.fillStyle = cssVar('--surface'); c.globalAlpha = 0.9;
    c.fillRect(8 * dpr, 8 * dpr, boxW, boxH);
    c.globalAlpha = 1;
    c.fillStyle = cssVar('--text'); c.textBaseline = 'middle';
    c.fillText(title, 16 * dpr, 20 * dpr);
    items.forEach((k, i) => {
      const y = 20 * dpr + (i + 1) * lineH;
      c.fillStyle = METHODS[k].color;
      c.beginPath(); c.arc(22 * dpr, y, 5 * dpr, 0, Math.PI * 2); c.fill();
      c.fillStyle = cssVar('--text');
      c.fillText(METHODS[k].label, 32 * dpr, y);
    });
    const safe = (P().builtin ? P().id : title).replace(/[\\/:*?"<>|\s]+/g, '_').slice(0, 40);
    out.toBlob((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `optimizer-${safe}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
  }

  // ---------- 初期化 ----------
  function init() {
    initProblems();
    const fromHash = importFromHash();

    const dl = $('rules');
    for (const k of METHOD_ORDER) {
      const dt = document.createElement('dt'); dt.textContent = METHODS[k].label; dt.style.color = METHODS[k].color;
      const dd = document.createElement('dd'); dd.textContent = METHODS[k].rule;
      dl.append(dt, dd);
    }

    $('playBtn').addEventListener('click', () => setPlaying(!app.playing));
    $('stepBtn').addEventListener('click', () => { setPlaying(false); stepAll(1); afterSteps(); updateTable(); draw(); });
    $('resetBtn').addEventListener('click', () => { setPlaying(false); resetRuns(); });
    $('pngBtn').addEventListener('click', savePng);
    const speed = $('speed');
    speed.addEventListener('input', () => { app.speed = +speed.value; $('speedVal').textContent = speed.value; });

    $('addBtn').addEventListener('click', () => openEditor('new'));
    $('dupBtn').addEventListener('click', () => openEditor('dup'));
    $('editBtn').addEventListener('click', () => openEditor('edit'));
    $('delBtn').addEventListener('click', deleteCurrent);
    $('shareBtn').addEventListener('click', shareCurrent);
    $('editor').addEventListener('submit', saveEditor);
    $('edCancel').addEventListener('click', closeEditor);
    let vt = 0;
    $('edExpr').addEventListener('input', () => { clearTimeout(vt); vt = setTimeout(validateEditor, 120); });
    $('batchBtn').addEventListener('click', runBatch);

    plot.addEventListener('click', (e) => {
      const r = plot.getBoundingClientRect();
      const [x, y] = toWorld((e.clientX - r.left) * app.dpr, (e.clientY - r.top) * app.dpr);
      if (!isFinite(fn().f(x, y))) { toast('その点では関数の値が計算できません'); return; }
      P().start = [x, y];
      if (!P().builtin) saveStore();
      const wasPlaying = app.playing;
      setPlaying(false);
      resetRuns();
      if (wasPlaying) setPlaying(true);
    });
    plot.addEventListener('pointermove', (e) => {
      const r = plot.getBoundingClientRect();
      const [x, y] = toWorld((e.clientX - r.left) * app.dpr, (e.clientY - r.top) * app.dpr);
      $('coord').textContent = `x=${x.toFixed(2)}  y=${y.toFixed(2)}  f=${fn().f(x, y).toPrecision(4)}`;
    });
    plot.addEventListener('pointerleave', () => { $('coord').textContent = ''; });

    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && !/INPUT|SELECT|BUTTON|TEXTAREA/.test(document.activeElement.tagName)) {
        e.preventDefault(); setPlaying(!app.playing);
      }
    });

    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    (mq.addEventListener ? mq.addEventListener.bind(mq, 'change') : mq.addListener.bind(mq))(() => { buildMap(); draw(); });

    new ResizeObserver(resize).observe($('canvasWrap'));
    new ResizeObserver(resize).observe(lossCanvas);

    selectProblem(fromHash || app.problems[0].id);
    resize();
  }

  // テストや埋め込み用に状態を見られるようにしておく
  window.OptimizerPlayground = { app, stepAll, resetRuns, selectProblem, runBatch, openEditor };
  init();
})();
