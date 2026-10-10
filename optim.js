/*
 * Optimizer Playground — 計算部分（描画から独立）
 * MIT License
 *
 * 目的関数の定義と、各最適化手法の1ステップ更新をまとめたファイル。
 * ブラウザでは window.Optim、Node では require('./optim.js') で使える。
 */
(function (root) {
  'use strict';

  // ---------- 目的関数 ----------
  // range は正方形（x と y の幅を揃える）にして、等高線がゆがまないようにする。
  // lr は手法ごとの初期学習率。値はその関数で「そこそこ動いて発散しない」目安。
  const FUNCTIONS = {
    rosenbrock: {
      label: 'Rosenbrock（細長い谷）',
      expr: '(1-x)^2 + 100(y-x^2)^2',
      note: '曲がった細い谷の底をたどる問題。勾配法は谷の壁を往復しやすく、ニュートン法は谷の曲がりを見て一気に進む。',
      f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
      grad: (x, y) => [
        -2 * (1 - x) - 400 * x * (y - x * x),
        200 * (y - x * x),
      ],
      range: [-2, 2, -1, 3],
      fmin: 0,
      minima: [[1, 1]],
      start: [-1.5, 2.0],
      lr: { gd: 0.0008, momentum: 0.0008, nesterov: 0.0008, rmsprop: 0.01, adam: 0.05 },
    },
    illcond: {
      label: '悪条件の二次関数',
      expr: '(x^2 + 25y^2)/2',
      note: 'f = (x² + 25y²)/2。縦と横で曲がり方が25倍違う。勾配法は急な方向に学習率を合わせざるをえず、緩い方向がなかなか進まない。',
      f: (x, y) => 0.5 * (x * x + 25 * y * y),
      grad: (x, y) => [x, 25 * y],
      range: [-3, 3, -3, 3],
      fmin: 0,
      minima: [[0, 0]],
      start: [-2.8, 1.6],
      lr: { gd: 0.075, momentum: 0.03, nesterov: 0.03, rmsprop: 0.1, adam: 0.2 },
    },
    himmelblau: {
      label: 'Himmelblau（谷が4つ）',
      expr: '(x^2+y-11)^2 + (x+y^2-7)^2',
      note: '最小値が4か所ある関数。開始点や手法によって、どの谷に落ちるかが変わる。',
      f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
      grad: (x, y) => {
        const a = x * x + y - 11, b = x + y * y - 7;
        return [4 * x * a + 2 * b, 2 * a + 4 * y * b];
      },
      range: [-5, 5, -5, 5],
      fmin: 0,
      minima: [[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]],
      start: [-0.5, -0.3],
      lr: { gd: 0.01, momentum: 0.004, nesterov: 0.004, rmsprop: 0.05, adam: 0.15 },
    },
    saddle: {
      label: '鞍点のある関数',
      expr: 'x^2 + y^4/4 - y^2',
      note: 'f = x² + y⁴/4 − y²。原点は鞍点で、本当の最小は y = ±√2。素のニュートン法は鞍点にも引き寄せられてしまい、減衰ニュートン法はそれを避ける。',
      f: (x, y) => x * x + 0.25 * y ** 4 - y * y,
      grad: (x, y) => [2 * x, y ** 3 - 2 * y],
      range: [-3, 3, -3, 3],
      fmin: -1,
      minima: [[0, Math.SQRT2], [0, -Math.SQRT2]],
      start: [2.5, 0.05],
      lr: { gd: 0.05, momentum: 0.02, nesterov: 0.02, rmsprop: 0.05, adam: 0.1 },
    },
    rastrigin: {
      label: 'Rastrigin（小さな谷だらけ）',
      expr: '20 + x^2 - 10cos(2pi x) + y^2 - 10cos(2pi y)',
      note: '細かい局所解が格子状に並ぶ。局所的な情報だけで動く手法は、近くの小さな谷に捕まりやすい。',
      f: (x, y) => 20 + x * x - 10 * Math.cos(2 * Math.PI * x) + y * y - 10 * Math.cos(2 * Math.PI * y),
      grad: (x, y) => [
        2 * x + 20 * Math.PI * Math.sin(2 * Math.PI * x),
        2 * y + 20 * Math.PI * Math.sin(2 * Math.PI * y),
      ],
      range: [-5.12, 5.12, -5.12, 5.12],
      fmin: 0,
      minima: [[0, 0]],
      start: [-3.3, 2.7],
      lr: { gd: 0.002, momentum: 0.001, nesterov: 0.001, rmsprop: 0.02, adam: 0.1 },
    },
  };

  // ---------- ヘッセ行列（勾配の中心差分） ----------
  function hessian(fn, x, y) {
    const hx = 1e-5 * (1 + Math.abs(x));
    const hy = 1e-5 * (1 + Math.abs(y));
    const gxp = fn.grad(x + hx, y), gxm = fn.grad(x - hx, y);
    const gyp = fn.grad(x, y + hy), gym = fn.grad(x, y - hy);
    const a = (gxp[0] - gxm[0]) / (2 * hx);
    const d = (gyp[1] - gym[1]) / (2 * hy);
    // 非対角は2通りの差分を平均して対称にする
    const b = 0.5 * ((gxp[1] - gxm[1]) / (2 * hx) + (gyp[0] - gym[0]) / (2 * hy));
    return [a, b, b, d]; // [[a,b],[b,d]]
  }

  // 2x2 対称行列の最小固有値
  function minEig(H) {
    const [a, b, , d] = H;
    const tr = a + d, det = a * d - b * b;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    return tr / 2 - disc;
  }

  // (H + lam I) d = -g を解く。解けなければ null
  function solve2(H, g, lam) {
    const a = H[0] + lam, b = H[1], c = H[2], d = H[3] + lam;
    const det = a * d - b * c;
    if (!isFinite(det) || Math.abs(det) < 1e-14) return null;
    return [-(d * g[0] - b * g[1]) / det, -(-c * g[0] + a * g[1]) / det];
  }

  // ---------- 最適化手法 ----------
  // init() で状態を作り、step(state, fn) で1回更新する。
  // state: { x, y, ... }。更新後に state.f, state.gnorm を入れておく。
  const METHODS = {
    gd: {
      label: '最急降下法',
      color: '#f76707',
      hasLr: true,
      rule: 'x ← x − η∇f',
      init: () => ({}),
      step(s, fn, p) {
        const g = fn.grad(s.x, s.y);
        s.x -= p.lr * g[0];
        s.y -= p.lr * g[1];
      },
    },
    momentum: {
      label: 'Momentum',
      color: '#37b24d',
      hasLr: true,
      rule: 'v ← βv − η∇f,  x ← x + v（β = 0.9）',
      init: () => ({ vx: 0, vy: 0 }),
      step(s, fn, p) {
        const g = fn.grad(s.x, s.y);
        s.vx = 0.9 * s.vx - p.lr * g[0];
        s.vy = 0.9 * s.vy - p.lr * g[1];
        s.x += s.vx;
        s.y += s.vy;
      },
    },
    nesterov: {
      label: 'Nesterov',
      color: '#15aabf',
      hasLr: true,
      rule: '先読み位置 x + βv で勾配を取ってから Momentum と同じ更新',
      init: () => ({ vx: 0, vy: 0 }),
      step(s, fn, p) {
        const g = fn.grad(s.x + 0.9 * s.vx, s.y + 0.9 * s.vy);
        s.vx = 0.9 * s.vx - p.lr * g[0];
        s.vy = 0.9 * s.vy - p.lr * g[1];
        s.x += s.vx;
        s.y += s.vy;
      },
    },
    rmsprop: {
      label: 'RMSProp',
      color: '#ae3ec9',
      hasLr: true,
      rule: 's ← ρs + (1−ρ)g²,  x ← x − η g / √s（ρ = 0.9）',
      init: () => ({ sx: 0, sy: 0 }),
      step(s, fn, p) {
        const g = fn.grad(s.x, s.y);
        s.sx = 0.9 * s.sx + 0.1 * g[0] * g[0];
        s.sy = 0.9 * s.sy + 0.1 * g[1] * g[1];
        s.x -= p.lr * g[0] / (Math.sqrt(s.sx) + 1e-8);
        s.y -= p.lr * g[1] / (Math.sqrt(s.sy) + 1e-8);
      },
    },
    adam: {
      label: 'Adam',
      color: '#1c7ed6',
      hasLr: true,
      rule: '勾配の移動平均 m と二乗の移動平均 v をバイアス補正して x ← x − η m̂ / √v̂',
      init: () => ({ mx: 0, my: 0, vx: 0, vy: 0, t: 0 }),
      step(s, fn, p) {
        const g = fn.grad(s.x, s.y);
        const b1 = 0.9, b2 = 0.999;
        s.t += 1;
        s.mx = b1 * s.mx + (1 - b1) * g[0];
        s.my = b1 * s.my + (1 - b1) * g[1];
        s.vx = b2 * s.vx + (1 - b2) * g[0] * g[0];
        s.vy = b2 * s.vy + (1 - b2) * g[1] * g[1];
        const c1 = 1 - b1 ** s.t, c2 = 1 - b2 ** s.t;
        s.x -= p.lr * (s.mx / c1) / (Math.sqrt(s.vx / c2) + 1e-8);
        s.y -= p.lr * (s.my / c1) / (Math.sqrt(s.vy / c2) + 1e-8);
      },
    },
    newton: {
      label: 'ニュートン法',
      color: '#f03e3e',
      hasLr: false,
      rule: 'x ← x − H⁻¹∇f（曲がり具合を見て一気に跳ぶ。凸でないと鞍点や山頂にも向かう）',
      init: () => ({}),
      step(s, fn) {
        const g = fn.grad(s.x, s.y);
        const d = solve2(hessian(fn, s.x, s.y), g, 0);
        if (!d) { s.stuck = 'ヘッセ行列が特異'; return; }
        s.x += d[0];
        s.y += d[1];
      },
    },
    damped: {
      label: '減衰ニュートン法',
      color: '#fab005',
      hasLr: false,
      rule: 'x ← x − (H + λI)⁻¹∇f。f が下がらなければ λ を大きく、下がれば小さく（Levenberg–Marquardt 型）',
      init: () => ({ lam: 1 }),
      step(s, fn) {
        const g = fn.grad(s.x, s.y);
        const H = hessian(fn, s.x, s.y);
        const f0 = fn.f(s.x, s.y);
        // H + λI が正定値になるように λ の下限を決める
        const floor = Math.max(0, -minEig(H)) + 1e-6;
        let lam = Math.max(s.lam, floor);
        for (let k = 0; k < 40; k++) {
          const d = solve2(H, g, lam);
          if (d) {
            const nx = s.x + d[0], ny = s.y + d[1];
            const f1 = fn.f(nx, ny);
            if (isFinite(f1) && f1 <= f0) {
              s.x = nx;
              s.y = ny;
              s.lam = Math.max(lam / 3, 1e-7);
              return;
            }
          }
          lam *= 4;
        }
        s.lam = lam; // 下がる方向が見つからない＝ほぼ停留点
      },
    },
  };

  const METHOD_ORDER = ['gd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'newton', 'damped'];

  // ---------- 自作関数 ----------
  // 範囲内の最小値を探す（格子で当たりをつけてから減衰ニュートン法で詰める）
  function estimateMin(fn) {
    const [x0, x1, y0, y1] = fn.range;
    const G = 160;
    let bx = 0, by = 0, bf = Infinity;
    for (let j = 0; j <= G; j++) {
      for (let i = 0; i <= G; i++) {
        const x = x0 + (x1 - x0) * i / G, y = y0 + (y1 - y0) * j / G;
        const v = fn.f(x, y);
        if (isFinite(v) && v < bf) { bf = v; bx = x; by = y; }
      }
    }
    if (!isFinite(bf)) return null;
    const run = createRun('damped', Object.assign({ fmin: 0 }, fn), [bx, by], {});
    for (let k = 0; k < 80 && run.status === 'running'; k++) stepRun(run, Object.assign({ fmin: 0 }, fn));
    const s = run.state;
    const w = x1 - x0, h = y1 - y0;
    const inside = s.x >= x0 - 0.1 * w && s.x <= x1 + 0.1 * w && s.y >= y0 - 0.1 * h && s.y <= y1 + 0.1 * h;
    const rf = fn.f(s.x, s.y);
    if (inside && isFinite(rf) && rf <= bf) return { x: s.x, y: s.y, f: rf };
    return { x: bx, y: by, f: bf };
  }

  // 開始点まわりの曲がり具合から、勾配法の学習率の目安を決める
  function suggestLr(fn, start) {
    const [x0, x1, y0, y1] = fn.range;
    const w = x1 - x0, h = y1 - y0;
    let L = 0;
    for (let j = -4; j <= 4; j++) {
      for (let i = -4; i <= 4; i++) {
        const x = start[0] + 0.2 * w * i / 4, y = start[1] + 0.2 * h * j / 4;
        const H = hessian(fn, x, y);
        const tr = H[0] + H[3], det = H[0] * H[3] - H[1] * H[2];
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const m = Math.max(Math.abs(tr / 2 + disc), Math.abs(tr / 2 - disc));
        if (isFinite(m)) L = Math.max(L, m);
      }
    }
    if (!(L > 0)) L = 1;
    const r2 = (v) => +v.toPrecision(2);
    const span = Math.max(w, h);
    return { gd: r2(0.9 / L), momentum: r2(0.5 / L), nesterov: r2(0.5 / L), rmsprop: r2(0.004 * span), adam: r2(0.02 * span) };
  }

  /**
   * 保存用の定義から問題を作る。
   * def: { id, name, expr, range:[x0,x1,y0,y1], start:[x,y], fmin?, minima?, note?, lr? }
   */
  function makeProblem(def, Expr) {
    const c = Expr.compile(def.expr);
    const fn = {
      id: def.id,
      custom: true,
      label: def.name || def.expr,
      note: def.note || '',
      expr: def.expr,
      exprText: c.text,
      dxText: c.dxText,
      dyText: c.dyText,
      f: c.f,
      grad: c.grad,
      range: def.range.slice(),
      start: def.start.slice(),
    };
    const v0 = fn.f(fn.start[0], fn.start[1]);
    if (!isFinite(v0)) throw new Error('開始点で関数の値が計算できません（定義域の外かもしれません）');
    if (def.fmin != null && def.fmin !== '' && isFinite(+def.fmin)) {
      fn.fmin = +def.fmin;
      fn.fminEstimated = false;
      fn.minima = (def.minima && def.minima.length) ? def.minima.map((p) => p.slice()) : [];
    } else {
      const m = estimateMin(fn);
      fn.fmin = m ? m.f : 0;
      fn.fminEstimated = true;
      fn.minima = (def.minima && def.minima.length) ? def.minima.map((p) => p.slice()) : (m ? [[m.x, m.y]] : []);
    }
    fn.lr = Object.assign(suggestLr(fn, fn.start), def.lr || {});
    return fn;
  }

  // ---------- 1本の軌跡を管理する ----------
  function createRun(methodKey, fn, start, params) {
    const m = METHODS[methodKey];
    const s = Object.assign({ x: start[0], y: start[1] }, m.init());
    const run = {
      key: methodKey,
      state: s,
      params: params || {},
      path: [[s.x, s.y]],
      vals: [fn.f(s.x, s.y)], // f の値そのもの（f* との差は表示のときに取る）
      iter: 0,
      status: 'running', // running | converged | diverged | stuck
    };
    return run;
  }

  function stepRun(run, fn) {
    if (run.status !== 'running') return;
    const m = METHODS[run.key];
    const s = run.state;
    m.step(s, fn, run.params);
    run.iter += 1;
    if (s.stuck) { run.status = 'stuck'; run.reason = s.stuck; return; }
    const fv = fn.f(s.x, s.y);
    if (!isFinite(s.x) || !isFinite(s.y) || !isFinite(fv) || Math.abs(s.x) > 1e6 || Math.abs(s.y) > 1e6) {
      run.status = 'diverged';
      return;
    }
    run.path.push([s.x, s.y]);
    run.vals.push(fv);
    const g = fn.grad(s.x, s.y);
    const gn = Math.hypot(g[0], g[1]);
    const prev = run.path[run.path.length - 2];
    const moved = Math.hypot(s.x - prev[0], s.y - prev[1]);
    if (gn < 1e-8 || (moved < 1e-12 && gn < 1e-5)) run.status = 'converged';
  }

  function gradNorm(fn, x, y) {
    const g = fn.grad(x, y);
    return Math.hypot(g[0], g[1]);
  }

  const api = { FUNCTIONS, METHODS, METHOD_ORDER, hessian, minEig, createRun, stepRun, gradNorm, makeProblem, estimateMin, suggestLr };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Optim = api;
})(typeof window !== 'undefined' ? window : globalThis);
