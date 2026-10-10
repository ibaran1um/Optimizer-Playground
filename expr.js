/*
 * Optimizer Playground — 自作関数のための数式パーサと記号微分
 * MIT License
 *
 * "(1-x)^2 + 100*(y-x^2)^2" のような x, y の式を構文木にし、
 * 値と ∂f/∂x・∂f/∂y を記号微分で求めて JavaScript の関数に変換する。
 * 生成するコードは構文木（数値・x・y・四則演算・許可した関数だけ）から
 * 組み立てるので、入力文字列がそのまま実行されることはない。
 */
(function (root) {
  'use strict';

  const FUNCS = {
    sin: 'Math.sin', cos: 'Math.cos', tan: 'Math.tan',
    asin: 'Math.asin', acos: 'Math.acos', atan: 'Math.atan',
    sinh: 'Math.sinh', cosh: 'Math.cosh', tanh: 'Math.tanh',
    exp: 'Math.exp', log: 'Math.log', ln: 'Math.log', log10: 'Math.log10',
    sqrt: 'Math.sqrt', abs: 'Math.abs', sign: 'Math.sign',
  };
  const CONSTS = { pi: Math.PI, e: Math.E };

  class ExprError extends Error {
    constructor(msg, pos) { super(msg); this.pos = pos; }
  }

  // ---------- 字句解析 ----------
  function tokenize(src) {
    const toks = [];
    let i = 0;
    const s = src.replace(/[（]/g, '(').replace(/[）]/g, ')').replace(/[×]/g, '*').replace(/[÷]/g, '/').replace(/[－−]/g, '-').replace(/[＋]/g, '+').replace(/[＾]/g, '^').replace(/[，]/g, ',');
    while (i < s.length) {
      const c = s[i];
      if (/\s/.test(c)) { i++; continue; }
      if (/[0-9.]/.test(c)) {
        const m = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(s.slice(i));
        if (!m || s[i + m[0].length] === '.') throw new ExprError(`数の書き方が正しくありません`, i);
        toks.push({ t: 'num', v: parseFloat(m[0]), pos: i });
        i += m[0].length; continue;
      }
      if (/[a-zA-Z_]/.test(c)) {
        const m = /^[a-zA-Z_][a-zA-Z_0-9]*/.exec(s.slice(i));
        toks.push({ t: 'id', v: m[0].toLowerCase(), pos: i });
        i += m[0].length; continue;
      }
      if (s.startsWith('**', i)) { toks.push({ t: 'op', v: '^', pos: i }); i += 2; continue; }
      if ('+-*/^(),'.includes(c)) { toks.push({ t: 'op', v: c, pos: i }); i++; continue; }
      throw new ExprError(`使えない文字「${c}」があります`, i);
    }
    toks.push({ t: 'end', pos: s.length });
    return toks;
  }

  // ---------- 構文解析 ----------
  // 優先順位：+ - ＜ * /（暗黙の掛け算 2x を含む）＜ 単項マイナス ＜ ^（右結合）
  function parse(src) {
    if (!src || !src.trim()) throw new ExprError('式が空です', 0);
    const toks = tokenize(src);
    let k = 0;
    const peek = () => toks[k];
    const isOp = (v) => peek().t === 'op' && peek().v === v;
    const expect = (v) => {
      if (!isOp(v)) throw new ExprError(`「${v}」が必要です`, peek().pos);
      k++;
    };

    function expr() {
      let a = term();
      while (isOp('+') || isOp('-')) {
        const op = toks[k++].v;
        a = { t: 'bin', op, a, b: term() };
      }
      return a;
    }
    function startsFactor() {
      const p = peek();
      return p.t === 'num' || p.t === 'id' || (p.t === 'op' && p.v === '(');
    }
    function term() {
      let a = unary();
      for (;;) {
        if (isOp('*') || isOp('/')) {
          const op = toks[k++].v;
          a = { t: 'bin', op, a, b: unary() };
        } else if (startsFactor()) {
          a = { t: 'bin', op: '*', a, b: power() }; // 2x, 3(x+1), x y
        } else break;
      }
      return a;
    }
    function unary() {
      if (isOp('-')) { k++; return { t: 'neg', a: unary() }; }
      if (isOp('+')) { k++; return unary(); }
      return power();
    }
    function power() {
      const base = atom();
      if (isOp('^')) { k++; return { t: 'bin', op: '^', a: base, b: unary() }; }
      return base;
    }
    function atom() {
      const p = peek();
      if (p.t === 'num') { k++; return { t: 'num', v: p.v }; }
      if (p.t === 'op' && p.v === '(') { k++; const e = expr(); expect(')'); return e; }
      if (p.t === 'id') {
        k++;
        if (p.v === 'x' || p.v === 'y') return { t: 'var', n: p.v };
        if (p.v in CONSTS) return { t: 'num', v: CONSTS[p.v] };
        if (p.v in FUNCS) {
          if (!isOp('(')) throw new ExprError(`${p.v} の後には ( ) で引数を書いてください`, peek().pos);
          k++;
          const a = expr();
          expect(')');
          return { t: 'call', f: p.v === 'ln' ? 'log' : p.v, a };
        }
        throw new ExprError(`「${p.v}」は使えません（使える変数は x と y）`, p.pos);
      }
      if (p.t === 'end') throw new ExprError('式が途中で終わっています', p.pos);
      throw new ExprError(`「${p.v}」の位置がおかしいです`, p.pos);
    }

    const tree = expr();
    if (peek().t !== 'end') throw new ExprError(`「${peek().v}」の位置がおかしいです`, peek().pos);
    return simplify(tree);
  }

  // ---------- 簡約 ----------
  const N = (v) => ({ t: 'num', v });
  const isNum = (n, v) => n.t === 'num' && (v === undefined || n.v === v);
  function simplify(n) {
    switch (n.t) {
      case 'num': case 'var': return n;
      case 'neg': {
        const a = simplify(n.a);
        if (isNum(a)) return N(-a.v);
        if (a.t === 'neg') return a.a;
        if (a.t === 'bin' && (a.op === '*' || a.op === '/') && isNum(a.a)) return simplify({ t: 'bin', op: a.op, a: N(-a.a.v), b: a.b });
        return { t: 'neg', a };
      }
      case 'call': {
        const a = simplify(n.a);
        if (isNum(a)) { const v = Math[n.f === 'log' ? 'log' : n.f](a.v); if (isFinite(v)) return N(v); }
        return { t: 'call', f: n.f, a };
      }
      case 'bin': {
        const a = simplify(n.a), b = simplify(n.b);
        if (isNum(a) && isNum(b)) {
          const v = n.op === '+' ? a.v + b.v : n.op === '-' ? a.v - b.v : n.op === '*' ? a.v * b.v : n.op === '/' ? a.v / b.v : Math.pow(a.v, b.v);
          if (isFinite(v)) return N(v);
        }
        switch (n.op) {
          case '+':
            if (isNum(a, 0)) return b;
            if (isNum(b, 0)) return a;
            if (b.t === 'neg') return simplify({ t: 'bin', op: '-', a, b: b.a });
            if (b.t === 'bin' && (b.op === '*' || b.op === '/') && isNum(b.a) && b.a.v < 0) return { t: 'bin', op: '-', a, b: simplify({ t: 'bin', op: b.op, a: N(-b.a.v), b: b.b }) };
            break;
          case '-':
            if (isNum(b, 0)) return a;
            if (isNum(a, 0)) return simplify({ t: 'neg', a: b });
            if (b.t === 'neg') return simplify({ t: 'bin', op: '+', a, b: b.a });
            if (isNum(b) && b.v < 0) return { t: 'bin', op: '+', a, b: N(-b.v) };
            if (b.t === 'bin' && (b.op === '*' || b.op === '/') && isNum(b.a) && b.a.v < 0) return { t: 'bin', op: '+', a, b: simplify({ t: 'bin', op: b.op, a: N(-b.a.v), b: b.b }) };
            break;
          case '*':
            if (isNum(a, 0) || isNum(b, 0)) return N(0);
            if (isNum(a, 1)) return b;
            if (isNum(b, 1)) return a;
            if (isNum(a, -1)) return simplify({ t: 'neg', a: b });
            if (isNum(b)) return simplify({ t: 'bin', op: '*', a: b, b: a }); // 数を前に
            if (isNum(a) && b.t === 'bin' && b.op === '*' && isNum(b.a)) return simplify({ t: 'bin', op: '*', a: N(a.v * b.a.v), b: b.b });
            if (a.t === 'neg') return simplify({ t: 'neg', a: { t: 'bin', op: '*', a: a.a, b } });
            if (b.t === 'neg') return simplify({ t: 'neg', a: { t: 'bin', op: '*', a, b: b.a } });
            break;
          case '/':
            if (isNum(a, 0)) return N(0);
            if (isNum(b, 1)) return a;
            break;
          case '^':
            if (isNum(b, 0)) return N(1);
            if (isNum(b, 1)) return a;
            break;
        }
        return { t: 'bin', op: n.op, a, b };
      }
    }
    throw new Error('unknown node');
  }

  const hasVar = (n) => n.t === 'var' || (n.a && hasVar(n.a)) || (n.b && hasVar(n.b));

  // ---------- 記号微分 ----------
  function diff(n, v) {
    const B = (op, a, b) => ({ t: 'bin', op, a, b });
    const C = (f, a) => ({ t: 'call', f, a });
    switch (n.t) {
      case 'num': return N(0);
      case 'var': return N(n.n === v ? 1 : 0);
      case 'neg': return { t: 'neg', a: diff(n.a, v) };
      case 'bin': {
        const { a, b } = n;
        const da = diff(a, v), db = diff(b, v);
        switch (n.op) {
          case '+': return B('+', da, db);
          case '-': return B('-', da, db);
          case '*': return B('+', B('*', da, b), B('*', a, db));
          case '/': return B('/', B('-', B('*', da, b), B('*', a, db)), B('^', b, N(2)));
          case '^':
            if (!hasVar(b)) return B('*', B('*', b, B('^', a, simplify(B('-', b, N(1))))), da);
            if (!hasVar(a)) return B('*', B('*', n, C('log', a)), db);
            return B('*', n, B('+', B('*', db, C('log', a)), B('/', B('*', b, da), a)));
        }
        break;
      }
      case 'call': {
        const a = n.a, da = diff(a, v);
        let d;
        switch (n.f) {
          case 'sin': d = C('cos', a); break;
          case 'cos': d = { t: 'neg', a: C('sin', a) }; break;
          case 'tan': d = B('/', N(1), B('^', C('cos', a), N(2))); break;
          case 'asin': d = B('/', N(1), C('sqrt', B('-', N(1), B('^', a, N(2))))); break;
          case 'acos': d = { t: 'neg', a: B('/', N(1), C('sqrt', B('-', N(1), B('^', a, N(2))))) }; break;
          case 'atan': d = B('/', N(1), B('+', N(1), B('^', a, N(2)))); break;
          case 'sinh': d = C('cosh', a); break;
          case 'cosh': d = C('sinh', a); break;
          case 'tanh': d = B('-', N(1), B('^', C('tanh', a), N(2))); break;
          case 'exp': d = C('exp', a); break;
          case 'log': d = B('/', N(1), a); break;
          case 'log10': d = B('/', N(1), B('*', a, N(Math.LN10))); break;
          case 'sqrt': d = B('/', N(1), B('*', N(2), C('sqrt', a))); break;
          case 'abs': d = C('sign', a); break;
          case 'sign': d = N(0); break;
        }
        return B('*', d, da);
      }
    }
    throw new Error('unknown node');
  }

  // ---------- JavaScript へ ----------
  function toJS(n) {
    switch (n.t) {
      case 'num': return n.v < 0 ? `(${n.v})` : String(n.v);
      case 'var': return n.n;
      case 'neg': return `(-${toJS(n.a)})`;
      case 'call': return `${FUNCS[n.f]}(${toJS(n.a)})`;
      case 'bin':
        if (n.op === '^') {
          if (isNum(n.b, 2)) { const s = toJS(n.a); return `((${s})*(${s}))`; }
          return `Math.pow(${toJS(n.a)},${toJS(n.b)})`;
        }
        return `(${toJS(n.a)}${n.op}${toJS(n.b)})`;
    }
    throw new Error('unknown node');
  }

  // ---------- 人が読む形へ ----------
  const PREC = { '+': 1, '-': 1, '*': 2, '/': 2, '^': 4 };
  function fmtNum(v) {
    if (Math.abs(v - Math.PI) < 1e-12) return 'pi';
    if (Number.isInteger(v)) return String(v);
    return String(+v.toPrecision(6));
  }
  function toText(n, parentPrec = 0, rightSide = false) {
    let s, p;
    switch (n.t) {
      case 'num': s = fmtNum(n.v); p = n.v < 0 ? 3 : 9; break;
      case 'var': s = n.n; p = 9; break;
      case 'call': s = `${n.f}(${toText(n.a)})`; p = 9; break;
      case 'neg': s = `-${toText(n.a, 3)}`; p = 3; break;
      case 'bin': {
        p = PREC[n.op];
        const l = toText(n.a, n.op === '^' ? p + 1 : p);
        const r = toText(n.b, n.op === '^' ? p : p, n.op !== '^');
        s = n.op === '*' ? `${l}*${r}` : n.op === '^' ? `${l}^${r}` : `${l} ${n.op} ${r}`;
        break;
      }
    }
    const need = p < parentPrec || (rightSide && p === parentPrec && parentPrec <= 2);
    return need ? `(${s})` : s;
  }

  /**
   * 式を値と勾配の関数にする。
   * 戻り値：{ f(x,y), grad(x,y) -> [gx, gy], text, dxText, dyText }
   */
  function compile(src) {
    const tree = parse(src);
    const dx = simplify(diff(tree, 'x'));
    const dy = simplify(diff(tree, 'y'));
    // eslint-disable-next-line no-new-func
    const f = new Function('x', 'y', `return ${toJS(tree)};`);
    // eslint-disable-next-line no-new-func
    const grad = new Function('x', 'y', `return [${toJS(dx)}, ${toJS(dy)}];`);
    return { f, grad, text: toText(tree), dxText: toText(dx), dyText: toText(dy) };
  }

  const api = { parse, compile, ExprError, FUNCS: Object.keys(FUNCS) };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Expr = api;
})(typeof window !== 'undefined' ? window : globalThis);
