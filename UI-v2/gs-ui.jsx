// Shared primitives. One deviation scale for the whole product lives here, so
// the colouring can never drift between panels the way it did in the current
// vanilla UI (code said 0.01/0.1, the legend said 0.02/0.1).

const TONE_OK = 0.02;
const TONE_WARN = 0.1;
const OXIDE_SCALE_FLOOR = 0.1; // same floor the feasibility LP uses

function deltaTone(delta) {
  const a = Math.abs(delta);
  if (a < TONE_OK) return 'ok';
  if (a < TONE_WARN) return 'warn';
  return 'bad';
}

function fmtOx(name) {
  const parts = String(name).split(/(\d+)/).filter(Boolean);
  return parts.map((p, i) => (/^\d+$/.test(p) ? <sub key={i}>{p}</sub> : <span key={i}>{p}</span>));
}

function num(v, d) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(d === undefined ? 3 : d);
}

function groupOf(oxide) {
  const g = window.GS.OXIDE_GROUPS;
  if (g.r2o.indexOf(oxide) >= 0 || g.ro.indexOf(oxide) >= 0) return 'r2o_ro';
  if (g.r2o3.indexOf(oxide) >= 0) return 'r2o3';
  if (g.ro2.indexOf(oxide) >= 0) return 'ro2';
  if (/2O[35]$/.test(oxide)) return 'r2o3';
  if (/O2$/.test(oxide)) return 'ro2';
  return 'r2o_ro';
}

function isR2O(oxide) {
  return window.GS.OXIDE_GROUPS.r2o.indexOf(oxide) >= 0;
}

function materialsByName() {
  const by = {};
  window.GS.MATERIALS.forEach((m) => { by[m.name] = m; });
  return by;
}

// Material x oxide contributions: the one thing practitioners say every glaze
// program hides, and the reason they go back to Excel.
function contributions(recipe) {
  const by = materialsByName();
  const oxides = [];
  const rows = Object.keys(recipe).sort((a, b) => recipe[b] - recipe[a]).map((name) => {
    const mat = by[name];
    const amt = recipe[name];
    const ox = {};
    let sum = 0;
    if (mat) {
      Object.keys(mat.formula).forEach((k) => {
        const v = (mat.formula[k] * amt) / 100;
        ox[k] = v;
        sum += v;
        if (oxides.indexOf(k) < 0) oxides.push(k);
      });
    }
    return { name, amt, ox, loi: amt - sum };
  });
  const totals = {};
  oxides.forEach((o) => { totals[o] = rows.reduce((s, r) => s + (r.ox[o] || 0), 0); });
  oxides.sort((a, b) => totals[b] - totals[a]);
  const loi = rows.reduce((s, r) => s + r.loi, 0);
  return { rows, oxides, totals, loi };
}

// "What is this material here for": the oxide it supplies the biggest share of.
// The share is damped by how much of that oxide the recipe holds at all, so a
// material does not win the label just for being the only source of a trace.
function whyMaterials(recipe) {
  const c = contributions(recipe);
  const out = {};
  c.rows.forEach((r) => {
    let best = null;
    let bestScore = 0;
    Object.keys(r.ox).forEach((o) => {
      const total = c.totals[o] || 0;
      if (total <= 0) return;
      const score = (r.ox[o] / total) * Math.min(1, total / 5);
      if (score > bestScore) { bestScore = score; best = o; }
    });
    out[r.name] = best;
  });
  return out;
}

// Which materials of a recipe have no trustworthy LOI. Until that list is
// empty the off-gassing estimate would silently answer "no pinholes", which is
// the most harmful of the possible answers.
function loiStatus(recipe) {
  const by = materialsByName();
  const unknown = [];
  Object.keys(recipe).forEach((n) => {
    const m = by[n];
    if (!m || m.loi === 'unknown') unknown.push(n);
  });
  return { unknown };
}

// ---------------------------------------------------------------- primitives

function Panel(props) {
  return (
    <section className="gs-panel" data-screen-label={props.title}>
      <header className="gs-panel-hd">
        <span className="gs-micro">{props.title}</span>
        <span className="gs-panel-hd-right">{props.right}</span>
      </header>
      <div className={'gs-panel-body' + (props.scroll ? ' is-scroll' : '')}>{props.children}</div>
    </section>
  );
}

function Seg(props) {
  return (
    <div className={'gs-seg' + (props.wide ? ' is-wide' : '')} role="tablist">
      {props.options.map((o) => (
        <button key={o.v} type="button" role="tab" aria-selected={props.value === o.v}
          className={'gs-seg-b' + (props.value === o.v ? ' on' : '')}
          title={o.title || o.label} onClick={() => props.onChange(o.v)}>
          {o.label}
          {o.count !== undefined ? <em className="gs-seg-n">{o.count}</em> : null}
        </button>
      ))}
    </div>
  );
}

// Signed deviation as a thin centre-anchored bar: the number stays readable,
// the bar carries the signal.
function DeltaBar(props) {
  const scale = Math.max(Math.abs(props.scale || 0), OXIDE_SCALE_FLOOR);
  const rel = Math.max(-1, Math.min(1, (props.delta || 0) / scale));
  const half = Math.abs(rel) * 50;
  const tone = props.tone || deltaTone(props.delta || 0);
  const band = props.spread ? Math.min(50, (props.spread / scale) * 50) : 0;
  return (
    <span className={'gs-dbar tone-' + tone} title={props.title}>
      {band ? <i className="gs-dbar-band" style={{ left: (50 - band) + '%', width: (band * 2) + '%' }} /> : null}
      <i className="gs-dbar-mid" />
      <i className="gs-dbar-fill"
        style={rel >= 0 ? { left: '50%', width: half + '%' } : { right: '50%', width: half + '%' }} />
    </span>
  );
}

function Bar(props) {
  const w = Math.max(0, Math.min(100, ((props.v || 0) / (props.max || 100)) * 100));
  return (
    <span className={'gs-bar' + (props.thin ? ' is-thin' : '')}>
      <i style={{ width: w + '%' }} className={'gs-bar-fill' + (props.tone ? ' tone-' + props.tone : '')} />
    </span>
  );
}

function Badge(props) {
  return (
    <span className={'gs-badge' + (props.tone ? ' tone-' + props.tone : '')} title={props.title}>
      <span className="gs-badge-k">{props.label}</span>
      {props.value !== undefined ? <span className="gs-badge-v">{props.value}</span> : null}
    </span>
  );
}

// Text-first numeric field. Never rewrites what is being typed: normalisation
// happens on blur / Enter only. Accepts a comma decimal and an expression.
function NumField(props) {
  const [draft, setDraft] = React.useState(null);
  const shown = draft === null ? String(props.value) : draft;
  const commit = (raw) => {
    setDraft(null);
    let s = String(raw).replace(',', '.').trim();
    if (/^[0-9+\-*/(). ]+$/.test(s) && /[+\-*/]/.test(s.slice(1))) {
      try { s = String(Function('"use strict";return (' + s + ')')()); } catch (e) { s = String(props.value); }
    }
    const v = parseFloat(s);
    props.onChange(Number.isNaN(v) ? String(props.value) : String(Math.max(0, v)));
  };
  return (
    <input
      className={'gs-inp' + (props.tone ? ' tone-' + props.tone : '')}
      type="text" inputMode="decimal" value={shown}
      placeholder={props.placeholder || ''}
      title="↑↓ — шаг 0.01, Shift+↑↓ — 0.1; можно ввести выражение, например 100/3"
      style={props.width ? { width: props.width } : null}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={(e) => commit(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') { commit(e.target.value); e.target.blur(); return; }
        if (e.key === 'Escape') { setDraft(null); e.target.blur(); return; }
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
        e.preventDefault();
        const step = (e.shiftKey ? 0.1 : 0.01) * (e.key === 'ArrowUp' ? 1 : -1);
        const base = parseFloat(String(shown).replace(',', '.')) || 0;
        setDraft(null);
        props.onChange(String(Math.round(Math.max(0, base + step) * 1000) / 1000));
      }}
    />
  );
}

function Check(props) {
  return (
    <label className="gs-check" title={props.title}>
      <input type="checkbox" checked={!!props.checked} onChange={(e) => props.onChange(e.target.checked)} />
      <span className="gs-check-box" />
      <span className="gs-check-l">{props.children}</span>
    </label>
  );
}

function Btn(props) {
  return (
    <button type="button"
      className={'gs-btn' + (props.kind ? ' is-' + props.kind : '') + (props.on ? ' on' : '')}
      title={props.title} onClick={props.onClick}>
      {props.children}
    </button>
  );
}

function Empty(props) {
  return <div className="gs-empty">{props.children}</div>;
}

window.GSUI = {
  TONE_OK, TONE_WARN, OXIDE_SCALE_FLOOR,
  deltaTone, fmtOx, num, groupOf, isR2O,
  contributions, whyMaterials, loiStatus, materialsByName,
  Panel, Seg, DeltaBar, Bar, Badge, NumField, Check, Btn, Empty,
};

function GSUi() {
  return null;
}

Object.assign(window, { GSUi });
