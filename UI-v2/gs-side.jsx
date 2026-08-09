// Left (target) and right (materials) columns.

// Requirement or passenger? Two signals, both computable, neither invented:
// 1. can this oxide arrive by accident at all - some are carried only by
//    materials that are essentially pure oxide, so any amount is intent;
// 2. is the amount below the level at which people add it on purpose.
// A formula computed from someone's recipe drags its passengers along; chasing
// them exactly is how a solver ends up sprinkling 0.3% of red iron oxide.
function guessRole(oxide, value) {
  const D = window.GS;
  if (D.NEVER_ACCIDENTAL.indexOf(oxide) >= 0) return 'req';
  const t = D.OXIDE_THRESHOLD[oxide];
  if (!t) return 'req';
  return value < t.notable ? 'pass' : 'req';
}

function OxideRow(props) {
  const U = window.GSUI;
  const { oxide, value, actual } = props;
  const known = actual !== undefined && actual !== null;
  const target = parseFloat(value) || 0;
  const delta = known ? actual - target : 0;
  const pass = props.role === 'pass';
  return (
    <div className={'gs-orow' + (props.divider ? ' has-div' : '') + (pass ? ' is-pass' : '') + (props.plain ? ' is-plain' : '')}>
      <span className="gs-ox">{U.fmtOx(oxide)}</span>
      <span className="gs-inp-wrap">
        {pass ? <span className="gs-le" title="верхняя граница, а не точное значение">≤</span> : null}
        <U.NumField value={value} onChange={props.onChange} width={pass ? 42 : 52} />
      </span>
      <button type="button" className={'gs-role is-' + props.role}
        title={pass
          ? 'попутный: солвер не гонится за точным значением, только держит сверху. Клик — сделать требованием'
          : 'требование: попасть в значение. Клик — сделать попутным'}
        onClick={() => props.onRole(oxide, pass ? 'req' : 'pass')}>
        {pass ? 'попут.' : 'треб.'}
      </button>
      {props.plain ? null : (
        <React.Fragment>
          <U.DeltaBar delta={delta} scale={target} spread={props.spread}
            title={known ? 'цель ' + value + ', факт ' + U.num(actual) : ''} />
          <span className={'gs-actual tone-' + (known ? U.deltaTone(delta) : 'none')}>
            {known ? U.num(actual) : '—'}
          </span>
        </React.Fragment>
      )}
      <button type="button" className="gs-x" title="убрать оксид" onClick={props.onRemove}>×</button>
    </div>
  );
}

function OxideGroup(props) {
  if (!props.oxides.length) return null;
  return (
    <div className="gs-ogroup">
      <div className="gs-ogroup-hd"><span>{props.title}</span>{props.right}</div>
      {props.oxides.map((ox, i) => (
        <OxideRow key={ox} oxide={ox}
          divider={props.dividerAfter === ox && i < props.oxides.length - 1}
          {...props.rowProps(ox)} />
      ))}
    </div>
  );
}

function RatiosStrip(props) {
  const u = props.umf;
  const g = window.GS.OXIDE_GROUPS;
  const sum = (list) => list.reduce((a, k) => a + (parseFloat(u[k]) || 0), 0);
  const si = parseFloat(u.SiO2) || 0;
  const al = parseFloat(u.Al2O3) || 0;
  const r2o = sum(g.r2o);
  const ro = sum(g.ro);
  const items = [
    { k: 'Si / Al', v: al ? (si / al).toFixed(2) : '—', hint: 'отношение кремнезёма к глинозёму' },
    { k: 'R₂O / RO', v: ro ? r2o.toFixed(2) + ' / ' + ro.toFixed(2) : '—', hint: 'баланс щёлочных и щёлочноземельных плавней' },
  ];
  return (
    <div className="gs-ratios">
      {items.map((it) => (
        <div className="gs-ratio" key={it.k} title={it.hint}>
          <span className="gs-ratio-k">{it.k}</span>
          <span className="gs-ratio-v">{it.v}</span>
        </div>
      ))}
    </div>
  );
}

// SiO2 x Al2O3 map. The classic Stull zones are NOT drawn: their boundaries
// hold only for one flux balance and one cone, and sketching them freehand
// would be decorative science. Axes and the target, nothing invented.
// Solution dots appear only when they are further from the target than the
// chart can resolve - with a single target they usually are not.
const COINCIDENT_PX = 5;

function StullChart(props) {
  const W = 288;
  const H = 150;
  const PAD = { l: 34, r: 12, t: 10, b: 22 };
  const xd = [1.5, 5.2];
  const yd = [0.1, 0.72];
  const x = (v) => PAD.l + ((v - xd[0]) / (xd[1] - xd[0])) * (W - PAD.l - PAD.r);
  const y = (v) => H - PAD.b - ((v - yd[0]) / (yd[1] - yd[0])) * (H - PAD.t - PAD.b);
  const tx = [2, 3, 4, 5];
  const ty = [0.2, 0.4, 0.6];

  const txp = x(props.target.x);
  const typ = y(props.target.y);
  let merged = 0;
  const shown = (props.points || []).filter((p) => {
    const d = Math.sqrt(Math.pow(x(p.x) - txp, 2) + Math.pow(y(p.y) - typ, 2));
    if (d < COINCIDENT_PX) { merged += 1; return false; }
    return true;
  });

  return (
    <div className="gs-chart">
      <div className="gs-chart-hd"><span className="gs-micro">SiO₂ × Al₂O₃</span></div>
      <svg viewBox={'0 0 ' + W + ' ' + H} width="100%" height={H} role="img" aria-label="Карта SiO2 × Al2O3">
        {tx.map((t) => <line key={'gx' + t} x1={x(t)} y1={PAD.t} x2={x(t)} y2={H - PAD.b} className="gs-grid" />)}
        {ty.map((t) => <line key={'gy' + t} x1={PAD.l} y1={y(t)} x2={W - PAD.r} y2={y(t)} className="gs-grid" />)}
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} className="gs-axis" />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} className="gs-axis" />
        {tx.map((t) => <text key={'tx' + t} x={x(t)} y={H - 8} className="gs-tick" textAnchor="middle">{t}</text>)}
        {ty.map((t) => <text key={'ty' + t} x={PAD.l - 5} y={y(t) + 3} className="gs-tick" textAnchor="end">{t}</text>)}
        {shown.map((p) => (
          <circle key={p.id} cx={x(Math.max(xd[0], Math.min(xd[1], p.x)))}
            cy={y(Math.max(yd[0], Math.min(yd[1], p.y)))} r="3.2" className="gs-pt">
            <title>{p.label}</title>
          </circle>
        ))}
        <path className="gs-target-mark"
          d={'M ' + txp + ' ' + (typ - 5.5) + ' l 5.5 5.5 l -5.5 5.5 l -5.5 -5.5 z'}>
          <title>{'цель: SiO2 ' + props.target.x + ', Al2O3 ' + props.target.y}</title>
        </path>
      </svg>
      {merged ? (
        <div className="gs-chart-note">решения лежат на цели точнее разрешения графика</div>
      ) : null}
    </div>
  );
}

function ImportedRef(props) {
  const U = window.GSUI;
  const r = props.data;
  if (!r) return null;
  return (
    <div className="gs-ref">
      <div className="gs-ref-hd">
        <span className="gs-micro">Импортировано</span>
        <button type="button" className="gs-x" onClick={props.onClear}>×</button>
      </div>
      <div className="gs-ref-name">{r.name}</div>
      <div className="gs-ref-meta">{r.author} · {r.cone} · источник: {r.source}</div>
      <div className="gs-ref-rows">
        {Object.keys(r.umf).map((k) => (
          <div className="gs-ref-row" key={k}>
            <span>{U.fmtOx(k)}</span><span className="gs-n">{U.num(r.umf[k])}</span>
          </div>
        ))}
      </div>
      <div className="gs-hint gs-ref-note">
        Пересчитано в нашу нормировку — делением на сумму наших плавней.
      </div>
      <div className="gs-ref-act"><U.Btn kind="ghost" onClick={props.onUse}>взять целью</U.Btn></div>
    </div>
  );
}

// --------------------------------------------------------------- left column

function TargetPanel(props) {
  const U = window.GSUI;
  const G = window.GS.OXIDE_GROUPS;
  const umf = props.umf;
  const keys = Object.keys(umf);
  const best = props.best;
  const plain = !best;
  const inGroup = (id) => keys.filter((k) => U.groupOf(k) === id)
    .sort((a, b) => (U.isR2O(b) ? 1 : 0) - (U.isR2O(a) ? 1 : 0));
  const r2oro = inGroup('r2o_ro');
  const lastR2O = r2oro.filter(U.isR2O).slice(-1)[0];

  const rowProps = (ox) => ({
    value: umf[ox],
    actual: best ? best.umf[ox] : undefined,
    spread: best && best.spread ? best.spread[ox] : undefined,
    role: props.roles[ox] || guessRole(ox, parseFloat(umf[ox]) || 0),
    plain: plain,
    onChange: (v) => props.onChange(ox, v),
    onRole: props.onRole,
    onRemove: () => props.onRemove(ox),
  });

  const fluxSum = [].concat(G.r2o, G.ro).reduce((a, k) => a + (parseFloat(umf[k]) || 0), 0);
  const normalized = Math.abs(fluxSum - 1) <= 0.0005;

  return (
    <U.Panel title="Целевая формула Зегера" scroll>
      <div className={'gs-colhead' + (plain ? ' is-plain' : '')}>
        <span /><span>цель</span><span>роль</span>
        {plain ? null : <span>разница</span>}
        {plain ? null : <span>факт</span>}
        <span />
      </div>
      <OxideGroup title="R₂O / RO" oxides={r2oro} dividerAfter={lastR2O} rowProps={rowProps}
        right={normalized ? null : (
          <button type="button" className="gs-normalize" onClick={props.onNormalize}
            title="в unity-нормировке сумма плавней равна 1.000 — привести">
            Σ {U.num(fluxSum, 3)} → 1.000
          </button>
        )} />
      <OxideGroup title="R₂O₃" oxides={inGroup('r2o3')} rowProps={rowProps} />
      <OxideGroup title="RO₂" oxides={inGroup('ro2')} rowProps={rowProps} />
      {props.addOxide}
      <RatiosStrip umf={umf} />
      <StullChart target={{ x: parseFloat(umf.SiO2) || 0, y: parseFloat(umf.Al2O3) || 0 }}
        points={props.points} />
      <ImportedRef data={props.importedRef} onClear={props.onClearRef} onUse={props.onUseRef} />
    </U.Panel>
  );
}

// -------------------------------------------------------------- right column

function MaterialsPanel(props) {
  const U = window.GSUI;
  const all = window.GS.MATERIALS;
  const [q, setQ] = React.useState('');
  const [scope, setScope] = React.useState('inv');
  const pool = all.filter((m) => {
    if (scope === 'inv' && !m.inv) return false;
    if (q && m.name.toLowerCase().indexOf(q.toLowerCase()) < 0) return false;
    return true;
  });
  const groups = [];
  pool.forEach((m) => {
    let g = groups.filter((x) => x.name === m.group)[0];
    if (!g) { g = { name: m.group, items: [] }; groups.push(g); }
    g.items.push(m);
  });
  const hint = (m) => Object.keys(m.formula).sort((a, b) => m.formula[b] - m.formula[a])
    .map((k) => k + ' ' + m.formula[k]).join(' · ');
  return (
    <U.Panel title="Мой инвентарь"
      right={<span className="gs-count">{props.selected.length} из {pool.length}</span>} scroll>
      <div className="gs-mfilter">
        <input className="gs-search" placeholder="поиск материала…" value={q}
          onChange={(e) => setQ(e.target.value)} />
        <U.Seg value={scope} onChange={setScope} options={[
          { v: 'inv', label: 'в наличии', count: all.filter((m) => m.inv).length },
          { v: 'all', label: 'вся база', count: all.length },
        ]} />
      </div>
      {groups.map((g) => (
        <div className="gs-mgroup" key={g.name}>
          <div className="gs-mgroup-hd">{g.name}</div>
          {g.items.map((m) => {
            const on = props.selected.indexOf(m.name) >= 0;
            return (
              <div className={'gs-mrow' + (on ? ' on' : '')} key={m.name}>
                <U.Check checked={on} onChange={() => props.onToggle(m.name)} title={hint(m)}>
                  <span className="gs-mname">{m.name}</span>
                </U.Check>
                {m.soluble ? <span className="gs-flag" title="водорастворимый — часть уйдёт в раствор при замесе">раств.</span> : null}
                <span className={'gs-tol' + (m.tol >= 0.08 ? ' is-loose' : '')}
                  title="правдоподобный разброс паспорта партии">±{(m.tol * 100).toFixed(0)}%</span>
              </div>
            );
          })}
        </div>
      ))}
      <div className="gs-mact">
        <U.Btn kind="ghost" onClick={() => props.onBulk('all', pool)}>выбрать всё</U.Btn>
        <U.Btn kind="ghost" onClick={() => props.onBulk('none', pool)}>сбросить</U.Btn>
      </div>
    </U.Panel>
  );
}

window.GSSIDE = { TargetPanel, MaterialsPanel, StullChart, guessRole };

function GSSide() {
  return null;
}

Object.assign(window, { GSSide });
