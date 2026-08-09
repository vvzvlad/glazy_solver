// Application shell. One screen, one job: a Seger formula in, a recipe out.
// The mode tabs and the solver settings panel are gone - neither had a use
// case a potter could name, and the engine choice was our own kitchen.

function ImportField(props) {
  const U = window.GSUI;
  const [text, setText] = React.useState('');
  const [state, setState] = React.useState('idle');
  const run = () => {
    if (!text.trim()) return;
    setState('loading');
    setTimeout(() => {
      setState('idle');
      props.onImport(window.GS.GLAZY_IMPORT);
      setText('');
    }, 600);
  };
  return (
    <div className="gs-import">
      <span className="gs-import-l">Формула</span>
      <input className="gs-import-i"
        placeholder="вставьте формулу Зегера или ссылку на рецепт"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter') run(); }} />
      <U.Seg value={props.source} onChange={props.onSource} options={[
        { v: 'glazy', label: 'Glazy', title: 'их конвенция unity отличается — пересчитаем' },
        { v: 'seger', label: 'Seger', title: 'Fe₂O₃ у них плавень — пересчитаем' },
        { v: 'raw', label: 'своя', title: 'уже в нашей нормировке' },
      ]} />
      <U.Btn kind="ghost" onClick={run}>{state === 'loading' ? '…' : '→'}</U.Btn>
    </div>
  );
}

function StatusChip(props) {
  const map = {
    ready: { t: 'готово', c: 'ok' },
    stale: { t: 'изменено', c: 'warn' },
    busy: { t: 'считается', c: 'busy' },
  };
  const s = map[props.state] || map.ready;
  return <span className={'gs-status is-' + s.c}><i className="gs-dot" />{s.t}</span>;
}

function AddOxide(props) {
  const U = window.GSUI;
  const [open, setOpen] = React.useState(false);
  const pool = ['Li2O', 'BaO', 'SrO', 'ZnO', 'MnO', 'FeO', 'CuO', 'CoO', 'NiO',
    'Fe2O3', 'Cr2O3', 'P2O5', 'TiO2', 'ZrO2', 'SnO2'].filter((o) => props.used.indexOf(o) < 0);
  if (!open) return <button type="button" className="gs-add" onClick={() => setOpen(true)}>+ оксид</button>;
  return (
    <div className="gs-addbox">
      <div className="gs-addlist">
        {pool.map((o) => (
          <button type="button" className="gs-addopt" key={o}
            onClick={() => { props.onAdd(o); setOpen(false); }}>{U.fmtOx(o)}</button>
        ))}
      </div>
    </div>
  );
}

function BatchPanel(props) {
  const U = window.GSUI;
  const [mass, setMass] = React.useState('1000');
  const m = parseFloat(mass) || 0;
  const r = props.recipe;
  const names = Object.keys(r).sort((a, b) => r[b] - r[a]);
  // Round to 0.1 g and give the rounding remainder to the largest component,
  // so the batch actually adds up to the mass that was asked for.
  const grams = {};
  let acc = 0;
  names.forEach((n, i) => {
    if (i === names.length - 1) { grams[n] = Math.round((m - acc) * 10) / 10; return; }
    const g = Math.round((r[n] / 100) * m * 10) / 10;
    grams[n] = g;
    acc += g;
  });
  const total = names.reduce((a, n) => a + grams[n], 0);
  return (
    <div className="gs-drawer">
      <div className="gs-drawer-hd">
        <span className="gs-micro">Замес</span>
        <span className="gs-drawer-in"><U.NumField value={mass} onChange={setMass} width={62} /> г</span>
        <button type="button" className="gs-x" onClick={props.onClose}>×</button>
      </div>
      <div className="gs-batch">
        {names.map((n) => (
          <div className="gs-batch-row" key={n}>
            <span className="gs-rname">{n}</span>
            <span className="gs-n gs-batch-g">{grams[n].toFixed(1)}</span>
          </div>
        ))}
        <div className="gs-batch-row is-total">
          <span className="gs-rname">ВСЕГО</span>
          <span className="gs-n gs-batch-g">{total.toFixed(1)}</span>
        </div>
      </div>
    </div>
  );
}

function Workspace() {
  const U = window.GSUI;
  const S = window.GSSIDE;
  const R = window.GSRES;
  const D = window.GS;

  const [theme, setTheme] = React.useState('light');
  const [name, setName] = React.useState('Прозрачная глазурь △6');
  const [umf, setUmf] = React.useState(
    Object.keys(D.TARGET).reduce((a, k) => { a[k] = String(D.TARGET[k]); return a; }, {}),
  );
  const [roles, setRoles] = React.useState({});
  const [inventory, setInventory] = React.useState(D.DEFAULT_INVENTORY.slice());
  const [status, setStatus] = React.useState('ready');
  const [activeId, setActiveId] = React.useState('s1');
  const [batchId, setBatchId] = React.useState(null);
  const [scenario, setScenario] = React.useState('ok');
  const [importedRef, setImportedRef] = React.useState(null);
  const [source, setSource] = React.useState('glazy');

  const touch = () => {
    setStatus('stale');
    clearTimeout(window.__gsTimer);
    window.__gsTimer = setTimeout(() => {
      setStatus('busy');
      setTimeout(() => setStatus('ready'), 400);
    }, 500);
  };

  const solutions = D.SOLUTIONS;
  const active = solutions.filter((s) => s.id === activeId)[0] || solutions[0];
  const feasibility = scenario === 'bad' ? D.FEASIBILITY_BAD : D.FEASIBILITY_OK;
  const numericUmf = Object.keys(umf).reduce((a, k) => { a[k] = parseFloat(umf[k]) || 0; return a; }, {});

  const normalize = () => {
    const G = D.OXIDE_GROUPS;
    const flux = [].concat(G.r2o, G.ro);
    const sum = flux.reduce((a, k) => a + (parseFloat(umf[k]) || 0), 0);
    if (!sum) return;
    const next = {};
    Object.keys(umf).forEach((k) => {
      next[k] = String(Math.round(((parseFloat(umf[k]) || 0) / sum) * 1000) / 1000);
    });
    setUmf(next);
    touch();
  };

  return (
    <div className={'gs ' + (theme === 'dark' ? 'gs-dark' : 'gs-light')}>
      <header className="gs-hd" data-screen-label="Шапка">
        <span className="gs-logo" aria-hidden="true"><i /><i /><i /></span>
        <input className="gs-title" value={name} onChange={(e) => setName(e.target.value)} />
        <ImportField source={source} onSource={setSource}
          onImport={(r) => { setImportedRef(r); touch(); }} />
        <span className="gs-hd-sp" />
        <span className="gs-scen">
          <span className="gs-micro">демо</span>
          <U.Seg value={scenario} onChange={setScenario} options={[
            { v: 'ok', label: 'достижимо' },
            { v: 'bad', label: 'недостижимо' },
          ]} />
        </span>
        <StatusChip state={status} />
        <U.Btn kind="ghost" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
          {theme === 'dark' ? '◐' : '◑'}
        </U.Btn>
        <U.Btn kind="ghost" title="ссылка на эту формулу">⇧</U.Btn>
      </header>

      <div className="gs-body">
        <div className="gs-col gs-col-l">
          <S.TargetPanel
            umf={umf}
            roles={roles}
            best={active}
            points={solutions.map((s) => ({
              id: s.id, x: s.umf.SiO2 || 0, y: s.umf.Al2O3 || 0, label: s.label,
            }))}
            importedRef={importedRef}
            onChange={(ox, v) => { setUmf(Object.assign({}, umf, { [ox]: v })); touch(); }}
            onRole={(ox, role) => { setRoles(Object.assign({}, roles, { [ox]: role })); touch(); }}
            onRemove={(ox) => {
              const next = Object.assign({}, umf);
              delete next[ox];
              setUmf(next);
              touch();
            }}
            onNormalize={normalize}
            onClearRef={() => setImportedRef(null)}
            onUseRef={() => {
              setUmf(Object.keys(importedRef.umf).reduce((a, k) => {
                a[k] = String(importedRef.umf[k]);
                return a;
              }, {}));
              touch();
            }}
            addOxide={<AddOxide used={Object.keys(umf)}
              onAdd={(ox) => { setUmf(Object.assign({}, umf, { [ox]: '0' })); touch(); }} />}
          />
        </div>

        <div className="gs-col gs-col-c">
          <R.SolutionsView
            solutions={solutions}
            target={numericUmf}
            feasibility={feasibility}
            activeId={active ? active.id : null}
            onSelect={setActiveId}
            onBatch={setBatchId}
            onExclude={(n) => { setInventory(inventory.filter((x) => x !== n)); touch(); }}
            onPromote={(ox, value) => {
              setUmf(Object.assign({}, umf, { [ox]: String(Math.round(value * 1000) / 1000) }));
              setRoles(Object.assign({}, roles, { [ox]: 'pass' }));
              touch();
            }}
            onRelax={() => {}}
          />
        </div>

        <div className="gs-col gs-col-r">
          <S.MaterialsPanel
            selected={inventory}
            onToggle={(n) => {
              setInventory(inventory.indexOf(n) >= 0 ? inventory.filter((x) => x !== n) : inventory.concat([n]));
              touch();
            }}
            onBulk={(what, pool) => {
              setInventory(what === 'all' ? pool.map((m) => m.name) : []);
              touch();
            }}
          />
        </div>
      </div>

      {batchId ? (
        <BatchPanel recipe={solutions.filter((s) => s.id === batchId)[0].recipe}
          onClose={() => setBatchId(null)} />
      ) : null}
    </div>
  );
}

function App() {
  const [ready, setReady] = React.useState(
    !!(window.GS && window.GSUI && window.GSSIDE && window.GSRES),
  );
  React.useEffect(() => {
    if (ready) return undefined;
    const t = setInterval(() => {
      if (window.GS && window.GSUI && window.GSSIDE && window.GSRES) {
        setReady(true);
        clearInterval(t);
      }
    }, 40);
    return () => clearInterval(t);
  }, [ready]);
  if (!ready) return <div className="gs-boot">загрузка модулей…</div>;
  return <Workspace />;
}

Object.assign(window, { App });
