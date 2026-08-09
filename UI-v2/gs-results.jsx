// Centre column: the verdict and the solutions.
// The step-by-step and interval modes were removed: neither had a use case a
// potter could name. What the step mode was really for - "why is this material
// here" - survives as one column in the recipe table.

function Verdict(props) {
  const U = window.GSUI;
  const f = props.feasibility;
  if (!f) return null;
  if (f.feasible) {
    return (
      <div className="gs-verdict tone-ok">
        <span className="gs-dot" />
        <span className="gs-verdict-t">Формула достижима из выбранных материалов</span>
        <span className="gs-verdict-s">макс. отклонение {(f.maxRelativeDeviation * 100).toFixed(1)}%</span>
      </div>
    );
  }
  return (
    <div className="gs-verdict tone-bad is-block">
      <div className="gs-verdict-line">
        <span className="gs-dot" />
        <span className="gs-verdict-t">Формула недостижима из этого набора</span>
        <span className="gs-verdict-s">лучшее возможное отклонение {(f.maxRelativeDeviation * 100).toFixed(0)}%</span>
      </div>
      <div className="gs-unreach">
        {f.perOxide.filter((o) => !o.reachable).map((o) => (
          <div className="gs-unreach-row" key={o.oxide}>
            <span className="gs-ox">{U.fmtOx(o.oxide)}</span>
            <span className="gs-unreach-need">нужно <b className="gs-n">{U.num(o.target)}</b></span>
            <span className="gs-unreach-can">достижимо <b className="gs-n">{U.num(o.closest)}</b></span>
            <U.Bar v={o.closest} max={o.target || 1} tone="bad" thin />
            <span className="gs-unreach-fix">{o.why}</span>
          </div>
        ))}
      </div>
      <div className="gs-verdict-act">
        <U.Btn onClick={props.onRelax}>Ослабить цель до достижимой</U.Btn>
      </div>
    </div>
  );
}

function QualityBadges(props) {
  const U = window.GSUI;
  const q = props.quality;
  const loi = props.loi;
  return (
    <div className="gs-badges">
      <U.Badge label="junk" value={q.junk} tone={q.junk > 0 ? 'bad' : 'none'}
        title="компоненты легче 2% — их невозможно точно отвесить" />
      <U.Badge label="min" value={U.num(q.minPortion, 1) + '%'}
        tone={q.minPortion < 1 ? 'bad' : q.minPortion < 2 ? 'warn' : 'none'}
        title="наименьшая доля компонента" />
      <U.Badge label="дрейф" value={U.num(q.drift, 3)}
        tone={q.drift > 0.02 ? 'bad' : q.drift > 0.01 ? 'warn' : 'none'}
        title="насколько уедет формула, если округлить доли до 0.5% при развеске" />
      <U.Badge label="₽/кг" value={q.cost}
        title="ориентировочно: розница РФ, часть позиций — аналоги" />
      <U.Badge label="глины" value={U.num(q.clay, 0) + '%'}
        tone={q.clay < 5 ? 'warn' : 'none'}
        title="без глинистых глазурь не держится в суспензии" />
      {loi.unknown.length ? (
        <U.Badge label="газы" value="нет данных"
          title={'ППП недостоверны у ' + loi.unknown.length + ': ' + loi.unknown.join(', ')
            + '. Без них оценка газовыделения соврёт в безопасную сторону'} />
      ) : null}
    </div>
  );
}

// Three levels instead of one alarm: a passenger oxide is normal, it starts to
// matter only at the amount where people add it on purpose.
function contamination(oxide, value, target, fluxSum) {
  const D = window.GS;
  const U = window.GSUI;
  if ((parseFloat(target[oxide]) || 0) > 0.0005) return null;
  if (U.groupOf(oxide) === 'r2o_ro') {
    const share = fluxSum ? value / fluxSum : 0;
    return {
      level: share > D.FLUX_SHARE_NOTABLE * 3 ? 'strong' : share > D.FLUX_SHARE_NOTABLE ? 'notable' : 'trace',
      effect: 'съел ' + (share * 100).toFixed(1) + '% бюджета плавней',
    };
  }
  const t = D.OXIDE_THRESHOLD[oxide];
  if (!t) return { level: 'trace', effect: '' };
  if (value >= t.strong) return { level: 'strong', effect: t.effect };
  if (value >= t.notable) return { level: 'notable', effect: t.effect };
  return { level: 'trace', effect: '' };
}

function UmfStrip(props) {
  const U = window.GSUI;
  const target = props.target;
  const umf = props.umf;
  const spread = props.spread || {};
  const G = window.GS.OXIDE_GROUPS;
  const fluxSum = [].concat(G.r2o, G.ro).reduce((a, k) => a + (umf[k] || 0), 0);

  const keys = Object.keys(umf).filter((k) => umf[k] > 0.0005);
  Object.keys(target).forEach((k) => {
    if (keys.indexOf(k) < 0 && (parseFloat(target[k]) || 0) > 0.0005) keys.push(k);
  });

  const cell = (ox) => {
    const has = umf[ox] !== undefined && umf[ox] > 0.0005;
    const t = parseFloat(target[ox]);
    const cont = has ? contamination(ox, umf[ox], target, fluxSum) : null;
    const delta = has && t > 0 ? umf[ox] - t : 0;
    const sig = spread[ox];
    const tone = !has ? 'miss' : cont ? 'extra-' + cont.level : U.deltaTone(delta);
    return (
      <div className={'gs-ucell tone-' + tone} key={ox}
        title={cont
          ? 'приехал с материалами' + (cont.effect ? ' · ' + cont.effect : '')
          : !has ? 'есть в цели, но решение его не даёт'
            : 'цель ' + U.num(t) + ' · факт ' + U.num(umf[ox])
              + (sig ? ' · разброс партий ±' + U.num(sig) : '')}>
        <span className="gs-ucell-n">{U.fmtOx(ox)}</span>
        <span className="gs-ucell-v gs-n">{has ? U.num(umf[ox]) : '—'}</span>
        {cont ? (
          <span className="gs-ucell-x">
            {cont.level === 'trace'
              ? <span className="gs-ucell-tr">след</span>
              : (
                <button type="button" className="gs-promote"
                  title="внести в цель — солвер перестанет считать это случайностью"
                  onClick={(e) => { e.stopPropagation(); props.onPromote(ox, umf[ox]); }}>
                  {cont.effect} → в цель
                </button>
              )}
          </span>
        ) : has && t > 0 ? (
          <span className="gs-ucell-x"><U.DeltaBar delta={delta} scale={t} /></span>
        ) : <span className="gs-ucell-x" />}
      </div>
    );
  };

  const groups = [{ id: 'r2o_ro', t: 'R₂O/RO' }, { id: 'r2o3', t: 'R₂O₃' }, { id: 'ro2', t: 'RO₂' }];
  return (
    <div className="gs-ustrip">
      {groups.map((g) => {
        const list = keys.filter((k) => U.groupOf(k) === g.id)
          .sort((a, b) => (U.isR2O(b) ? 1 : 0) - (U.isR2O(a) ? 1 : 0));
        if (!list.length) return null;
        return (
          <div className="gs-ustrip-g" key={g.id}>
            <div className="gs-ustrip-t">{g.t}</div>
            {list.map(cell)}
          </div>
        );
      })}
    </div>
  );
}

// "What is this material here for" - the whole point of the deleted step mode,
// compressed into one column.
function RecipeRows(props) {
  const U = window.GSUI;
  const r = props.recipe;
  const names = Object.keys(r).sort((a, b) => r[b] - r[a]);
  const max = Math.max.apply(null, names.map((n) => r[n]).concat([1]));
  const why = props.why || {};
  return (
    <div className="gs-recipe">
      {names.map((n) => (
        <div className={'gs-rrow' + (r[n] < 2 ? ' is-junk' : '')} key={n}>
          <span className="gs-rname" title={n}>{n}</span>
          <span className="gs-rwhy" title="этот оксид материал даёт в наибольшей доле">
            {why[n] ? <React.Fragment>ради {U.fmtOx(why[n])}</React.Fragment> : null}
          </span>
          <U.Bar v={r[n]} max={max} />
          <span className="gs-rval gs-n">{U.num(r[n], 1)}</span>
          {props.onExclude ? (
            <button type="button" className="gs-x" title="исключить из инвентаря и пересчитать"
              onClick={(e) => { e.stopPropagation(); props.onExclude(n); }}>×</button>
          ) : <span />}
        </div>
      ))}
    </div>
  );
}

// "What the recipe rests on": leverage x passport uncertainty, not leverage.
function SensitivityBlock(props) {
  const U = window.GSUI;
  const list = props.sensitivity || [];
  if (!list.length) return null;
  const top = list[0];
  return (
    <div className="gs-sens">
      <div className="gs-micro">На чём держится рецепт</div>
      <div className="gs-sens-lead">
        Главный риск — <b>{top.material}</b>: {(top.share * 100).toFixed(0)}% разброса формулы приходится на него.
      </div>
      <div className="gs-sens-rows">
        {list.map((s) => (
          <div className="gs-sens-row" key={s.material}>
            <span className="gs-rname" title={s.material}>{s.material}</span>
            <U.Bar v={s.share} max={1} tone={s.share > 0.4 ? 'bad' : s.share > 0.15 ? 'warn' : null} />
            <span className="gs-n gs-sens-v">{(s.share * 100).toFixed(1)}%</span>
            <span className="gs-sens-why">
              {s.flux
                ? <React.Fragment>{U.fmtOx(s.via)} — задаёт единицу, двигает формулу целиком</React.Fragment>
                : <React.Fragment>{U.fmtOx(s.via)} · ±{(s.sigma * 100).toFixed(0)}% по паспорту</React.Fragment>}
            </span>
          </div>
        ))}
      </div>
      <div className="gs-hint gs-sens-note">
        Сигмы — оценки по классам сырья, их можно править под свои паспорта.
      </div>
    </div>
  );
}

function ContributionMatrix(props) {
  const U = window.GSUI;
  const c = U.contributions(props.recipe);
  const cols = c.oxides.slice(0, 9);
  return (
    <div className="gs-matrix-wrap">
      <table className="gs-matrix">
        <thead>
          <tr>
            <th className="gs-matrix-name">Материал</th>
            <th>доля</th>
            {cols.map((o) => <th key={o}>{U.fmtOx(o)}</th>)}
            <th title="потери при прокаливании">ППП</th>
          </tr>
        </thead>
        <tbody>
          {c.rows.map((r) => (
            <tr key={r.name}>
              <td className="gs-matrix-name"><span className="gs-rname" title={r.name}>{r.name}</span></td>
              <td className="gs-n">{U.num(r.amt, 1)}</td>
              {cols.map((o) => <td key={o} className="gs-n">{r.ox[o] > 0.005 ? U.num(r.ox[o], 2) : ''}</td>)}
              <td className="gs-n">{r.loi > 0.05 ? U.num(r.loi, 2) : ''}</td>
            </tr>
          ))}
          <tr className="gs-matrix-total">
            <td className="gs-matrix-name">Сырой итог</td>
            <td className="gs-n">100.0</td>
            {cols.map((o) => <td key={o} className="gs-n">{U.num(c.totals[o], 2)}</td>)}
            <td className="gs-n">{U.num(c.loi, 2)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function SolutionCard(props) {
  const U = window.GSUI;
  const s = props.solution;
  const active = props.active;
  const [matrix, setMatrix] = React.useState(false);
  const loi = U.loiStatus(s.recipe);
  return (
    <article className={'gs-card' + (active ? ' on' : '') + (s.recommended ? ' is-rec' : '')}
      onClick={props.onSelect} data-screen-label={'Решение: ' + s.label}>
      <header className="gs-card-hd">
        <span className={'gs-label' + (s.recommended ? ' is-rec' : '')}>{s.label}</span>
        <span className="gs-labelwhy">{s.labelWhy}</span>
        <span className="gs-card-act">
          <span className="gs-n gs-err-v" title="расхождение с целью">{(s.error * 100).toFixed(2)}%</span>
          <span className="gs-mcount">{Object.keys(s.recipe).length} матер.</span>
          {active ? (
            <U.Btn kind="icon" on={matrix} title="матрица вкладов: какой материал что даёт"
              onClick={(e) => { e.stopPropagation(); setMatrix(!matrix); }}>▦</U.Btn>
          ) : null}
          <U.Btn kind="icon" title="замес на вес"
            onClick={(e) => { e.stopPropagation(); props.onBatch(); }}>⚖</U.Btn>
        </span>
      </header>
      {s.collapsed ? (
        <div className="gs-collapsed">свёрнуто ещё {s.collapsed} почти такое же решение</div>
      ) : null}
      <QualityBadges quality={s.quality} loi={loi} />
      <RecipeRows recipe={s.recipe} why={U.whyMaterials(s.recipe)}
        onExclude={active ? props.onExclude : null} />
      {active ? <SensitivityBlock sensitivity={s.sensitivity} /> : null}
      {active && matrix ? <ContributionMatrix recipe={s.recipe} /> : null}
      {active ? <UmfStrip umf={s.umf} target={props.target} spread={s.spread} onPromote={props.onPromote} /> : null}
    </article>
  );
}

function SolutionsView(props) {
  const U = window.GSUI;
  return (
    <div className="gs-results">
      <Verdict feasibility={props.feasibility} onRelax={props.onRelax} />
      <div className="gs-toolbar">
        <span className="gs-micro">Варианты</span>
        <span className="gs-hint">похожие свёрнуты — показаны те, что отличаются по существу</span>
      </div>
      <div className="gs-cards">
        {props.solutions.map((s) => (
          <SolutionCard key={s.id} solution={s} target={props.target}
            active={props.activeId === s.id}
            onSelect={() => props.onSelect(s.id)}
            onBatch={() => props.onBatch(s.id)}
            onExclude={props.onExclude}
            onPromote={props.onPromote} />
        ))}
        {!props.solutions.length ? <U.Empty>Решений не найдено</U.Empty> : null}
      </div>
    </div>
  );
}

window.GSRES = { SolutionsView, Verdict, RecipeRows, UmfStrip, ContributionMatrix, SensitivityBlock };

function GSRes() {
  return null;
}

Object.assign(window, { GSRes });
