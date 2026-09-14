import { useCallback, useEffect, useMemo, useState } from 'react';

const ACTIONS = [
  { key: 'h', action: 'ship', label: 'Ship', tone: 'ship' },
  { key: 's', action: 'scrap', label: 'Scrap', tone: 'scrap' },
  { key: 'r', action: 'senior_review', label: 'Senior review', tone: 'review' },
];
const ACTION_BY_KEY = Object.fromEntries(ACTIONS.map((a) => [a.key, a.action]));
const LABEL_BY_ACTION = Object.fromEntries(ACTIONS.map((a) => [a.action, a.label]));

async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

const fmt = (v, digits = 3) => (v === null || v === undefined ? '–' : Number(v).toFixed(digits));

function Kbd({ children }) {
  return <kbd className="kbd">{children}</kbd>;
}

function ScoreGauge({ score, policy }) {
  if (!policy?.band_high) return null;
  const max = Math.max(policy.band_high * 1.6, score * 1.1);
  const pos = (v) => Math.sqrt(Math.max(0, v) / max) * 100;  // square-root scale keeps small thresholds readable
  const low = pos(policy.band_low), high = pos(policy.band_high);
  return (
    <div className="gauge" aria-label={`score ${fmt(score)} within band ${fmt(policy.band_low)} to ${fmt(policy.band_high)}`}>
      <div className="gauge-track">
        <div className="gauge-zone ship" style={{ left: 0, width: `${low}%` }} />
        <div className="gauge-zone band" style={{ left: `${low}%`, width: `${high - low}%` }} />
        <div className="gauge-zone inspect" style={{ left: `${high}%`, right: 0 }} />
        <div className="gauge-threshold" style={{ left: `${pos(policy.threshold)}%` }} />
        <div className="gauge-marker" style={{ left: `${pos(score)}%` }} />
      </div>
      <div className="gauge-ticks">
        {[0, policy.band_low, policy.threshold, policy.band_high].map((t) => (
          <span key={t} style={{ left: `${pos(t)}%` }} className="mono">{fmt(t)}</span>
        ))}
      </div>
      <div className="gauge-legend">
        <span><i className="swatch ship" /> ship</span>
        <span><i className="swatch band" /> review band</span>
        <span><i className="swatch threshold" /> cost-optimal threshold</span>
        <span><i className="swatch inspect" /> inspect</span>
        <span className="muted">square-root scale</span>
      </div>
    </div>
  );
}

function Route({ route }) {
  if (!route) return <p className="muted">No stations recorded for this part.</p>;
  const lines = [];
  for (const stop of route.split('-')) {
    const [line, station] = stop.split('_');
    const last = lines[lines.length - 1];
    if (last && last.line === line) last.stations.push(station);
    else lines.push({ line, stations: [station] });
  }
  return (
    <div className="route">
      {lines.map((group, i) => (
        <div className="route-line" key={`${group.line}-${i}`}>
          <span className="route-label">{group.line}</span>
          <div className="route-stations">
            {group.stations.map((s) => <span className="chip" key={s}>{s}</span>)}
          </div>
        </div>
      ))}
    </div>
  );
}

function Deviation({ m }) {
  const width = Math.max(m.high - m.low, 1e-9);
  const beyond = m.value > m.high ? (m.value - m.high) / width : (m.low - m.value) / width;
  const side = m.value > m.high ? 'high' : 'low';
  return (
    <span className={`deviation ${side}`} title={`${(beyond * 100).toFixed(0)}% of the normal range beyond the ${side === 'high' ? 'upper' : 'lower'} edge`}>
      <span className="deviation-track"><span className="deviation-bar" style={{ width: `${Math.min(100, 6 + beyond * 94)}%` }} /></span>
      <span className="deviation-text mono">{side === 'high' ? 'above' : 'below'} {(beyond * 100).toFixed(0)}%</span>
    </span>
  );
}

function Measurements({ items }) {
  if (!items.length) return <p className="muted">Every recorded measurement is inside its normal range.</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr><th>Column</th><th className="num">Value</th><th className="num">Normal range (p1–p99)</th><th>Outside by</th></tr>
        </thead>
        <tbody>
          {items.map((m) => (
            <tr key={m.column}>
              <td className="mono">{m.column}</td>
              <td className="num mono">{fmt(m.value)}</td>
              <td className="num mono muted">{fmt(m.low)} to {fmt(m.high)}</td>
              <td><Deviation m={m} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Neighbors({ neighbors }) {
  const passed = neighbors.k - neighbors.failed;
  return (
    <div>
      <div className="split-bar" aria-label={`${neighbors.failed} failed, ${passed} passed`}>
        <span className="split-fail" style={{ flexGrow: neighbors.failed }} />
        <span className="split-pass" style={{ flexGrow: passed }} />
      </div>
      <p className="split-caption"><strong>{neighbors.failed}</strong> of {neighbors.k} similar parts failed
        <span className="muted"> · {neighbors.same_route ? 'same route' : 'any route'}</span></p>
      <ul className="neighbor-list">
        {neighbors.neighbors.slice(0, 10).map((n) => (
          <li key={n.part_id}>
            <span className="mono">#{n.part_id}</span>
            <span className={`pill ${n.response ? 'fail' : 'pass'}`}>{n.response ? 'failed' : 'passed'}</span>
            <span className="muted mono">d {fmt(n.distance, 2)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Disposition({ disposition }) {
  if (disposition.status === 'valid') {
    return (
      <section className="card agent valid" data-testid="disposition-valid">
        <header className="card-head">
          <h3>Agent write-up</h3>
          <span className={`badge ${disposition.recommendation}`}>{LABEL_BY_ACTION[disposition.recommendation]}</span>
        </header>
        <p className="lead">{disposition.reason}</p>
        <p className="muted">{disposition.neighbor_summary}</p>
        <p className="fine">Citations checked against the part record · the model does not score parts</p>
      </section>
    );
  }
  if (disposition.status === 'rejected') {
    return (
      <section className="card agent rejected" data-testid="disposition-rejected">
        <header className="card-head">
          <h3>Agent write-up rejected</h3>
          <span className="badge senior_review">Senior review</span>
        </header>
        <p>The write-up cited evidence the tools never returned, so it is withheld.</p>
        <ul className="errors">{disposition.errors.map((e) => <li key={e} className="mono">{e}</li>)}</ul>
      </section>
    );
  }
  return (
    <section className="card agent missing" data-testid="disposition-missing">
      <header className="card-head">
        <h3>No agent write-up</h3>
        <span className="badge senior_review">Senior review</span>
      </header>
      <p className="muted">Decide from the evidence above, or send the part to senior review.</p>
    </section>
  );
}

function Detail({ detail, policy, onDecide, busy }) {
  const { part, neighbors, disposition } = detail;
  return (
    <div className="detail" data-testid="detail" data-part-id={part.part_id}>
      <section className="card hero">
        <div className="hero-top">
          <div>
            <p className="eyebrow">Part in review band</p>
            <h2 className="mono">#{part.part_id}</h2>
          </div>
          <div className="score">
            <span className="score-value mono">{fmt(part.score)}</span>
            <span className="muted">failure score</span>
          </div>
        </div>
        <ScoreGauge score={part.score} policy={policy} />
        <div className="actions">
          {ACTIONS.map((a) => (
            <button key={a.action} type="button" className={`btn ${a.tone}`} disabled={busy} onClick={() => onDecide(a.action)}>
              {a.label} <Kbd>{a.key}</Kbd>
            </button>
          ))}
        </div>
      </section>

      <Disposition disposition={disposition} />

      <section className="card">
        <header className="card-head"><h3>Route</h3><span className="muted">{part.measurements_present} measurements recorded</span></header>
        <Route route={part.route} />
      </section>

      <div className="grid-2">
        <section className="card">
          <header className="card-head"><h3>Outside normal range</h3><span className="count">{part.out_of_range.length}</span></header>
          <Measurements items={part.out_of_range} />
        </section>
        <section className="card">
          <header className="card-head"><h3>Similar train parts</h3></header>
          <Neighbors neighbors={neighbors} />
        </section>
      </div>
    </div>
  );
}

export default function App() {
  const [queue, setQueue] = useState({ total: 0, decided: 0, items: [] });
  const [policy, setPolicy] = useState(null);
  const [index, setIndex] = useState(0);
  const [detail, setDetail] = useState(null);
  const [lastAction, setLastAction] = useState('');
  const [busy, setBusy] = useState(false);

  const loadQueue = useCallback(async () => {
    const data = await getJSON('/api/queue?limit=200');
    setQueue(data);
    setIndex((i) => Math.min(i, Math.max(0, data.items.length - 1)));
  }, []);

  useEffect(() => { loadQueue(); getJSON('/api/policy').then(setPolicy).catch(() => setPolicy(null)); }, [loadQueue]);

  const selected = queue.items[index];
  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    let cancelled = false;
    getJSON(`/api/parts/${selected.part_id}`).then((d) => { if (!cancelled) setDetail(d); });
    return () => { cancelled = true; };
  }, [selected?.part_id]);

  useEffect(() => {
    document.querySelector('.queue li.selected')?.scrollIntoView({ block: 'nearest' });
  }, [index]);

  const decide = useCallback(async (action) => {
    if (!selected || busy) return;
    setBusy(true);
    try {
      await getJSON(`/api/parts/${selected.part_id}/disposition`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }),
      });
      setLastAction(`Part ${selected.part_id}: ${action.replace('_', ' ')}`);
      await loadQueue();
    } finally {
      setBusy(false);
    }
  }, [selected, busy, loadQueue]);

  useEffect(() => {
    const onKey = (event) => {
      if (event.target instanceof HTMLInputElement || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === 'j' || event.key === 'ArrowDown') { event.preventDefault(); setIndex((i) => Math.min(i + 1, queue.items.length - 1)); }
      else if (event.key === 'k' || event.key === 'ArrowUp') { event.preventDefault(); setIndex((i) => Math.max(i - 1, 0)); }
      else if (ACTION_BY_KEY[event.key]) decide(ACTION_BY_KEY[event.key]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [queue.items.length, decide]);

  const writeUps = useMemo(() => queue.items.filter((q) => q.has_disposition).length, [queue.items]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <div>
            <h1>linegate</h1>
            <p className="muted">Review queue · parts the cost policy will not decide alone</p>
          </div>
        </div>
        <dl className="stats">
          <div><dt>In queue</dt><dd className="mono">{queue.total.toLocaleString()}</dd></div>
          <div><dt>Decided</dt><dd className="mono">{queue.decided.toLocaleString()}</dd></div>
          <div><dt>Review band</dt><dd className="mono">{policy ? `${fmt(policy.band_low)}–${fmt(policy.band_high)}` : '–'}</dd></div>
          <div><dt>Policy</dt><dd className="mono small">{policy?.version ?? '–'}</dd></div>
        </dl>
      </header>
      <div className="statusbar">
        <span className="keys"><Kbd>j</Kbd><Kbd>k</Kbd> move <Kbd>h</Kbd> ship <Kbd>s</Kbd> scrap <Kbd>r</Kbd> senior review</span>
        <span className="last" data-testid="last-action" aria-live="polite">{lastAction}</span>
      </div>
      <main>
        <aside className="sidebar">
          <div className="sidebar-head">
            <span>Highest score first</span>
            <span className="muted">{writeUps} with write-up</span>
          </div>
          <ol className="queue" data-testid="queue">
            {queue.items.map((item, i) => (
              <li key={item.part_id} data-testid="queue-item" data-part-id={item.part_id}
                  className={i === index ? 'selected' : ''} onClick={() => setIndex(i)}>
                <span className="mono">#{item.part_id}</span>
                <span className="queue-meta">
                  {item.has_disposition && <span className="dot" title="agent write-up available" />}
                  <span className="mono score-small">{fmt(item.score)}</span>
                </span>
              </li>
            ))}
          </ol>
        </aside>
        {detail
          ? <Detail detail={detail} policy={policy} onDecide={decide} busy={busy} />
          : <div className="detail empty">{queue.items.length ? 'Loading part…' : 'The review queue is empty.'}</div>}
      </main>
    </div>
  );
}
