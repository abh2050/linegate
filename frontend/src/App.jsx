import { useCallback, useEffect, useState } from 'react';

const ACTIONS = { h: 'ship', s: 'scrap', r: 'senior_review' };

async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

function Disposition({ disposition }) {
  if (disposition.status === 'valid') {
    return (
      <section className="disposition valid" data-testid="disposition-valid">
        <h3>Agent recommendation: {disposition.recommendation.replace('_', ' ')}</h3>
        <p>{disposition.reason}</p>
        <p className="muted">{disposition.neighbor_summary}</p>
      </section>
    );
  }
  if (disposition.status === 'rejected') {
    return (
      <section className="disposition rejected" data-testid="disposition-rejected">
        <h3>Agent output rejected: route to senior review</h3>
        <ul>{disposition.errors.map((e) => <li key={e}>{e}</li>)}</ul>
      </section>
    );
  }
  return (
    <section className="disposition missing" data-testid="disposition-missing">
      <h3>No agent write-up: route to senior review</h3>
    </section>
  );
}

function Detail({ detail }) {
  const { part, neighbors, disposition } = detail;
  return (
    <div className="detail" data-testid="detail" data-part-id={part.part_id}>
      <h2>Part {part.part_id} <span className="score">score {part.score.toFixed(3)}</span></h2>
      <p><strong>Route</strong> {part.route || '(no stations recorded)'}</p>
      <h3>Out of normal range ({part.out_of_range.length})</h3>
      <table>
        <thead><tr><th>Column</th><th>Value</th><th>Normal range</th></tr></thead>
        <tbody>
          {part.out_of_range.map((m) => (
            <tr key={m.column}><td>{m.column}</td><td>{m.value}</td><td>{m.low} to {m.high}</td></tr>
          ))}
        </tbody>
      </table>
      <p><strong>Similar parts</strong> {neighbors.failed} of {neighbors.k} failed</p>
      <Disposition disposition={disposition} />
    </div>
  );
}

export default function App() {
  const [queue, setQueue] = useState([]);
  const [index, setIndex] = useState(0);
  const [detail, setDetail] = useState(null);
  const [lastAction, setLastAction] = useState('');

  const loadQueue = useCallback(async () => {
    const data = await getJSON('/api/queue?limit=200');
    setQueue(data.items);
    setIndex((i) => Math.min(i, Math.max(0, data.items.length - 1)));
  }, []);

  useEffect(() => { loadQueue(); }, [loadQueue]);

  const selected = queue[index];
  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    let cancelled = false;
    getJSON(`/api/parts/${selected.part_id}`).then((d) => { if (!cancelled) setDetail(d); });
    return () => { cancelled = true; };
  }, [selected?.part_id]);

  const decide = useCallback(async (action) => {
    if (!selected) return;
    await getJSON(`/api/parts/${selected.part_id}/disposition`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }),
    });
    setLastAction(`Part ${selected.part_id}: ${action.replace('_', ' ')}`);
    await loadQueue();
  }, [selected, loadQueue]);

  useEffect(() => {
    const onKey = (event) => {
      if (event.target instanceof HTMLInputElement || event.metaKey || event.ctrlKey) return;
      if (event.key === 'j' || event.key === 'ArrowDown') setIndex((i) => Math.min(i + 1, queue.length - 1));
      else if (event.key === 'k' || event.key === 'ArrowUp') setIndex((i) => Math.max(i - 1, 0));
      else if (ACTIONS[event.key]) decide(ACTIONS[event.key]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [queue.length, decide]);

  return (
    <div className="app">
      <header>
        <h1>linegate review queue</h1>
        <p className="keys">j / k move · h ship · s scrap · r senior review</p>
        <p className="last" data-testid="last-action" aria-live="polite">{lastAction}</p>
      </header>
      <main>
        <ol className="queue" data-testid="queue">
          {queue.map((item, i) => (
            <li key={item.part_id} data-testid="queue-item" data-part-id={item.part_id}
                className={i === index ? 'selected' : ''} onClick={() => setIndex(i)}>
              <span>#{item.part_id}</span><span className="score">{item.score.toFixed(3)}</span>
            </li>
          ))}
        </ol>
        {detail ? <Detail detail={detail} /> : <div className="detail empty">Queue is empty</div>}
      </main>
    </div>
  );
}
