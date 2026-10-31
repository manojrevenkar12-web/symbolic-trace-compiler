import type { Check } from '../../shared/lib/results';
import styles from './gate.module.css';

const fmt = (v: number | null) => (v == null ? 'n/a' : Number.isInteger(v) ? String(v) : v.toPrecision(3));

function List({ items, kind }: { items: Check[]; kind: 'floor' | 'target' }) {
  return (
    <ul className={styles.list}>
      {items.map((c) => (
        <li key={c.name} data-ok={c.ok}>
          <span className={styles.mark}>{c.ok ? (kind === 'floor' ? 'Pass' : 'Met') : kind === 'floor' ? 'Fail' : 'Missed'}</span>
          <span>{c.name}</span>
          <span className={styles.nums}>
            {fmt(c.actual)}
            <span className={styles.need}>
              needs {c.op} {c.value}
            </span>
          </span>
        </li>
      ))}
    </ul>
  );
}

export function Gate({ floors, targets }: { floors: Check[]; targets: Check[] }) {
  return (
    <div className={styles.gate}>
      <h3>Floors: a failure stops the release</h3>
      <List items={floors} kind="floor" />
      <h3>Targets: reported, met or not</h3>
      <List items={targets} kind="target" />
    </div>
  );
}
