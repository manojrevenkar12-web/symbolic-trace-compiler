import { useState } from 'react';
import { human, num, pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { CHECK_LABEL, POLICY, WORKLOAD_LABEL } from '../../shared/theme/tokens';
import { ScrollRegion, tableClass } from '../../shared/ui/Section';
import styles from './soundness.module.css';

export function Soundness({ r }: { r: Results }) {
  return (
    <ScrollRegion label="Fuzzing the cached programs">
      <table className={tableClass}>
        <caption className="visually-hidden">Fuzz draws per workload and policy: accepted, violations and coverage</caption>
        <thead>
          <tr>
            <th scope="col">Workload</th>
            <th scope="col">Policy</th>
            <th scope="col">Draws</th>
            <th scope="col">Eager accepts</th>
            <th scope="col">A program accepts</th>
            <th scope="col">Wrong results</th>
            <th scope="col">Coverage</th>
          </tr>
        </thead>
        <tbody>
          {(r.fuzz ?? []).map((f) => (
            <tr key={`${f.workload}-${f.policy}`}>
              <th scope="row">{WORKLOAD_LABEL[f.workload] ?? human(f.workload)}</th>
              <td style={{ color: POLICY[f.policy].color }}>{POLICY[f.policy].label}</td>
              <td>{num(f.draws)}</td>
              <td>{num(f.eager_ok)}</td>
              <td>{num(f.accepted)}</td>
              <td className={f.violations ? styles.bad : styles.good}>{num(f.violations)}</td>
              <td>{pct(f.coverage)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollRegion>
  );
}

export function Necessity({ r }: { r: Results }) {
  const names = Object.keys(r.necessity ?? {});
  const [open, setOpen] = useState(false);
  if (!names.length) return null;
  return (
    <div className={styles.necessity}>
      <button type="button" className={styles.toggle} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        {open ? 'Hide' : 'Show'} what each size check protects
      </button>
      {open && (
        <ScrollRegion label="What each size check protects">
          <table className={tableClass}>
            <caption className="visually-hidden">Each check removed in turn, and what the fuzzer then found</caption>
            <thead>
              <tr>
                <th scope="col">Workload</th>
                <th scope="col">Check</th>
                <th scope="col">Without it</th>
              </tr>
            </thead>
            <tbody>
              {names.flatMap((n) =>
                (r.necessity?.[n] ?? []).map((c) => (
                  <tr key={`${n}-${c.check}`}>
                    <th scope="row">{WORKLOAD_LABEL[n] ?? human(n)}</th>
                    <td className={styles.code}>{c.check}</td>
                    <td className={c.without_it === 'wrong results' ? styles.bad : undefined}>{CHECK_LABEL[c.without_it]}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </ScrollRegion>
      )}
    </div>
  );
}
