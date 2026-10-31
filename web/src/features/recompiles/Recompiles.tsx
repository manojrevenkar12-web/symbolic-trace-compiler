import { human, num } from '../../shared/lib/format';
import { byPolicy, POLICIES, type Results } from '../../shared/lib/results';
import { POLICY, WORKLOAD_LABEL } from '../../shared/theme/tokens';
import { ScrollRegion, subClass, tableClass } from '../../shared/ui/Section';
import styles from './recompiles.module.css';

/** Programs compiled per workload and policy, with the share of calls that compiled as a bar. */
export function Recompiles({ r }: { r: Results }) {
  const names = Object.keys(r.workloads ?? {});
  return (
    <ScrollRegion label="Programs compiled per workload">
      <table className={tableClass}>
        <caption className="visually-hidden">Programs compiled for each workload under each policy</caption>
        <thead>
          <tr>
            <th scope="col">Workload</th>
            <th scope="col">Calls</th>
            <th scope="col">Distinct shapes</th>
            {POLICIES.map((p) => (
              <th scope="col" key={p}>
                {POLICY[p].label}
                <span className={subClass}>{POLICY[p].note}</span>
              </th>
            ))}
            <th scope="col">Compile time, static → symbolic</th>
          </tr>
        </thead>
        <tbody>
          {names.map((n) => {
            const row = byPolicy(r.rows ?? [], n);
            const calls = row.static?.calls ?? 0;
            return (
              <tr key={n}>
                <th scope="row">
                  {WORKLOAD_LABEL[n] ?? human(n)}
                  <span className={subClass}>{r.workloads?.[n]?.about}</span>
                </th>
                <td>{num(calls)}</td>
                <td>{num(row.static?.distinct_shapes)}</td>
                {POLICIES.map((p) => {
                  const c = row[p]?.compiles ?? 0;
                  return (
                    <td key={p} className={styles.cell}>
                      <span className={styles.value} style={{ color: POLICY[p].color }}>
                        {num(c)}
                      </span>
                      <span className={styles.track} aria-hidden="true">
                        <span className={styles.bar} style={{ width: `${calls ? (100 * c) / calls : 0}%`, background: POLICY[p].color }} />
                      </span>
                    </td>
                  );
                })}
                <td>
                  {num((row.static?.compile_ms ?? 0) / 1000, 1)} s → {num((row.symbolic?.compile_ms ?? 0) / 1000, 1)} s
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </ScrollRegion>
  );
}
