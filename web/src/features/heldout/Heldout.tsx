import { day, num } from '../../shared/lib/format';
import type { HeldSet } from '../../shared/lib/results';
import { POLICY } from '../../shared/theme/tokens';
import { ScrollRegion, subClass, tableClass } from '../../shared/ui/Section';
import styles from './heldout.module.css';

export function HeldoutTable({ name, set }: { name: string; set: HeldSet }) {
  return (
    <div className={styles.set}>
      <h3>
        Set {name}{' '}
        <span className={styles.status}>
          {set.blind_first_run
            ? `blind first run (${day(set.recorded_at)}), kept as recorded${set.status === 'heldout' ? '' : `; now a regression set, rerun today: ${set.rerun.supported} of ${set.programs} supported, ${set.rerun.wrong_or_violations} wrong`}`
            : 'regression set: studied before a first run was recorded, rerun every time'}
        </span>
      </h3>
      <ScrollRegion label={`Programs of set ${name}`}>
        <table className={tableClass}>
          <caption className="visually-hidden">Programs compiled and wrong calls per program</caption>
          <thead>
            <tr>
              <th scope="col">Program</th>
              <th scope="col">Calls</th>
              <th scope="col">{POLICY.static.label}</th>
              <th scope="col">{POLICY.auto.label}</th>
              <th scope="col">{POLICY.symbolic.label}</th>
              <th scope="col">Wrong calls</th>
              <th scope="col">Fuzz: wrong results</th>
            </tr>
          </thead>
          <tbody>
            {set.rows.map((x) => {
              const wrong = x.policies.static.wrong_calls + x.policies.auto.wrong_calls + x.policies.symbolic.wrong_calls;
              const uns = x.policies.symbolic.unsupported;
              return (
                <tr key={x.program}>
                  <th scope="row">
                    {x.program}
                    <span className={subClass}>{x.about}</span>
                    {uns && <span className={styles.unsupported}>Unsupported: {uns}</span>}
                  </th>
                  <td>{num(x.calls)}</td>
                  <td>{num(x.policies.static.compiles)}</td>
                  <td>{num(x.policies.auto.compiles)}</td>
                  <td className={styles.sym}>{num(x.policies.symbolic.compiles)}</td>
                  <td className={wrong ? styles.bad : undefined}>{num(wrong)}</td>
                  <td className={x.fuzz?.violations ? styles.bad : undefined}>{x.fuzz ? num(x.fuzz.violations) : 'n/a'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </ScrollRegion>
    </div>
  );
}
