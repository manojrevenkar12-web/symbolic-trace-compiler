import { human, num } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import { WORKLOAD_LABEL } from '../../shared/theme/tokens';
import { ScrollRegion, subClass, tableClass } from '../../shared/ui/Section';
import { Code, EmptyState } from '../../shared/ui/StateBlock';

const SYSTEM: Record<string, string> = {
  'thunder-constant': 'Thunder, default cache',
  'thunder-symbolic': 'Thunder, cache="symbolic values"',
  'stc-symbolic': 'This compiler, symbolic',
};

export function Thunder({ t }: { t: Results['thunder'] }) {
  if (!t)
    return (
      <EmptyState title="No Thunder comparison in this run">
        It is measured in CI (<Code>scripts/thunder_bridge.py</Code>), where torch and lightning-thunder versions agree, and
        imported with <Code>scripts/import_bridge.py</Code>.
      </EmptyState>
    );
  return (
    <ScrollRegion label="Thunder on the same calls">
      <table className={tableClass}>
        <caption className="visually-hidden">Programs compiled by Thunder and by this compiler on the same calls</caption>
        <thead>
          <tr>
            <th scope="col">Workload</th>
            <th scope="col">System</th>
            <th scope="col">Calls</th>
            <th scope="col">Programs compiled</th>
            <th scope="col">Wall time (s)</th>
            <th scope="col">Largest difference from eager</th>
          </tr>
        </thead>
        <tbody>
          {t.rows.map((x) => (
            <tr key={`${x.workload}-${x.system}`}>
              <th scope="row">
                {WORKLOAD_LABEL[x.workload] ?? human(x.workload)}
                {x.error && <span className={subClass}>Error: {x.error}</span>}
              </th>
              <td>{SYSTEM[x.system] ?? x.system}</td>
              <td>{num(x.calls)}</td>
              <td>{num(x.compiles)}</td>
              <td>{num(x.wall_s, 1)}</td>
              <td>{x.max_abs_diff.toExponential(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollRegion>
  );
}
