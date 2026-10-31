import { num } from '../../shared/lib/format';
import { byPolicy, type Results } from '../../shared/lib/results';
import { Listing } from './Listing';
import styles from './program.module.css';

/** `p0, p1, ..., p28` (the module's parameters) shown as `p0, …, p28`. */
export const compactParams = (src: string) => src.replace(/\bp0, (?:p\d+, )+(p\d+)\b/, 'p0, …, $1');

/** The page's opening: one workload, its compile counts, and the program that served it. */
export function ProgramHero({ r, workload }: { r: Results; workload: string }) {
  const ex = r.examples?.[workload];
  const rows = byPolicy(r.rows ?? [], workload);
  return (
    <div className={styles.hero}>
      {rows.static && rows.symbolic && (
        <div className={styles.counts} role="group" aria-label="Programs compiled for greedy decoding">
          <div className={`${styles.count} ${styles.static}`}>
            <b>{num(rows.static.compiles)}</b>
            <span>programs compiled with sizes as constants, one per length</span>
          </div>
          <div className={`${styles.count} ${styles.symbolic}`}>
            <b>{num(rows.symbolic.compiles)}</b>
            <span>program compiled with sizes as symbols, for all {num(rows.symbolic.calls)} calls</span>
          </div>
        </div>
      )}
      {ex && (
        <div className={styles.split}>
          <div className={styles.picker}>
            <p className={styles.label}>Prologue: the checks that decide whether this program may run</p>
            <Listing source={ex.prologue_checks} label="Prologue size checks" />
          </div>
          <div className={styles.picker}>
            <p className={styles.label}>
              Computation: {ex.nodes} PyTorch calls, sizes written as <code>{ex.symbols.join(', ') || 'constants'}</code>
            </p>
            <Listing source={compactParams(ex.computation)} label="Generated computation" reveal />
          </div>
        </div>
      )}
    </div>
  );
}
