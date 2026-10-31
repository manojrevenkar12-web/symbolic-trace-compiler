import { useState } from 'react';
import { Gate } from '../features/gate/Gate';
import { HeldoutTable } from '../features/heldout/Heldout';
import { ProgramHero } from '../features/program/ProgramHero';
import { Recompiles } from '../features/recompiles/Recompiles';
import { Necessity, Soundness } from '../features/soundness/Soundness';
import { Thunder } from '../features/thunder/Thunder';
import { day, human, num } from '../shared/lib/format';
import type { Results } from '../shared/lib/results';
import { useData } from '../shared/lib/useData';
import { WORKLOAD_LABEL } from '../shared/theme/tokens';
import { Card, Section } from '../shared/ui/Section';
import { Segmented } from '../shared/ui/Segmented';
import { Code, EmptyState, ErrorState, Skeleton } from '../shared/ui/StateBlock';
import styles from './app.module.css';

function Loading() {
  return (
    <div className={styles.loading}>
      <Skeleton height={56} width="35%" label="Loading results" />
      <Skeleton height={300} />
      <Skeleton height={200} />
    </div>
  );
}

function Study({ r }: { r: Results }) {
  const names = Object.keys(r.examples ?? {});
  const [workload, setWorkload] = useState(names.includes('greedy_decode') ? 'greedy_decode' : (names[0] ?? ''));
  const s = r.summary;
  const sets = Object.entries(r.sets ?? {}).sort(([a], [b]) => b.localeCompare(a));
  return (
    <>
      {names.length > 0 ? (
        <Card>
          <div className={styles.heroCard}>
            <Segmented
              label="Workload"
              options={names}
              value={workload}
              onChange={setWorkload}
              format={(n) => WORKLOAD_LABEL[n] ?? human(n)}
            />
            <ProgramHero r={r} workload={workload} />
          </div>
        </Card>
      ) : (
        <EmptyState title="No programs in this run">
          Run <Code>stc report</Code> to generate them.
        </EmptyState>
      )}

      {r.rows && s ? (
        <Section
          id="recompiles"
          title="Programs compiled per workload"
          lede="Each workload is a stream of calls with realistic size variation, replayed under three caching policies. Every call is compared with eager PyTorch on the same weights."
          answer={
            <>
              Across {num(s.calls)} calls, constants compiled <strong>{num(s.static_compiles)}</strong> programs and symbols{' '}
              <strong>{num(s.symbolic_compiles)}</strong>, with no difference from eager (largest{' '}
              {s.max_abs_diff.toExponential(1)}). The extra symbolic programs are for batch size 1, which broadcasting treats
              differently.
            </>
          }
        >
          <Card>
            <Recompiles r={r} />
          </Card>
        </Section>
      ) : (
        <EmptyState title="No study in this run">
          Run <Code>stc study</Code>.
        </EmptyState>
      )}

      {r.fuzz && s && (
        <Section
          id="soundness"
          title="Are the guards sound?"
          lede="After each workload its cached programs are fuzzed with new sizes: some keep the relations the trace found, some break them, from 1 to far past any limit in the model. A wrong result is a draw a program accepted and answered differently from eager."
          answer={
            <>
              <strong>{num(s.fuzz_violations)}</strong> wrong results in {num(s.fuzz_draws)} draws. Coverage is the share of draws
              eager accepts that a cached program also accepts, without compiling.
            </>
          }
        >
          <Card>
            <Soundness r={r} />
          </Card>
          <Necessity r={r} />
        </Section>
      )}

      {sets.length > 0 && (
        <Section
          id="heldout"
          title="Programs written by someone who never saw the compiler"
          lede="Ten small programs per set, written from a one-page brief and the list of supported operations, each with size traps: branches on sizes, slices past the end, sizes of 1, int arguments that set sizes."
        >
          {sets.map(([name, set]) => (
            <Card key={name}>
              <HeldoutTable name={name} set={set} />
            </Card>
          ))}
        </Section>
      )}

      <Section id="thunder" title="Thunder on the same calls" lede="Measured in CI on CPU, in one process, on the same calls.">
        <Card>
          <Thunder t={r.thunder} />
        </Card>
      </Section>

      {r.gate && (
        <Section id="gate" title="Release gate">
          <Card>
            <Gate floors={r.gate.floors} targets={r.gate.targets} />
          </Card>
        </Section>
      )}
    </>
  );
}

export function App({ base = '.' }: { base?: string }) {
  const data = useData(base);
  return (
    <main className={styles.page}>
      <header className={styles.hero}>
        <h1>One compiled program for every sequence length.</h1>
        <p className={styles.lede}>
          A source-to-source trace compiler for PyTorch whose programs name sizes as symbols. It traces once, writes the
          conditions under which the program stays valid as a Python prologue, and proves them by fuzzing against eager PyTorch.
        </p>
        {data.status === 'ready' && (
          <p className={styles.meta}>
            Generated by <Code>stc report</Code> on {day(data.results.generated_at)}, profile <Code>{data.results.profile}</Code>.
          </p>
        )}
      </header>
      {data.status === 'loading' && <Loading />}
      {data.status === 'error' && <ErrorState message={data.message} onRetry={data.retry} />}
      {data.status === 'empty' && (
        <EmptyState title="No results yet">
          Run <Code>stc all --profile full</Code>, then <Code>npm run build</Code> to copy the results here.
        </EmptyState>
      )}
      {data.status === 'ready' && <Study r={data.results} />}
      <footer className={styles.footer}>
        <p>
          Measured on CPU. The compiled programs run the same PyTorch calls as eager: this project is about the cache, not about
          kernels.
        </p>
      </footer>
    </main>
  );
}
