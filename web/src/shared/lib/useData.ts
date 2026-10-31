import { useCallback, useEffect, useState } from 'react';
import { isResults, type Results } from './results';

export type Load =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'empty' }
  | { status: 'ready'; results: Results };

/** Load the generated results, with loading, error (retry) and empty states. */
export function useData(base = '.'): Load & { retry: () => void } {
  const [state, setState] = useState<Load>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `attempt` is the retry trigger
  useEffect(() => {
    const ctl = new AbortController();
    setState({ status: 'loading' });
    fetch(`${base}/results.json`, { signal: ctl.signal })
      .then(async (r) => {
        if (r.status === 404) return setState({ status: 'empty' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const body: unknown = await r.json();
        if (!isResults(body)) return setState({ status: 'error', message: 'The results file is not in the expected format.' });
        setState({ status: 'ready', results: body });
      })
      .catch((e: Error) => {
        if (e.name !== 'AbortError')
          setState({
            status: 'error',
            message: `The results could not be loaded (${e.message}). Check the connection and try again.`,
          });
      });
    return () => ctl.abort();
  }, [base, attempt]);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { ...state, retry };
}
