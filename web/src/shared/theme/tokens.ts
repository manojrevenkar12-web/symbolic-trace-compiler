import type { Policy } from '../lib/results';

export const MOTION = {
  base: 0.3,
  stagger: 0.025,
  ease: [0.2, 0.8, 0.3, 1] as [number, number, number, number],
} as const;

export const LISTING = {
  previewLines: 18,
} as const;

export const POLICY: Record<Policy, { label: string; note: string; color: string }> = {
  static: { label: 'Static', note: 'sizes are constants (Thunder’s default cache)', color: 'var(--static)' },
  auto: { label: 'Auto', note: 'sizes that changed become symbols', color: 'var(--auto)' },
  symbolic: { label: 'Symbolic', note: 'sizes are symbols from the first call', color: 'var(--symbolic)' },
};

export const WORKLOAD_LABEL: Record<string, string> = {
  chat_prefill: 'Chat prefill (GPT)',
  llama_prefill: 'Chat prefill (Llama)',
  greedy_decode: 'Greedy decoding, no KV cache',
  padded_encoder: 'Padded encoder batches',
  train_variable_length: 'Training, variable length',
};

export const CHECK_LABEL = {
  'wrong results': 'wrong results',
  errors: 'errors inside the program',
  'none seen': 'nothing seen',
} as const;
