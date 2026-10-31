import { motion, useReducedMotion } from 'motion/react';
import { Fragment, useState } from 'react';
import { LISTING, MOTION } from '../../shared/theme/tokens';
import styles from './program.module.css';

const TOKEN = /(#.*$)|(\bs\d+\b)|(\b(?:def|return|if|not|and|or|None)\b)/g;

/** Highlights three things only: size symbols, comments (the shapes), and keywords. */
export function Line({ text }: { text: string }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    const i = m.index ?? 0;
    if (i > last) parts.push(text.slice(last, i));
    const cls = m[1] ? styles.comment : m[2] ? styles.sym : styles.kw;
    parts.push(
      <span key={i} className={cls}>
        {m[0]}
      </span>,
    );
    last = i + m[0].length;
  }
  parts.push(text.slice(last));
  return <>{parts.map((p, i) => (typeof p === 'string' ? <Fragment key={`t${i}`}>{p}</Fragment> : p))}</>;
}

export function Listing({ source, label, reveal = false }: { source: string; label: string; reveal?: boolean }) {
  const reduce = useReducedMotion();
  const lines = source.split('\n');
  const [all, setAll] = useState(false);
  const shown = all ? lines : lines.slice(0, LISTING.previewLines);
  return (
    <figure className={styles.figure}>
      <figcaption className="visually-hidden">{label}</figcaption>
      <pre className={styles.pre} tabIndex={0} aria-label={label}>
        {shown.map((ln, i) => (
          <motion.span
            // biome-ignore lint/suspicious/noArrayIndexKey: lines of a fixed listing
            key={i}
            className={styles.line}
            initial={reveal && !reduce ? { opacity: 0, x: -6 } : false}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: MOTION.base, delay: reveal && !reduce ? i * MOTION.stagger : 0, ease: MOTION.ease }}
          >
            <Line text={ln} />
            {'\n'}
          </motion.span>
        ))}
      </pre>
      {lines.length > LISTING.previewLines && (
        <button type="button" className={styles.more} onClick={() => setAll((v) => !v)} aria-expanded={all}>
          {all ? 'Show the first lines only' : `Show all ${lines.length} lines`}
        </button>
      )}
    </figure>
  );
}
