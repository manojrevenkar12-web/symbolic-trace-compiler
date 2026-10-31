import type { ReactNode } from 'react';
import styles from './ui.module.css';

/** One question of the study: what was asked, the evidence, and the answer in a sentence or two. */
export function Section({ id, title, lede, answer, children }: { id: string; title: string; lede?: ReactNode; answer?: ReactNode; children: ReactNode }) {
  return (
    <section className={styles.section} aria-labelledby={`${id}-h`} id={id}>
      <div className={styles.sectionHead}>
        <h2 id={`${id}-h`}>{title}</h2>
        {lede && <p className={styles.lede}>{lede}</p>}
      </div>
      {children}
      {answer && <p className={styles.answer}>{answer}</p>}
    </section>
  );
}

export function Card({ children }: { children: ReactNode }) {
  return <div className={styles.card}>{children}</div>;
}

/** A table wider than a phone scrolls inside itself; it is focusable so keyboards can scroll it. */
export function ScrollRegion({ label, children }: { label: string; children: ReactNode }) {
  return (
    // biome-ignore lint/a11y/noNoninteractiveTabindex: a scrollable region must be reachable by keyboard (WCAG 2.1.1)
    <section className={styles.scroll} aria-label={label} tabIndex={0}>
      {children}
    </section>
  );
}

export const tableClass = styles.table;
export const subClass = styles.sub;
