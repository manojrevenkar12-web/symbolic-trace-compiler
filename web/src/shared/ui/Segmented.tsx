import styles from './ui.module.css';

export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
  format,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  format: (v: T) => string;
}) {
  return (
    <fieldset className={styles.segmented}>
      <legend>{label}</legend>
      {options.map((o) => (
        <button key={o} type="button" className={styles.segment} aria-pressed={o === value} onClick={() => onChange(o)}>
          {format(o)}
        </button>
      ))}
    </fieldset>
  );
}
