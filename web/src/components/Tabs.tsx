import { useRef } from "react";
import type { KeyboardEvent, ReactNode } from "react";

export interface TabItem {
  id: string;
  label: string;
}

interface TabsProps {
  label: string;
  tabs: TabItem[];
  active: string;
  onChange: (id: string) => void;
  idPrefix: string;
  children: ReactNode;
}

const KEYS: Record<string, (index: number, last: number) => number> = {
  ArrowRight: (index, last) => (index === last ? 0 : index + 1),
  ArrowLeft: (index, last) => (index === 0 ? last : index - 1),
  Home: () => 0,
  End: (_, last) => last,
};

/** WAI-ARIA tabs with roving focus: arrow keys, Home and End move between tabs. */
export function Tabs({ label, tabs, active, onChange, idPrefix, children }: TabsProps) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const index = Math.max(
    0,
    tabs.findIndex((tab) => tab.id === active),
  );

  function move(event: KeyboardEvent<HTMLButtonElement>) {
    const step = KEYS[event.key];
    if (!step) return;
    event.preventDefault();
    const next = step(index, tabs.length - 1);
    const tab = tabs[next];
    if (!tab) return;
    onChange(tab.id);
    refs.current[next]?.focus();
  }

  return (
    <div className="tabs">
      <div className="tablist-scroll">
        <div role="tablist" aria-label={label} className="tablist">
          {tabs.map((tab, position) => (
            <button
              key={tab.id}
              ref={(node) => {
                refs.current[position] = node;
              }}
              role="tab"
              type="button"
              id={`${idPrefix}-tab-${tab.id}`}
              aria-selected={tab.id === active}
              aria-controls={`${idPrefix}-panel-${tab.id}`}
              tabIndex={tab.id === active ? 0 : -1}
              className="tab"
              onClick={() => onChange(tab.id)}
              onKeyDown={move}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      <div
        role="tabpanel"
        id={`${idPrefix}-panel-${active}`}
        aria-labelledby={`${idPrefix}-tab-${active}`}
        className="tabpanel"
      >
        {children}
      </div>
    </div>
  );
}
