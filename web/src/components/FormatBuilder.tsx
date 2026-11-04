import { useState } from "react";
import { knockoutSize, koWinsFor, roundsFor, type FormatKind, type FormatState } from "../formats";
import type { StageConfig } from "../engine/types";

interface Props {
  n: number;
  format: FormatState;
  onChange: (f: FormatState) => void;
  stages: StageConfig[];
  customJson: string | null;
  onCustomJson: (s: string | null) => void;
}

const KINDS: { kind: FormatKind; label: string; hint: string }[] = [
  { kind: "groups_ko", label: "Groups + playoff", hint: "Round-robin groups, best players advance to a knockout" },
  { kind: "ko", label: "Knockout", hint: "Straight playoff from the seeding" },
  { kind: "league", label: "League", hint: "Everyone plays everyone" },
  { kind: "swiss_ko", label: "Swiss + playoff", hint: "Swiss rounds paired by score, then a knockout" },
];

export function FormatBuilder({ n, format: f, onChange, stages, customJson, onCustomJson }: Props) {
  const [showJson, setShowJson] = useState(false);
  const set = (patch: Partial<FormatState>) => onChange({ ...f, ...patch });
  const koSize = knockoutSize(f, n);
  const rounds = roundsFor(koSize);
  const wins = koWinsFor(f, koSize);
  const groupSize = f.groups > 0 ? Math.ceil(n / f.groups) : n;
  return (
    <section className="card format">
      <header className="card-head">
        <h2>
          <span className="step">3</span> Format
        </h2>
      </header>
      <div className="segmented" role="radiogroup" aria-label="Tournament format">
        {KINDS.map((k) => (
          <button key={k.kind} role="radio" aria-checked={f.kind === k.kind} className={f.kind === k.kind ? "on" : ""} title={k.hint} onClick={() => set({ kind: k.kind })}>
            {k.label}
          </button>
        ))}
      </div>
      <div className="form-grid">
        {f.kind === "groups_ko" && (
          <label>
            Groups
            <input type="number" min={1} max={Math.max(1, Math.floor(n / 2))} value={f.groups} onChange={(e) => set({ groups: Math.max(1, +e.target.value) })} />
            <span className="muted small">≈ {groupSize} per group</span>
          </label>
        )}
        {f.kind === "groups_ko" && (
          <label>
            Group seeding
            <select value={f.seeding ?? "snake"} onChange={(e) => set({ seeding: e.target.value as "snake" | "abba" })}>
              <option value="snake">Snake</option>
              <option value="abba">ABBA</option>
            </select>
            <span className="muted small">by world rank</span>
          </label>
        )}
        {(f.kind === "groups_ko" || f.kind === "league") && (
          <label>
            Round-robin
            <select value={f.legs} onChange={(e) => set({ legs: +e.target.value })}>
              <option value={1}>Single</option>
              <option value={2}>Double</option>
            </select>
          </label>
        )}
        {f.kind === "swiss_ko" && (
          <label>
            Swiss rounds
            <input type="number" min={1} max={20} value={f.swissRounds} onChange={(e) => set({ swissRounds: Math.max(1, +e.target.value) })} />
          </label>
        )}
        {(f.kind === "groups_ko" || f.kind === "league" || f.kind === "swiss_ko") && (
          <label>
            Points W-D-L
            <select value={f.points.join("-")} onChange={(e) => set({ points: e.target.value.split("-").map(Number) as [number, number, number] })}>
              <option>2-1-0</option>
              <option>3-1-0</option>
            </select>
          </label>
        )}
        {(f.kind === "groups_ko" || f.kind === "swiss_ko") && (
          <label>
            To playoff
            <select value={f.advance} onChange={(e) => set({ advance: +e.target.value, koWins: [] })}>
              {[2, 4, 6, 8, 12, 16, 24, 32, 48, 64].filter((k) => k <= n).map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
        )}
        {(f.kind === "groups_ko" || f.kind === "swiss_ko") && (
          <label className="check">
            <input type="checkbox" checked={f.bPlayoff} onChange={(e) => set({ bPlayoff: e.target.checked })} /> B playoff for the next {Math.min(koSize, Math.max(0, n - koSize))}
          </label>
        )}
        {rounds.length > 0 && (
          <label className="check">
            <input type="checkbox" checked={f.bronze} onChange={(e) => set({ bronze: e.target.checked })} /> Bronze match
          </label>
        )}
      </div>
      {rounds.length > 0 && (
        <div className="rounds">
          <span className="muted small">Series length (first to N wins)</span>
          <div className="round-row">
            {rounds.map((name, r) => (
              <label key={name}>
                {name}
                <select
                  value={wins[r]}
                  onChange={(e) => {
                    const next = wins.slice();
                    next[r] = +e.target.value;
                    set({ koWins: next });
                  }}
                >
                  {[1, 2, 3, 4].map((w) => (
                    <option key={w} value={w}>
                      first to {w}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
        </div>
      )}
      <p className="muted small">
        Tiebreaks in groups: points, then head-to-head among tied players, goal difference, goals scored, lot. Games are 2-1-0 by
        default, playoff games are decided by golden-goal overtime.
      </p>
      <button className="link" onClick={() => setShowJson(!showJson)}>
        {showJson ? "Hide" : "Edit"} the raw config (advanced)
      </button>
      {showJson && (
        <div className="json-edit">
          <textarea
            spellCheck={false}
            value={customJson ?? JSON.stringify(stages, null, 2)}
            onChange={(e) => onCustomJson(e.target.value)}
            rows={14}
            aria-label="Stage config JSON"
          />
          <div className="quick">
            <button className="chip ghost" onClick={() => onCustomJson(null)}>
              Reset to builder
            </button>
            <span className="muted small">
              {customJson ? "Using your config." : "Edit to override the builder."} Stage types: groups, swiss, knockout; `source` slices earlier rankings; `places` sets final places.
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
