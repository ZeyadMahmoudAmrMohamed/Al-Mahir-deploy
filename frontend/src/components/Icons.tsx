/** Line icons, drawn to one grid: 24px box, 1.6 stroke, round caps. */

type P = { size?: number };
const box = (size = 20) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export const SearchIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <circle cx="11" cy="11" r="6.5" />
    <path d="m16 16 4 4" />
  </svg>
);

export const SlidersIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h8M16 18h4" />
    <circle cx="16" cy="6" r="2" />
    <circle cx="8" cy="12" r="2" />
    <circle cx="14" cy="18" r="2" />
  </svg>
);

export const MicIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <rect x="9" y="2.5" width="6" height="11" rx="3" />
    <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
  </svg>
);

export const StopIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none" />
  </svg>
);

/** Past mistakes — a brain. */
export const BrainIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="M12 5.5a3 3 0 0 0-5.7-1.3A2.8 2.8 0 0 0 4 9.4a3 3 0 0 0 .6 5.2A2.8 2.8 0 0 0 9 19a3 3 0 0 0 3-2.2z" />
    <path d="M12 5.5a3 3 0 0 1 5.7-1.3A2.8 2.8 0 0 1 20 9.4a3 3 0 0 1-.6 5.2A2.8 2.8 0 0 1 15 19a3 3 0 0 1-3-2.2z" />
    <path d="M12 5.5v11.3" />
  </svg>
);

export const EyeIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" />
    <circle cx="12" cy="12" r="2.75" />
  </svg>
);

export const EyeOffIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="M9.9 5.7A9.6 9.6 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-2.7 3.6M6.3 7.8A16.7 16.7 0 0 0 2.5 12S6 18.5 12 18.5c1.2 0 2.3-.2 3.3-.6" />
    <path d="M10 10a2.75 2.75 0 0 0 3.9 3.9M3 3l18 18" />
  </svg>
);

/** Highlight mistakes — a highlighter pen. */
export const HighlighterIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="m14.5 3.5 6 6-7.8 7.8-6-6z" />
    <path d="m6.7 11.3-2.2 5.4a1 1 0 0 0 1.3 1.3l5.4-2.2" />
    <path d="M4 21h7" />
  </svg>
);

/** Show next word. */
export const ArrowIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="M19 12H5M12 5l-7 7 7 7" />
  </svg>
);

/** Show the rest of the āyah. */
export const DoubleArrowIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="m17 5-7 7 7 7M10 5l-7 7 7 7" />
  </svg>
);

export const CloseIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

/** Model picker — two sliders, standing in for "which engine is listening". */
export const ChipIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <rect x="7" y="7" width="10" height="10" rx="2" />
    <path d="M10 4v3M14 4v3M10 17v3M14 17v3M4 10h3M4 14h3M17 10h3M17 14h3" />
  </svg>
);

export const ChevronIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="m9 5 7 7-7 7" />
  </svg>
);

export const SoundIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="M11 5 6.5 8.5H3v7h3.5L11 19z" />
    <path d="M15 9.5a3.5 3.5 0 0 1 0 5M17.5 7a7 7 0 0 1 0 10" />
  </svg>
);

export const SoundOffIcon = ({ size }: P) => (
  <svg {...box(size)}>
    <path d="M11 5 6.5 8.5H3v7h3.5L11 19z" />
    <path d="m16 9.5 5 5M21 9.5l-5 5" />
  </svg>
);

/**
 * The madd stroke: 2 / 4 / 6, the counts an elongation is held for. The house motif,
 * and the app's one ornament. `level` (0..1) drives it live from the mic, so the mark
 * that means "length, measured" is measuring length.
 */
export const MaddStroke = ({ level = 0.15 }: { level?: number }) => (
  <span className="madd" style={{ ["--m" as string]: level }} aria-hidden="true">
    <i />
    <i />
    <i />
  </span>
);
