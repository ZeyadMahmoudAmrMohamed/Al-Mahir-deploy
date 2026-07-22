import {
  ArrowIcon,
  BrainIcon,
  DoubleArrowIcon,
  EyeIcon,
  EyeOffIcon,
  HighlighterIcon,
} from "./Icons";

type Props = {
  mistakesOnPage: number;
  accuracy: number | null;
  hidden: boolean;
  showMarks: boolean;
  canReveal: boolean;
  onHistory: () => void;
  onToggleHidden: () => void;
  onToggleMarks: () => void;
  onNextWord: () => void;
  onWholeAyah: () => void;
};

export function FeedbackBar({
  mistakesOnPage,
  accuracy,
  hidden,
  showMarks,
  canReveal,
  onHistory,
  onToggleHidden,
  onToggleMarks,
  onNextWord,
  onWholeAyah,
}: Props) {
  return (
    <nav className="feedbar" aria-label="أدوات التلاوة">
      <button className="iconbtn" onClick={onHistory} title="الأخطاء السابقة">
        <BrainIcon />
      </button>

      <div className={`stat${mistakesOnPage ? " stat--marked" : ""}`}>
        <span className="stat__value">{mistakesOnPage}</span>
        <span className="stat__label">في الصفحة</span>
      </div>

      {/* Before anything is scored there is no score. Showing 100% would be a claim
          we have not earned — the reciter has not recited yet. */}
      <div
        className={`stat${accuracy === null ? "" : accuracy === 100 ? " stat--clean" : " stat--marked"}`}
      >
        <span className="stat__value">{accuracy === null ? "—" : `${accuracy}%`}</span>
        <span className="stat__label">الدقّة</span>
      </div>

      <button
        className="iconbtn"
        onClick={onToggleHidden}
        aria-pressed={hidden}
        title={hidden ? "إظهار الآيات" : "إخفاء الآيات — اقرأ من حفظك"}
      >
        {hidden ? <EyeOffIcon /> : <EyeIcon />}
      </button>

      <button
        className="iconbtn"
        onClick={onToggleMarks}
        aria-pressed={showMarks}
        title="تمييز الأخطاء"
      >
        <HighlighterIcon />
      </button>

      <button
        className="iconbtn"
        onClick={onNextWord}
        disabled={!canReveal}
        title="إظهار الكلمة التالية"
      >
        <ArrowIcon />
      </button>

      <button
        className="iconbtn"
        onClick={onWholeAyah}
        disabled={!canReveal}
        title="إظهار بقيّة الآية"
      >
        <DoubleArrowIcon />
      </button>
    </nav>
  );
}
