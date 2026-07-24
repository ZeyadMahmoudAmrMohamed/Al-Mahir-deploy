import { ENGINES, liveAvailable } from "../lib/engines";
import type { EngineChoice } from "../lib/types";

/**
 * A compact popover, not a full sheet: two options is a menu, not a page. Anchored
 * under the topbar button that opens it (App.tsx positions it via CSS).
 */
export function EnginePicker({
  choice,
  onChoose,
  available,
  live,
  onLiveChange,
  capture,
  onCaptureChange,
  captureAvailable,
  locked,
  onClose,
}: {
  choice: EngineChoice;
  onChoose: (choice: EngineChoice) => void;
  /** available_engines from /health, or null while that request is still in flight —
   *  null means "unknown yet", so nothing is disabled on a guess. */
  available: Set<string> | null;
  live: boolean;
  onLiveChange: (on: boolean) => void;
  capture: boolean;
  onCaptureChange: (on: boolean) => void;
  /** Whether the server can record at all (/health's capture_available); null while
   *  that request is in flight, so nothing is disabled on a guess. */
  captureAvailable: boolean | null;
  /** A session is running. Both settings are read at session start, so changing one
   *  mid-recitation would silently do nothing — say so instead of lying. */
  locked: boolean;
  onClose: () => void;
}) {
  const liveOk = liveAvailable(choice, available);
  const liveHint = locked
    ? "أوقف التلاوة لتغيير هذا الإعداد"
    : choice === "zipformer"
      ? "يرافق تقييم المُعلِّم فقط"
      : !liveOk
        ? "غير متاح على هذا الخادم"
        : "تُملأ الكلمات فور نطقها، دون انتظار الوقف";
  return (
    <>
      {/* Click-outside-to-close. Transparent — this is a menu, not a modal. */}
      <div className="enginemenu-backdrop" onClick={onClose} />
      <div className="enginemenu" role="menu" aria-label="اختر محرك التعرف الصوتي">
        <p className="enginemenu__title">محرك التعرّف الصوتي</p>
        {ENGINES.map((e) => {
          const known = available !== null;
          const off = known && !available!.has(e.id);
          return (
            <button
              key={e.id}
              className="enginemenu__opt"
              role="menuitemradio"
              aria-checked={choice === e.id}
              disabled={off || locked}
              onClick={() => {
                onChoose(e.id);
                onClose();
              }}
            >
              <span className="enginemenu__label">
                <span className="enginemenu__dot" data-on={choice === e.id} />
                {e.label}
              </span>
              <span className="enginemenu__hint">
                {locked
                  ? "أوقف التلاوة لتغيير المحرّك"
                  : off
                    ? "غير متاح على هذا الخادم الآن"
                    : e.hint}
              </span>
            </button>
          );
        })}

        {/* The live tier is a companion to the grade, not an engine of its own, so it
            is a switch under the list rather than a fourth option. Disabled — not
            hidden — when it cannot apply, so the reason is visible. */}
        <div className="enginemenu__sep" role="separator" />
        <button
          className="enginemenu__opt"
          role="menuitemcheckbox"
          aria-checked={live && liveOk && !locked}
          disabled={!liveOk || locked}
          onClick={() => onLiveChange(!live)}
        >
          <span className="enginemenu__label">
            <span className="enginemenu__dot" data-on={live && liveOk && !locked} />
            الوضع التفاعلي
          </span>
          <span className="enginemenu__hint">{liveHint}</span>
        </button>

        {/* Diagnostic recording. Kept apart from the live switch because it is not a
            recitation setting at all — it changes nothing about the feedback, it only
            asks the server to keep the audio for later review. Disabled rather than
            hidden when the server cannot record, so the reason is visible. */}
        <button
          className="enginemenu__opt"
          role="menuitemcheckbox"
          aria-checked={capture && captureAvailable !== false && !locked}
          disabled={captureAvailable === false || locked}
          onClick={() => onCaptureChange(!capture)}
        >
          <span className="enginemenu__label">
            <span
              className="enginemenu__dot"
              data-on={capture && captureAvailable !== false && !locked}
            />
            وضع التشخيص
          </span>
          <span className="enginemenu__hint">
            {locked
              ? "أوقف التلاوة لتغيير هذا الإعداد"
              : captureAvailable === false
                ? "غير متاح على هذا الخادم"
                : "يحفظ صوت التلاوة لمراجعتها لاحقًا"}
          </span>
        </button>
      </div>
    </>
  );
}
