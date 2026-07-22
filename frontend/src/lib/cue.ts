/**
 * The sound feedback: two short tones, synthesised — no audio files to ship.
 *
 * Deliberately quiet and rare. A reciter is listening to themselves; a chime on every
 * word would be intolerable, and a harsh one on a mistake would punish. Only a
 * CONFIDENT mistake makes a sound, and it is a soft low pair — a nudge, not a buzzer.
 * `almost` is silent, for the same reason it is never painted red: we are not sure.
 */

let ctx: AudioContext | null = null;

const context = () => (ctx ??= new AudioContext());

function tone(freq: number, start: number, dur: number, gain: number) {
  const c = context();
  const osc = c.createOscillator();
  const amp = c.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  // Ramped, never gated: a square-edged gain change clicks.
  amp.gain.setValueAtTime(0, start);
  amp.gain.linearRampToValueAtTime(gain, start + 0.015);
  amp.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  osc.connect(amp).connect(c.destination);
  osc.start(start);
  osc.stop(start + dur + 0.02);
}

export function cueMistake(): void {
  const t = context().currentTime;
  tone(392, t, 0.12, 0.05); // G4
  tone(311, t + 0.1, 0.16, 0.05); // Eb4 — falling, the shape of "not quite"
}

/** Reaching the end of an āyah cleanly. Rare and rising. */
export function cueAyahClean(): void {
  const t = context().currentTime;
  tone(523, t, 0.1, 0.035); // C5
  tone(784, t + 0.08, 0.14, 0.03); // G5
}
