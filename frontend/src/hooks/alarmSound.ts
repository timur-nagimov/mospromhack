// Звук критической тревоги на WebAudio: без аудиофайлов, работает офлайн.
// Браузер разрешает звук только после жеста пользователя — контекст
// «размораживается» первым кликом или нажатием клавиши.

let ctx: AudioContext | null = null;

function context(): AudioContext | null {
  if (ctx) return ctx;
  const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;
  ctx = new Ctor();
  return ctx;
}

export function unlockAudioOnGesture(): () => void {
  const unlock = () => {
    const c = context();
    if (c && c.state === 'suspended') void c.resume();
  };
  window.addEventListener('pointerdown', unlock);
  window.addEventListener('keydown', unlock);
  return () => {
    window.removeEventListener('pointerdown', unlock);
    window.removeEventListener('keydown', unlock);
  };
}

export function isAudioBlocked(): boolean {
  return !ctx || ctx.state !== 'running';
}

/** Двухтональный сигнал ×3 — узнаваемо, но не оглушающе */
export function playAlarm(): void {
  const c = context();
  if (!c || c.state !== 'running') return;
  const start = c.currentTime + 0.02;
  for (let i = 0; i < 6; i += 1) {
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = 'square';
    osc.frequency.value = i % 2 === 0 ? 880 : 622;
    const t0 = start + i * 0.17;
    gain.gain.setValueAtTime(0, t0);
    gain.gain.linearRampToValueAtTime(0.06, t0 + 0.015);
    gain.gain.setValueAtTime(0.06, t0 + 0.13);
    gain.gain.linearRampToValueAtTime(0, t0 + 0.155);
    osc.connect(gain).connect(c.destination);
    osc.start(t0);
    osc.stop(t0 + 0.16);
  }
}

/** Короткий щелчок для некритичных событий */
export function playChime(): void {
  const c = context();
  if (!c || c.state !== 'running') return;
  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.type = 'sine';
  osc.frequency.value = 1046;
  const t0 = c.currentTime + 0.01;
  gain.gain.setValueAtTime(0, t0);
  gain.gain.linearRampToValueAtTime(0.035, t0 + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.25);
  osc.connect(gain).connect(c.destination);
  osc.start(t0);
  osc.stop(t0 + 0.26);
}
