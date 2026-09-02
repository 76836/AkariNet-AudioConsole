/**
 * AKARINET AUDIO CONSOLE v4.1.2
 * ====================================================================
 * v4.1.1 compatibility wrapper with a fix for manual/continued wake arms:
 * preserve wake/session bookkeeping before dispatching the wake event.
 */

export * from './audioConsole-4.1.1.js';
export { default } from './audioConsole-4.1.1.js';

import { AkarinetVoice } from './audioConsole-4.1.1.js';

const _activateWakeWord411 = AkarinetVoice.prototype.activateWakeWord;

AkarinetVoice.prototype.activateWakeWord = function activateWakeWord411Fixed(opts) {
    const o = opts && typeof opts === 'object' ? opts : {};
    const listenMs = typeof o.listenMs === 'number' ? o.listenMs
        : (typeof this.config?.manualListenMs === 'number' ? this.config.manualListenMs : 12000);
    const now = Date.now();

    // Keep the same state bookkeeping used by the automatic wake path.
    this.wakeSoundDetectedTime = now;
    this.lastWakeSoundScore = 1;

    // v4.1.1's implementation handles the arm window and UI class.
    // Calling it after stamping state preserves both behaviors.
    const result = _activateWakeWord411.call(this, {
        ...o,
        listenMs,
        kind: o.kind || 'manual'
    });

    // _onWakeDetect may replace the timestamp; ensure the explicit activation
    // remains a valid wake for downstream in-session checks.
    if (!this.wakeSoundDetectedTime) this.wakeSoundDetectedTime = now;
    if (this.lastWakeSoundScore == null) this.lastWakeSoundScore = 1;
    return result;
};

export { AkarinetVoice };
