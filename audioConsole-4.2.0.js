/**
 * AKARINET AUDIO CONSOLE v4.2.0
 * ====================================================================
 * Fork of v4.1.2 with one additive feature: Vosk segment-based STT.
 *
 * What this file does NOT do:
 *   - Does not rewrite 4.1.0 / 4.1.1 / 4.1.2 behavior
 *   - Does not attach Vosk to the live AudioBus
 *   - Does not use partial results as commands
 *   - Does not mark Vosk as session-based (no parallel result path)
 *
 * Privacy / pipeline (identical to Moonshine / transformers):
 *   AudioBus → BusVAD → _handleSpeech (wake gate) → srProvider.transcribe(segment)
 *   Only FINAL Vosk text is returned from transcribe().
 *
 * Config:
 *   speechRecognitionProvider: 'vosk'
 *   vosk: { modelUrl?, sampleRate?, grammar? }
 *
 * Default model (CORS-friendly tar.gz):
 *   https://ccoreilly.github.io/vosk-browser/models/vosk-model-small-en-us-0.15.tar.gz
 */

// ── Preserve entire 4.1.2 surface (includes activateWakeWord fix) ──
export * from './audioConsole-4.1.2.js';
export { default } from './audioConsole-4.1.2.js';

import {
    SpeechRecognitionProvider,
    AkarinetVoice
} from './audioConsole-4.1.2.js';

// ── Vosk loader (once) ─────────────────────────────────────────────
const VOSK_CDN = 'https://cdn.jsdelivr.net/npm/vosk-browser@0.0.8/dist/vosk.js';
const DEFAULT_VOSK_MODEL =
    'https://ccoreilly.github.io/vosk-browser/models/vosk-model-small-en-us-0.15.tar.gz';

let _voskScriptPromise = null;

function loadVoskScript() {
    if (typeof window !== 'undefined' && window.Vosk) {
        return Promise.resolve(window.Vosk);
    }
    if (_voskScriptPromise) return _voskScriptPromise;
    _voskScriptPromise = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = VOSK_CDN;
        s.async = true;
        s.onload = () => {
            if (window.Vosk) resolve(window.Vosk);
            else reject(new Error('vosk-browser loaded but window.Vosk missing'));
        };
        s.onerror = () => reject(new Error('Failed to load vosk-browser CDN'));
        document.head.appendChild(s);
    });
    return _voskScriptPromise;
}

/**
 * Segment-based Vosk STT — same contract as TransformersProvider (Moonshine).
 * isSessionBased === false → orchestrator only calls transcribe() after VAD + wake gate.
 */
export class VoskProvider extends SpeechRecognitionProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._model = null;
        this._recognizer = null;
        this._sampleRate = config.sampleRate || 16000;
        this._modelUrl = config.modelUrl || DEFAULT_VOSK_MODEL;
        this._grammar = config.grammar || null;
        this._ready = false;
        this._busy = false;
        this._onResult = null;
        this._onError = null;
        this._onPartial = null;
    }

    get isSessionBased() {
        return false;
    }

    async init() {
        this._log('INFO', `Vosk loading model: ${this._modelUrl}`);
        const Vosk = await loadVoskScript();
        this._model = await Vosk.createModel(this._modelUrl);
        await this._ensureRecognizer();
        this._ready = true;
        this._log('OK', 'Vosk ready (segment-based, recognizer worker-acked).');
    }

    /**
     * Create KaldiRecognizer and wait until the worker confirms it exists.
     * No fixed delays: probe with retrieveFinalResult; worker replies result or error.
     * "Does not exist" → probe again on the next microtask after the error event.
     */
    async _ensureRecognizer() {
        if (this._recognizer) return this._recognizer;
        if (!this._model || !this._model.ready) {
            throw new Error('Vosk model not ready');
        }

        const Kaldi = this._model.KaldiRecognizer;
        const recognizer = this._grammar
            ? new Kaldi(this._sampleRate, this._grammar)
            : new Kaldi(this._sampleRate);

        await new Promise((resolve, reject) => {
            let settled = false;
            let probes = 0;
            const maxProbes = 200; // event-driven retries, not timed waits

            const cleanup = () => {
                if (this._onResult) {
                    try { recognizer.removeEventListener('result', this._onResult); } catch (_) {}
                }
                if (this._onError) {
                    try { recognizer.removeEventListener('error', this._onError); } catch (_) {}
                }
                this._onResult = null;
                this._onError = null;
            };

            const finishOk = () => {
                if (settled) return;
                settled = true;
                cleanup();
                resolve();
            };
            const finishErr = (e) => {
                if (settled) return;
                settled = true;
                cleanup();
                reject(e instanceof Error ? e : new Error(String(e)));
            };

            // KaldiRecognizer.on wraps addEventListener and passes event.detail
            recognizer.on('result', () => {
                // Empty final is fine — means recognizer exists and FinalResult ran
                finishOk();
            });
            recognizer.on('error', (message) => {
                const errText = String((message && (message.error || message.message)) || message || '');
                if (/does not exist|not exist|already been deleted/i.test(errText)) {
                    probes += 1;
                    if (probes > maxProbes) {
                        finishErr(new Error('Vosk recognizer never became ready in worker'));
                        return;
                    }
                    // Retry only after the worker answered — no sleep
                    queueMicrotask(() => {
                        try { recognizer.retrieveFinalResult(); } catch (e) { finishErr(e); }
                    });
                    return;
                }
                finishErr(new Error(errText || 'Vosk recognizer error'));
            });

            // First probe: if create is already done, result arrives; else error → retry
            try {
                recognizer.retrieveFinalResult();
            } catch (e) {
                finishErr(e);
            }
        });

        this._recognizer = recognizer;
        // Persistent listeners for transcription results
        this._bindRecognizerEvents();
        return recognizer;
    }

    _bindRecognizerEvents() {
        const recognizer = this._recognizer;
        if (!recognizer) return;

        // Track last final / partial for the in-flight transcribe() promise
        this._lastFinal = '';
        this._lastPartial = '';
        this._resultWaiters = [];

        recognizer.on('result', (message) => {
            const t = (message && message.result && message.result.text) || '';
            if (t && t.trim()) this._lastFinal = t.trim();
            const waiters = this._resultWaiters.splice(0);
            for (const w of waiters) {
                try { w(this._lastFinal); } catch (_) {}
            }
        });
        recognizer.on('partialresult', (message) => {
            const p = (message && message.result && message.result.partial) || '';
            if (p) this._lastPartial = String(p);
        });
        recognizer.on('error', (message) => {
            const errText = String((message && (message.error || message.message)) || message || '');
            this._log('WARN', `recognizer error: ${errText}`);
            const waiters = this._resultWaiters.splice(0);
            for (const w of waiters) {
                try { w(''); } catch (_) {}
            }
        });
    }

    /**
     * Wait for the next worker `result` event after retrieveFinalResult.
     * Event-driven; safety timeout only for a dead worker (not for readiness).
     */
    _waitNextFinal(timeoutMs = 10000) {
        return new Promise((resolve) => {
            let done = false;
            const finish = (text) => {
                if (done) return;
                done = true;
                resolve(text || '');
            };
            this._resultWaiters.push(finish);
            // Safety only: hung worker. Not used for create-race.
            setTimeout(() => finish(this._lastFinal || ''), timeoutMs);
        });
    }

    /**
     * Transcribe one VAD segment. Reuses the init-time recognizer (worker already acked).
     * Finals only. No fixed create delays.
     */
    async transcribe(audio) {
        if (!this._ready || !this._model) {
            throw new Error('VoskProvider not initialized');
        }
        if (!audio || !audio.length) return '';
        if (this._busy) {
            this._log('WARN', 'transcribe skipped (busy)');
            return '';
        }
        this._busy = true;
        this._lastFinal = '';
        this._lastPartial = '';

        try {
            const recognizer = await this._ensureRecognizer();
            const samples = audio instanceof Float32Array ? audio : new Float32Array(audio);

            let peak = 0;
            for (let i = 0; i < samples.length; i++) {
                const a = samples[i] < 0 ? -samples[i] : samples[i];
                if (a > peak) peak = a;
            }
            this._log('INFO', `transcribe start: ${samples.length} samples, peak=${peak.toFixed(4)}`);
            if (peak < 0.001) {
                this._log('WARN', 'audio peak near zero — VAD segment may be silence');
            }

            // Drain any stale waiters from a previous call
            this._resultWaiters.splice(0);

            recognizer.acceptWaveformFloat(samples, this._sampleRate);

            // Endpoint: short silence helps Kaldi finalize; still event-driven after
            const silence = new Float32Array(Math.floor(this._sampleRate * 0.3));
            recognizer.acceptWaveformFloat(silence, this._sampleRate);

            const finalPromise = this._waitNextFinal(10000);
            recognizer.retrieveFinalResult();
            const text = await finalPromise;

            this._log('INFO', `transcribe final: "${(text || '').slice(0, 80)}" (${samples.length} samples)`);
            return text || '';
        } catch (e) {
            this._log('WARN', `transcribe failed: ${e.message || e}`);
            return '';
        } finally {
            this._busy = false;
        }
    }

    async startSession() { /* segment-based no-op */ }
    async stopSession() { /* segment-based no-op */ }

    async destroy() {
        this._busy = false;
        this._ready = false;
        this._resultWaiters = [];
        try {
            if (this._recognizer && typeof this._recognizer.remove === 'function') {
                this._recognizer.remove();
            }
        } catch (_) { /* ignore */ }
        this._recognizer = null;
        try {
            if (this._model && typeof this._model.terminate === 'function') {
                this._model.terminate();
            }
        } catch (_) { /* ignore */ }
        this._model = null;
        this._log('INFO', 'VoskProvider destroyed.');
    }
}

// ── Selective edits (do not rewrite 4.1.x core) ────────────────────
//
// Core 4.1.0 hardcodes:
//   needsVad = needsBus && (srProv === 'transformers' || srProv === 'whispercpp')
// So speechRecognitionProvider === 'vosk' never gets BusVAD → no VAD lights,
// no speech-end segments, no transcribe(). Moonshine works because it matches.
//
// Fix: for vosk only, borrow the transformers bus/VAD path during init, while
// still constructing VoskProvider from the factory.

const _createSpeechProvider412 = AkarinetVoice.prototype._createSpeechProvider;
const _init412 = AkarinetVoice.prototype.init;

AkarinetVoice.prototype._createSpeechProvider = function _createSpeechProvider420() {
    if (this.config._wantVosk || this.config.speechRecognitionProvider === 'vosk') {
        const v = this.config.vosk || {};
        return new VoskProvider({
            modelUrl: v.modelUrl || this.config.voskModelUrl || DEFAULT_VOSK_MODEL,
            sampleRate: v.sampleRate || 16000,
            grammar: v.grammar || null
        }, this.config.debugWakeSound);
    }
    return _createSpeechProvider412.call(this);
};

AkarinetVoice.prototype.init = async function init420() {
    const wantVosk = this.config.speechRecognitionProvider === 'vosk';
    if (wantVosk) {
        // Enable needsBus / needsVad / needsOrt checks inside 4.1.0 init
        this.config._wantVosk = true;
        this.config.speechRecognitionProvider = 'transformers';
    }
    try {
        await _init412.call(this);
    } finally {
        if (wantVosk) {
            this.config.speechRecognitionProvider = 'vosk';
        }
    }
};

// Do NOT re-export AkarinetVoice / default — already provided by export * / export { default }.
