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
        this._sampleRate = config.sampleRate || 16000;
        this._modelUrl = config.modelUrl || DEFAULT_VOSK_MODEL;
        this._grammar = config.grammar || null;
        this._ready = false;
        this._busy = false;
    }

    get isSessionBased() {
        return false;
    }

    async init() {
        this._log('INFO', `Vosk loading model: ${this._modelUrl}`);
        const Vosk = await loadVoskScript();
        this._model = await Vosk.createModel(this._modelUrl);
        this._ready = true;
        this._log('OK', 'Vosk ready (segment-based).');
    }

    /**
     * Transcribe one VAD segment. Returns FINAL text only.
     * Fresh KaldiRecognizer per call so state cannot leak between utterances.
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

        const Kaldi = this._model.KaldiRecognizer;
        const recognizer = this._grammar
            ? new Kaldi(this._sampleRate, this._grammar)
            : new Kaldi(this._sampleRate);

        let finalText = '';

        try {
            await new Promise((resolve) => {
                let settled = false;
                const done = () => {
                    if (settled) return;
                    settled = true;
                    resolve();
                };

                recognizer.on('result', (message) => {
                    const t = (message && message.result && message.result.text) || '';
                    if (t && t.trim()) finalText = t.trim();
                    done();
                });

                // Partials intentionally ignored for the command path
                recognizer.on('partialresult', () => {});

                try {
                    // Feed in worklet-sized chunks (1280 @ 16 kHz ≈ 80 ms)
                    const step = 1280;
                    for (let i = 0; i < audio.length; i += step) {
                        const slice = audio.subarray(i, Math.min(i + step, audio.length));
                        recognizer.acceptWaveformFloat(slice, this._sampleRate);
                    }
                    if (typeof recognizer.retrieveFinalResult === 'function') {
                        recognizer.retrieveFinalResult();
                    }
                } catch (e) {
                    this._log('WARN', `acceptWaveformFloat: ${e.message || e}`);
                    done();
                    return;
                }

                setTimeout(done, 3500);
            });
        } finally {
            try {
                if (typeof recognizer.remove === 'function') recognizer.remove();
            } catch (_) { /* ignore */ }
            this._busy = false;
        }

        return finalText;
    }

    async startSession() { /* segment-based no-op */ }
    async stopSession() { /* segment-based no-op */ }

    async destroy() {
        this._busy = false;
        this._ready = false;
        try {
            if (this._model && typeof this._model.terminate === 'function') {
                this._model.terminate();
            }
        } catch (_) { /* ignore */ }
        this._model = null;
        this._log('INFO', 'VoskProvider destroyed.');
    }
}

// ── Selective edit: only the speech-provider factory ───────────────
const _createSpeechProvider412 = AkarinetVoice.prototype._createSpeechProvider;

AkarinetVoice.prototype._createSpeechProvider = function _createSpeechProvider420() {
    if (this.config.speechRecognitionProvider === 'vosk') {
        const v = this.config.vosk || {};
        return new VoskProvider({
            modelUrl: v.modelUrl || this.config.voskModelUrl || DEFAULT_VOSK_MODEL,
            sampleRate: v.sampleRate || 16000,
            grammar: v.grammar || null
        }, this.config.debugWakeSound);
    }
    return _createSpeechProvider412.call(this);
};

// Do NOT re-export AkarinetVoice / default — already provided by export * / export { default }.
