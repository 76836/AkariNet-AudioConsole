/**
 * AKARINET AUDIO CONSOLE v4.2.0
 * ====================================================================
 * Drop-in compatible with v4.1.1 / v4.1.2.
 *
 * Vosk-Browser speech recognition provider — SEGMENT-BASED (like Moonshine)
 * ====================================================================
 * Privacy contract (same as transformers / Moonshine):
 *   - Vosk does NOT listen to the live mic continuously.
 *   - AudioBus + BusVAD capture speech segments only.
 *   - ASR runs ONLY inside _handleSpeech after the wake gate
 *     (openWakeWord / manual / continued) when requireWakeSound is true.
 *   - Only FINAL transcripts are returned from transcribe(); partials are
 *     ignored for commands (they can still change as context grows).
 *   - One transcribe() call → one _parse() → one 'result' event max.
 *
 * speechRecognitionProvider: 'vosk'
 * vosk: { modelUrl, sampleRate, grammar? }
 *
 * Default model (CORS-friendly .tar.gz):
 *   https://ccoreilly.github.io/vosk-browser/models/vosk-model-small-en-us-0.15.tar.gz
 */

export * from './audioConsole-4.1.1.js';
export { default } from './audioConsole-4.1.1.js';

import {
    SpeechRecognitionProvider,
    AkarinetVoice
} from './audioConsole-4.1.1.js';

const VOSK_CDN = 'https://cdn.jsdelivr.net/npm/vosk-browser@0.0.8/dist/vosk.js';
const DEFAULT_VOSK_MODEL =
    'https://ccoreilly.github.io/vosk-browser/models/vosk-model-small-en-us-0.15.tar.gz';

let _voskLoadPromise = null;

function loadVoskScript() {
    if (typeof window !== 'undefined' && window.Vosk) {
        return Promise.resolve(window.Vosk);
    }
    if (_voskLoadPromise) return _voskLoadPromise;
    _voskLoadPromise = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = VOSK_CDN;
        s.async = true;
        s.onload = () => {
            if (window.Vosk) resolve(window.Vosk);
            else reject(new Error('vosk-browser loaded but window.Vosk is missing'));
        };
        s.onerror = () => reject(new Error('Failed to load vosk-browser from CDN'));
        document.head.appendChild(s);
    });
    return _voskLoadPromise;
}

/**
 * VoskProvider — segment-based offline STT (Moonshine-compatible contract).
 *
 * isSessionBased = false  → orchestrator uses BusVAD + _handleSpeech + transcribe()
 * Never attaches to the live AudioBus for continuous recognition.
 * Partials are never used as commands.
 */
export class VoskProvider extends SpeechRecognitionProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._model = null;
        this._recognizer = null;
        this._vosk = null;
        this._sampleRate = config.sampleRate || 16000;
        this._modelUrl = config.modelUrl || DEFAULT_VOSK_MODEL;
        this._grammar = config.grammar || null;
        this._destroyed = false;
        this._ready = false;
        this._busy = false;
    }

    /** Segment-based: orchestrator calls transcribe() on VAD clips only. */
    get isSessionBased() {
        return false;
    }

    async init() {
        this._log('INFO', `Loading Vosk model (segment mode): ${this._modelUrl}`);
        this._vosk = await loadVoskScript();
        this._model = await this._vosk.createModel(this._modelUrl);
        this._createRecognizer();
        this._ready = true;
        this._log('OK', 'Vosk model ready (segment-based; no continuous mic feed).');
    }

    _createRecognizer() {
        if (!this._model) return;
        try {
            if (this._recognizer && typeof this._recognizer.remove === 'function') {
                this._recognizer.remove();
            }
        } catch (_) { /* ignore */ }
        const Kaldi = this._model.KaldiRecognizer;
        if (this._grammar) {
            this._recognizer = new Kaldi(this._sampleRate, this._grammar);
        } else {
            this._recognizer = new Kaldi(this._sampleRate);
        }
    }

    /**
     * Transcribe a single VAD-captured Float32Array segment.
     * Waits for the FINAL result only (ignores partialresult).
     * Fresh recognizer per call so prior audio cannot leak into the transcript.
     */
    async transcribe(audio) {
        if (!this._ready || !this._model) {
            throw new Error('VoskProvider not initialized');
        }
        if (!audio || !audio.length) return '';
        if (this._busy) {
            this._log('WARN', 'transcribe() ignored — already busy');
            return '';
        }
        this._busy = true;

        // New recognizer per utterance: no residual state / double finals
        this._createRecognizer();
        const recognizer = this._recognizer;

        return new Promise((resolve) => {
            let settled = false;
            const finish = (text) => {
                if (settled) return;
                settled = true;
                this._busy = false;
                try {
                    if (recognizer && typeof recognizer.remove === 'function') {
                        recognizer.remove();
                    }
                } catch (_) { /* ignore */ }
                resolve(String(text || '').trim());
            };

            // FINAL only — never resolve on partialresult
            recognizer.on('result', (message) => {
                const text = (message && message.result && message.result.text) || '';
                finish(text);
            });

            // Ignore partials for command path (they change as context grows)
            recognizer.on('partialresult', () => { /* intentional no-op */ });

            try {
                if (typeof recognizer.acceptWaveformFloat === 'function') {
                    recognizer.acceptWaveformFloat(audio, this._sampleRate);
                } else {
                    throw new Error('acceptWaveformFloat not available');
                }
                // Force end-of-utterance finalization
                if (typeof recognizer.retrieveFinalResult === 'function') {
                    recognizer.retrieveFinalResult();
                }
            } catch (e) {
                this._log('WARN', `transcribe feed failed: ${e.message || e}`);
                finish('');
                return;
            }

            // Safety timeout if no final arrives
            setTimeout(() => {
                if (!settled) {
                    this._log('WARN', 'transcribe() timed out waiting for final result');
                    finish('');
                }
            }, 4000);
        });
    }

    async startSession() { /* segment-based: no-op */ }
    async stopSession() { /* segment-based: no-op */ }

    async destroy() {
        this._destroyed = true;
        this._busy = false;
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
        this._ready = false;
        this._log('INFO', 'VoskProvider destroyed.');
    }
}

// ---------------------------------------------------------------------------
// Factory only — no continuous bus attach, no partial wake, no result listeners
// ---------------------------------------------------------------------------

const _origCreateSpeech = AkarinetVoice.prototype._createSpeechProvider;
AkarinetVoice.prototype._createSpeechProvider = function _createSpeechProviderVosk() {
    if (this.config.speechRecognitionProvider === 'vosk') {
        const v = this.config.vosk || {};
        return new VoskProvider({
            modelUrl: v.modelUrl || this.config.voskModelUrl || DEFAULT_VOSK_MODEL,
            sampleRate: v.sampleRate || 16000,
            grammar: v.grammar || null
        }, this.config.debugWakeSound);
    }
    return _origCreateSpeech.call(this);
};

// VoskProvider already exported via `export class`.
// AkarinetVoice + default already re-exported from 4.1.1 above.
