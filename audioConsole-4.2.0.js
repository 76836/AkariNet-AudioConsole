/**
 * AKARINET AUDIO CONSOLE v4.2.0
 * ====================================================================
 * Drop-in compatible with v4.1.1 / v4.1.2.
 *
 * NEW: Vosk-Browser speech recognition provider
 *   - speechRecognitionProvider: 'vosk'  (NEW default recommended for offline)
 *   - Continuous real-time STT on the shared AudioBus
 *   - Partial + final results power plain-text wake words reliably
 *   - Full replacement option for Moonshine / transformers
 *   - Still supports transformers, whispercpp, webspeech, none
 *
 * Config additions:
 *   vosk: {
 *     modelUrl: string,   // gzipped tar / zip of a Vosk model folder
 *     sampleRate: 16000,  // must match bus (16 kHz)
 *     continuous: true,   // always-on recognition (default true)
 *     grammar: string|null // optional Kaldi grammar JSON string
 *   }
 *
 * Default model (small English, ~40 MB, cached by browser after first load):
 *   https://ccoreilly.github.io/vosk-browser/models/vosk-model-small-en-us-0.15.tar.gz
 * Alternative used elsewhere in Akari:
 *   https://huggingface.co/ambind/vosk-model-small-en-us-0.15/resolve/main/vosk-model-small-en-us-0.15_c_.zip
 */

export * from './audioConsole-4.1.1.js';
export { default } from './audioConsole-4.1.1.js';

import {
    SpeechRecognitionProvider,
    AkarinetVoice,
    AudioBus
} from './audioConsole-4.1.1.js';

// ---------------------------------------------------------------------------
// Vosk script loader (CDN). Loaded once, then Vosk global is available.
// ---------------------------------------------------------------------------
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
 * VoskProvider — continuous or segment-based offline STT via vosk-browser WASM.
 *
 * Continuous mode (default):
 *   Subscribes to AudioBus chunks, feeds acceptWaveformFloat continuously,
 *   emits 'partial' and 'result' events. Ideal for plain-text wake words.
 *
 * Segment mode:
 *   Implements transcribe(Float32Array) for VAD-gated clips (Moonshine-style).
 *
 * isSessionBased is false so the orchestrator treats it as segment-capable,
 * but when continuous=true the provider also self-feeds from the bus and
 * pushes results through the same event path used by WebSpeech.
 */
export class VoskProvider extends SpeechRecognitionProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._model = null;
        this._recognizer = null;
        this._vosk = null;
        this._busUnsub = null;
        this._continuous = config.continuous !== false;
        this._sampleRate = config.sampleRate || 16000;
        this._modelUrl = config.modelUrl || DEFAULT_VOSK_MODEL;
        this._grammar = config.grammar || null;
        this._lastPartial = '';
        this._destroyed = false;
        this._pendingAudio = []; // small pre-roll while model loads
        this._ready = false;
    }

    get isSessionBased() {
        // Continuous Vosk still exposes transcribe() for buffer/XL cache use.
        // Orchestrator wires result events when continuous.
        return this._continuous;
    }

    async init() {
        this._log('INFO', `Loading Vosk model: ${this._modelUrl}`);
        this._vosk = await loadVoskScript();
        this._model = await this._vosk.createModel(this._modelUrl);
        const Kaldi = this._model.KaldiRecognizer;
        if (this._grammar) {
            this._recognizer = new Kaldi(this._sampleRate, this._grammar);
        } else {
            this._recognizer = new Kaldi(this._sampleRate);
        }

        this._recognizer.on('result', (message) => {
            const text = (message && message.result && message.result.text) || '';
            if (!text.trim()) return;
            this._lastPartial = '';
            this._emitter.emit('result', text.trim());
        });

        this._recognizer.on('partialresult', (message) => {
            const partial = (message && message.result && message.result.partial) || '';
            if (!partial.trim()) return;
            this._lastPartial = partial.trim();
            this._emitter.emit('partial', this._lastPartial);
        });

        this._ready = true;
        // Drain any audio that arrived while the model was loading
        if (this._pendingAudio.length) {
            for (const chunk of this._pendingAudio) {
                this._feedFloat(chunk);
            }
            this._pendingAudio = [];
        }
        this._log('OK', 'Vosk model + recognizer ready.');
    }

    /** Feed a Float32Array @ sampleRate into the recognizer. */
    _feedFloat(samples) {
        if (!this._recognizer || this._destroyed) return;
        try {
            if (typeof this._recognizer.acceptWaveformFloat === 'function') {
                this._recognizer.acceptWaveformFloat(samples, this._sampleRate);
            } else {
                this._recognizer.acceptWaveformFloat(samples, this._sampleRate);
            }
        } catch (e) {
            this._log('WARN', `acceptWaveformFloat failed: ${e.message || e}`);
        }
    }

    /**
     * Attach to an AudioBus (or any chunk emitter). Used for continuous mode.
     * chunk is Float32Array of mono samples at bus sampleRate.
     */
    attachBus(bus) {
        if (!bus || typeof bus.addSubscriber !== 'function') {
            this._log('WARN', 'attachBus: invalid bus');
            return;
        }
        if (this._busUnsub) return;
        const handler = (chunk) => {
            if (this._destroyed) return;
            if (!this._ready) {
                this._pendingAudio.push(chunk);
                if (this._pendingAudio.length > 20) this._pendingAudio.shift();
                return;
            }
            this._feedFloat(chunk);
        };
        bus.addSubscriber(handler);
        this._busUnsub = () => {
            try {
                if (typeof bus.removeSubscriber === 'function') {
                    bus.removeSubscriber(handler);
                }
            } catch (_) { /* ignore */ }
            this._busUnsub = null;
        };
        this._log('INFO', 'Attached to AudioBus for continuous recognition.');
    }

    detachBus() {
        if (this._busUnsub) {
            this._busUnsub();
            this._busUnsub = null;
        }
    }

    /**
     * Segment-based transcription (VAD clip / XL cache).
     * Feeds the whole buffer then forces a final result.
     */
    async transcribe(audio) {
        if (!this._recognizer) {
            throw new Error('VoskProvider not initialized');
        }
        return new Promise((resolve, reject) => {
            let settled = false;
            const onResult = (text) => {
                if (settled) return;
                settled = true;
                this._emitter.off && this._emitter.off('result', onResult);
                resolve(typeof text === 'string' ? text : String(text || ''));
            };
            const off = this._emitter.on('result', onResult);
            try {
                this._feedFloat(audio);
                if (typeof this._recognizer.retrieveFinalResult === 'function') {
                    this._recognizer.retrieveFinalResult();
                }
                setTimeout(() => {
                    if (!settled) {
                        settled = true;
                        try { if (typeof off === 'function') off(); } catch (_) {}
                        resolve(this._lastPartial || '');
                    }
                }, 2500);
            } catch (e) {
                settled = true;
                reject(e);
            }
        });
    }

    async startSession() {
        this._log('INFO', 'startSession (vosk continuous — already active)');
    }

    async stopSession() {
        this._lastPartial = '';
    }

    async destroy() {
        this._destroyed = true;
        this.detachBus();
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
// Patch AkarinetVoice factory + init path for 'vosk'
// ---------------------------------------------------------------------------

const _origCreateSpeech = AkarinetVoice.prototype._createSpeechProvider;
AkarinetVoice.prototype._createSpeechProvider = function _createSpeechProviderVosk() {
    if (this.config.speechRecognitionProvider === 'vosk') {
        const v = this.config.vosk || {};
        return new VoskProvider({
            modelUrl: v.modelUrl || this.config.voskModelUrl || DEFAULT_VOSK_MODEL,
            sampleRate: v.sampleRate || 16000,
            continuous: v.continuous !== false,
            grammar: v.grammar || null
        }, this.config.debugWakeSound);
    }
    return _origCreateSpeech.call(this);
};

const _origInit = AkarinetVoice.prototype.init;
AkarinetVoice.prototype.init = async function initWithVosk() {
    await _origInit.call(this);

    const sr = this.srProvider;
    if (sr && sr instanceof VoskProvider && sr._continuous) {
        if (this.bus) {
            sr.attachBus(this.bus);
        }

        sr.on('result', (text) => {
            if (typeof this._onAsrResult === 'function') {
                this._onAsrResult(text);
            } else if (typeof this._parse === 'function') {
                const via = !!(this.wakeSoundDetectedTime ||
                    (this._armUntil && Date.now() < this._armUntil) ||
                    !this.config.requireWakeSound);
                this._parse(text, via);
            }
        });

        sr.on('partial', (partial) => {
            if (!partial) return;
            const words = this.config.wakewords || [];
            if (!words.length) return;
            const lower = partial.toLowerCase();
            let hit = false;
            for (const w of words) {
                if (w && lower.includes(String(w).toLowerCase())) {
                    hit = true;
                    break;
                }
            }
            if (hit && typeof this._onWakeDetect === 'function') {
                this.wakeSoundDetectedTime = Date.now();
                this.lastWakeSoundScore = 0.9;
                this._log && this._log('INFO', `Text wake (partial): "${partial}"`);
            }
        });

        this._log && this._log('OK', 'Vosk continuous recognition attached to AudioBus.');
    }
};

const _origDestroy = AkarinetVoice.prototype.destroy;
AkarinetVoice.prototype.destroy = async function destroyWithVosk() {
    try {
        if (this.srProvider && typeof this.srProvider.detachBus === 'function') {
            this.srProvider.detachBus();
        }
    } catch (_) { /* ignore */ }
    if (typeof _origDestroy === 'function') {
        return _origDestroy.call(this);
    }
};


// VoskProvider already exported via `export class`.
// AkarinetVoice + default already re-exported from 4.1.1 above.
