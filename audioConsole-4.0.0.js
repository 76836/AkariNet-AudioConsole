/**
 * AKARINET AUDIO CONSOLE v4.0
 * ====================================================================
 * A modular voice recognition engine with pluggable wake word providers.
 *
 * Pipeline:
 *   [Mic] → [VAD] ──→ [Wake Provider (parallel)] ──→ [ASR] → [Parse] → [Result]
 *
 * Wake Word Providers (plug-in via `wakeWordProvider` config):
 *   - 'teachablemachine' (DEFAULT, backwards-compatible with v3.4)
 *       Uses TensorFlow.js + Teachable Machine audio models.
 *       Config: wakesoundURL, wakesoundThreshold, wakesoundIndex, etc.
 *   - 'openwakeword' (NEW in v4.0)
 *       Uses ONNX Runtime + openWakeWord models (speaker-invariant,
 *       trained on diverse synthetic + real data).
 *       Config: openWakeWord.baseAssetUrl, .keywords, .detectionThreshold, etc.
 *   - 'none'
 *       No audio wake detection. Only text wake words via ASR transcript.
 *
 * Backwards compatibility with v3.4:
 *   - Same constructor signature: new AkarinetVoice(config)
 *   - Same events: ready, speechstart, speechend, wakesound, result,
 *                  speechdiscarded, error, processing, processingend
 *   - Same config keys (wakesoundURL, wakesoundThreshold, wakesoundIndex,
 *     wakesoundDuration, wakesoundDelay, vadThreshold, requireWakeSound, etc.)
 *   - Same activateWakeWord() manual trigger
 *   - Same destroy() cleanup
 *
 * Migration from v3.4 → v4.0:
 *   Change your import URL from audioConsole-3.4.0.js to audioConsole-4.0.0.js.
 *   Everything else works unchanged. To switch to openWakeWord, add:
 *     wakeWordProvider: 'openwakeword',
 *     openWakeWord: {
 *       baseAssetUrl: 'https://your-host/openwakeword/models',
 *       keywords: ['hey_jarvis'],
 *       detectionThreshold: 0.5
 *     }
 *
 * Licenses:
 *   - OpenWakeWordProvider (inlined WakeWordEngine port): Apache 2.0
 *     Adapted from https://github.com/dnavarrom/openwakeword_wasm
 */

// ==================================================================
// SECTION 1: UTILITIES
// ==================================================================

/**
 * Dynamically load an external <script> tag, deduped by URL.
 * @param {string} src
 * @returns {Promise<void>}
 */
function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) return resolve();
        const script = document.createElement('script');
        script.src = src;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
        document.head.appendChild(script);
    });
}

/**
 * Minimal event emitter used by wake word providers.
 * @returns {{on: Function, off: Function, emit: Function}}
 */
function createEmitter() {
    const listeners = new Map();
    return {
        on(event, handler) {
            if (!listeners.has(event)) listeners.set(event, new Set());
            listeners.get(event).add(handler);
            return () => this.off(event, handler);
        },
        off(event, handler) {
            const set = listeners.get(event);
            if (set) set.delete(handler);
        },
        emit(event, payload) {
            const set = listeners.get(event);
            if (!set) return;
            for (const handler of Array.from(set)) {
                try { handler(payload); } catch (err) { console.error('[Provider] listener error:', err); }
            }
        }
    };
}


// ==================================================================
// SECTION 2: PROVIDER INTERFACE
// ==================================================================
//
// All wake word providers implement this interface. The orchestrator
// (AkarinetVoice) talks to providers through this contract only — it
// never reaches into provider internals.
//
// Lifecycle:
//   const provider = new SomeProvider(config, debug);
//   await provider.init();
//   provider.on('detect', (detail) => { ... });
//   await provider.startListening();
//   ...
//   await provider.stopListening();   // can be restarted
//   ...
//   await provider.destroy();         // final teardown
//
// 'detect' event payload:
//   { score: number, class: string, timestamp: number, raw?: any }
//   - score:     0..1 confidence (some providers may emit raw logits — see provider docs)
//   - class:     human-readable label (e.g. "hey_jarvis" or "_background_noise_")
//   - timestamp: Date.now() at detection (wall clock — used for correlation with VAD events)
//   - raw:       optional provider-specific extras for debugging

class WakeWordProvider {
    /**
     * @param {Object} config  Provider-specific configuration.
     * @param {boolean} debug  Enable verbose logging.
     */
    constructor(config = {}, debug = false) {
        this.config = config;
        this.debug = debug;
        this._emitter = createEmitter();
        this._lastDetection = null;
    }

    on(event, handler) { return this._emitter.on(event, handler); }
    off(event, handler) { this._emitter.off(event, handler); }

    _log(level, msg) {
        if (this.debug) console.log(`[${this.constructor.name}:${level}] ${msg}`);
    }

    /** Subclasses MUST override: load models, prepare for listening. */
    async init() { throw new Error(`${this.constructor.name}.init() not implemented`); }

    /** Subclasses MUST override: start continuous wake word listening. */
    async startListening() { throw new Error(`${this.constructor.name}.startListening() not implemented`); }

    /** Subclasses MUST override: stop listening (must be restartable). */
    async stopListening() { throw new Error(`${this.constructor.name}.stopListening() not implemented`); }

    /** Subclasses MUST override: release all resources. */
    async destroy() { throw new Error(`${this.constructor.name}.destroy() not implemented`); }

    /** Returns the last detection detail, or null. */
    getLastDetection() { return this._lastDetection; }
}


// ==================================================================
// SECTION 3: TEACHABLE MACHINE PROVIDER
// ==================================================================
//
// Drop-in port of the v3.4 wake-sound detection logic. Same TensorFlow.js
// speech-commands library, same .listen() options, same score check.
//
// Config keys (all optional, all match v3.4):
//   url:         string  URL to Teachable Machine model directory (must end with /)
//   threshold:   number  Minimum score to emit a detection (default 0.75)
//   index:       number  Class index in the TM model to monitor (default 2)
//   overlapFactor:       number  Frame overlap for .listen() (default 0.65)
//   probabilityThreshold: number  Inner threshold for .listen() (default 0.75)
//   invokeCallbackOnNoiseAndUnknown: boolean (default true)

class TeachableMachineProvider extends WakeWordProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._recognizer = null;
        this._classLabels = null;
        this._isListening = false;
    }

    async init() {
        this._log('INFO', 'Loading TensorFlow.js + speech-commands...');

        // speech-commands 0.4.0 expects tf 1.3.1 — load both, in order.
        await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.4.0/dist/speech-commands.min.js');

        const url = this.config.url;
        if (!url) throw new Error('TeachableMachineProvider: config.url is required');

        const checkpointURL = url + 'model.json';
        const metadataURL = url + 'metadata.json';

        this._recognizer = speechCommands.create(
            'BROWSER_FFT',
            undefined,
            checkpointURL,
            metadataURL
        );
        await this._recognizer.ensureModelLoaded();

        this._classLabels = this._recognizer.wordLabels();
        const targetClass = this._classLabels[this.config.index ?? 2];

        this._log('OK', `Model loaded. Monitoring class ${this.config.index ?? 2}: "${targetClass}"`);
        this._log('INFO', `Threshold: ${this.config.threshold ?? 0.75}`);
    }

    async startListening() {
        if (!this._recognizer) throw new Error('TeachableMachineProvider not initialized');
        if (this._isListening) return;
        this._isListening = true;

        const threshold = this.config.threshold ?? 0.75;
        const targetIndex = this.config.index ?? 2;

        this._recognizer.listen(result => {
            const scores = result.scores;
            const targetScore = scores[targetIndex];

            if (this.debug && targetScore > 0.5) {
                this._log('DEBUG', `Score: ${targetScore.toFixed(4)} (threshold: ${threshold})`);
            }

            if (targetScore > threshold && targetScore < 1.62) {
                const detail = {
                    score: targetScore,
                    class: this._classLabels[targetIndex],
                    timestamp: Date.now(),
                    raw: { scores }
                };
                this._lastDetection = detail;
                this._emitter.emit('detect', detail);
                if (this.debug) this._log('OK', `Detect: ${targetScore.toFixed(4)}`);
            }
        }, {
            includeSpectrogram: false,
            probabilityThreshold: this.config.probabilityThreshold ?? 0.75,
            invokeCallbackOnNoiseAndUnknown: this.config.invokeCallbackOnNoiseAndUnknown ?? true,
            overlapFactor: this.config.overlapFactor ?? 0.65
        });

        this._log('INFO', 'Listening started.');
    }

    async stopListening() {
        if (!this._recognizer || !this._isListening) return;
        this._isListening = false;
        try { this._recognizer.stopListening(); } catch (e) { /* ignore */ }
        this._log('INFO', 'Listening stopped.');
    }

    async destroy() {
        await this.stopListening();
        this._recognizer = null;
        this._classLabels = null;
        this._log('INFO', 'Destroyed.');
    }
}


// ==================================================================
// SECTION 4: OPEN WAKE WORD PROVIDER
// ==================================================================
//
// Inlined port of WakeWordEngine from openwakeword_wasm (Apache 2.0).
// Source: https://github.com/dnavarrom/openwakeword_wasm/blob/main/src/WakeWordEngine.js
//
// Modifications from upstream:
//   - Uses global `ort` (loaded via <script> by AkarinetVoice) instead of ESM import
//   - Wrapped in a private class `_WakeWordEngine` (underscore prefix = internal)
//   - Public surface is `OpenWakeWordProvider` which implements WakeWordProvider
//
// The 3-stage ONNX pipeline (mel → embedding → keyword head) is preserved
// exactly as designed by the upstream authors. DO NOT modify the math
// (mel normalization `value / 10 + 2`, 76-frame window, 8-frame splice step,
// embedding window sizes 16/22/34, etc.) — these are calibrated to the
// trained model weights and any deviation will produce garbage scores.

const OWW_MODEL_FILE_MAP = {
    alexa: 'alexa_v0.1.onnx',
    hey_mycroft: 'hey_mycroft_v0.1.onnx',
    hey_jarvis: 'hey_jarvis_v0.1.onnx',
    hey_rhasspy: 'hey_rhasspy_v0.1.onnx',
    timer: 'timer_v0.1.onnx',
    weather: 'weather_v0.1.onnx'
};

const OWW_AUDIO_PROCESSOR_CODE = `
class AudioProcessor extends AudioWorkletProcessor {
    bufferSize = 1280;
    _buffer = new Float32Array(this.bufferSize);
    _pos = 0;
    process(inputs) {
        const input = inputs[0][0];
        if (input) {
            for (let i = 0; i < input.length; i++) {
                this._buffer[this._pos++] = input[i];
                if (this._pos === this.bufferSize) {
                    this.port.postMessage(this._buffer);
                    this._pos = 0;
                }
            }
        }
        return true;
    }
}
registerProcessor('audio-processor', AudioProcessor);
`;

/**
 * Internal — port of WakeWordEngine from openwakeword_wasm.
 * Uses the global `ort` object (window.ort) loaded externally.
 */
class _WakeWordEngine {
    constructor({
        keywords = ['hey_jarvis'],
        modelFiles = OWW_MODEL_FILE_MAP,
        baseAssetUrl = '/models',
        ortWasmPath,
        frameSize = 1280,
        sampleRate = 16000,
        vadHangoverFrames = 12,
        detectionThreshold = 0.5,
        cooldownMs = 2000,
        executionProviders = ['wasm'],
        embeddingWindowSize = 16,
        debug = false
    } = {}) {
        this.config = {
            keywords, modelFiles, baseAssetUrl, frameSize, sampleRate,
            vadHangoverFrames, detectionThreshold, cooldownMs,
            executionProviders, embeddingWindowSize, debug
        };
        if (ortWasmPath && typeof ort !== 'undefined') {
            ort.env.wasm.wasmPaths = ortWasmPath;
        }
        this._emitter = createEmitter();
        this._melBuffer = [];
        this._embeddingWindowSize = embeddingWindowSize;
        this._activeKeywords = new Set(keywords);
        this._vadState = { h: null, c: null };
        this._isSpeechActive = false;
        this._vadHangover = 0;
        this._mediaStream = null;
        this._audioContext = null;
        this._workletNode = null;
        this._gainNode = null;
        this._processingQueue = Promise.resolve();
        this._isDetectionCoolingDown = false;
        this._loaded = false;
    }

    on(event, handler) { return this._emitter.on(event, handler); }
    off(event, handler) { this._emitter.off(event, handler); }

    async load() {
        if (this._loaded) return;
        if (typeof ort === 'undefined') {
            throw new Error('_WakeWordEngine: global `ort` (onnxruntime-web) not found. Load it before calling load().');
        }
        const sessionOptions = { executionProviders: this.config.executionProviders };
        const resolver = (file) => `${this.config.baseAssetUrl.replace(/\/+$/, '')}/${file}`;
        this._debug('Loading core models with options', sessionOptions);

        this._melspecModel = await ort.InferenceSession.create(resolver('melspectrogram.onnx'), sessionOptions);
        this._embeddingModel = await ort.InferenceSession.create(resolver('embedding_model.onnx'), sessionOptions);
        this._vadModel = await ort.InferenceSession.create(resolver('silero_vad.onnx'), sessionOptions);

        this._keywordModels = {};
        let maxWindowSize = this.config.embeddingWindowSize;
        for (const keyword of this.config.keywords) {
            const file = this.config.modelFiles[keyword];
            if (!file) throw new Error(`No model file configured for keyword "${keyword}"`);
            const session = await ort.InferenceSession.create(resolver(file), sessionOptions);
            const windowSize = this._inferKeywordWindowSize(session) ?? this.config.embeddingWindowSize;
            maxWindowSize = Math.max(maxWindowSize, windowSize);
            const history = [];
            for (let i = 0; i < windowSize; i++) history.push(new Float32Array(96).fill(0));
            this._keywordModels[keyword] = { session, scores: new Array(50).fill(0), windowSize, history };
            this._debug('Loaded keyword model', { keyword, file, windowSize });
        }
        this._embeddingWindowSize = maxWindowSize;
        this._debug('Embedding window size resolved', this._embeddingWindowSize);
        this._resetState();
        this._loaded = true;
        this._emitter.emit('ready');
    }

    async start({ deviceId, gain = 1.0 } = {}) {
        if (!this._loaded) throw new Error('Call load() before start()');
        if (this._workletNode) return;

        this._resetState();
        this._mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: deviceId ? { deviceId: { exact: deviceId } } : true
        });

        this._audioContext = new AudioContext({ sampleRate: this.config.sampleRate });
        const source = this._audioContext.createMediaStreamSource(this._mediaStream);
        this._gainNode = this._audioContext.createGain();
        this._gainNode.gain.value = gain;

        const blob = new Blob([OWW_AUDIO_PROCESSOR_CODE], { type: 'application/javascript' });
        const workletURL = URL.createObjectURL(blob);
        await this._audioContext.audioWorklet.addModule(workletURL);
        this._workletNode = new AudioWorkletNode(this._audioContext, 'audio-processor');

        this._workletNode.port.onmessage = (event) => {
            const chunk = event.data;
            if (!chunk) return;
            this._processingQueue = this._processingQueue
                .then(() => this._processChunk(chunk))
                .catch((err) => this._emitter.emit('error', err));
        };

        source.connect(this._gainNode);
        this._gainNode.connect(this._workletNode);
        this._workletNode.connect(this._audioContext.destination);
        this._debug('Microphone stream started', { deviceId: deviceId ?? 'default', gain });
    }

    async stop() {
        if (this._workletNode) {
            this._workletNode.port.onmessage = null;
            this._workletNode.disconnect();
            this._workletNode = null;
        }
        if (this._gainNode) { this._gainNode.disconnect(); this._gainNode = null; }
        if (this._audioContext && this._audioContext.state !== 'closed') {
            await this._audioContext.close();
        }
        this._audioContext = null;
        if (this._mediaStream) {
            this._mediaStream.getTracks().forEach((track) => track.stop());
            this._mediaStream = null;
        }
        this._isDetectionCoolingDown = false;
        this._debug('Engine stopped and media stream closed');
    }

    setGain(value) {
        if (this._gainNode) this._gainNode.gain.value = value;
    }

    setActiveKeywords(keywords) {
        const next = Array.isArray(keywords) && keywords.length ? keywords : this.config.keywords;
        this._activeKeywords = new Set(next);
        this._debug('Active keywords updated', Array.from(this._activeKeywords));
    }

    _resetState() {
        this._melBuffer = [];
        const vadShape = [2, 1, 64];
        if (!this._vadState.h) {
            this._vadState.h = new ort.Tensor('float32', new Float32Array(128).fill(0), vadShape);
            this._vadState.c = new ort.Tensor('float32', new Float32Array(128).fill(0), vadShape);
        } else {
            this._vadState.h.data.fill(0);
            this._vadState.c.data.fill(0);
        }
        this._isSpeechActive = false;
        this._vadHangover = 0;
        this._isDetectionCoolingDown = false;
        if (this._keywordModels) {
            for (const key of Object.keys(this._keywordModels)) {
                this._keywordModels[key].scores.fill(0);
                const history = this._keywordModels[key].history;
                if (history) for (let i = 0; i < history.length; i++) history[i].fill(0);
            }
        }
        this._debug('Internal buffers reset');
    }

    async _processChunk(chunk, { emitEvents = true } = {}) {
        if (this.config.debug) {
            let peak = 0, sumSquares = 0;
            for (let i = 0; i < chunk.length; i++) {
                const s = chunk[i];
                sumSquares += s * s;
                const abs = Math.abs(s);
                if (abs > peak) peak = abs;
            }
            const rms = Math.sqrt(sumSquares / chunk.length);
            this._debug('Chunk received', { rms: Number(rms.toFixed(4)), peak: Number(peak.toFixed(4)) });
        }
        const vadTriggered = await this._runVad(chunk);
        if (vadTriggered) {
            if (!this._isSpeechActive && emitEvents) this._emitter.emit('speech-start');
            this._isSpeechActive = true;
            this._vadHangover = this.config.vadHangoverFrames;
        } else if (this._isSpeechActive) {
            this._vadHangover -= 1;
            if (this._vadHangover <= 0) {
                this._isSpeechActive = false;
                if (emitEvents) this._emitter.emit('speech-end');
            }
        }
        await this._runInference(chunk, this._isSpeechActive, emitEvents);
    }

    async _runVad(chunk) {
        try {
            const tensor = new ort.Tensor('float32', chunk, [1, chunk.length]);
            const sr = new ort.Tensor('int64', [BigInt(this.config.sampleRate)], []);
            const res = await this._vadModel.run({ input: tensor, sr, h: this._vadState.h, c: this._vadState.c });
            this._vadState.h = res.hn;
            this._vadState.c = res.cn;
            const confidence = res.output.data[0];
            this._debug('VAD result', { confidence: Number(confidence.toFixed(3)) });
            return confidence > 0.5;
        } catch (err) {
            this._emitter.emit('error', err);
            return false;
        }
    }

    async _runInference(chunk, isSpeechActive, emitEvents) {
        const melspecTensor = new ort.Tensor('float32', chunk, [1, this.config.frameSize]);
        const melspecResults = await this._melspecModel.run({ [this._melspecModel.inputNames[0]]: melspecTensor });
        const newMelData = melspecResults[this._melspecModel.outputNames[0]].data;

        // Calibration: mel normalization — DO NOT MODIFY.
        for (let j = 0; j < newMelData.length; j++) newMelData[j] = newMelData[j] / 10.0 + 2.0;
        // ONNX Runtime reuses output buffers — always copy.
        for (let j = 0; j < 5; j++) {
            this._melBuffer.push(new Float32Array(newMelData.subarray(j * 32, (j + 1) * 32)));
        }

        while (this._melBuffer.length >= 76) {
            const windowFrames = this._melBuffer.slice(0, 76);
            const flattenedMel = new Float32Array(76 * 32);
            for (let j = 0; j < windowFrames.length; j++) flattenedMel.set(windowFrames[j], j * 32);

            const embeddingFeeds = { [this._embeddingModel.inputNames[0]]: new ort.Tensor('float32', flattenedMel, [1, 76, 32, 1]) };
            const embeddingOut = await this._embeddingModel.run(embeddingFeeds);
            const newEmbedding = new Float32Array(embeddingOut[this._embeddingModel.outputNames[0]].data);

            for (const name of Object.keys(this._keywordModels)) {
                const km = this._keywordModels[name];
                km.history.shift();
                km.history.push(newEmbedding);

                const flattenedEmbeddings = new Float32Array(km.windowSize * 96);
                for (let j = 0; j < km.history.length; j++) flattenedEmbeddings.set(km.history[j], j * 96);
                const finalInput = new ort.Tensor('float32', flattenedEmbeddings, [1, km.windowSize, 96]);
                const results = await km.session.run({ [km.session.inputNames[0]]: finalInput });
                const score = results[km.session.outputNames[0]].data[0];
                km.scores.shift();
                km.scores.push(score);
                this._debug('Keyword score', { keyword: name, score: Number(score.toFixed(3)), windowSize: km.windowSize });

                const keywordActive = this._activeKeywords.has(name);
                if (emitEvents && keywordActive && score > this.config.detectionThreshold && isSpeechActive && !this._isDetectionCoolingDown) {
                    this._isDetectionCoolingDown = true;
                    this._debug('Detection emitted', { keyword: name, score });
                    this._emitter.emit('detect', { keyword: name, score, at: performance.now() });
                    setTimeout(() => { this._isDetectionCoolingDown = false; }, this.config.cooldownMs);
                } else if (emitEvents && !keywordActive) {
                    this._debug('Detection suppressed (inactive keyword)', { keyword: name, score });
                }
            }
            this._melBuffer.splice(0, 8);
        }
    }

    _inferKeywordWindowSize(session) {
        if (!session) return undefined;
        const metadata = session.inputMetadata;
        const inputName = session.inputNames?.[0];
        if (!metadata || !inputName) return undefined;
        let meta;
        if (Array.isArray(metadata)) {
            meta = metadata.find((m) => m?.name === inputName) || metadata[0];
        } else {
            meta = metadata[inputName];
        }
        if (!meta || !meta.isTensor || !Array.isArray(meta.shape)) return undefined;
        const dim = meta.shape[1];
        return typeof dim === 'number' && Number.isFinite(dim) ? dim : undefined;
    }

    _debug(...args) {
        if (this.config.debug) console.debug('[_WakeWordEngine]', ...args);
    }
}

/**
 * OpenWakeWord provider — wraps _WakeWordEngine with the WakeWordProvider contract.
 *
 * Config keys (all optional unless noted):
 *   baseAssetUrl:        string  URL to folder containing the .onnx files (REQUIRED)
 *   keywords:            string[]  e.g. ['hey_jarvis'] (default: ['hey_jarvis'])
 *   modelFiles:          Object  Map of keyword → filename (default: OWW_MODEL_FILE_MAP)
 *   detectionThreshold:  number  0..1 (default 0.5)
 *   cooldownMs:          number  ms between detections (default 2000)
 *   vadHangoverFrames:   number  keep speech-active window open this many frames after VAD drops (default 12)
 *   embeddingWindowSize: number  fallback window size if model metadata is missing (default 16)
 *   executionProviders:  string[]  ONNX RT providers (default ['wasm'])
 *   ortWasmPath:         string  optional path to ORT wasm binaries
 *   deviceId:            string  optional mic device ID
 *   gain:                number  mic gain (default 1.0)
 */
class OpenWakeWordProvider extends WakeWordProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._engine = null;
    }

    async init() {
        this._log('INFO', 'Initializing openWakeWord engine...');

        // AkarinetVoice loads onnxruntime-web globally before instantiating providers,
        // but be defensive — load it ourselves if missing.
        if (typeof ort === 'undefined') {
            this._log('INFO', 'Global ort not found — loading onnxruntime-web...');
            await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js');
        }

        const engineConfig = {
            keywords: this.config.keywords ?? ['hey_jarvis'],
            modelFiles: this.config.modelFiles ?? OWW_MODEL_FILE_MAP,
            baseAssetUrl: this.config.baseAssetUrl ?? '/models',
            ortWasmPath: this.config.ortWasmPath,
            detectionThreshold: this.config.detectionThreshold ?? 0.5,
            cooldownMs: this.config.cooldownMs ?? 2000,
            vadHangoverFrames: this.config.vadHangoverFrames ?? 12,
            embeddingWindowSize: this.config.embeddingWindowSize ?? 16,
            executionProviders: this.config.executionProviders ?? ['wasm'],
            debug: this.debug
        };

        this._engine = new _WakeWordEngine(engineConfig);

        // Bridge engine events → provider events.
        this._engine.on('detect', ({ keyword, score, at }) => {
            const detail = {
                score,
                class: keyword,         // normalize to the provider contract
                timestamp: Date.now(),  // wall clock for correlation with VAD
                raw: { keyword, at }
            };
            this._lastDetection = detail;
            this._emitter.emit('detect', detail);
            this._log('OK', `Detect: ${keyword} @ ${score.toFixed(4)}`);
        });
        this._engine.on('error', (err) => {
            this._emitter.emit('error', err);
            this._log('ERROR', `Engine error: ${err.message || err}`);
        });
        this._engine.on('speech-start', () => this._emitter.emit('speech-start'));
        this._engine.on('speech-end', () => this._emitter.emit('speech-end'));

        await this._engine.load();
        this._log('OK', `Engine loaded. Keywords: ${engineConfig.keywords.join(', ')}`);
    }

    async startListening() {
        if (!this._engine) throw new Error('OpenWakeWordProvider not initialized');
        await this._engine.start({
            deviceId: this.config.deviceId,
            gain: this.config.gain ?? 1.0
        });
        this._log('INFO', 'Listening started.');
    }

    async stopListening() {
        if (!this._engine) return;
        await this._engine.stop();
        this._log('INFO', 'Listening stopped.');
    }

    async destroy() {
        if (this._engine) {
            await this._engine.stop();
            this._engine = null;
        }
        this._log('INFO', 'Destroyed.');
    }

    /** Update which keywords may emit detections (without reloading models). */
    setActiveKeywords(keywords) {
        if (this._engine) this._engine.setActiveKeywords(keywords);
    }

    /** Live-adjust mic gain. */
    setGain(value) {
        if (this._engine) this._engine.setGain(value);
    }
}


// ==================================================================
// SECTION 5: AKARINETVOICE ORCHESTRATOR
// ==================================================================
//
// Backwards-compatible with v3.4. New config keys are additive:
//   wakeWordProvider: 'teachablemachine' | 'openwakeword' | 'none'  (default: 'teachablemachine')
//   openWakeWord: { ... }   (only used when wakeWordProvider === 'openwakeword')
//
// Existing v3.4 config keys are mapped to the appropriate provider:
//   wakesoundURL          → TeachableMachineProvider.url
//   wakesoundThreshold    → TeachableMachineProvider.threshold
//   wakesoundIndex        → TeachableMachineProvider.index
//   wakesoundDuration     → orchestrator's wake-sound session window (unchanged)
//   wakesoundDelay        → orchestrator's session delay (unchanged)
//
// The orchestrator still owns:
//   - VAD (vad-web) for speech boundary detection + ASR audio capture
//   - ASR worker (Moonshine via transformers.js)
//   - Wake-sound ↔ speech correlation logic (the inSession / lateDetection dance)
//   - Text wake word parsing (the _parse() method)
//
// The provider only owns:
//   - Continuous wake word listening
//   - Emitting 'detect' events when the wake word fires

export class AkarinetVoice extends EventTarget {
    constructor(config = {}) {
        super();

        // --- Provider selection (NEW in v4.0) ---
        this.config = {
            // ASR (same as v3.4)
            modelId: config.modelId || 'onnx-community/moonshine-tiny-ONNX',
            modelQuantization: config.modelQuantization || 'q8',

            // Text wake words (same as v3.4)
            wakewords: config.wakewords || (config.wakeword ? [config.wakeword] : []),

            // Wake word provider selection (NEW in v4.0, defaults to v3.4 behavior)
            wakeWordProvider: config.wakeWordProvider || (config.wakesoundURL ? 'teachablemachine' : 'none'),

            // v3.4 wake-sound config (mapped to TeachableMachineProvider)
            wakesoundURL: config.wakesoundURL || null,
            wakesoundThreshold: config.wakesoundThreshold ?? 0.75,
            wakesoundIndex: config.wakesoundIndex ?? 2,
            wakesoundDuration: config.wakesoundDuration ?? 100,
            wakesoundDelay: config.wakesoundDelay ?? 0,

            // openWakeWord config (NEW in v4.0)
            openWakeWord: config.openWakeWord || null,

            // VAD + parse (same as v3.4)
            vadThreshold: config.vadThreshold ?? 0.75,
            cleanup: config.cleanup !== undefined ? config.cleanup : true,
            debugWakeSound: config.debugWakeSound || false,
            requireWakeSound: config.requireWakeSound || false
        };

        // Orchestrator state (same as v3.4)
        this._asrWorker = null;
        this._asrCallbacks = new Map();
        this._asrCallId = 0;
        this.vad = null;
        this.wakeWordProvider = null;       // NEW: the provider instance
        this.wakeSoundDetectedTime = null;  // same as v3.4
        this.lastWakeSoundScore = 0;        // same as v3.4
        this.speechStartTime = 0;           // same as v3.4
        this._isProcessing = false;         // same as v3.4
        this._logInitialized = false;
    }

    async init() {
        try {
            this._log('INFO', 'Welcome to AkariNet Audio Console v4.0!');
            this._log('INFO', `Wake word provider: ${this.config.wakeWordProvider}`);

            const needOrt = (this.config.wakeWordProvider === 'openwakeword');
            const modelCount = needOrt ? 3 : 2;
            this._log('OK', `Loading ${modelCount} model${modelCount > 1 ? 's' : ''}...`);

            if (!this.config.debugWakeSound) {
                this._suppressLibraryLogs();
            }

            // 1) Load shared external scripts (ORT if needed, vad-web always).
            await loadScript('https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js');
            if (needOrt) {
                await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js');
            }

            // 2) Instantiate + init the wake word provider.
            this.wakeWordProvider = this._createProvider();
            if (this.wakeWordProvider) {
                await this.wakeWordProvider.init();

                // Wire provider → orchestrator's wake-sound state + events.
                // This is the v3.4 contract: a wake detection sets wakeSoundDetectedTime
                // and dispatches 'wakesound' so engine/audioConsole.js's listener still fires.
                this.wakeWordProvider.on('detect', (detail) => {
                    this.wakeSoundDetectedTime = detail.timestamp;
                    this.lastWakeSoundScore = detail.score;
                    this.dispatchEvent(new CustomEvent('wakesound', {
                        detail: {
                            score: detail.score,
                            class: detail.class,
                            timestamp: detail.timestamp
                        }
                    }));
                });

                await this.wakeWordProvider.startListening();
                this._log('OK', `Wake word provider "${this.config.wakeWordProvider}" listening.`);
            } else {
                this._log('INFO', 'No wake word provider — relying on text wake words via ASR only.');
            }

            // 3) ASR worker (same as v3.4).
            this._log('INFO', `Loading ASR model: ${this.config.modelId} (${this.config.modelQuantization})...`);
            await this._initASRWorker();
            this._log('OK', 'ASR model loaded.');

            // 4) VAD (same as v3.4).
            this._log('INFO', 'Initializing Voice Activity Detection...');
            this.vad = await vad.MicVAD.new({
                positiveSpeechThreshold: this.config.vadThreshold,
                redemptionFrames: 15,
                onSpeechStart: () => {
                    this.speechStartTime = Date.now();
                    this.dispatchEvent(new Event('speechstart'));
                },
                onVADMisfire: () => {
                    this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: '(VAD Misfire)' }));
                },
                onSpeechEnd: async (audio) => {
                    this.dispatchEvent(new Event('speechend'));
                    await this._handleSpeech(audio);
                }
            });
            await this.vad.start();

            this._log('OK', 'AkariNet Audio Console ready.');
            this._log('INFO', `Wake words: ${this.config.wakewords.join(', ') || '(none — text fallback disabled)'}`);
            this._log('INFO', `Wake sound detection: ${this.wakeWordProvider ? `enabled (${this.config.wakeWordProvider})` : 'disabled'}`);

            this.dispatchEvent(new Event('ready'));
        } catch (e) {
            this._log('ERROR', `Initialization failed: ${e.message}`);
            this.dispatchEvent(new CustomEvent('error', { detail: e.message }));
        }
    }

    /**
     * Factory: build the configured wake word provider.
     * @returns {WakeWordProvider|null}
     */
    _createProvider() {
        switch (this.config.wakeWordProvider) {
            case 'teachablemachine':
                if (!this.config.wakesoundURL) {
                    this._log('WARN', 'wakeWordProvider=teachablemachine but no wakesoundURL — skipping provider.');
                    return null;
                }
                return new TeachableMachineProvider({
                    url: this.config.wakesoundURL,
                    threshold: this.config.wakesoundThreshold,
                    index: this.config.wakesoundIndex,
                    // Preserve v3.4 .listen() options exactly.
                    probabilityThreshold: 0.75,
                    invokeCallbackOnNoiseAndUnknown: true,
                    overlapFactor: 0.65
                }, this.config.debugWakeSound);

            case 'openwakeword': {
                const oww = this.config.openWakeWord || {};
                if (!oww.baseAssetUrl) {
                    throw new Error("wakeWordProvider='openwakeword' requires openWakeWord.baseAssetUrl");
                }
                return new OpenWakeWordProvider({
                    baseAssetUrl: oww.baseAssetUrl,
                    keywords: oww.keywords ?? ['hey_jarvis'],
                    modelFiles: oww.modelFiles,
                    detectionThreshold: oww.detectionThreshold ?? 0.5,
                    cooldownMs: oww.cooldownMs ?? 2000,
                    vadHangoverFrames: oww.vadHangoverFrames ?? 12,
                    embeddingWindowSize: oww.embeddingWindowSize ?? 16,
                    executionProviders: oww.executionProviders ?? ['wasm'],
                    ortWasmPath: oww.ortWasmPath,
                    deviceId: oww.deviceId,
                    gain: oww.gain ?? 1.0
                }, this.config.debugWakeSound);
            }

            case 'none':
            case null:
            case undefined:
                return null;

            default:
                throw new Error(`Unknown wakeWordProvider: "${this.config.wakeWordProvider}"`);
        }
    }

    // ----------------------------------------------------------------
    // Speech handling — verbatim port from v3.4 (do not modify logic).
    // ----------------------------------------------------------------

    async _handleSpeech(audio) {
        if (audio.length < 2000) {
            this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: '(Audio Too Short)' }));
            return;
        }
        if (this._isProcessing) {
            this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: '(Busy Processing)' }));
            return;
        }

        // Snapshot before the sync wait loop. The continuous wake sound listener keeps
        // running during speech and can overwrite wakeSoundDetectedTime with a spurious
        // detection mid-utterance. Capturing here ensures long speech isn't penalised.
        const wakeSoundAtSpeechStart = this.wakeSoundDetectedTime;

        const maxSyncWait = 600;
        const startSync = Date.now();
        while (Date.now() - startSync < maxSyncWait) {
            if (this.wakeSoundDetectedTime && this.wakeSoundDetectedTime > this.speechStartTime) break;
            if (this.wakeSoundDetectedTime && (Date.now() - this.wakeSoundDetectedTime > this.config.wakesoundDuration)) break;
            await new Promise(r => setTimeout(r, 50));
        }

        const lateDetection = (this.wakeSoundDetectedTime && this.wakeSoundDetectedTime > this.speechStartTime)
            ? this.wakeSoundDetectedTime
            : null;
        const relevantWakeTime = lateDetection || wakeSoundAtSpeechStart;

        let inSession = false;
        if (relevantWakeTime) {
            if (lateDetection) {
                inSession = true;
            } else {
                const timeSinceWakeSound = this.speechStartTime - relevantWakeTime;
                inSession = timeSinceWakeSound >= 0 && timeSinceWakeSound < this.config.wakesoundDuration;
            }
        }

        if (inSession) {
            let audioToProcess = audio;
            if (lateDetection) {
                const offsetMs = lateDetection - this.speechStartTime;
                const trimSamples = Math.floor((offsetMs / 1000) * 16000);
                if (audio.length > trimSamples) {
                    audioToProcess = audio.slice(trimSamples);
                    if (this.config.debugWakeSound) {
                        this._log('DEBUG', `Trimmed ${offsetMs}ms overlap from audio segment.`);
                    }
                }
            }
            await this._transcribeAndParse(audioToProcess, true);
        } else if (!this.config.requireWakeSound) {
            await this._transcribeAndParse(audio, false);
        } else {
            if (this.config.debugWakeSound) {
                this._log('INFO', 'Skipping ASR (requireWakeSound is true, no wake sound detected).');
            }
            this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: '(Waiting for Wake Sound)' }));
        }
    }

    async _initASRWorker() {
        const workerSrc = `
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
env.allowLocalModels = false;

let asr = null;

self.onmessage = async ({ data }) => {
    if (data.type === 'init') {
        try {
            asr = await pipeline('automatic-speech-recognition', data.modelId, {
                dtype: data.modelQuantization
            });
            self.postMessage({ type: 'ready' });
        } catch (e) {
            self.postMessage({ type: 'initError', message: e.message });
        }
    } else if (data.type === 'transcribe') {
        try {
            const { text } = await asr(data.audio);
            self.postMessage({ type: 'result', id: data.id, text });
        } catch (e) {
            self.postMessage({ type: 'transcribeError', id: data.id, message: e.message });
        }
    }
};
`;
        const blob = new Blob([workerSrc], { type: 'text/javascript' });
        this._asrWorker = new Worker(URL.createObjectURL(blob), { type: 'module' });

        await new Promise((resolve, reject) => {
            const onInit = ({ data }) => {
                if (data.type !== 'ready' && data.type !== 'initError') return;
                this._asrWorker.removeEventListener('message', onInit);
                data.type === 'ready' ? resolve() : reject(new Error(data.message));
            };
            this._asrWorker.addEventListener('message', onInit);
            this._asrWorker.postMessage({
                type: 'init',
                modelId: this.config.modelId,
                modelQuantization: this.config.modelQuantization
            });
        });

        this._asrWorker.addEventListener('message', ({ data }) => {
            if (data.type !== 'result' && data.type !== 'transcribeError') return;
            const cb = this._asrCallbacks.get(data.id);
            if (cb) {
                this._asrCallbacks.delete(data.id);
                cb(data);
            }
        });
    }

    _runASR(audio) {
        return new Promise((resolve, reject) => {
            const id = this._asrCallId++;
            this._asrCallbacks.set(id, (data) => {
                data.type === 'result'
                    ? resolve(data.text)
                    : reject(new Error(data.message));
            });
            const copy = new Float32Array(audio);
            this._asrWorker.postMessage({ type: 'transcribe', id, audio: copy }, [copy.buffer]);
        });
    }

    async _transcribeAndParse(audio, wakeSoundDetected) {
        this._isProcessing = true;
        this.dispatchEvent(new Event('processing'));
        try {
            const text = await this._runASR(audio);
            if (!text) {
                this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: '(Empty Recognition)' }));
                return;
            }
            this._parse(text, wakeSoundDetected);
        } catch (e) {
            this._log('ERROR', `ASR transcription failed: ${e.message}`);
            this.dispatchEvent(new CustomEvent('error', { detail: 'ASR Fail: ' + e.message }));
        } finally {
            this._isProcessing = false;
            this.dispatchEvent(new Event('processingend'));
        }
    }

    _parse(raw, wakeSoundDetected = false) {
        const lower = raw.toLowerCase();
        let cmd = lower;
        let wakeWordFound = wakeSoundDetected;

        if (!wakeWordFound) {
            for (let word of this.config.wakewords) {
                const idx = lower.indexOf(word.toLowerCase());
                if (idx !== -1) {
                    wakeWordFound = true;
                    cmd = lower.substring(idx + word.length).trim();
                    break;
                }
            }
        } else {
            for (let word of this.config.wakewords) {
                const idx = lower.indexOf(word.toLowerCase());
                if (idx !== -1) {
                    cmd = lower.substring(idx + word.length).trim();
                    break;
                }
            }
        }

        if (!wakeWordFound) {
            const detail = this.config.debugWakeSound ? raw : (raw = null, '(No Wake Word Detected)');
            this.dispatchEvent(new CustomEvent('speechdiscarded', { detail }));
            return;
        }

        if (this.config.cleanup) {
            cmd = cmd.replace(/[^\w\s\+\-\*\/\(\)\.]/g, ' ').replace(/\s+/g, ' ').trim();
        }

        this.dispatchEvent(new CustomEvent('result', {
            detail: {
                text: cmd.toUpperCase(),
                original: raw,
                viaSound: wakeSoundDetected
            }
        }));

        if (wakeWordFound) {
            this.wakeSoundDetectedTime = null;
        }
    }

    /**
     * Manually trigger a wake word session — call this from a button click or
     * any other UI event. Behaves identically to a real wake sound detection:
     * the next utterance captured by VAD will be transcribed and parsed as if
     * the wake word had been spoken aloud.
     */
    activateWakeWord() {
        const now = Date.now();
        this.wakeSoundDetectedTime = now;
        this.dispatchEvent(new CustomEvent('wakesound', {
            detail: { score: 1, class: 'manual', timestamp: now }
        }));
    }

    async destroy() {
        this._log('INFO', 'Shutting down AkariNet Audio Console...');

        if (this.wakeWordProvider) {
            try { await this.wakeWordProvider.destroy(); } catch (e) { /* ignore */ }
            this.wakeWordProvider = null;
        }
        if (this.vad) {
            try { await this.vad.pause(); } catch (e) { /* ignore */ }
        }
        if (this._asrWorker) {
            this._asrWorker.terminate();
            this._asrWorker = null;
        }

        this._restoreConsoleLogs();
        this._log('OK', 'Shutdown complete.');
    }

    // ----------------------------------------------------------------
    // Logging (same as v3.4).
    // ----------------------------------------------------------------

    _log(level, message) {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        console.log(`[${level}] ${message}`);
    }

    _suppressLibraryLogs() {
        this._originalConsole = {
            log: console.log,
            warn: console.warn,
            info: console.info
        };
        const filter = (originalMethod) => {
            return (...args) => {
                const message = args.join(' ');
                if (message.match(/^\[(OK|INFO|ERROR|DEBUG|WARN)\]/)) {
                    originalMethod.apply(console, args);
                }
            };
        };
        console.log = filter(this._originalConsole.log);
        console.warn = filter(this._originalConsole.warn);
        console.info = filter(this._originalConsole.info);
    }

    _restoreConsoleLogs() {
        if (this._originalConsole) {
            console.log = this._originalConsole.log;
            console.warn = this._originalConsole.warn;
            console.info = this._originalConsole.info;
        }
    }
}


// ==================================================================
// SECTION 6: EXPORTS
// ==================================================================
//
// Default export: AkarinetVoice (drop-in replacement for v3.4)
// Named exports:  provider classes for advanced users who want to
//                 instantiate providers directly (e.g. for testing
//                 or custom orchestrator wiring).

export { WakeWordProvider, TeachableMachineProvider, OpenWakeWordProvider };

export default AkarinetVoice;
