/**
 * AKARINET AUDIO CONSOLE v4.1
 * ====================================================================
 * Unified, security-hardened voice recognition engine.
 *
 * WHAT'S NEW IN v4.1 (vs v4.0):
 *   - ONE microphone. A single AudioBus owns getUserMedia + the 16 kHz
 *     AudioContext. VAD, wake-word detection, and the optional XL cache
 *     all subscribe to the same chunk stream — no duplicate mic taps.
 *   - Pluggable SPEECH RECOGNITION providers (mirrors the wake-word
 *     provider pattern introduced in v4.0):
 *       'transformers'  (DEFAULT — Moonshine/Whisper via transformers.js, = v4.0)
 *       'whispercpp'    (HTTP POST to a whisper.cpp server — includes llamafile whisperfile)
 *       'webspeech'     (browser SpeechRecognition — wake-gated for privacy, own mic)
 *       'none'          (no ASR; wake word events only)
 *   - XL CACHE: an optional large sliding ring (1 second to 1 hour of
 *     audio) kept in RAM, retrievable after the fact via retrieveCache().
 *     Must be configured before init(); can optionally store only
 *     VAD-verified speech.
 *   - BusVAD: replaces the vad-web dependency with a direct Silero VAD
 *     on the bus chunk stream (same model openWakeWord uses internally).
 *
 * PIPELINE:
 *                          ┌─── AudioBus (ONE getUserMedia, 16 kHz mono) ───┐
 *                          │     AudioWorklet → 1280-sample chunks (80 ms)    │
 *                          └───────────────────────┬──────────────────────────┘
 *                                                  │ chunk stream
 *                  ┌───────────────┬───────────────┼───────────────┬──────────────┐
 *                  ▼               ▼               ▼               ▼              ▼
 *            LiveRing(2 s)    BusVAD(Silero)   WakeProvider   XlCache(opt)   (future taps)
 *            pre-roll for     speech-start/   .feedChunk()    ring, vadOnly
 *            late-detect      speech-end      → 'detect'
 *                             ▼
 *                       on speech-end:
 *                         assemble segment
 *                         → SpeechRecognitionProvider.transcribe(segment)
 *                         → _parse() (text wake words)
 *                         → 'result' event
 *
 * WEB SPEECH API & PRIVACY:
 *   The Web Speech API cannot transcribe pre-recorded clips — it opens
 *   its own microphone and streams to the recognition backend.  To
 *   preserve privacy, WebSpeechProvider is SESSION-BASED: it only starts
 *   a recognition session after a wake event (from a wake-word provider
 *   or a manual activateWakeWord() call).  It never listens continuously.
 *   In Chrome, audio is sent to Google servers during a session; in
 *   Safari, recognition is on-device.  This is acceptable because a
 *   wake event is required first — there is no always-on cloud listening.
 *
 * TEACHABLE MACHINE:
 *   TeachableMachineProvider is preserved for backwards compatibility but
 *   runs on its OWN microphone (TensorFlow.js speech-commands owns its
 *   audio path).  It is NOT fed from the bus.  If you use it alongside
 *   the bus, you will temporarily have two mic consumers.  Prefer
 *   openwakeword for new projects.
 *
 * NON-HTTPS CONTEXTS (http://, file://):
 *   We are optimistic: we try getUserMedia and surface whatever error
 *   the browser gives us.  No pre-emptive "secure context required"
 *   warnings.  On http://localhost and file://, mic access typically
 *   works.  On http:// (non-localhost) the browser may allow one-time
 *   access without persisting the grant — we just try it.
 *
 * BACKWARDS COMPATIBILITY with v4.0:
 *   - Same constructor: new AkarinetVoice(config)
 *   - Same events: ready, speechstart, speechend, wakesound, result,
 *     speechdiscarded, error, processing, processingend
 *   - Same config keys (all v4.0 keys work unchanged):
 *     modelId, modelQuantization, wakewords, wakeWordProvider, openWakeWord,
 *     wakesoundURL, wakesoundThreshold, wakesoundIndex, wakesoundDuration,
 *     wakesoundDelay, vadThreshold, requireWakeSound, cleanup, debugWakeSound
 *   - Same activateWakeWord() and destroy()
 *   - Default speechRecognitionProvider='transformers' reproduces v4.0 ASR
 *   - Default unifiedMic=true (hardened); set false to fall back to v4.0's
 *     separate-mic behavior (not recommended — disables bus/VAD/cache)
 *   - vadThreshold auto-maps: values >= 0.6 (vad-web scale) are shifted to
 *     Silero's confidence scale so existing configs work without changes
 *
 * MIGRATION from v4.0 → v4.1:
 *   Change your import URL from audioConsole-4.0.0.js to audioConsole-4.1.0.js.
 *   Everything else works unchanged.  To use new features, add the new
 *   config keys (all optional):
 *     speechRecognitionProvider, whisperCpp, webSpeech,
 *     unifiedMic, liveCacheMs, xlCache
 *
 * Licenses:
 *   - OpenWakeWordProvider (inlined WakeWordEngine port): Apache 2.0
 *     Adapted from https://github.com/dnavarrom/openwakeword_wasm
 *   - Silero VAD model: MIT (via openwakeword_wasm CDN)
 */


// ==================================================================
// SECTION 1: UTILITIES
// ==================================================================

/**
 * Dynamically load an external <script> tag, deduped by URL.
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
 * Minimal event emitter used by providers and internal components.
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
                try { handler(payload); } catch (err) { console.error('[Emitter] listener error:', err); }
            }
        }
    };
}

/** Returns true if a string is an absolute URL (http:// or https://). */
function isURL(s) {
    return typeof s === 'string' && /^https?:\/\//i.test(s);
}

/**
 * Extract a keyword name from a model URL filename.
 *   '.../hey_akari_v0.1.onnx' → 'hey_akari'
 *   '.../alexa_v0.1.onnx'     → 'alexa'
 *   '.../custom.onnx'         → 'custom'
 */
function extractKeywordNameFromURL(url) {
    const filename = url.split('/').pop() || 'keyword';
    const withoutExt = filename.replace(/\.onnx$/i, '');
    const withoutVersion = withoutExt.replace(/_v\d+\.\d+$/i, '');
    return withoutVersion || 'keyword';
}

/**
 * Encode a Float32Array of mono samples as a 16-bit PCM WAV Blob.
 * Used by WhisperCppProvider to send VAD-captured audio to the server.
 */
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');

    // fmt subchunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);        // subchunk size
    view.setUint16(20, 1, true);         // audio format = PCM
    view.setUint16(22, 1, true);         // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);         // block align
    view.setUint16(34, 16, true);        // bits per sample

    // data subchunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM samples (float -1..1 → int16)
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

/**
 * v4.0 used vad-web whose positiveSpeechThreshold scale differs from
 * Silero's raw confidence.  Silero: ~0.5 is a typical speech threshold,
 * ~0.9 is clear speech.  vad-web: ~0.75 typical, ~0.95 strict.
 *
 * To keep v4.0 configs working without changes, values >= 0.6 (vad-web
 * territory) are shifted into Silero's range by subtracting 0.15.
 * Values < 0.6 are treated as raw Silero confidence (new v4.1 style).
 *
 *   0.85 (v4.0 experimental adapter) → 0.70 (Silero)
 *   0.95 (v4.0 strict)              → 0.80 (Silero)
 *   0.50 (v4.1 default)             → 0.50 (raw)
 */
function normalizeVadThreshold(t) {
    if (t >= 0.6) return t - 0.15;
    return t;
}


// ==================================================================
// SECTION 2: RING BUFFER
// ==================================================================
//
// A fixed-capacity circular Float32Array used by XlCache (and internally
// by AudioBus for the small live pre-roll ring).  Capacity is set at
// construction time and never changes — growing a ring mid-stream would
// require allocating + copying, which stalls audio.

class RingBuffer {
    /**
     * @param {number} durationMs  How many milliseconds of audio to retain.
     * @param {number} sampleRate  Expected sample rate (16000 for the bus).
     */
    constructor(durationMs, sampleRate) {
        this.sampleRate = sampleRate;
        this.capacity = Math.max(1, Math.ceil((durationMs / 1000) * sampleRate));
        this.buf = new Float32Array(this.capacity);
        this.writePos = 0;
        this.filled = 0; // total samples ever written (for availability)
    }

    /** Push a chunk of samples (Float32Array) into the ring. */
    push(samples) {
        const n = samples.length;
        for (let i = 0; i < n; i++) {
            this.buf[this.writePos] = samples[i];
            this.writePos = (this.writePos + 1) % this.capacity;
        }
        this.filled += n;
    }

    /**
     * Retrieve audio from the recent past.
     * @param {number} fromMsAgo  Start of the range (ms ago, larger = further back).
     * @param {number} toMsAgo    End of the range (ms ago, 0 = now). Default 0.
     * @returns {Float32Array}    Samples in [now-fromMsAgo, now-toMsAgo], oldest first.
     */
    retrieve(fromMsAgo, toMsAgo = 0) {
        const samplesPerMs = this.sampleRate / 1000;
        const fromIdx = Math.floor(fromMsAgo * samplesPerMs);
        const toIdx = Math.floor(toMsAgo * samplesPerMs);
        const available = Math.min(this.filled, this.capacity);
        const from = Math.min(fromIdx, available);
        const to = Math.min(toIdx, available);
        if (from <= to) return new Float32Array(0);

        const len = from - to;
        const out = new Float32Array(len);
        // Most recent sample is at (writePos - 1 + capacity) % capacity.
        // Sample k positions back from most recent: (writePos - 1 - k + capacity) % capacity.
        // Read oldest-first: k goes from (from-1) down to to.
        for (let i = 0; i < len; i++) {
            const k = from - 1 - i;
            const idx = (((this.writePos - 1 - k) % this.capacity) + this.capacity) % this.capacity;
            out[i] = this.buf[idx];
        }
        return out;
    }

    clear() {
        this.buf.fill(0);
        this.writePos = 0;
        this.filled = 0;
    }

    /** How many ms of audio are currently available in the ring. */
    get availableMs() {
        return Math.min(this.filled, this.capacity) / this.sampleRate * 1000;
    }
}


// ==================================================================
// SECTION 3: AUDIO BUS
// ==================================================================
//
// Owns ONE getUserMedia stream and ONE 16 kHz mono AudioContext.
// An AudioWorklet taps the mic and emits fixed-size Float32Array chunks
// (1280 samples = 80 ms at 16 kHz, matching openWakeWord's frame size).
//
// Subscribers register via addSubscriber(fn) and receive every chunk.
// Subscribers: BusVAD, XlCache, OpenWakeWordProvider (chunk-fed).
//
// This is the "one mic input" in the unified pipeline.  No other
// component should call getUserMedia — except TeachableMachineProvider
// (which runs on its own mic for legacy reasons) and WebSpeechProvider
// (which opens its own mic only during a wake-gated session).

const BUS_WORKLET_CODE = `
class AudioBusProcessor extends AudioWorkletProcessor {
    bufferSize = 1280; // 80 ms at 16 kHz — matches openWakeWord's frame size
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
registerProcessor('audio-bus-processor', AudioBusProcessor);
`;

class AudioBus {
    /**
     * @param {Object} config
     * @param {number} config.sampleRate      Target AudioContext rate (default 16000).
     * @param {number} config.chunkSize       Worklet buffer size (default 1280).
     * @param {string} [config.deviceId]      Optional mic device ID.
     * @param {number} config.gain            Mic gain (default 1.0).
     * @param {number} config.liveCacheMs     Small pre-roll ring size (default 2000).
     * @param {boolean} debug                 Verbose logging.
     */
    constructor(config = {}, debug = false) {
        this.config = {
            sampleRate: config.sampleRate ?? 16000,
            chunkSize: config.chunkSize ?? 1280,
            deviceId: config.deviceId ?? null,
            gain: config.gain ?? 1.0,
            liveCacheMs: config.liveCacheMs ?? 2000
        };
        this.debug = debug;
        this._subscribers = new Set();
        this._mediaStream = null;
        this._audioContext = null;
        this._sourceNode = null;
        this._gainNode = null;
        this._workletNode = null;
        this._active = false;
        // Small pre-roll ring — always available when the bus is running.
        // Used for late wake-detection reslicing and as a general "recent audio"
        // buffer.  Distinct from XlCache (which is the large optional ring).
        this.liveRing = new RingBuffer(this.config.liveCacheMs, this.config.sampleRate);
    }

    /**
     * Register a chunk subscriber.  fn(Float32Array) is called for every
     * 1280-sample chunk emitted by the worklet.  Returns an unsubscribe fn.
     */
    addSubscriber(fn) {
        this._subscribers.add(fn);
        return () => this._subscribers.delete(fn);
    }

    async start() {
        if (this._active) return;

        // --- getUserMedia (optimistic: just try it) ---
        try {
            this._mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: this.config.deviceId
                    ? { deviceId: { exact: this.config.deviceId } }
                    : true
            });
        } catch (e) {
            throw new Error(
                `Microphone access failed: ${e.message || e.name}. ` +
                `If you're on http://, try localhost or https://.`
            );
        }

        // --- AudioContext at 16 kHz ---
        // Modern Chrome/Firefox/Safari support arbitrary sample rates.
        // The browser resamples the mic input to the requested rate.
        try {
            this._audioContext = new AudioContext({ sampleRate: this.config.sampleRate });
        } catch (e) {
            // Fallback: default-rate context (chunks will be at the wrong rate
            // for openWakeWord, but at least the bus runs).
            this._audioContext = new AudioContext();
            if (this.debug) console.warn('[AudioBus] Could not create 16 kHz context, using default:', this._audioContext.sampleRate);
        }

        // Some browsers create the context in "suspended" state until a user
        // gesture resumes it.  init() is typically called from a click handler.
        if (this._audioContext.state === 'suspended') {
            try { await this._audioContext.resume(); } catch (e) { /* will retry on first chunk */ }
        }

        const source = this._audioContext.createMediaStreamSource(this._mediaStream);
        this._sourceNode = source;
        this._gainNode = this._audioContext.createGain();
        this._gainNode.gain.value = this.config.gain;

        // --- AudioWorklet ---
        const blob = new Blob([BUS_WORKLET_CODE], { type: 'application/javascript' });
        const workletURL = URL.createObjectURL(blob);
        await this._audioContext.audioWorklet.addModule(workletURL);
        URL.revokeObjectURL(workletURL); // module is registered, URL no longer needed
        this._workletNode = new AudioWorkletNode(this._audioContext, 'audio-bus-processor');

        this._workletNode.port.onmessage = (event) => {
            const chunk = event.data;
            if (!chunk) return;
            // Always push to live ring (pre-roll buffer).
            this.liveRing.push(chunk);
            // Dispatch to all subscribers (VAD, cache, wake provider).
            for (const fn of this._subscribers) {
                try { fn(chunk); } catch (err) { console.error('[AudioBus] subscriber error:', err); }
            }
        };

        source.connect(this._gainNode);
        this._gainNode.connect(this._workletNode);
        // Connect to destination so the worklet's process() is called.
        // (AudioWorklet nodes must be connected to the destination graph to run.)
        // We use a zero-gain node to avoid echoing the mic back to the speaker.
        const muteGain = this._audioContext.createGain();
        muteGain.gain.value = 0;
        this._workletNode.connect(muteGain);
        muteGain.connect(this._audioContext.destination);

        this._active = true;
        if (this.debug) console.log('[AudioBus] Started. Sample rate:', this._audioContext.sampleRate);
    }

    /** Live-adjust mic gain. */
    setGain(value) {
        if (this._gainNode) this._gainNode.gain.value = value;
    }

    async stop() {
        this._active = false;
        if (this._workletNode) {
            this._workletNode.port.onmessage = null;
            try { this._workletNode.disconnect(); } catch (e) {}
            this._workletNode = null;
        }
        if (this._gainNode) { try { this._gainNode.disconnect(); } catch (e) {} this._gainNode = null; }
        if (this._sourceNode) { try { this._sourceNode.disconnect(); } catch (e) {} this._sourceNode = null; }
        if (this._audioContext && this._audioContext.state !== 'closed') {
            try { await this._audioContext.close(); } catch (e) {}
        }
        this._audioContext = null;
        if (this._mediaStream) {
            this._mediaStream.getTracks().forEach(t => t.stop());
            this._mediaStream = null;
        }
        if (this.debug) console.log('[AudioBus] Stopped.');
    }

    get isActive() { return this._active; }
    get sampleRate() { return this._audioContext?.sampleRate ?? this.config.sampleRate; }
}


// ==================================================================
// SECTION 4: XL CACHE
// ==================================================================
//
// A large optional ring buffer (1 second to 1 hour) that keeps a
// rolling window of mic audio in RAM.  Retrievable after the fact via
// AkarinetVoice.retrieveCache() — "start recording back in time".
//
// MUST be configured before init() (ring size is fixed at construction).
// After init, only clear() and retrieve() are available.
//
// vadOnly: if true, only chunks where VAD detected speech are stored.
// This dramatically reduces RAM usage for long durations (e.g. 1 hour
// of pure speech is far less than 1 hour of continuous audio).

class XlCache {
    /**
     * @param {Object} config
     * @param {number} config.durationMs   1000 .. 3600000 (1s .. 1h).
     * @param {boolean} config.vadOnly     Only store VAD-positive chunks.
     * @param {number} config.sampleRate   16000 (from bus).
     * @param {boolean} debug
     */
    constructor(config = {}, debug = false) {
        this.debug = debug;
        // Validate duration range.
        const dur = config.durationMs ?? 60000;
        if (dur < 1000 || dur > 3600000) {
            throw new Error(`XlCache: durationMs must be between 1000 and 3600000 (1s to 1h), got ${dur}`);
        }
        this.config = {
            durationMs: dur,
            vadOnly: config.vadOnly ?? false,
            sampleRate: config.sampleRate ?? 16000
        };
        this._ring = new RingBuffer(this.config.durationMs, this.config.sampleRate);
        this._purgedCount = 0; // how many chunks were evicted (oldest)
    }

    /**
     * Push a chunk.  If vadOnly is true, isSpeech must be true for the
     * chunk to be stored.
     * @param {Float32Array} chunk
     * @param {boolean} isSpeech   Whether VAD currently detects speech.
     */
    push(chunk, isSpeech = false) {
        if (this.config.vadOnly && !isSpeech) return;
        const before = this._ring.filled;
        this._ring.push(chunk);
        // RingBuffer auto-evicts; we don't track exact evictions but
        // availableMs tells the caller how much is retrievable.
    }

    /**
     * Retrieve a time range from the cache.
     * @param {Object} opts
     * @param {number} opts.fromMsAgo  Start (ms ago, larger = further back).
     * @param {number} [opts.toMsAgo]  End (ms ago, 0 = now). Default 0.
     * @returns {{ audio: Float32Array, sampleRate: number, availableMs: number }}
     */
    retrieve({ fromMsAgo, toMsAgo = 0 } = {}) {
        if (typeof fromMsAgo !== 'number' || fromMsAgo < 0) {
            throw new Error('XlCache.retrieve: fromMsAgo is required and must be >= 0');
        }
        const audio = this._ring.retrieve(fromMsAgo, toMsAgo);
        return {
            audio,
            sampleRate: this.config.sampleRate,
            availableMs: this._ring.availableMs
        };
    }

    clear() {
        this._ring.clear();
        this._purgedCount = 0;
    }

    get availableMs() { return this._ring.availableMs; }
}


// ==================================================================
// SECTION 5: BUS VAD (Silero Voice Activity Detection)
// ==================================================================
//
// Replaces the vad-web dependency from v4.0.  Runs the Silero VAD ONNX
// model directly on bus chunks using onnxruntime-web (which we load
// for openWakeWord anyway).  This gives us:
//   - Full control over the VAD pipeline (no external mic).
//   - Same model openWakeWord uses internally (browser-cached after first fetch).
//   - Speech segment assembly (like vad-web's onSpeechEnd(audio)).
//
// Emits:
//   'speech-start'   — when speech begins (confidence > threshold after silence)
//   'speech-end'     — { audio: Float32Array } when speech ends after redemption period
//   'misfire'        — { audio } when speech was too short (< MIN_SPEECH_SAMPLES)
//   'error'          — VAD inference errors

const VAD_MODEL_URL = 'https://cdn.jsdelivr.net/gh/dnavarrom/openwakeword_wasm@main/models/silero_vad.onnx';
const MIN_SPEECH_SAMPLES = 2000; // ~125 ms at 16 kHz — same as v4.0's misfire threshold

class BusVAD {
    /**
     * @param {Object} config
     * @param {Object} config.ort           onnxruntime-web namespace (REQUIRED).
     * @param {number} config.threshold     Silero confidence threshold (0..1).
     * @param {number} config.redemptionMs  Silence before declaring speech-end (default 480).
     * @param {number} config.sampleRate    16000.
     * @param {number} config.chunkSize     1280 (from bus).
     * @param {boolean} debug
     */
    constructor(config = {}, debug = false) {
        this.debug = debug;
        this.config = {
            ort: config.ort,
            threshold: config.threshold ?? 0.5,
            redemptionMs: config.redemptionMs ?? 480,
            sampleRate: config.sampleRate ?? 16000,
            chunkSize: config.chunkSize ?? 1280
        };
        this._session = null;
        this._h = null; // Silero LSTM hidden state
        this._c = null; // Silero LSTM cell state
        this._isSpeech = false;
        this._redemptionCount = 0;
        this._redemptionFrames = Math.ceil(this.config.redemptionMs / (this.config.chunkSize / this.config.sampleRate * 1000));
        this._speechBuffer = [];
        this._emitter = createEmitter();
    }

    on(event, handler) { return this._emitter.on(event, handler); }

    async init() {
        if (!this.config.ort) throw new Error('BusVAD: config.ort (onnxruntime-web) is required');
        const sessionOptions = { executionProviders: ['wasm'] };
        this._session = await this.config.ort.InferenceSession.create(VAD_MODEL_URL, sessionOptions);
        // Silero VAD state: h and c tensors, shape [2, 1, 64], 128 floats each.
        this._h = new this.config.ort.Tensor('float32', new Float32Array(128).fill(0), [2, 1, 64]);
        this._c = new this.config.ort.Tensor('float32', new Float32Array(128).fill(0), [2, 1, 64]);
        if (this.debug) console.log('[BusVAD] Silero VAD loaded. Threshold:', this.config.threshold);
    }

    /**
     * Feed a chunk from the bus.  Runs VAD inference and manages speech
     * state.  Called for every 1280-sample chunk.
     */
    feedChunk(chunk) {
        // Process asynchronously; results update state and may emit events.
        // We don't serialize with a queue — Silero inference is fast (~1-5ms)
        // and the bus emits chunks at 80ms intervals, so there's ample headroom.
        this._runVad(chunk).then(triggered => {
            if (triggered) {
                if (!this._isSpeech) {
                    this._isSpeech = true;
                    this._speechBuffer = [];
                    this._emitter.emit('speech-start');
                }
                this._redemptionCount = this._redemptionFrames;
                this._speechBuffer.push(chunk);
            } else if (this._isSpeech) {
                this._redemptionCount--;
                this._speechBuffer.push(chunk); // include trailing silence in segment
                if (this._redemptionCount <= 0) {
                    this._isSpeech = false;
                    // Assemble the speech segment.
                    const total = this._speechBuffer.reduce((s, a) => s + a.length, 0);
                    const audio = new Float32Array(total);
                    let off = 0;
                    for (const c of this._speechBuffer) { audio.set(c, off); off += c.length; }
                    this._speechBuffer = [];

                    if (audio.length < MIN_SPEECH_SAMPLES) {
                        this._emitter.emit('misfire', { audio });
                    } else {
                        this._emitter.emit('speech-end', { audio });
                    }
                }
            }
        }).catch(err => {
            this._emitter.emit('error', err);
        });
    }

    async _runVad(chunk) {
        const ort = this.config.ort;
        try {
            const tensor = new ort.Tensor('float32', chunk, [1, chunk.length]);
            const sr = new ort.Tensor('int64', [BigInt(this.config.sampleRate)], []);
            // Silero VAD input/output names match the openwakeword export:
            //   inputs:  input, sr, h, c
            //   outputs: output (confidence), hn, cn (new state)
            const res = await this._session.run({ input: tensor, sr, h: this._h, c: this._c });
            this._h = res.hn;
            this._c = res.cn;
            const confidence = res.output.data[0];
            return confidence > this.config.threshold;
        } catch (err) {
            this._emitter.emit('error', err);
            return false;
        }
    }

    get isSpeechActive() { return this._isSpeech; }

    reset() {
        this._isSpeech = false;
        this._redemptionCount = 0;
        this._speechBuffer = [];
        if (this._h) this._h.data.fill(0);
        if (this._c) this._c.data.fill(0);
    }

    async destroy() {
        this._session = null;
        this._h = null;
        this._c = null;
        this._speechBuffer = [];
    }
}


// ==================================================================
// SECTION 6: WAKE WORD PROVIDER INTERFACE
// ==================================================================
//
// All wake word providers implement this interface.  The orchestrator
// talks to providers through this contract only.
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
//
// CHUNK-FED PROVIDERS (NEW in v4.1):
//   Providers that can accept audio chunks from the bus implement
//   feedChunk(chunk).  The orchestrator checks for this method and, if
//   present, feeds bus chunks to the provider instead of expecting it
//   to open its own microphone.
//
//   Currently chunk-fed: OpenWakeWordProvider
//   Own-mic (legacy):   TeachableMachineProvider

class WakeWordProvider {
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

    async init() { throw new Error(`${this.constructor.name}.init() not implemented`); }
    async startListening() { throw new Error(`${this.constructor.name}.startListening() not implemented`); }
    async stopListening() { throw new Error(`${this.constructor.name}.stopListening() not implemented`); }
    async destroy() { throw new Error(`${this.constructor.name}.destroy() not implemented`); }

    getLastDetection() { return this._lastDetection; }

    /**
     * Optional: receive a chunk from the bus.  Providers that don't
     * implement this are assumed to manage their own microphone.
     * @param {Float32Array} _chunk
     */
    feedChunk(_chunk) { /* default: no-op — provider manages its own mic */ }
}


// ==================================================================
// SECTION 7: TEACHABLE MACHINE PROVIDER (own mic, legacy)
// ==================================================================
//
// Drop-in port of the v4.0 wake-sound detection logic.  Same TensorFlow.js
// speech-commands library, same .listen() options, same score check.
//
// This provider runs on its OWN microphone — TensorFlow.js speech-commands
// owns its audio analyser path and cannot be fed external chunks.  It is
// preserved for backwards compatibility but is NOT recommended for new
// projects.  Prefer openwakeword (chunk-fed, speaker-invariant, no TFJS).
//
// Config keys (all optional, all match v4.0):
//   url, threshold, index, overlapFactor, probabilityThreshold,
//   invokeCallbackOnNoiseAndUnknown

class TeachableMachineProvider extends WakeWordProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._recognizer = null;
        this._classLabels = null;
        this._isListening = false;
    }

    async init() {
        this._log('INFO', 'Loading TensorFlow.js + speech-commands (own mic)...');
        await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.4.0/dist/speech-commands.min.js');

        const url = this.config.url;
        if (!url) throw new Error('TeachableMachineProvider: config.url is required');

        const checkpointURL = url + 'model.json';
        const metadataURL = url + 'metadata.json';

        this._recognizer = speechCommands.create('BROWSER_FFT', undefined, checkpointURL, metadataURL);
        await this._recognizer.ensureModelLoaded();

        this._classLabels = this._recognizer.wordLabels();
        const targetClass = this._classLabels[this.config.index ?? 2];
        this._log('OK', `Model loaded. Monitoring class ${this.config.index ?? 2}: "${targetClass}"`);
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
            if (targetScore > threshold && targetScore < 1.62) {
                const detail = {
                    score: targetScore,
                    class: this._classLabels[targetIndex],
                    timestamp: Date.now(),
                    raw: { scores }
                };
                this._lastDetection = detail;
                this._emitter.emit('detect', detail);
            }
        }, {
            includeSpectrogram: false,
            probabilityThreshold: this.config.probabilityThreshold ?? 0.75,
            invokeCallbackOnNoiseAndUnknown: this.config.invokeCallbackOnNoiseAndUnknown ?? true,
            overlapFactor: this.config.overlapFactor ?? 0.65
        });
        this._log('INFO', 'Listening started (own mic).');
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
// SECTION 8: OPEN WAKE WORD PROVIDER (chunk-fed from bus)
// ==================================================================
//
// Inlined port of WakeWordEngine from openwakeword_wasm (Apache 2.0).
// Source: https://github.com/dnavarrom/openwakeword_wasm/blob/main/src/WakeWordEngine.js
//
// v4.1 MODIFICATION: The engine no longer opens its own microphone.
// Instead, it receives chunks via feedChunk() from the AudioBus.  This
// is the "one mic" hardening — openWakeWord processes the same chunks
// as VAD and the cache, with no duplicate getUserMedia.
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

const OWW_DEFAULT_CORE_URLS = {
    melspectrogram: 'https://cdn.jsdelivr.net/gh/dnavarrom/openwakeword_wasm@main/models/melspectrogram.onnx',
    embedding:      'https://cdn.jsdelivr.net/gh/dnavarrom/openwakeword_wasm@main/models/embedding_model.onnx',
    vad:            'https://cdn.jsdelivr.net/gh/dnavarrom/openwakeword_wasm@main/models/silero_vad.onnx'
};

/**
 * Internal — port of WakeWordEngine from openwakeword_wasm, modified to
 * be chunk-fed (no internal mic management).  Uses global `ort` passed
 * in explicitly via constructor config.
 */
class _WakeWordEngine {
    constructor({
        keywords = ['hey_jarvis'],
        modelFiles = OWW_MODEL_FILE_MAP,
        baseAssetUrl = '/models',
        coreModelURLs = OWW_DEFAULT_CORE_URLS,
        ort,
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
        if (!ort) throw new Error('_WakeWordEngine: `ort` (onnxruntime-web namespace) must be passed in the constructor config.');
        this._ort = ort;
        this.config = {
            keywords, modelFiles, baseAssetUrl, coreModelURLs, frameSize, sampleRate,
            vadHangoverFrames, detectionThreshold, cooldownMs,
            executionProviders, embeddingWindowSize, debug
        };
        if (ortWasmPath) { ort.env.wasm.wasmPaths = ortWasmPath; }
        this._emitter = createEmitter();
        this._melBuffer = [];
        this._embeddingWindowSize = embeddingWindowSize;
        this._activeKeywords = new Set(keywords);
        this._vadState = { h: null, c: null };
        this._isSpeechActive = false;
        this._vadHangover = 0;
        // Serialization queue — ensures chunks are processed in order even
        // though ONNX inference is async.  Same pattern as v4.0.
        this._processingQueue = Promise.resolve();
        this._isDetectionCoolingDown = false;
        this._loaded = false;
    }

    on(event, handler) { return this._emitter.on(event, handler); }
    off(event, handler) { this._emitter.off(event, handler); }

    async load() {
        if (this._loaded) return;
        const ort = this._ort;
        const sessionOptions = { executionProviders: this.config.executionProviders };
        const resolver = (file) => `${this.config.baseAssetUrl.replace(/\/+$/, '')}/${file}`;
        const resolveSource = (fileOrUrl) => isURL(fileOrUrl) ? fileOrUrl : resolver(fileOrUrl);

        this._melspecModel = await ort.InferenceSession.create(this.config.coreModelURLs.melspectrogram, sessionOptions);
        this._embeddingModel = await ort.InferenceSession.create(this.config.coreModelURLs.embedding, sessionOptions);
        this._vadModel = await ort.InferenceSession.create(this.config.coreModelURLs.vad, sessionOptions);

        this._keywordModels = {};
        let maxWindowSize = this.config.embeddingWindowSize;
        for (const keyword of this.config.keywords) {
            const source = this.config.modelFiles[keyword];
            if (!source) throw new Error(`No model file or URL configured for keyword "${keyword}"`);
            const session = await ort.InferenceSession.create(resolveSource(source), sessionOptions);
            const windowSize = this._inferKeywordWindowSize(session) ?? this.config.embeddingWindowSize;
            maxWindowSize = Math.max(maxWindowSize, windowSize);
            const history = [];
            for (let i = 0; i < windowSize; i++) history.push(new Float32Array(96).fill(0));
            this._keywordModels[keyword] = { session, scores: new Array(50).fill(0), windowSize, history };
        }
        this._embeddingWindowSize = maxWindowSize;
        this._resetState();
        this._loaded = true;
        this._emitter.emit('ready');
    }

    /**
     * v4.1: Receive a chunk from the bus.  Enqueues processing via the
     * serialization queue (same as v4.0's worklet onmessage handler).
     */
    feedChunk(chunk) {
        this._processingQueue = this._processingQueue
            .then(() => this._processChunk(chunk))
            .catch((err) => this._emitter.emit('error', err));
    }

    setActiveKeywords(keywords) {
        const next = Array.isArray(keywords) && keywords.length ? keywords : this.config.keywords;
        this._activeKeywords = new Set(next);
    }

    _resetState() {
        const ort = this._ort;
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
    }

    async _processChunk(chunk, { emitEvents = true } = {}) {
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
        const ort = this._ort;
        try {
            const tensor = new ort.Tensor('float32', chunk, [1, chunk.length]);
            const sr = new ort.Tensor('int64', [BigInt(this.config.sampleRate)], []);
            const res = await this._vadModel.run({ input: tensor, sr, h: this._vadState.h, c: this._vadState.c });
            this._vadState.h = res.hn;
            this._vadState.c = res.cn;
            const confidence = res.output.data[0];
            return confidence > 0.5;
        } catch (err) {
            this._emitter.emit('error', err);
            return false;
        }
    }

    async _runInference(chunk, isSpeechActive, emitEvents) {
        const ort = this._ort;
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

                const keywordActive = this._activeKeywords.has(name);
                if (emitEvents && keywordActive && score > this.config.detectionThreshold && isSpeechActive && !this._isDetectionCoolingDown) {
                    this._isDetectionCoolingDown = true;
                    this._emitter.emit('detect', { keyword: name, score, at: performance.now() });
                    setTimeout(() => { this._isDetectionCoolingDown = false; }, this.config.cooldownMs);
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
 * openWakeWord provider — chunk-fed from the AudioBus (v4.1).
 *
 * THREE WAYS TO SPECIFY KEYWORD MODELS (pick one):
 *
 * 1. Single URL (simplest):
 *    { keywordURL: 'https://.../hey_akari_v0.1.onnx' }
 *
 * 2. Multiple URLs:
 *    { keywordURLs: ['https://...', { url: 'https://...', name: 'alexa' }] }
 *
 * 3. Legacy (filename + baseAssetUrl):
 *    { baseAssetUrl: 'https://host/models', keywords: ['hey_jarvis'] }
 *
 * Core models (melspectrogram, embedding, VAD) auto-loaded from jsdelivr CDN.
 */
class OpenWakeWordProvider extends WakeWordProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._engine = null;
        this._isListening = false;
    }

    _resolveKeywordConfig() {
        const cfg = this.config;
        if (cfg.keywordURL) {
            const name = cfg.keywordName || extractKeywordNameFromURL(cfg.keywordURL);
            return { keywords: [name], modelFiles: { [name]: cfg.keywordURL } };
        }
        if (Array.isArray(cfg.keywordURLs) && cfg.keywordURLs.length > 0) {
            const keywords = [];
            const modelFiles = {};
            for (const entry of cfg.keywordURLs) {
                if (typeof entry === 'string') {
                    const name = extractKeywordNameFromURL(entry);
                    keywords.push(name);
                    modelFiles[name] = entry;
                } else if (entry && typeof entry === 'object' && entry.url) {
                    const name = entry.name || extractKeywordNameFromURL(entry.url);
                    keywords.push(name);
                    modelFiles[name] = entry.url;
                }
            }
            return { keywords, modelFiles };
        }
        return {
            keywords: cfg.keywords ?? ['hey_jarvis'],
            modelFiles: cfg.modelFiles ?? OWW_MODEL_FILE_MAP
        };
    }

    async init() {
        this._log('INFO', 'Initializing openWakeWord engine (chunk-fed)...');

        let ortNS = this.config.ort;
        if (!ortNS && typeof globalThis !== 'undefined' && globalThis.ort) ortNS = globalThis.ort;
        if (!ortNS && typeof window !== 'undefined' && window.ort) ortNS = window.ort;
        if (!ortNS) {
            this._log('INFO', 'Global ort not found — loading onnxruntime-web ESM...');
            try {
                const mod = await import(/* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/+esm');
                ortNS = mod.default || mod;
            } catch (e) {
                throw new Error(`OpenWakeWordProvider: failed to load onnxruntime-web: ${e.message}`);
            }
        }
        if (!ortNS || !ortNS.InferenceSession) {
            throw new Error('OpenWakeWordProvider: onnxruntime-web loaded but InferenceSession not found.');
        }

        const { keywords, modelFiles } = this._resolveKeywordConfig();
        const engineConfig = {
            keywords,
            modelFiles,
            baseAssetUrl: this.config.baseAssetUrl ?? '/models',
            coreModelURLs: { ...OWW_DEFAULT_CORE_URLS, ...(this.config.coreModelURLs || {}) },
            ort: ortNS,
            ortWasmPath: this.config.ortWasmPath,
            detectionThreshold: this.config.detectionThreshold ?? 0.5,
            cooldownMs: this.config.cooldownMs ?? 2000,
            vadHangoverFrames: this.config.vadHangoverFrames ?? 12,
            embeddingWindowSize: this.config.embeddingWindowSize ?? 16,
            executionProviders: this.config.executionProviders ?? ['wasm'],
            debug: this.debug
        };

        this._engine = new _WakeWordEngine(engineConfig);

        this._engine.on('detect', ({ keyword, score, at }) => {
            const detail = {
                score,
                class: keyword,
                timestamp: Date.now(),
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
        // OWW's internal VAD speech-start/end events are NOT forwarded to
        // the orchestrator — BusVAD is the authority for ASR gating.
        // OWW's VAD only gates keyword detection internally.

        await this._engine.load();
        this._log('OK', `Engine loaded. Keywords: ${engineConfig.keywords.join(', ')}`);
    }

    async startListening() {
        if (!this._engine) throw new Error('OpenWakeWordProvider not initialized');
        this._isListening = true;
        this._log('INFO', 'Listening (chunk-fed from bus).');
    }

    async stopListening() {
        this._isListening = false;
        this._log('INFO', 'Listening stopped.');
    }

    /** Receive a chunk from the AudioBus. */
    feedChunk(chunk) {
        if (this._isListening && this._engine) {
            this._engine.feedChunk(chunk);
        }
    }

    async destroy() {
        this._isListening = false;
        this._engine = null;
        this._log('INFO', 'Destroyed.');
    }

    setActiveKeywords(keywords) {
        if (this._engine) this._engine.setActiveKeywords(keywords);
    }
}


// ==================================================================
// SECTION 9: SPEECH RECOGNITION PROVIDER INTERFACE
// ==================================================================
//
// Mirrors the wake-word provider pattern.  Two modes:
//
// SEGMENT-BASED (transformers, whispercpp):
//   The orchestrator captures a speech segment via VAD and calls
//   transcribe(Float32Array) → Promise<string>.  The provider returns
//   the transcript.  This is the v4.0 model.
//
// SESSION-BASED (webspeech):
//   The provider opens its own mic for a short recognition session,
//   triggered by a wake event.  It emits 'result' (transcript) and
//   'end' (session ended).  VAD is not used — the browser's own
//   endpointing handles utterance boundaries.  This mode is required
//   because the Web Speech API cannot accept pre-recorded audio.
//
//   isSessionBased getter returns true for session-based providers.

class SpeechRecognitionProvider {
    constructor(config = {}, debug = false) {
        this.config = config;
        this.debug = debug;
        this._emitter = createEmitter();
    }

    on(event, handler) { return this._emitter.on(event, handler); }

    _log(level, msg) {
        if (this.debug) console.log(`[${this.constructor.name}:${level}] ${msg}`);
    }

    async init() { throw new Error(`${this.constructor.name}.init() not implemented`); }

    /** Segment-based providers override this. Returns transcript string. */
    async transcribe(_audio) { throw new Error(`${this.constructor.name}.transcribe() not implemented`); }

    /** Session-based providers override these. */
    async startSession() { /* default: no-op */ }
    async stopSession() { /* default: no-op */ }

    /** True for session-based providers (webspeech). */
    get isSessionBased() { return false; }

    async destroy() { throw new Error(`${this.constructor.name}.destroy() not implemented`); }
}


// ==================================================================
// SECTION 10: TRANSFORMERS PROVIDER (Moonshine/Whisper via transformers.js)
// ==================================================================
//
// Direct port of v4.0's ASR worker.  Loads a transformers.js ASR pipeline
// in a Web Worker, sends VAD-captured Float32Array segments for
// transcription, receives text back.  Default ASR backend in v4.1.

class TransformersProvider extends SpeechRecognitionProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._worker = null;
        this._callbacks = new Map();
        this._callId = 0;
    }

    async init() {
        this._log('INFO', `Loading ASR model: ${this.config.modelId} (${this.config.modelQuantization})...`);

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
        this._worker = new Worker(URL.createObjectURL(blob), { type: 'module' });

        await new Promise((resolve, reject) => {
            const onInit = ({ data }) => {
                if (data.type !== 'ready' && data.type !== 'initError') return;
                this._worker.removeEventListener('message', onInit);
                data.type === 'ready' ? resolve() : reject(new Error(data.message));
            };
            this._worker.addEventListener('message', onInit);
            this._worker.postMessage({
                type: 'init',
                modelId: this.config.modelId,
                modelQuantization: this.config.modelQuantization
            });
        });

        this._worker.addEventListener('message', ({ data }) => {
            if (data.type !== 'result' && data.type !== 'transcribeError') return;
            const cb = this._callbacks.get(data.id);
            if (cb) {
                this._callbacks.delete(data.id);
                cb(data);
            }
        });

        this._log('OK', 'ASR model loaded.');
    }

    transcribe(audio) {
        return new Promise((resolve, reject) => {
            const id = this._callId++;
            this._callbacks.set(id, (data) => {
                data.type === 'result'
                    ? resolve(data.text)
                    : reject(new Error(data.message));
            });
            // Transfer the buffer to avoid copying (worker gets ownership).
            const copy = new Float32Array(audio);
            this._worker.postMessage({ type: 'transcribe', id, audio: copy }, [copy.buffer]);
        });
    }

    async destroy() {
        if (this._worker) {
            this._worker.terminate();
            this._worker = null;
        }
        this._log('INFO', 'Destroyed.');
    }
}


// ==================================================================
// SECTION 11: WHISPER.CPP PROVIDER (llamafile whisperfile / whisper-server)
// ==================================================================
//
// Sends VAD-captured audio to an external whisper.cpp HTTP server for
// transcription.  This is the path for llamafile's whisperfile, which
// bundles whisper.cpp's whisper-server with Cosmopolitan packaging.
//
// API (whisper.cpp server, default port 8080):
//   POST /inference   (multipart/form-data)
//     file=<audio.wav>   response_format=json   [temperature=0.0]   [language=en]
//   → { "text": "transcribed text" }
//   GET  /load        (swap models at runtime)
//   GET  /status
//
// The server sets Access-Control-Allow-Origin: * and handles OPTIONS
// preflight, so cross-origin browser fetch works.  If your setup blocks
// CORS, set proxyUrl to a same-origin proxy.
//
// Audio is encoded as 16-bit PCM WAV (16 kHz mono) before sending.
// The server expects a complete utterance — this provider is called
// once per VAD speech segment, not streaming.

class WhisperCppProvider extends SpeechRecognitionProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        // config: { baseUrl, inferencePath, responseFormat, temperature, language, proxyUrl, timeoutMs }
    }

    async init() {
        if (!this.config.baseUrl) {
            throw new Error('WhisperCppProvider: config.baseUrl is required (e.g. "http://127.0.0.1:8080")');
        }
        this._log('OK', `whisper.cpp server: ${this.config.baseUrl}${this.config.inferencePath || '/inference'}`);
    }

    async transcribe(audio) {
        const sampleRate = this.config.sampleRate ?? 16000;
        const wavBlob = encodeWAV(audio, sampleRate);

        const formData = new FormData();
        formData.append('file', wavBlob, 'audio.wav');
        formData.append('response_format', this.config.responseFormat || 'json');
        if (this.config.temperature !== undefined) {
            formData.append('temperature', String(this.config.temperature));
        }
        if (this.config.language) {
            formData.append('language', this.config.language);
        }

        const base = this.config.proxyUrl || this.config.baseUrl;
        const path = this.config.inferencePath || '/inference';
        const url = base.replace(/\/+$/, '') + path;

        const controller = new AbortController();
        const timeoutMs = this.config.timeoutMs ?? 30000;
        const timeout = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const res = await fetch(url, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            if (!res.ok) {
                const body = await res.text().catch(() => '');
                throw new Error(`whisper.cpp server returned ${res.status}: ${body.slice(0, 200)}`);
            }
            const data = await res.json();
            return data.text || '';
        } finally {
            clearTimeout(timeout);
        }
    }

    async destroy() {
        this._log('INFO', 'Destroyed (no resources to release).');
    }
}


// ==================================================================
// SECTION 12: WEB SPEECH PROVIDER (wake-gated, own mic)
// ==================================================================
//
// Uses the browser's built-in SpeechRecognition API.  This provider is
// SESSION-BASED: it only starts listening after a wake event, and stops
// after the first result (or a timeout).  This preserves privacy —
// there is no always-on cloud transcription.
//
// PRIVACY:
//   The Web Speech API opens its own microphone and (in Chrome) sends
//   audio to Google's servers during a session.  By gating sessions
//   behind a wake event, we ensure the cloud is only listening AFTER
//   the user has deliberately triggered recognition (via a local wake
//   word detector like openWakeWord, or a manual activateWakeWord()
//   button press).  This is the same privacy contract as pressing a
//   "push to talk" button.
//
// LIMITATIONS:
//   - Cannot transcribe pre-recorded clips (browser API limitation).
//   - Opens a SECOND microphone during each session (unavoidable).
//   - Chrome: online (cloud).  Safari: on-device.  Firefox: not supported.
//   - continuous=false ensures one utterance per session (privacy default).
//
// Config:
//   lang            BCP-47 language tag (default 'en-US')
//   continuous      If true, keeps listening until stopSession() or timeout.
//                   DEFAULT false (one utterance per wake — privacy).
//   interimResults  If true, emits partial results. Default false.
//   maxAlternatives Max result alternatives. Default 1.
//   maxSessionMs    Safety timeout — force-stop after this many ms.
//                   Default 10000 (10s). Prevents runaway sessions.

class WebSpeechProvider extends SpeechRecognitionProvider {
    constructor(config = {}, debug = false) {
        super(config, debug);
        this._SRClass = null;
        this._recognition = null;
        this._sessionTimer = null;
        this._active = false;
    }

    get isSessionBased() { return true; }

    async init() {
        const SR = (typeof window !== 'undefined') &&
            (window.SpeechRecognition || window.webkitSpeechRecognition);
        if (!SR) {
            throw new Error(
                'WebSpeechProvider: SpeechRecognition API not available in this browser. ' +
                'Try Chrome or Safari, or use a different speechRecognitionProvider.'
            );
        }
        this._SRClass = SR;
        this._log('OK', 'Web Speech API available.');
    }

    /**
     * Start a recognition session.  Called by the orchestrator after a
     * wake event.  Opens the browser's SpeechRecognition mic.
     */
    startSession() {
        if (this._active) return; // already in a session

        const rec = new this._SRClass();
        rec.lang = this.config.lang || 'en-US';
        rec.continuous = this.config.continuous ?? false;
        rec.interimResults = this.config.interimResults ?? false;
        rec.maxAlternatives = this.config.maxAlternatives ?? 1;

        rec.onresult = (event) => {
            // Get the last result in the results list.
            const last = event.results[event.results.length - 1];
            if (last.isFinal) {
                const text = last[0].transcript;
                this._emitter.emit('result', text);
            }
        };

        rec.onerror = (e) => {
            this._emitter.emit('error', e.error || 'unknown');
        };

        rec.onend = () => {
            this._cleanup();
            this._emitter.emit('end');
        };

        this._recognition = rec;
        this._active = true;

        // Safety timeout: force-stop after maxSessionMs to prevent runaway
        // sessions (e.g. if continuous=true and the user walks away).
        const maxMs = this.config.maxSessionMs ?? 10000;
        this._sessionTimer = setTimeout(() => {
            if (this._active) {
                this._log('WARN', `Session timed out after ${maxMs}ms.`);
                this.stopSession();
            }
        }, maxMs);

        try {
            rec.start();
            this._log('INFO', 'Recognition session started.');
        } catch (e) {
            this._cleanup();
            this._emitter.emit('error', e.message || 'Failed to start recognition');
            this._emitter.emit('end');
        }
    }

    stopSession() {
        if (this._recognition) {
            try { this._recognition.stop(); } catch (e) { /* ignore */ }
            // onend will fire and call _cleanup().
        }
    }

    _cleanup() {
        this._recognition = null;
        this._active = false;
        if (this._sessionTimer) {
            clearTimeout(this._sessionTimer);
            this._sessionTimer = null;
        }
    }

    get isActive() { return this._active; }

    async destroy() {
        this.stopSession();
        this._log('INFO', 'Destroyed.');
    }
}


// ==================================================================
// SECTION 13: AKARINETVOICE ORCHESTRATOR
// ==================================================================
//
// Backwards-compatible with v4.0.  New config keys are additive.
//
// The orchestrator owns:
//   - AudioBus (one mic) — if unifiedMic is true and any bus consumer is active
//   - BusVAD (Silero) — for segment-based ASR (transformers/whispercpp)
//   - XlCache (optional) — large RAM ring
//   - Wake word provider (pluggable)
//   - Speech recognition provider (pluggable, NEW in v4.1)
//   - Wake ↔ speech correlation logic (preserved from v4.0)
//   - Text wake word parsing (_parse, preserved from v4.0)
//
// The orchestrator does NOT own:
//   - TeachableMachineProvider's mic (it manages its own)
//   - WebSpeechProvider's mic (it manages its own, per session)

export class AkarinetVoice extends EventTarget {
    constructor(config = {}) {
        super();

        // ── ASR config (v4.0 keys, unchanged) ──
        this.config = {
            modelId: config.modelId || 'onnx-community/moonshine-tiny-ONNX',
            modelQuantization: config.modelQuantization || 'q8',

            // ── Text wake words (v4.0, unchanged) ──
            wakewords: config.wakewords || (config.wakeword ? [config.wakeword] : []),

            // ── Wake word provider (v4.0, unchanged) ──
            wakeWordProvider: config.wakeWordProvider || (config.wakesoundURL ? 'teachablemachine' : 'none'),

            // v3.4/v4.0 wake-sound config (mapped to TeachableMachineProvider)
            wakesoundURL: config.wakesoundURL || null,
            wakesoundThreshold: config.wakesoundThreshold ?? 0.75,
            wakesoundIndex: config.wakesoundIndex ?? 2,
            wakesoundDuration: config.wakesoundDuration ?? 100,
            wakesoundDelay: config.wakesoundDelay ?? 0,

            // openWakeWord config (v4.0, unchanged)
            openWakeWord: config.openWakeWord || null,

            // ── VAD + parse (v4.0 keys, unchanged) ──
            vadThreshold: config.vadThreshold ?? 0.5,
            vadRedemptionMs: config.vadRedemptionMs ?? 480,
            cleanup: config.cleanup !== undefined ? config.cleanup : true,
            debugWakeSound: config.debugWakeSound || false,
            requireWakeSound: config.requireWakeSound || false,

            // ── NEW in v4.1: Speech recognition provider ──
            speechRecognitionProvider: config.speechRecognitionProvider || 'transformers',
            whisperCpp: config.whisperCpp || null,
            webSpeech: config.webSpeech || null,

            // ── NEW in v4.1: Unified audio bus + caches ──
            unifiedMic: config.unifiedMic !== undefined ? config.unifiedMic : true,
            liveCacheMs: config.liveCacheMs ?? 2000,
            xlCache: config.xlCache || null
        };

        // Orchestrator state (same shape as v4.0)
        this._asrWorker = null;      // kept for v4.0 compat (TransformersProvider uses its own worker)
        this._asrCallbacks = new Map();
        this._asrCallId = 0;
        this.vad = null;             // v4.0 compat reference (null in v4.1 — BusVAD replaces it)
        this.wakeWordProvider = null;
        this.srProvider = null;      // NEW: speech recognition provider
        this.wakeSoundDetectedTime = null;
        this.lastWakeSoundScore = 0;
        this.speechStartTime = 0;
        this._isProcessing = false;
        this._asrSessionActive = false; // NEW: tracks webspeech session state
        this._logInitialized = false;

        // NEW in v4.1: bus + VAD + cache
        this.bus = null;
        this.busVad = null;
        this.xlCache = null;
    }

    async init() {
        try {
            this._log('INFO', 'Welcome to AkariNet Audio Console v4.1!');
            this._log('INFO', `Wake word provider: ${this.config.wakeWordProvider}`);
            this._log('INFO', `Speech recognition provider: ${this.config.speechRecognitionProvider}`);

            if (!this.config.debugWakeSound) {
                this._suppressLibraryLogs();
            }

            // ── Determine what needs the bus ──
            const srProv = this.config.speechRecognitionProvider;
            const wwProv = this.config.wakeWordProvider;
            const needsOrt = (wwProv === 'openwakeword') ||
                (this.config.unifiedMic && (srProv === 'transformers' || srProv === 'whispercpp'));
            const needsBus = this.config.unifiedMic && (
                srProv === 'transformers' ||
                srProv === 'whispercpp' ||
                wwProv === 'openwakeword' ||
                (this.config.xlCache && this.config.xlCache.enabled)
            );
            const needsVad = needsBus && (srProv === 'transformers' || srProv === 'whispercpp');

            // ── 1. Load onnxruntime-web (needed for OWW and/or BusVAD) ──
            if (needsOrt) {
                this._log('INFO', 'Loading onnxruntime-web...');
                await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js');
            }

            // ── 2. Create AudioBus (don't start yet — wire subscribers first) ──
            if (needsBus) {
                this.bus = new AudioBus({
                    sampleRate: 16000,
                    chunkSize: 1280,
                    liveCacheMs: this.config.liveCacheMs
                }, this.config.debugWakeSound);
            }

            // ── 3. Init BusVAD (loads Silero model) ──
            if (needsVad) {
                this._log('INFO', 'Initializing BusVAD (Silero)...');
                const vadThreshold = normalizeVadThreshold(this.config.vadThreshold);
                this.busVad = new BusVAD({
                    ort: window.ort,
                    threshold: vadThreshold,
                    redemptionMs: this.config.vadRedemptionMs,
                    sampleRate: 16000,
                    chunkSize: 1280
                }, this.config.debugWakeSound);
                await this.busVad.init();

                this.busVad.on('speech-start', () => {
                    this.speechStartTime = Date.now();
                    this.dispatchEvent(new Event('speechstart'));
                });
                this.busVad.on('misfire', () => {
                    this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: '(VAD Misfire)' }));
                });
                this.busVad.on('speech-end', ({ audio }) => {
                    this.dispatchEvent(new Event('speechend'));
                    this._handleSpeech(audio);
                });
                this.busVad.on('error', (err) => {
                    this._log('ERROR', `BusVAD error: ${err.message || err}`);
                });
            }

            // ── 4. Init XlCache (if configured) ──
            if (this.config.xlCache && this.config.xlCache.enabled) {
                this.xlCache = new XlCache({
                    durationMs: this.config.xlCache.durationMs ?? 60000,
                    vadOnly: this.config.xlCache.vadOnly ?? false,
                    sampleRate: 16000
                }, this.config.debugWakeSound);
                this._log('OK', `XL cache enabled: ${this.config.xlCache.durationMs ?? 60000}ms (vadOnly=${this.config.xlCache.vadOnly ?? false})`);
            }

            // ── 5. Init wake word provider ──
            this.wakeWordProvider = this._createWakeProvider();
            if (this.wakeWordProvider) {
                await this.wakeWordProvider.init();
                this.wakeWordProvider.on('detect', (detail) => this._onWakeDetect(detail));
                await this.wakeWordProvider.startListening();
                this._log('OK', `Wake word provider "${this.config.wakeWordProvider}" listening.`);
            } else {
                this._log('INFO', 'No wake word provider — manual activateWakeWord() only.');
            }

            // ── 6. Init speech recognition provider ──
            this.srProvider = this._createSpeechProvider();
            if (this.srProvider) {
                await this.srProvider.init();
                if (this.srProvider.isSessionBased) {
                    // Session-based (webspeech): wire result/end/error.
                    this.srProvider.on('result', (text) => this._onAsrResult(text));
                    this.srProvider.on('end', () => this._onAsrSessionEnd());
                    this.srProvider.on('error', (err) => {
                        this._log('ERROR', `ASR session error: ${err}`);
                        this.dispatchEvent(new CustomEvent('error', { detail: 'ASR: ' + err }));
                    });
                }
                // Segment-based providers don't emit events — orchestrator calls transcribe().
                this._log('OK', `Speech recognition provider "${this.config.speechRecognitionProvider}" ready.`);
            }

            // ── 7. Wire bus subscribers + start bus ──
            if (this.bus) {
                if (this.busVad) {
                    this.bus.addSubscriber(chunk => this.busVad.feedChunk(chunk));
                }
                if (this.xlCache) {
                    this.bus.addSubscriber(chunk => this.xlCache.push(chunk, this.busVad?.isSpeechActive));
                }
                if (this.wakeWordProvider && typeof this.wakeWordProvider.feedChunk === 'function') {
                    this.bus.addSubscriber(chunk => this.wakeWordProvider.feedChunk(chunk));
                }
                this._log('INFO', 'Starting AudioBus (mic permission may be requested)...');
                await this.bus.start();
                this._log('OK', 'AudioBus started.');
            }

            // ── 8. Ready ──
            this._log('OK', 'AkariNet Audio Console ready.');
            this._log('INFO', `Wake words: ${this.config.wakewords.join(', ') || '(none — text fallback disabled)'}`);
            this._log('INFO', `Wake sound: ${this.wakeWordProvider ? `enabled (${this.config.wakeWordProvider})` : 'disabled'}`);
            this._log('INFO', `ASR: ${this.srProvider ? this.config.speechRecognitionProvider : 'disabled'}`);
            if (this.xlCache) this._log('INFO', `XL cache: ${this.xlCache.availableMs === 0 ? 'filling' : 'available'}`);

            this.dispatchEvent(new Event('ready'));
        } catch (e) {
            this._log('ERROR', `Initialization failed: ${e.message}`);
            this.dispatchEvent(new CustomEvent('error', { detail: e.message }));
        }
    }

    // ----------------------------------------------------------------
    // Wake word provider factory (preserved from v4.0, unchanged logic)
    // ----------------------------------------------------------------

    _createWakeProvider() {
        switch (this.config.wakeWordProvider) {
            case 'teachablemachine':
                if (!this.config.wakesoundURL) {
                    this._log('WARN', 'wakeWordProvider=teachablemachine but no wakesoundURL — skipping.');
                    return null;
                }
                return new TeachableMachineProvider({
                    url: this.config.wakesoundURL,
                    threshold: this.config.wakesoundThreshold,
                    index: this.config.wakesoundIndex,
                    probabilityThreshold: 0.75,
                    invokeCallbackOnNoiseAndUnknown: true,
                    overlapFactor: 0.65
                }, this.config.debugWakeSound);

            case 'openwakeword': {
                const oww = this.config.openWakeWord || {};
                const hasKeywordSource = oww.keywordURL
                    || (Array.isArray(oww.keywordURLs) && oww.keywordURLs.length > 0)
                    || oww.baseAssetUrl;
                if (!hasKeywordSource) {
                    throw new Error(
                        "wakeWordProvider='openwakeword' requires one of: " +
                        "openWakeWord.keywordURL, openWakeWord.keywordURLs, or openWakeWord.baseAssetUrl"
                    );
                }
                return new OpenWakeWordProvider({
                    keywordURL: oww.keywordURL,
                    keywordName: oww.keywordName,
                    keywordURLs: oww.keywordURLs,
                    baseAssetUrl: oww.baseAssetUrl,
                    keywords: oww.keywords,
                    modelFiles: oww.modelFiles,
                    coreModelURLs: oww.coreModelURLs,
                    ort: oww.ort,
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
    // Speech recognition provider factory (NEW in v4.1)
    // ----------------------------------------------------------------

    _createSpeechProvider() {
        switch (this.config.speechRecognitionProvider) {
            case 'transformers':
                return new TransformersProvider({
                    modelId: this.config.modelId,
                    modelQuantization: this.config.modelQuantization
                }, this.config.debugWakeSound);

            case 'whispercpp': {
                const wc = this.config.whisperCpp || {};
                if (!wc.baseUrl) {
                    throw new Error(
                        "speechRecognitionProvider='whispercpp' requires whisperCpp.baseUrl " +
                        "(e.g. 'http://127.0.0.1:8080')"
                    );
                }
                return new WhisperCppProvider({
                    baseUrl: wc.baseUrl,
                    inferencePath: wc.inferencePath ?? '/inference',
                    responseFormat: wc.responseFormat ?? 'json',
                    temperature: wc.temperature ?? 0.0,
                    language: wc.language ?? 'en',
                    proxyUrl: wc.proxyUrl ?? null,
                    timeoutMs: wc.timeoutMs ?? 30000,
                    sampleRate: 16000
                }, this.config.debugWakeSound);
            }

            case 'webspeech': {
                const ws = this.config.webSpeech || {};
                return new WebSpeechProvider({
                    lang: ws.lang ?? 'en-US',
                    continuous: ws.continuous ?? false,
                    interimResults: ws.interimResults ?? false,
                    maxAlternatives: ws.maxAlternatives ?? 1,
                    maxSessionMs: ws.maxSessionMs ?? 10000
                }, this.config.debugWakeSound);
            }

            case 'none':
            case null:
            case undefined:
                return null;

            default:
                throw new Error(`Unknown speechRecognitionProvider: "${this.config.speechRecognitionProvider}"`);
        }
    }

    // ----------------------------------------------------------------
    // Wake event handling (unified for manual + provider detections)
    // ----------------------------------------------------------------

    /**
     * Called when ANY wake event fires — from a wake word provider's
     * 'detect' event, or from activateWakeWord() (manual trigger).
     *
     * For segment-based ASR: records the wake time for _handleSpeech
     * correlation (same as v4.0).
     *
     * For session-based ASR (webspeech): starts a recognition session.
     * The session opens its own mic, transcribes one utterance, then
     * closes.  Privacy: no listening happens without this wake event.
     */
    _onWakeDetect(detail) {
        // Ignore wake events during an active webspeech session (prevents double-trigger).
        if (this._asrSessionActive) {
            if (this.config.debugWakeSound) this._log('DEBUG', 'Wake ignored (ASR session active).');
            return;
        }

        this.wakeSoundDetectedTime = detail.timestamp;
        this.lastWakeSoundScore = detail.score;
        this.dispatchEvent(new CustomEvent('wakesound', {
            detail: {
                score: detail.score,
                class: detail.class,
                timestamp: detail.timestamp
            }
        }));

        if (this.srProvider && this.srProvider.isSessionBased) {
            this._asrSessionActive = true;
            this.srProvider.startSession();
        }
    }

    // ----------------------------------------------------------------
    // ASR session callbacks (for session-based providers like webspeech)
    // ----------------------------------------------------------------

    _onAsrResult(text) {
        this._isProcessing = true;
        this.dispatchEvent(new Event('processing'));
        try {
            // Wake was already detected (that's why the session started),
            // so wakeSoundDetected=true.  _parse will still strip text wake
            // words from the transcript if present.
            this._parse(text, true);
        } catch (e) {
            this._log('ERROR', `Parse failed: ${e.message}`);
            this.dispatchEvent(new CustomEvent('error', { detail: e.message }));
        } finally {
            this._isProcessing = false;
            this.dispatchEvent(new Event('processingend'));
        }
    }

    _onAsrSessionEnd() {
        this._asrSessionActive = false;
        // Reset wake state for the next session.
        this.wakeSoundDetectedTime = null;
    }

    // ----------------------------------------------------------------
    // Speech handling (segment-based ASR — preserved from v4.0)
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

        const wakeSoundAtSpeechStart = this.wakeSoundDetectedTime;

        // Wait briefly for a late wake detection (wake provider may process
        // chunks slightly after VAD's speech-end due to async inference).
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

    async _transcribeAndParse(audio, wakeSoundDetected) {
        this._isProcessing = true;
        this.dispatchEvent(new Event('processing'));
        try {
            const text = await this.srProvider.transcribe(audio);
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

    // ----------------------------------------------------------------
    // Text wake word parsing (preserved from v4.0, unchanged logic)
    // ----------------------------------------------------------------

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
            const detail = this.config.debugWakeSound ? raw : '(No Wake Word Detected)';
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

    // ----------------------------------------------------------------
    // Manual wake trigger (v4.0 compatible + webspeech session start)
    // ----------------------------------------------------------------

    /**
     * Manually trigger a wake word session — call this from a button click
     * or any other UI event.  Behaves identically to a real wake sound
     * detection: the next utterance will be transcribed and parsed.
     *
     * For segment-based ASR (transformers/whispercpp): the next VAD
     * speech segment will be transcribed (same as v4.0).
     *
     * For session-based ASR (webspeech): a recognition session starts
     * immediately (opens the browser SpeechRecognition mic).
     */
    activateWakeWord() {
        this._onWakeDetect({
            score: 1,
            class: 'manual',
            timestamp: Date.now()
        });
    }

    // ----------------------------------------------------------------
    // XL Cache retrieval (NEW in v4.1)
    // ----------------------------------------------------------------

    /**
     * Retrieve audio from the XL cache's recent past.
     * "Start recording back in time" — pulls from the RAM ring buffer
     * that has been continuously filling since init().
     *
     * @param {Object} opts
     * @param {number} opts.fromMsAgo  Start of range (ms ago, larger = further back).
     * @param {number} [opts.toMsAgo]  End of range (ms ago, 0 = now). Default 0.
     * @param {string} [opts.format]   'float32' (default) or 'wav'.
     * @returns {{ audio: Float32Array|Blob, sampleRate: number, availableMs: number, format: string }}
     */
    retrieveCache({ fromMsAgo, toMsAgo = 0, format = 'float32' } = {}) {
        if (!this.xlCache) {
            throw new Error('retrieveCache: XL cache is not enabled. Set xlCache.enabled=true in config before init().');
        }
        const result = this.xlCache.retrieve({ fromMsAgo, toMsAgo });
        const sampleRate = this.bus ? this.bus.sampleRate : 16000;
        if (format === 'wav') {
            return {
                audio: encodeWAV(result.audio, sampleRate),
                sampleRate,
                availableMs: result.availableMs,
                format: 'wav'
            };
        }
        return {
            audio: result.audio,
            sampleRate,
            availableMs: result.availableMs,
            format: 'float32'
        };
    }

    /** Clear the XL cache (audio is discarded, ring resets). */
    clearXlCache() {
        if (this.xlCache) {
            this.xlCache.clear();
            this.dispatchEvent(new Event('xlcachepurge'));
        }
    }

    /** Returns current mic/bus state for UI display. */
    getMicState() {
        return {
            active: !!(this.bus && this.bus.isActive),
            sampleRate: this.bus ? this.bus.sampleRate : null,
            xlCacheMs: this.xlCache ? this.xlCache.availableMs : 0,
            asrSessionActive: this._asrSessionActive
        };
    }

    // ----------------------------------------------------------------
    // Shutdown (preserved from v4.0 + new component cleanup)
    // ----------------------------------------------------------------

    async destroy() {
        this._log('INFO', 'Shutting down AkariNet Audio Console...');

        if (this.wakeWordProvider) {
            try { await this.wakeWordProvider.destroy(); } catch (e) { /* ignore */ }
            this.wakeWordProvider = null;
        }
        if (this.srProvider) {
            try { await this.srProvider.destroy(); } catch (e) { /* ignore */ }
            this.srProvider = null;
        }
        if (this.busVad) {
            try { await this.busVad.destroy(); } catch (e) { /* ignore */ }
            this.busVad = null;
        }
        if (this.bus) {
            try { await this.bus.stop(); } catch (e) { /* ignore */ }
            this.bus = null;
        }
        this.xlCache = null;

        this._restoreConsoleLogs();
        this._log('OK', 'Shutdown complete.');
    }

    // ----------------------------------------------------------------
    // Logging (preserved from v4.0)
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
// SECTION 14: EXPORTS
// ==================================================================
//
// Default export: AkarinetVoice (drop-in replacement for v4.0)
// Named exports: all provider classes + bus/VAD/cache for advanced users.

export {
    WakeWordProvider,
    TeachableMachineProvider,
    OpenWakeWordProvider,
    SpeechRecognitionProvider,
    TransformersProvider,
    WhisperCppProvider,
    WebSpeechProvider,
    AudioBus,
    BusVAD,
    XlCache,
    RingBuffer
};

export default AkarinetVoice;
