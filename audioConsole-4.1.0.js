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
        this._redemptionFrames = Math.ceil(this.config.redemptionMs / (this.config.chunkSize / 