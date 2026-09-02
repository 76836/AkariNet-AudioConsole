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
