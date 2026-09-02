/**
 * AKARINET AUDIO CONSOLE v4.1.1
 * ====================================================================
 * Drop-in compatible with v4.1.0. Fixes Firefox/Linux sample-rate error:
 *   AudioContext.createMediaStreamSource: Connecting AudioNodes from
 *   AudioContexts with different sample-rate is currently not supported.
 *
 * Approach: re-export v4.1.0, then patch AudioBus.prototype.start so the
 * AudioContext uses the hardware rate and the worklet resamples to 16 kHz.
 *
 * Also pins the 4.1.0 core to a known-good commit via jsDelivr and adds
 * versatile whisper.cpp response parsing (plain text, verbose_json, nested
 * keys, segments, OpenAI-style, forks).
 */

const _AC410 = 'https://cdn.jsdelivr.net/gh/76836/AkariNet-AudioConsole@fb8936347264c4e15154d0dd4d358b7a4d350199/audioConsole-4.1.0.js';

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
} from 'https://cdn.jsdelivr.net/gh/76836/AkariNet-AudioConsole@fb8936347264c4e15154d0dd4d358b7a4d350199/audioConsole-4.1.0.js';

export { default, AkarinetVoice } from 'https://cdn.jsdelivr.net/gh/76836/AkariNet-AudioConsole@fb8936347264c4e15154d0dd4d358b7a4d350199/audioConsole-4.1.0.js';

import { AudioBus, WhisperCppProvider } from 'https://cdn.jsdelivr.net/gh/76836/AkariNet-AudioConsole@fb8936347264c4e15154d0dd4d358b7a4d350199/audioConsole-4.1.0.js';

/** Worklet that emits fixed 1280-sample @ targetRate chunks, resampling if needed. */
const FIXED_BUS_WORKLET_CODE = `
class AudioBusProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const opts = (options && options.processorOptions) || {};
        this.targetRate = opts.targetSampleRate || 16000;
        this.inRate = sampleRate;
        this.step = this.inRate / this.targetRate;
        this.bufferSize = 1280;
        this._out = new Float32Array(this.bufferSize);
        this._outPos = 0;
        this._phase = 0;
        this._buf = [];
    }
    process(inputs) {
        const input = inputs[0] && inputs[0][0];
        if (!input) return true;

        if (Math.abs(this.step - 1) < 0.001) {
            for (let i = 0; i < input.length; i++) {
                this._out[this._outPos++] = input[i];
                if (this._outPos === this.bufferSize) {
                    this.port.postMessage(this._out.slice(0));
                    this._outPos = 0;
                }
            }
            return true;
        }

        for (let i = 0; i < input.length; i++) this._buf.push(input[i]);

        while (this._phase + this.step < this._buf.length - 1) {
            const idx = this._phase;
            const i0 = Math.floor(idx);
            const frac = idx - i0;
            const s0 = this._buf[i0];
            const s1 = this._buf[i0 + 1];
            const sample = s0 + (s1 - s0) * frac;
            this._out[this._outPos++] = sample;
            if (this._outPos === this.bufferSize) {
                this.port.postMessage(this._out.slice(0));
                this._outPos = 0;
            }
            this._phase += this.step;
        }

        const drop = Math.floor(this._phase);
        if (drop > 0) {
            this._buf.splice(0, drop);
            this._phase -= drop;
        }
        if (this._buf.length > this.inRate) {
            const excess = this._buf.length - Math.ceil(this.inRate / 10);
            this._buf.splice(0, excess);
            this._phase = Math.max(0, this._phase - excess);
        }
        return true;
    }
}
registerProcessor('audio-bus-processor', AudioBusProcessor);
`;

/**
 * Firefox-safe AudioBus.start: never force sampleRate on the AudioContext
 * (createMediaStreamSource fails when rates differ). Resample in the worklet.
 */
AudioBus.prototype.start = async function start() {
    if (this._active) return;

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

    // Default (hardware) rate — required for Firefox MediaStreamSource connect.
    // Chrome still works; worklet resamples to config.sampleRate (16 kHz).
    this._audioContext = new AudioContext();

    if (this._audioContext.state === 'suspended') {
        try { await this._audioContext.resume(); } catch (e) { /* retry later */ }
    }

    const source = this._audioContext.createMediaStreamSource(this._mediaStream);
    this._sourceNode = source;
    this._gainNode = this._audioContext.createGain();
    this._gainNode.gain.value = this.config.gain;

    const blob = new Blob([FIXED_BUS_WORKLET_CODE], { type: 'application/javascript' });
    const workletURL = URL.createObjectURL(blob);
    await this._audioContext.audioWorklet.addModule(workletURL);
    URL.revokeObjectURL(workletURL);

    this._workletNode = new AudioWorkletNode(this._audioContext, 'audio-bus-processor', {
        processorOptions: { targetSampleRate: this.config.sampleRate || 16000 }
    });

    this._workletNode.port.onmessage = (event) => {
        const chunk = event.data;
        if (!chunk) return;
        this.liveRing.push(chunk);
        for (const fn of this._subscribers) {
            try { fn(chunk); } catch (err) { console.error('[AudioBus] subscriber error:', err); }
        }
    };

    source.connect(this._gainNode);
    this._gainNode.connect(this._workletNode);
    const muteGain = this._audioContext.createGain();
    muteGain.gain.value = 0;
    this._workletNode.connect(muteGain);
    muteGain.connect(this._audioContext.destination);

    this._active = true;
    if (this.debug) {
        console.log(
            '[AudioBus] Started (v4.1.1 Firefox-safe). Context rate:',
            this._audioContext.sampleRate,
            '→ target',
            this.config.sampleRate
        );
    }
};

// Downstream (VAD / OWW / cache) always sees the target (16 kHz) rate.
Object.defineProperty(AudioBus.prototype, 'sampleRate', {
    get: function () {
        return this.config?.sampleRate ?? 16000;
    },
    configurable: true
});

// ---------------------------------------------------------------------------
// Robust whisper.cpp response parsing (plain text, verbose_json, nested keys,
// segments, OpenAI-style, forks). Overrides the strict data.text parser.
// ---------------------------------------------------------------------------
function _ac411_encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
}

(function patchWhisperCppParser() {
    if (typeof WhisperCppProvider === 'undefined' || !WhisperCppProvider.prototype) return;

    WhisperCppProvider.prototype.transcribe = async function (audio) {
        const sampleRate = this.config.sampleRate ?? 16000;
        const wavBlob = _ac411_encodeWAV(audio, sampleRate);

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
            const bodyText = await res.text().catch(() => '');
            if (!res.ok) {
                throw new Error(`whisper.cpp server returned ${res.status}: ${bodyText.slice(0, 200)}`);
            }
            return this._extractTranscript(bodyText);
        } finally {
            clearTimeout(timeout);
        }
    };

    WhisperCppProvider.prototype._extractTranscript = function (bodyText) {
        if (bodyText == null) return '';
        const raw = String(bodyText).trim();
        if (!raw) return '';

        const looksJson = raw.startsWith('{') || raw.startsWith('[');
        if (!looksJson) return this._cleanPlainText(raw);

        let data;
        try {
            data = JSON.parse(raw);
        } catch (e) {
            this._log('WARN', `whisper.cpp JSON parse failed, using raw text: ${e.message}`);
            return this._cleanPlainText(raw);
        }

        const text = this._digForText(data);
        if (text) return text;

        this._log('WARN', 'whisper.cpp response had no recognizable text field; using cleaned body');
        return this._cleanPlainText(raw);
    };

    WhisperCppProvider.prototype._digForText = function (data) {
        if (data == null) return '';
        if (typeof data === 'string') return data.trim();

        const KEYS = [
            'text', 'transcription', 'transcript', 'result', 'output',
            'message', 'content', 'transcription_text', 'asr'
        ];
        for (const k of KEYS) {
            if (typeof data[k] === 'string' && data[k].trim()) return data[k].trim();
        }

        for (const nest of ['data', 'response', 'result', 'output', 'body']) {
            if (data[nest] && typeof data[nest] === 'object') {
                const nested = this._digForText(data[nest]);
                if (nested) return nested;
            }
        }

        if (Array.isArray(data.segments) && data.segments.length) {
            const parts = data.segments
                .map(s => (s && (s.text || s.transcript || s.transcription)) || '')
                .map(t => String(t).trim())
                .filter(Boolean);
            if (parts.length) return parts.join(' ').replace(/\s+/g, ' ').trim();
        }

        if (Array.isArray(data)) {
            const parts = data.map(item => this._digForText(item)).filter(Boolean);
            if (parts.length) return parts.join(' ').replace(/\s+/g, ' ').trim();
        }
        if (Array.isArray(data.results)) {
            const parts = data.results.map(item => this._digForText(item)).filter(Boolean);
            if (parts.length) return parts.join(' ').replace(/\s+/g, ' ').trim();
        }
        if (Array.isArray(data.alternatives)) {
            let best = '';
            let bestConf = -1;
            for (const alt of data.alternatives) {
                const t = this._digForText(alt);
                const conf = typeof alt.confidence === 'number' ? alt.confidence
                    : (typeof alt.score === 'number' ? alt.score : 0);
                if (t && conf >= bestConf) { best = t; bestConf = conf; }
            }
            if (best) return best;
        }
        return '';
    };

    WhisperCppProvider.prototype._cleanPlainText = function (raw) {
        let t = String(raw).trim();
        if (/^\d+\s*\n\d{2}:\d{2}/.test(t) || /-->/.test(t)) {
            t = t.split(/\r?\n/)
                .filter(line => {
                    const s = line.trim();
                    if (!s) return false;
                    if (/^\d+$/.test(s)) return false;
                    if (/\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->/.test(s)) return false;
                    if (/^WEBVTT/i.test(s)) return false;
                    return true;
                })
                .join(' ');
        }
        return t.replace(/\s+/g, ' ').trim();
    };
})();
