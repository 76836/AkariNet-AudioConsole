/**
 * AKARINET AUDIO CONSOLE v4.1.1
 * ====================================================================
 * Drop-in compatible with v4.1.0. Fixes Firefox/Linux sample-rate error:
 *   AudioContext.createMediaStreamSource: Connecting AudioNodes from
 *   AudioContexts with different sample-rate is currently not supported.
 *
 * Approach: re-export v4.1.0, then patch AudioBus.prototype.start so the
 * AudioContext uses the hardware rate and the worklet resamples to 16 kHz.
 */

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
} from './audioConsole-4.1.0.js';

export { default, AkarinetVoice } from './audioConsole-4.1.0.js';

import { AudioBus } from './audioConsole-4.1.0.js';

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
