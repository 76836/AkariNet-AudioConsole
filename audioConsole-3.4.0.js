/**
 * AKARINET AUDIO CONSOLE v3.4
 * A hybrid voice recognition engine that supports:
 * - Multiple text wake words
 * - Optional TensorFlow.js Wake Sound detection (continuous listening, correlated with VAD)
 * - ONNX Runtime ASR model for transcription
 * - Voice Activity Detection to save energy and processing time
 */
export class AkarinetVoice extends EventTarget {
    constructor(config = {}) {
        super();
        this.config = {
            modelId: config.modelId || "onnx-community/moonshine-tiny-ONNX",
            modelQuantization: config.modelQuantization || "q8",
            wakewords: config.wakewords || (config.wakeword ? [config.wakeword] : []),
            wakesoundURL: config.wakesoundURL || null,
            wakesoundThreshold: config.wakesoundThreshold || 0.75,
            wakesoundIndex: config.wakesoundIndex || 2,
            wakesoundDuration: config.wakesoundDuration || 100,
            wakesoundDelay: config.wakesoundDelay || 0,
            vadThreshold: config.vadThreshold || 0.75,
            cleanup: config.cleanup !== undefined ? config.cleanup : true,
            debugWakeSound: config.debugWakeSound || false,
            requireWakeSound: config.requireWakeSound || false
        };
        
        this._asrWorker = null;
        this._asrCallbacks = new Map();
        this._asrCallId = 0;
        this.vad = null;
        this.recognizer = null;
        this.wakeSoundDetectedTime = null;
        this.lastWakeSoundScore = 0;
        this.speechStartTime = 0;
        this._isProcessing = false;
        
        this._logInitialized = false;
    }

    async init() {
        try {
            this._log('INFO', 'Welcome to AkariNet Audio Console v3.4!');
            
            const modelCount = this.config.wakesoundURL ? 3 : 2;
            this._log('OK', `Loading ${modelCount} model${modelCount > 1 ? 's' : ''}...`);

            // Suppress library console output unless in debug mode
            if (!this.config.debugWakeSound) {
                this._suppressLibraryLogs();
            }

            await this._loadExternalScripts();

            if (this.config.wakesoundURL) {
                await this._loadTensorFlowScripts();
                await this._initWakeSoundModel();
                this._startContinuousWakeSoundListening();
            }

            this._log('INFO', `Loading ASR model: ${this.config.modelId} (${this.config.modelQuantization})...`);
            await this._initASRWorker();
            this._log('OK', 'ASR model loaded.');

            this._log('INFO', 'Initializing Voice Activity Detection...');
            this.vad = await vad.MicVAD.new({
                positiveSpeechThreshold: this.config.vadThreshold,
                redemptionFrames: 15,
                onSpeechStart: () => {
                    this.speechStartTime = Date.now();
                    this.dispatchEvent(new Event('speechstart'));
                },
                onVADMisfire: () => {
                    this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: "(VAD Misfire)" }));
                },
                onSpeechEnd: async (audio) => {
                    this.dispatchEvent(new Event('speechend'));
                    await this._handleSpeech(audio);
                }
            });

            await this.vad.start();
            
            this._log('OK', 'AkariNet Audio Console ready.');
            this._log('INFO', `Wake words: ${this.config.wakewords.join(', ')}`);
            this._log('INFO', `Wake sound detection: ${this.config.wakesoundURL ? 'enabled' : 'disabled'}`);
            
            this.dispatchEvent(new Event('ready'));

        } catch (e) {
            this._log('ERROR', `Initialization failed: ${e.message}`);
            this.dispatchEvent(new CustomEvent('error', { detail: e.message }));
        }
    }

    async _loadExternalScripts() {
        const loadScript = (src) => new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) return resolve();
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });

        await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js');
    }

    async _loadTensorFlowScripts() {
        const loadScript = (src) => new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) return resolve();
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });

        await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.4.0/dist/speech-commands.min.js');
    }

    async _initWakeSoundModel() {
        this._log('INFO', 'Loading wake sound model...');
        
        const checkpointURL = this.config.wakesoundURL + "model.json";
        const metadataURL = this.config.wakesoundURL + "metadata.json";

        this.recognizer = speechCommands.create(
            "BROWSER_FFT",
            undefined,
            checkpointURL,
            metadataURL
        );
        await this.recognizer.ensureModelLoaded();

        const classLabels = this.recognizer.wordLabels();
        const targetClass = classLabels[this.config.wakesoundIndex];
        
        this._log('OK', 'Wake sound model loaded.');
        this._log('INFO', `Monitoring class index ${this.config.wakesoundIndex}: "${targetClass}"`);
        this._log('INFO', `Wake sound threshold: ${this.config.wakesoundThreshold}`);
    }

    _startContinuousWakeSoundListening() {
        if (!this.recognizer) return;

        const classLabels = this.recognizer.wordLabels();

        this.recognizer.listen(result => {
            const scores = result.scores;
            const targetScore = scores[this.config.wakesoundIndex];

            if (this.config.debugWakeSound && targetScore > 0.5) {
                this._log('DEBUG', `Wake sound score: ${targetScore.toFixed(4)} (threshold: ${this.config.wakesoundThreshold})`);
            }

            if (targetScore > this.config.wakesoundThreshold && targetScore < 1.62) {
                this.wakeSoundDetectedTime = Date.now();
                this.lastWakeSoundScore = targetScore;

                if (this.config.debugWakeSound) {
                    this._log('OK', `Wake sound detected with score: ${targetScore.toFixed(4)}`);
                }

                this.dispatchEvent(new CustomEvent('wakesound', {
                    detail: {
                        score: targetScore,
                        class: classLabels[this.config.wakesoundIndex],
                        timestamp: this.wakeSoundDetectedTime
                    }
                }));
            }
        }, {
            includeSpectrogram: false,
            probabilityThreshold: 0.75,
            invokeCallbackOnNoiseAndUnknown: true,
            overlapFactor: 0.65
        });
    }

    async _handleSpeech(audio) {
        if (audio.length < 2000) {
            this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: "(Audio Too Short)" }));
            return;
        }

        if (this._isProcessing) {
            this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: "(Busy Processing)" }));
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

        // A late detection is one that arrived after speech started (caught by the wait loop).
        // For everything else use the pre-speech snapshot so spurious in-speech detections
        // don't invalidate the session.
        const lateDetection = (this.wakeSoundDetectedTime && this.wakeSoundDetectedTime > this.speechStartTime)
            ? this.wakeSoundDetectedTime
            : null;
        const relevantWakeTime = lateDetection || wakeSoundAtSpeechStart;

        let inSession = false;
        if (relevantWakeTime) {
            if (lateDetection) {
                // Fired after speech started — always counts as in-session
                inSession = true;
            } else {
                // Fired before speech — check it was within the session window
                const timeSinceWakeSound = this.speechStartTime - relevantWakeTime;
                inSession = timeSinceWakeSound >= 0 && timeSinceWakeSound < this.config.wakesoundDuration;
            }
        }

        if (inSession) {
            let audioToProcess = audio;

            if (lateDetection) {
                // Wake sound fired during the utterance — trim the leading overlap
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
            this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: "(Waiting for Wake Sound)" }));
        }
    }

    async _initASRWorker() {
        // Inline worker source — loaded as an ES module so it can import transformers.js.
        // Runs in its own thread, keeping WASM computation off the main thread entirely.
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

        // One-time listener just for the init handshake
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

        // Persistent listener for transcription responses — routes by call ID
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
            // Copy the Float32Array so we own the buffer before transferring.
            // Transferring (not copying) across postMessage means zero overhead
            // regardless of audio length.
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
                this.dispatchEvent(new CustomEvent('speechdiscarded', { detail: "(Empty Recognition)" }));
                return;
            }
            this._parse(text, wakeSoundDetected);
        } catch (e) {
            this._log('ERROR', `ASR transcription failed: ${e.message}`);
            this.dispatchEvent(new CustomEvent('error', { detail: "ASR Fail: " + e.message }));
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
            const detail = this.config.debugWakeSound ? raw : (raw = null, "(No Wake Word Detected)");
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
        
        if (this.recognizer) {
            this.recognizer.stopListening();
        }

        if (this.vad) {
            await this.vad.pause();
        }

        if (this._asrWorker) {
            this._asrWorker.terminate();
            this._asrWorker = null;
        }
        
        this._restoreConsoleLogs();
        
        this._log('OK', 'Shutdown complete.');
    }

    _log(level, message) {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        console.log(`[${level}] ${message}`);
    }

    _suppressLibraryLogs() {
        // Store original console methods
        this._originalConsole = {
            log: console.log,
            warn: console.warn,
            info: console.info
        };

        // Filter out library logs while keeping AkariNet logs
        const filter = (originalMethod) => {
            return (...args) => {
                const message = args.join(' ');
                // Keep AkariNet logs (those starting with [LEVEL])
                if (message.match(/^\[(OK|INFO|ERROR|DEBUG|WARN)\]/)) {
                    originalMethod.apply(console, args);
                }
                // Suppress library logs like [VAD], onnxruntime warnings, etc.
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
