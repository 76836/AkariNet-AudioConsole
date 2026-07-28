<p align="left">
  <img src="https://raw.githubusercontent.com/76836/Akari/main/images/banner.png" width="100%" alt="AkariNet"/>
</p>

> ### **This repository is a part of the [AkariNet](https://github.com/76836/Akari/tree/main) project.**

# AkariNet Audio Console

**v4.1** — A unified, security-hardened voice recognition engine for the browser.

One microphone. Pluggable wake-word detection. Pluggable speech recognition. Optional long-term audio cache.

## Features

- **Single shared AudioBus** — One `getUserMedia` + 16 kHz AudioWorklet. VAD, wake-word, and the optional XL cache all share the same chunk stream.
- **Pluggable wake-word providers**
  - `openwakeword` (recommended) — chunk-fed, speaker-invariant ONNX models
  - `teachablemachine` — legacy TensorFlow.js speech-commands (own mic)
  - `none` — manual trigger only
- **Pluggable speech recognition**
  - `transformers` (default) — Moonshine / Whisper via transformers.js
  - `whispercpp` — HTTP POST to a local whisper.cpp / llamafile server
  - `webspeech` — browser SpeechRecognition, wake-gated for privacy
  - `none` — wake events only
- **BusVAD (Silero)** — Direct ONNX VAD on the bus (replaces the old vad-web dependency)
- **XL Cache** — Optional rolling ring buffer (1 second → 1 hour) that can be retrieved after the fact
- **Text wake words** still supported as a fallback / filter
- Full backwards compatibility with v4.0 configs

## Demo

https://76836.github.io/AkariNet-AudioConsole

## Quick Start (v4.1)

```javascript
import { AkarinetVoice } from 'https://76836.github.io/AkariNet-AudioConsole/audioConsole-4.1.0.js';

const voice = new AkarinetVoice({
  // Text wake words (optional)
  wakewords: ['hey akari', 'akari'],

  // Wake-word provider (recommended)
  wakeWordProvider: 'openwakeword',
  openWakeWord: {
    keywordURL: 'https://cdn.jsdelivr.net/gh/dnavarrom/openwakeword_wasm@main/models/hey_jarvis_v0.1.onnx',
    // or use keywordURLs for multiple models
  },

  // Speech recognition (default is transformers / Moonshine)
  speechRecognitionProvider: 'transformers',
  modelId: 'onnx-community/moonshine-tiny-ONNX',
  modelQuantization: 'q8',

  // Optional long audio cache
  // xlCache: { enabled: true, durationMs: 60000, vadOnly: true }
});

voice.addEventListener('ready', () => console.log('Ready'));
voice.addEventListener('wakesound', (e) => console.log('Wake:', e.detail));
voice.addEventListener('result', (e) => {
  console.log('Command:', e.detail.text);
  console.log('Original:', e.detail.original);
  console.log('Via sound:', e.detail.viaSound);
});
voice.addEventListener('error', (e) => console.error(e.detail));

await voice.init();
```

When finished:

```javascript
await voice.destroy();
```

## Configuration

### Core options (v4.0 compatible)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelId` | string | `"onnx-community/moonshine-tiny-ONNX"` | ASR model (transformers provider) |
| `modelQuantization` | string | `"q8"` | Quantization for transformers.js |
| `wakewords` | string[] | `[]` | Text wake words |
| `wakeWordProvider` | string | `'none'` or `'teachablemachine'` if `wakesoundURL` is set | `'openwakeword'`, `'teachablemachine'`, or `'none'` |
| `wakesoundURL` | string | `null` | Teachable Machine model URL (legacy) |
| `wakesoundThreshold` | number | `0.75` | Confidence threshold for Teachable Machine |
| `wakesoundIndex` | number | `2` | Class index for Teachable Machine |
| `wakesoundDuration` | number | `100` | Time window used for wake ↔ speech correlation (ms) |
| `wakesoundDelay` | number | `0` | Extra correlation window (ms) |
| `vadThreshold` | number | `0.5` | Silero confidence (values ≥ 0.6 are auto-mapped from old vad-web scale) |
| `vadRedemptionMs` | number | `480` | Silence before speech-end is declared |
| `cleanup` | boolean | `true` | Strip punctuation / normalize result text |
| `debugWakeSound` | boolean | `false` | Verbose logging |
| `requireWakeSound` | boolean | `false` | Only run ASR after a wake event |

### New in v4.1

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `speechRecognitionProvider` | string | `'transformers'` | `'transformers'`, `'whispercpp'`, `'webspeech'`, or `'none'` |
| `whisperCpp` | object | `null` | `{ baseUrl, inferencePath?, language?, temperature?, timeoutMs?, proxyUrl? }` |
| `webSpeech` | object | `null` | `{ lang?, continuous?, interimResults?, maxSessionMs? }` |
| `unifiedMic` | boolean | `true` | Use the shared AudioBus (recommended) |
| `liveCacheMs` | number | `2000` | Size of the small pre-roll ring |
| `xlCache` | object | `null` | `{ enabled: true, durationMs: 1000–3600000, vadOnly?: boolean }` |
| `openWakeWord` | object | `null` | See openWakeWord section below |

### openWakeWord config

```javascript
openWakeWord: {
  // Easiest — single model URL
  keywordURL: 'https://.../hey_akari_v0.1.onnx',
  keywordName: 'hey_akari',          // optional, auto-extracted from filename

  // Or multiple
  // keywordURLs: ['https://.../a.onnx', { url: 'https://.../b.onnx', name: 'alexa' }],

  // Or classic filename + base path
  // baseAssetUrl: 'https://host/models',
  // keywords: ['hey_jarvis'],

  detectionThreshold: 0.5,
  cooldownMs: 2000,
}
```

Built-in keyword names: `alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy`, `timer`, `weather`.

## Events

| Event | Detail | When |
|-------|--------|------|
| `ready` | — | Engine fully initialized |
| `speechstart` | — | VAD detects speech |
| `speechend` | — | VAD decides speech has ended |
| `wakesound` | `{ score, class, timestamp }` | Wake word / sound detected |
| `result` | `{ text, original, viaSound }` | Successful command after wake-word filtering |
| `speechdiscarded` | string reason | Speech captured but discarded |
| `error` | string / Error | Fatal or recoverable error |
| `processing` / `processingend` | — | ASR is running |
| `xlcachepurge` | — | XL cache was cleared |

## Manual wake trigger

```javascript
voice.activateWakeWord();   // same as a real wake detection
```

Useful for a push-to-talk button. For `webspeech` this immediately starts a recognition session.

## XL Cache (optional)

```javascript
const voice = new AkarinetVoice({
  xlCache: { enabled: true, durationMs: 120000, vadOnly: true }
});

await voice.init();

// Later… pull the last 30 seconds of audio
const { audio, sampleRate, availableMs } = voice.retrieveCache({
  fromMsAgo: 30000,
  format: 'float32'   // or 'wav'
});

voice.clearXlCache();
```

`durationMs` must be between 1 000 and 3 600 000 (1 s – 1 h). RAM usage scales with duration; `vadOnly: true` stores only speech segments.

## Privacy notes (Web Speech)

The Web Speech API cannot accept pre-recorded audio — it opens its own microphone and (in Chrome) streams to Google.  
To keep privacy intact, the `webspeech` provider is **session-based**: it only starts after a wake event (or `activateWakeWord()`). There is never continuous cloud listening.

## Migration from v4.0 → v4.1

1. Change the import URL to `audioConsole-4.1.0.js`.
2. Everything else continues to work.
3. Optionally adopt the new keys: `speechRecognitionProvider`, `whisperCpp`, `webSpeech`, `unifiedMic`, `liveCacheMs`, `xlCache`.

`vadThreshold` values ≥ 0.6 are automatically shifted into Silero’s scale so old configs keep the same sensitivity.

## Architecture (simplified)

```
                    ┌── AudioBus (one getUserMedia, 16 kHz) ──┐
                    │     AudioWorklet → 1280-sample chunks     │
                    └──────────────────┬───────────────────────┘
                                       │
          ┌──────────────┬─────────────┼─────────────┬────────────┐
          ▼              ▼             ▼             ▼            ▼
     LiveRing(2s)   BusVAD(Silero)  WakeProvider  XlCache    (future)
          │              │             │
          │         speech-end ──► SpeechRecognitionProvider.transcribe()
          │                              │
          └──────────────────────────────┴──► _parse() → 'result'
```

## Cleanup

```javascript
await voice.destroy();
```

Stops the bus, releases the microphone, tears down providers and the XL cache.

## License notes

- OpenWakeWordProvider (inlined engine): Apache 2.0  
  Adapted from [openwakeword_wasm](https://github.com/dnavarrom/openwakeword_wasm)
- Silero VAD model: MIT
