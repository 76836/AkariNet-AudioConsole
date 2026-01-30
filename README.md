# AkariNet Audio Console

A lightweight, hybrid voice recognition engine for web applications that combines wake word detection, voice activity detection, and automatic speech recognition.

## Features

- **Multiple Wake Word Support** - Trigger commands with custom text wake words
- **Wake Sound Detection** - Optional audio-based wake word detection using TensorFlow.js
- **Voice Activity Detection** - Efficient speech detection to save processing power
- **ONNX ASR** - Fast, on-device speech transcription using Moonshine models
- **Continuous Listening** - Always-on wake word monitoring with minimal resource usage

## Demo

Simple demo at: https://76836.github.io/AkariNet-AudioConsole

## Quick Start

```javascript
import { AkarinetVoice } from 'https://76836.github.io/AkariNet-AudioConsole/audioConsole-3.3.0.js';

const voice = new AkarinetVoice({
    wakewords: ['hey akari', 'computer', 'jarvis'],
    wakesoundURL: 'https://teachablemachine.withgoogle.com/models/YOUR_MODEL/',
    wakesoundThreshold: 0.75,
    wakesoundIndex: 2
});

voice.addEventListener('ready', () => {
    console.log('Voice assistant ready!');
});

voice.addEventListener('result', (e) => {
    console.log('Command:', e.detail.text);
    console.log('Original:', e.detail.original);
    console.log('Via Sound:', e.detail.viaSound);
});

await voice.init();
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelId` | string | `"onnx-community/moonshine-tiny-ONNX"` | ASR model to use |
| `modelQuantization` | string | `"q8"` | Model quantization level |
| `wakewords` | array | `null` | Text wake words to listen for |
| `wakesoundURL` | string | `null` | URL to TensorFlow.js wake sound model |
| `wakesoundThreshold` | number | `0.75` | Confidence threshold for wake sound |
| `wakesoundIndex` | number | `2` | Class index in wake sound model |
| `wakesoundDelay` | number | `0` | Time window (ms) to correlate wake sound with speech |
| `vadThreshold` | number | `0.75` | Voice activity detection sensitivity |
| `cleanup` | boolean | `true` | Clean up recognized text (remove punctuation) |
| `debugWakeSound` | boolean | `false` | Enable detailed wake sound logging |
| `requireWakeSound` | boolean | `false` | Only transcribe after wake sound detected |

## Events

### `ready`
Fired when the voice engine is fully initialized and ready to listen.

### `speechstart`
Fired when voice activity is detected.

### `speechend`
Fired when voice activity stops.

### `wakesound`
Fired when the wake sound is detected (if enabled).

```javascript
voice.addEventListener('wakesound', (e) => {
    console.log('Score:', e.detail.score);
    console.log('Class:', e.detail.class);
    console.log('Timestamp:', e.detail.timestamp);
});
```

### `result`
Fired when a command is successfully recognized.

```javascript
voice.addEventListener('result', (e) => {
    console.log(e.detail.text);      // Cleaned command text (uppercase)
    console.log(e.detail.original);  // Original transcription
    console.log(e.detail.viaSound);  // True if triggered by wake sound
});
```

### `speechdiscarded`
Fired when speech is captured but not processed (no wake word, too short, etc.).

### `error`
Fired when an error occurs.

## Wake Sound Models

You can train custom wake sound models using [Teachable Machine](https://teachablemachine.withgoogle.com/):

1. Create a new Audio Project
2. Record your wake sound samples
3. Train the model
4. Export and use the model URL

```javascript
const voice = new AkarinetVoice({
    wakesoundURL: 'https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/',
    wakesoundIndex: 2  // Your trained class index
});
```

## How It Works

1. **Continuous Wake Sound Monitoring** - TensorFlow.js listens continuously for your custom audio wake word
2. **Voice Activity Detection** - When speech is detected, VAD captures the audio
3. **Wake Word Correlation** - Audio is checked for text or sound wake words
4. **Transcription** - Matching audio is transcribed using the ASR model
5. **Command Output** - Recognized commands are emitted via the `result` event

## Cleanup

When you're done, properly shut down the voice engine:

```javascript
await voice.destroy();
```
