# Real-Time Audio Transcription

A browser-based application that uses the Whisper model to transcribe audio in real-time from your microphone or system audio.

## How to Use

1. Select a Whisper model (smaller models are faster but less accurate)
2. Click "Start Microphone" to transcribe from your microphone
3. Click "Start System Audio" to transcribe audio playing on your system
4. The transcription will appear in the text area
5. Click "Stop" when you want to end recording
6. Click "Clear" to reset the transcription

## GitHub Pages Requirements

When running on GitHub Pages:

1. You must access the page via HTTPS
2. You must grant permission for microphone access when prompted
3. System audio capture requires Chrome or Edge browser
4. The first run will download the Whisper model, which might take some time

## Browser Support

- Google Chrome (recommended)
- Microsoft Edge
- Firefox (limited functionality)
- Safari (limited functionality)

## Technology

This application uses:
- Transformers.js - Browser-based machine learning
- Web Audio API - Audio capture and processing
- WebRTC - Microphone access
- Screen Capture API - System audio capture
