// Import the transformers library properly
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0/dist/transformers.min.js';

// Elements
const startMicrophoneBtn = document.getElementById('startMicrophone');
const startSystemAudioBtn = document.getElementById('startSystemAudio');
const stopAudioBtn = document.getElementById('stopAudio');
const clearBtn = document.getElementById('clearTranscription');
const transcriptionArea = document.getElementById('transcription');
const statusMessage = document.getElementById('status-message');
const errorMessage = document.getElementById('error-message');
const modelSelect = document.getElementById('modelSelect');
const filterNonSpeechCheckbox = document.getElementById('filterNonSpeech');
const transcriptionModeIndicator = document.getElementById('transcription-mode-indicator');

// Variables
let audioContext;
let isRecording = false;
let rawTranscriptionBuffer = '';
let filteredTranscriptionBuffer = '';
let currentModel = modelSelect.value;
let whisperProcessor = null;
let audioProcessor;
let microphoneStream;
let analyser;
let audioQueue = [];
let processingAudio = false;
let audioBuffer = [];
let audioWorkletSupported = false;

// Initialize Whisper model
async function initWhisper() {
    try {
        updateStatus('Loading Whisper model...');
        
        whisperProcessor = await pipeline('automatic-speech-recognition', currentModel, {
            quantized: false,
            revision: 'main'
        });
        
        updateStatus(`Whisper model ${currentModel.split('/')[1]} loaded. Ready to transcribe.`);
    } catch (error) {
        updateStatus('Error loading Whisper model.');
        showError(`${error.message}. Make sure you're using a modern browser and running from a local server.`);
        console.error('Error loading Whisper model:', error);
    }
}

// Update status message
function updateStatus(message) {
    statusMessage.textContent = message;
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
}

// Clear error message
function clearError() {
    errorMessage.textContent = '';
}

// Handle model change
async function handleModelChange() {
    const newModel = modelSelect.value;
    if (newModel !== currentModel) {
        currentModel = newModel;
        updateStatus(`Loading ${currentModel.split('/')[1]} model...`);
        
        try {
            whisperProcessor = await pipeline('automatic-speech-recognition', currentModel);
            updateStatus(`Model changed to ${currentModel.split('/')[1]}. Ready to transcribe.`);
            clearError();
        } catch (error) {
            updateStatus(`Error loading model.`);
            showError(`${error.message}`);
            console.error('Error loading model:', error);
        }
    }
}

// Process audio for transcription
async function processAudioForTranscription(audioData, sampleRate) {
    if (!whisperProcessor) {
        updateStatus('Whisper model not initialized yet. Please wait or reload the page.');
        return;
    }
    
    processingAudio = true;
    
    try {
        // Resample audio if needed (Whisper works best with 16kHz)
        const targetSampleRate = 16000;
        let processedAudio = audioData;
        
        if (sampleRate !== targetSampleRate) {
            processedAudio = resampleAudio(audioData, sampleRate, targetSampleRate);
        }
        
        // Get transcription with better parameters for speech recognition
        const result = await whisperProcessor(processedAudio, {
            sampling_rate: targetSampleRate,
            return_timestamps: true,
            chunk_length_s: 30,
            stride_length_s: 5,
            language: 'en',
            task: 'transcribe',
            // Additional parameters to improve speech detection
            no_speech_threshold: 0.6,
            logprob_threshold: -1.0,
            compression_ratio_threshold: 2.4,
            condition_on_previous_text: true,
            temperature: 0
        });
        
        // Process the transcription result
        handleTranscriptionResult(result);
    } catch (error) {
        console.error('Error during transcription:', error);
        updateStatus(`Transcription error.`);
        showError(error.message);
    } finally {
        processingAudio = false;
        
        // Process next chunk if available
        if (audioQueue.length > 0) {
            const nextChunk = audioQueue.shift();
            processAudioForTranscription(nextChunk.audio, nextChunk.sampleRate);
        }
    }
}

// Simple linear resampling function
function resampleAudio(audioData, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) {
        return audioData;
    }
    
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        const position = i * ratio;
        const index = Math.floor(position);
        const fraction = position - index;
        
        if (index >= audioData.length - 1) {
            result[i] = audioData[audioData.length - 1];
        } else {
            result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
        }
    }
    
    return result;
}

// Handle transcription results
function handleTranscriptionResult(result) {
    if (!result || !result.text) return;
    
    // Get the raw text
    const rawText = result.text.trim();
    
    if (rawText) {
        // Add to raw buffer
        rawTranscriptionBuffer += rawText + ' ';
        
        // Create filtered version
        let filteredText = rawText.replace(/\[MUSIC\]|\[Music\]|\[music\]|\[SOUND\]|\[Sound\]|\[sound\]|\[NOISE\]|\[Noise\]|\[noise\]/g, '');
        filteredText = filteredText.replace(/\(\s*clicking\s*\)|\(\s*music playing\s*\)/gi, '');
        filteredText = filteredText.replace(/\[.*?\]|\(.*?\)/g, '');
        filteredText = filteredText.replace(/\s+/g, ' ').trim();
        
        if (filteredText) {
            filteredTranscriptionBuffer += filteredText + ' ';
        }
        
        // Update display
        updateTranscriptionDisplay();
    }
}

// Start microphone recording
async function startMicrophoneRecording() {
    if (isRecording) return;
    
    try {
        // Initialize audio context if needed
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        clearError();
        
        // Get microphone stream with better audio settings for speech
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: { 
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        
        microphoneStream = stream;
        setupAudioProcessing(stream);
        updateStatus('Recording and transcribing from microphone...');
    } catch (error) {
        updateStatus('Error accessing microphone.');
        showError(error.message);
        console.error('Error accessing microphone:', error);
    }
}

// Start system audio recording (when supported)
async function startSystemAudioRecording() {
    if (isRecording) return;
    
    try {
        // Check if getDisplayMedia is supported
        if (!navigator.mediaDevices.getDisplayMedia) {
            updateStatus('System audio capture is not supported in your browser.');
            showError('Your browser does not support screen capture with audio. Try using Chrome or Edge.');
            return;
        }
        
        clearError();
        updateStatus('Please select a window or tab that contains audio to capture...');
        
        // Initialize audio context if needed
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        // Request screen sharing with audio
        const stream = await navigator.mediaDevices.getDisplayMedia({
            video: true,
            audio: true
        });
        
        // Check if audio track exists
        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length === 0) {
            updateStatus('No system audio track available.');
            showError('Please try again and select a source with audio enabled.');
            stream.getTracks().forEach(track => track.stop());
            return;
        }
        
        // Stop video tracks as we only need audio
        stream.getVideoTracks().forEach(track => track.stop());
        
        // Create a new stream with only audio tracks
        const audioStream = new MediaStream(audioTracks);
        
        microphoneStream = audioStream;
        setupAudioProcessing(audioStream);
        updateStatus('Recording and transcribing system audio...');
    } catch (error) {
        // Handle permission denied specifically
        if (error.name === 'NotAllowedError') {
            updateStatus('Permission to capture system audio was denied.');
            showError('You must allow access to continue. Please try again.');
        } else {
            updateStatus('Error capturing system audio.');
            showError(error.message);
        }
        console.error('Error capturing system audio:', error);
    }
}

// Set up real-time audio processing
function setupAudioProcessing(stream) {
    isRecording = true;
    audioBuffer = [];
    audioQueue = [];
    
    // Update button states
    startMicrophoneBtn.disabled = true;
    startSystemAudioBtn.disabled = true;
    stopAudioBtn.disabled = false;
    modelSelect.disabled = true;
    
    // Create media source from stream
    const source = audioContext.createMediaStreamSource(stream);
    
    // Create analyzer
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    
    // Create processor for real-time processing
    // Note: We're still using ScriptProcessorNode because AudioWorklet requires more setup
    // and may not be available in all browsers
    const bufferSize = 4096;
    audioProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
    
    let lastProcessingTime = Date.now();
    const PROCESSING_INTERVAL = 2000; // Process every 2 seconds
    
    audioProcessor.onaudioprocess = function(e) {
        if (!isRecording) return;
        
        // Get audio data from first channel
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Clone the data since it's a Float32Array view and would be gone by next callback
        const audioData = new Float32Array(inputData.length);
        audioData.set(inputData);
        
        // Add to buffer
        audioBuffer.push(audioData);
        
        // Check if it's time to process
        const currentTime = Date.now();
        if (currentTime - lastProcessingTime >= PROCESSING_INTERVAL) {
            // Concatenate all audio chunks
            const totalLength = audioBuffer.reduce((acc, val) => acc + val.length, 0);
            const concatenatedAudio = new Float32Array(totalLength);
            
            let offset = 0;
            for (const buffer of audioBuffer) {
                concatenatedAudio.set(buffer, offset);
                offset += buffer.length;
            }
            
            // Reset the buffer
            audioBuffer = [];
            lastProcessingTime = currentTime;
            
            // If previous processing is done, process directly, otherwise queue
            if (!processingAudio && whisperProcessor) {
                processAudioForTranscription(concatenatedAudio, audioContext.sampleRate);
            } else {
                audioQueue.push({
                    audio: concatenatedAudio,
                    sampleRate: audioContext.sampleRate
                });
            }
        }
    };
    
    // Connect the nodes: source -> analyser -> processor -> destination (silent)
    source.connect(analyser);
    analyser.connect(audioProcessor);
    audioProcessor.connect(audioContext.destination);
}

// Update the transcription display based on filter checkbox state
function updateTranscriptionDisplay() {
    if (filterNonSpeechCheckbox.checked) {
        transcriptionArea.value = filteredTranscriptionBuffer;
        transcriptionModeIndicator.textContent = "Showing: Filtered Transcription";
    } else {
        transcriptionArea.value = rawTranscriptionBuffer;
        transcriptionModeIndicator.textContent = "Showing: Raw Transcription";
    }
    
    transcriptionArea.scrollTop = transcriptionArea.scrollHeight;
}

// Stop recording
function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    
    // Disconnect and clean up audio processing
    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }
    
    if (analyser) {
        analyser.disconnect();
        analyser = null;
    }
    
    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        microphoneStream = null;
    }
    
    // Update button states
    startMicrophoneBtn.disabled = false;
    startSystemAudioBtn.disabled = false;
    stopAudioBtn.disabled = true;
    modelSelect.disabled = false;
    
    updateStatus('Recording stopped.');
}

// Clear transcription
function clearTranscription() {
    rawTranscriptionBuffer = '';
    filteredTranscriptionBuffer = '';
    updateTranscriptionDisplay();
    updateStatus('Transcription cleared.');
    clearError();
}

// Event listeners
startMicrophoneBtn.addEventListener('click', startMicrophoneRecording);
startSystemAudioBtn.addEventListener('click', startSystemAudioRecording);
stopAudioBtn.addEventListener('click', stopRecording);
clearBtn.addEventListener('click', clearTranscription);
modelSelect.addEventListener('change', handleModelChange);
filterNonSpeechCheckbox.addEventListener('change', updateTranscriptionDisplay);

// Initialize the app
document.addEventListener('DOMContentLoaded', initWhisper);
