// Whisper Web Worker for background processing

// Import the transformers library
importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0/dist/transformers.min.js');

// State variables
let whisperProcessor = null;
let currentModel = null;
let isInitializing = false;

// Initialize the Whisper model
async function initializeWhisperModel(modelName) {
    if (isInitializing) return;
    
    isInitializing = true;
    currentModel = modelName;
    
    try {
        postMessage({ type: 'status', message: `Loading model ${modelName.split('/')[1]}...` });
        
        const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0/dist/transformers.min.js');
        
        whisperProcessor = await pipeline('automatic-speech-recognition', modelName, {
            quantized: false, // Better quality at the cost of size
            revision: 'main'
        });
        
        postMessage({ type: 'ready' });
    } catch (error) {
        postMessage({ 
            type: 'error', 
            error: `Failed to load Whisper model: ${error.message}`
        });
        console.error('Error initializing Whisper model:', error);
    } finally {
        isInitializing = false;
    }
}

// Transcribe audio data using Whisper
async function transcribeAudio(audioData, sampleRate) {
    if (!whisperProcessor) {
        postMessage({ 
            type: 'error', 
            error: 'Whisper model not initialized yet'
        });
        return;
    }
    
    try {
        // Resample if needed
        const resampledAudio = (sampleRate !== 16000) 
            ? resampleAudio(audioData, sampleRate, 16000) 
            : audioData;
        
        // Get transcription with better parameters for speech recognition
        const result = await whisperProcessor(resampledAudio, {
            sampling_rate: 16000,
            return_timestamps: true,
            chunk_length_s: 30,
            stride_length_s: 5,
            language: 'en',
            task: 'transcribe',
            // Additional parameters to improve speech detection
            no_speech_threshold: 0.6,    // Higher value = less sensitive
            logprob_threshold: -1.0,     // Lower value = more willing to include uncertain segments
            compression_ratio_threshold: 2.4,  // Higher = more willing to include silence
            condition_on_previous_text: true,  // Use context from previous segments
            temperature: 0               // Use greedy decoding for clearer results
        });
        
        postMessage({ 
            type: 'transcription', 
            result 
        });
    } catch (error) {
        postMessage({ 
            type: 'error', 
            error: `Transcription failed: ${error.message}`
        });
        console.error('Error during transcription:', error);
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

// Handle messages from the main thread
self.onmessage = async function(e) {
    const data = e.data;
    
    switch (data.command) {
        case 'initialize':
            await initializeWhisperModel(data.model);
            break;
            
        case 'changeModel':
            await initializeWhisperModel(data.model);
            break;
            
        case 'transcribe':
            await transcribeAudio(data.audio, data.sampleRate);
            break;
            
        default:
            console.error('Unknown command:', data.command);
            postMessage({
                type: 'error',
                error: `Unknown command: ${data.command}`
            });
    }
};

// Let the main thread know the worker is ready
postMessage({ type: 'status', message: 'Whisper worker started' });
