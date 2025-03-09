// We'll try to import the library, but also provide a fallback
let pipeline;
try {
  // Try to import dynamically when the script runs
  import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0/dist/transformers.min.js')
    .then(module => {
      pipeline = module.pipeline;
      console.log("Transformers library loaded successfully through ES module");
      // Initialize once the library is loaded
      initWhisper();
    })
    .catch(err => {
      console.error("Failed to import Transformers module:", err);
      showError("Failed to load Whisper transcription engine. Try a different browser or check console for details.");
    });
} catch (e) {
  console.error("Error in import statement:", e);
}

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
let currentModel = modelSelect ? modelSelect.value : 'Xenova/whisper-small.en';
let whisperProcessor = null;
let audioProcessor;
let microphoneStream;
let analyser;
let audioQueue = [];
let processingAudio = false;
let audioBuffer = [];
let transformersLoaded = false;
let transformersPipeline = null;

// Check for Transformers availability and set up pipeline
function checkTransformers() {
    updateStatus('Initializing transcription engine...');
    
    // Check if Transformers is available globally (from script tag)
    if (typeof Transformers !== 'undefined') {
        console.log("Using globally loaded Transformers library");
        transformersLoaded = true;
        transformersPipeline = Transformers.pipeline;
        initWhisper();
        return;
    }
    
    console.log("Transformers not available yet, will retry...");
    
    // If not immediately available, check again in a moment
    setTimeout(function() {
        if (typeof Transformers !== 'undefined') {
            console.log("Transformers found on retry");
            transformersLoaded = true;
            transformersPipeline = Transformers.pipeline;
            initWhisper();
        } else {
            // Final fallback - try to load it one more time with a different CDN
            const script = document.createElement('script');
            script.src = "https://cdn.jsdelivr.net/npm/@xenova/transformers@latest/dist/transformers.min.js";
            script.onload = function() {
                if (typeof Transformers !== 'undefined') {
                    console.log("Transformers loaded via fallback script");
                    transformersLoaded = true;
                    transformersPipeline = Transformers.pipeline;
                    initWhisper();
                } else {
                    showError("Failed to load transcription engine. Please try a different browser (Chrome recommended) or check your internet connection.");
                    updateStatus("Error: Transcription engine not available");
                }
            };
            script.onerror = function() {
                showError("Failed to load transcription library. Please check your internet connection and try again.");
                updateStatus("Error: Unable to load required resources");
            };
            document.head.appendChild(script);
        }
    }, 2000);
}

// Initialize Whisper model using the imported pipeline
async function initWhisper() {
    try {
        updateStatus('Loading Whisper model...');
        
        if (!transformersLoaded || !transformersPipeline) {
            throw new Error("Transformers library not loaded. Please wait or reload the page.");
        }
        
        whisperProcessor = await transformersPipeline('automatic-speech-recognition', currentModel, {
            quantized: false,
            revision: 'main'
        });
        
        updateStatus(`Whisper model ${currentModel.split('/')[1]} loaded. Ready to transcribe.`);
    } catch (error) {
        updateStatus('Error loading Whisper model.');
        showError(`${error.message} Try using Chrome and ensure you're on a secure connection.`);
        console.error('Error loading Whisper model:', error);
    }
}

// Update status message
function updateStatus(message) {
    if (statusMessage) statusMessage.textContent = message;
}

// Show error message
function showError(message) {
    if (errorMessage) errorMessage.textContent = message;
}

// Clear error message
function clearError() {
    if (errorMessage) errorMessage.textContent = '';
}

// Handle model change
async function handleModelChange() {
    const newModel = modelSelect.value;
    if (newModel !== currentModel) {
        currentModel = newModel;
        updateStatus(`Loading ${currentModel.split('/')[1]} model...`);
        
        try {
            if (!transformersLoaded || !transformersPipeline) {
                throw new Error('Transformers library not loaded');
            }
            
            whisperProcessor = await transformersPipeline('automatic-speech-recognition', currentModel);
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
    if (!transformersLoaded || !whisperProcessor) {
        showError("Transcription engine is not ready yet. Please wait for initialization to complete.");
        return;
    }
    
    if (isRecording) return;
    
    try {
        // Initialize audio context if needed
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // GitHub Pages requires user interaction before audio context can start
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
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
        // Specific handling for GitHub Pages common issues
        if (error.name === 'NotAllowedError') {
            updateStatus('Microphone access denied.');
            showError('Please allow microphone access in your browser. On GitHub Pages, this requires HTTPS.');
        } else {
            updateStatus('Error accessing microphone.');
            showError(error.message);
        }
        console.error('Error accessing microphone:', error);
    }
}

// Start system audio recording (when supported)
async function startSystemAudioRecording() {
    if (!transformersLoaded || !whisperProcessor) {
        showError("Transcription engine is not ready yet. Please wait for initialization to complete.");
        return;
    }
    
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
            
            // GitHub Pages requires user interaction before audio context can start
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
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
            showError('You must allow access to continue. On GitHub Pages, this requires HTTPS.');
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
if (startMicrophoneBtn) startMicrophoneBtn.addEventListener('click', startMicrophoneRecording);
if (startSystemAudioBtn) startSystemAudioBtn.addEventListener('click', startSystemAudioRecording);
if (stopAudioBtn) stopAudioBtn.addEventListener('click', stopRecording);
if (clearBtn) clearBtn.addEventListener('click', clearTranscription);
if (modelSelect) modelSelect.addEventListener('change', handleModelChange);
if (filterNonSpeechCheckbox) filterNonSpeechCheckbox.addEventListener('change', updateTranscriptionDisplay);

// Detect if running on GitHub Pages
const isGitHubPages = window.location.hostname.includes('github.io');

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add GitHub Pages specific information
    if (isGitHubPages) {
        updateStatus('Running on GitHub Pages. Initializing transcription engine...');
        const note = document.querySelector('.note');
        if (note) {
            note.innerHTML = '<strong>GitHub Pages Notice:</strong> For microphone access and audio processing to work properly, ' +
                'you must access this page via HTTPS and grant the necessary permissions when prompted.';
        }
    } else {
        updateStatus('Loading transcription engine...');
    }
    
    // The initialization will happen after the library loads (in the import's then() callback)
});
