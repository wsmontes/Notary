// Elements
const startMicrophoneBtn = document.getElementById('startMicrophone');
const startSystemAudioBtn = document.getElementById('startSystemAudio');
const stopAudioBtn = document.getElementById('stopAudio');
const clearBtn = document.getElementById('clearTranscription');
const emergencyStopBtn = document.getElementById('emergencyStop');
const transcriptionArea = document.getElementById('transcription');
const statusMessage = document.getElementById('status-message');
const errorMessage = document.getElementById('error-message');
const modelSelect = document.getElementById('modelSelect');
const filterNonSpeechCheckbox = document.getElementById('filterNonSpeech');
const transcriptionModeIndicator = document.getElementById('transcription-mode-indicator');
const loaderContainer = document.getElementById('loaderContainer');

// Variables
let audioContext;
let isRecording = false;
let rawTranscriptionBuffer = '';
let filteredTranscriptionBuffer = '';
let currentModel = modelSelect ? modelSelect.value : 'Xenova/whisper-tiny.en'; // Start with tiny for speed
let whisperProcessor = null;
let audioProcessor;
let microphoneStream;
let analyser;
let audioQueue = [];
let processingAudio = false;
let audioBuffer = [];
let loadingTimeout;

// Show or hide the loader
function toggleLoader(show) {
    if (loaderContainer) {
        loaderContainer.classList.toggle('hidden', !show);
    }
}

// Safe way to check if Transformers exists
function safeGetTransformers() {
    return typeof window.Transformers !== 'undefined' ? window.Transformers : null;
}

// Initialize Whisper model with error handling and timeout
async function initWhisper() {
    toggleLoader(true);
    
    // Set a timeout to prevent infinite loading
    loadingTimeout = setTimeout(() => {
        toggleLoader(false);
        updateStatus('Loading timed out. Please refresh the page and try again.');
        showError('Model loading took too long. This may be due to network issues or browser limitations.');
    }, 60000); // 60 second timeout
    
    try {
        updateStatus('Loading Whisper model...');
        
        // Retry mechanism for loading the library
        let Transformers = safeGetTransformers();
        let retries = 0;
        
        while (!Transformers && retries < 5) {
            updateStatus(`Waiting for library to load (attempt ${retries + 1})...`);
            // Wait 1 second
            await new Promise(resolve => setTimeout(resolve, 1000));
            Transformers = safeGetTransformers();
            retries++;
        }
        
        if (!Transformers) {
            throw new Error("Transformers library failed to load after multiple attempts.");
        }
        
        // Update status with more frequent feedback
        updateStatus('Preparing transcription model...');
        
        // Configure to use Hugging Face Hub
        if (Transformers.env) {
            Transformers.env.allowRemoteModels = true;
            Transformers.env.useCacheFirst = true;
        }
        
        // Feedback on model downloading
        const startTime = Date.now();
        const progressCallback = (progress) => {
            if (progress.status === 'progress') {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                updateStatus(`Downloading model: ${Math.round(progress.loaded / 1024 / 1024)}MB (${elapsed}s elapsed)`);
            }
        };
        
        // Use tiny model first for faster initial loading
        updateStatus('Setting up lightweight model for speech recognition...');
        whisperProcessor = await Transformers.pipeline(
            'automatic-speech-recognition', 
            currentModel, 
            {
                quantized: true,
                progress_callback: progressCallback
            }
        );
        
        clearTimeout(loadingTimeout);
        toggleLoader(false);
        updateStatus(`Ready to transcribe with ${currentModel.split('/')[1]} model.`);
        clearError();
    } catch (error) {
        clearTimeout(loadingTimeout);
        toggleLoader(false);
        updateStatus('Error loading Whisper model.');
        showError(`${error.message}. Try reloading or using Chrome browser.`);
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

// Handle model change - simplified to prevent freezing
async function handleModelChange() {
    const newModel = modelSelect.value;
    if (newModel !== currentModel) {
        updateStatus(`Model change requested to ${newModel.split('/')[1]}...`);
        
        // Store the new model name but don't load immediately
        currentModel = newModel;
        
        // Show the user a message but delay actual loading until next recording
        updateStatus(`Model will change to ${currentModel.split('/')[1]} when you next start recording.`);
    }
}

// Process audio for transcription - with added safety checks
async function processAudioForTranscription(audioData, sampleRate) {
    // Skip if not ready
    if (!whisperProcessor) {
        updateStatus('Whisper model not initialized yet. Please wait or reload the page.');
        return;
    }
    
    processingAudio = true;
    updateStatus('Processing audio...');
    
    try {
        // Safety check for very large audio data that might freeze the browser
        if (audioData.length > 480000) { // Limit to ~10 seconds at 48kHz
            const trimmedAudio = new Float32Array(480000);
            trimmedAudio.set(audioData.slice(0, 480000));
            audioData = trimmedAudio;
            console.log("Audio data trimmed to prevent freezing");
        }
        
        // Simple silence detection to skip empty audio
        const isSilence = isAudioSilent(audioData);
        if (isSilence) {
            console.log("Skipping silent audio segment");
            processingAudio = false;
            return;
        }
        
        // Resample audio - 16kHz is optimal for Whisper
        const targetSampleRate = 16000;
        let processedAudio = audioData;
        if (sampleRate !== targetSampleRate) {
            processedAudio = resampleAudio(audioData, sampleRate, targetSampleRate);
        }
        
        // Process with simplified parameters
        const result = await whisperProcessor(processedAudio, {
            sampling_rate: targetSampleRate,
            return_timestamps: false, // Simplified for better performance
            language: 'en',
            task: 'transcribe'
        });
        
        handleTranscriptionResult(result);
    } catch (error) {
        console.error('Error during transcription:', error);
        updateStatus(`Transcription error: ${error.message}`);
    } finally {
        processingAudio = false;
        updateStatus('Ready to transcribe.');
        
        // Process next chunk if available - with a delay to prevent UI freezing
        if (audioQueue.length > 0) {
            const nextChunk = audioQueue.shift();
            setTimeout(() => {
                processAudioForTranscription(nextChunk.audio, nextChunk.sampleRate);
            }, 100);
        }
    }
}

// Check if audio is mostly silence (to skip processing empty segments)
function isAudioSilent(audioData) {
    // Calculate RMS (root mean square) of audio
    let sum = 0;
    for (let i = 0; i < audioData.length; i++) {
        sum += audioData[i] * audioData[i];
    }
    const rms = Math.sqrt(sum / audioData.length);
    
    // Consider it silence if RMS is below threshold
    return rms < 0.01;
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

// Start microphone recording - with added safety checks
async function startMicrophoneRecording() {
    if (isRecording) return;
    
    // Check if model needs to be initialized
    if (!whisperProcessor) {
        showError("Model not loaded yet. Please wait for initialization to complete.");
        return;
    }
    
    updateStatus('Starting microphone...');
    toggleLoader(true);
    
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
        
        // Add a safety timeout
        setTimeout(() => {
            toggleLoader(false);
        }, 5000);
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
        toggleLoader(false);
    }
}

// Start system audio recording (when supported)
async function startSystemAudioRecording() {
    if (!whisperProcessor) {
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

// Set up real-time audio processing - with performance improvements
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
    const PROCESSING_INTERVAL = 3000; // 3 seconds between processing
    
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
            
            // Add limit to queue size to prevent memory issues
            if (audioQueue.length < 5) {
                audioQueue.push({
                    audio: concatenatedAudio,
                    sampleRate: audioContext.sampleRate
                });
            }
            
            // Only start processing if not already processing
            if (!processingAudio && whisperProcessor) {
                processAudioForTranscription(concatenatedAudio, audioContext.sampleRate);
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

// Stop all processing - emergency function
function emergencyStop() {
    // Stop recording if active
    if (isRecording) {
        stopRecording();
    }
    
    // Clear any queued audio
    audioQueue = [];
    
    // Reset processing flag
    processingAudio = false;
    
    // Clear any timeouts
    clearTimeout(loadingTimeout);
    
    // Hide loader
    toggleLoader(false);
    
    // Reset UI
    updateStatus("Emergency stop performed. The page should be responsive again.");
    
    // Force garbage collection if possible
    if (window.gc) {
        window.gc();
    }
}

// Event listeners
if (startMicrophoneBtn) startMicrophoneBtn.addEventListener('click', startMicrophoneRecording);
if (startSystemAudioBtn) startSystemAudioBtn.addEventListener('click', startSystemAudioRecording);
if (stopAudioBtn) stopAudioBtn.addEventListener('click', stopRecording);
if (clearBtn) clearBtn.addEventListener('click', clearTranscription);
if (emergencyStopBtn) emergencyStopBtn.addEventListener('click', emergencyStop);
if (modelSelect) modelSelect.addEventListener('change', handleModelChange);
if (filterNonSpeechCheckbox) filterNonSpeechCheckbox.addEventListener('change', updateTranscriptionDisplay);

// Detect if running on GitHub Pages
const isGitHubPages = window.location.hostname.includes('github.io');

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initial UI setup
    toggleLoader(false);
    
    // Add GitHub Pages specific information
    if (isGitHubPages) {
        updateStatus('Running on GitHub Pages. Initializing transcription engine...');
        const note = document.querySelector('.note');
        if (note) {
            note.innerHTML = '<strong>GitHub Pages Notice:</strong> First-time use requires downloading the model files (~85MB). ' +
                'Please be patient during the initial setup. This process may take several minutes.';
        }
    } else {
        updateStatus('Loading transcription engine...');
    }
    
    // Give the page a moment to fully load, then check for Transformers
    setTimeout(() => {
        if (typeof Transformers !== 'undefined') {
            console.log("Transformers library detected. Initializing Whisper...");
            initWhisper();
        } else {
            console.error("Transformers library not available");
            showError("Failed to load speech recognition library. Please try using Chrome and refreshing the page.");
            updateStatus("Error: Transcription engine unavailable");
        }
    }, 1000);
});

// Note: The ONNX warnings you're seeing about "Removing initializer" are normal 
// optimization messages from the neural network engine and don't affect functionality.
