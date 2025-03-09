// Elements
const startMicrophoneBtn = document.getElementById('startMicrophone');
const startSystemAudioBtn = document.getElementById('startSystemAudio');
const stopAudioBtn = document.getElementById('stopAudio');
const clearBtn = document.getElementById('clearTranscription');
const transcriptionArea = document.getElementById('transcription');
const statusMessage = document.getElementById('status-message');
const modelSelect = document.getElementById('modelSelect');
const filterNonSpeechCheckbox = document.getElementById('filterNonSpeech');
const transcriptionModeIndicator = document.getElementById('transcription-mode-indicator');

// Variables
let audioContext;
let mediaRecorder;
let isRecording = false;
let rawTranscriptionBuffer = '';
let filteredTranscriptionBuffer = '';
let currentModel = modelSelect.value;
let whisperWorker;
let audioProcessor;
let microphoneStream;
let analyser;
let audioQueue = [];
let processingAudio = false;

// Initialize Whisper model via worker
function initWhisper() {
    updateStatus('Starting Whisper transcription engine...');
    
    // Create the worker
    whisperWorker = new Worker('whisper-worker.js');
    
    // Handle messages from the worker
    whisperWorker.onmessage = function(e) {
        const message = e.data;
        
        switch (message.type) {
            case 'ready':
                updateStatus(`Whisper model ${currentModel.split('/')[1]} ready. You can start transcribing.`);
                break;
                
            case 'transcription':
                handleTranscriptionResult(message.result);
                processingAudio = false;
                processNextAudioChunk();
                break;
                
            case 'error':
                console.error('Whisper error:', message.error);
                updateStatus(`Error in transcription: ${message.error}`);
                processingAudio = false;
                processNextAudioChunk();
                break;
                
            case 'status':
                updateStatus(message.message);
                break;
        }
    };
    
    // Initialize the worker with the model
    whisperWorker.postMessage({
        command: 'initialize',
        model: currentModel
    });
}

// Update status message
function updateStatus(message) {
    statusMessage.textContent = message;
}

// Handle model change
function handleModelChange() {
    const newModel = modelSelect.value;
    if (newModel !== currentModel) {
        currentModel = newModel;
        updateStatus(`Loading ${currentModel.split('/')[1]} model...`);
        
        // Tell the worker to change the model
        whisperWorker.postMessage({
            command: 'changeModel',
            model: currentModel
        });
    }
}

// Process the next audio chunk in the queue
function processNextAudioChunk() {
    if (audioQueue.length > 0 && !processingAudio) {
        processingAudio = true;
        const audioData = audioQueue.shift();
        
        whisperWorker.postMessage({
            command: 'transcribe',
            audio: audioData.audio,
            sampleRate: audioData.sampleRate
        });
    }
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
        
        // Get microphone stream with better audio settings for speech
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: { 
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                channelCount: 1, // Mono for better speech processing
                sampleRate: 16000 // Optimal for Whisper
            } 
        });
        
        microphoneStream = stream;
        setupAudioProcessing(stream);
        updateStatus('Recording and transcribing from microphone...');
    } catch (error) {
        updateStatus(`Error accessing microphone: ${error.message}`);
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
            return;
        }
        
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
            updateStatus('No system audio track available. Please try again and select a source with audio.');
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
        updateStatus(`Error capturing system audio: ${error.message}`);
        console.error('Error capturing system audio:', error);
    }
}

// Set up real-time audio processing
function setupAudioProcessing(stream) {
    isRecording = true;
    
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
    
    // Create script processor for real-time processing
    // Note: ScriptProcessorNode is deprecated but still widely supported
    // AudioWorklet would be better but has more complex setup
    audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    
    let audioBuffer = [];
    let lastProcessingTime = Date.now();
    const PROCESSING_INTERVAL = 2000; // Process every 2 seconds
    
    audioProcessor.onaudioprocess = function(e) {
        // Get audio data
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
            
            // Push to processing queue
            audioQueue.push({
                audio: concatenatedAudio,
                sampleRate: audioContext.sampleRate
            });
            
            // Reset the buffer
            audioBuffer = [];
            
            // Process if not already processing
            if (!processingAudio) {
                processNextAudioChunk();
            }
            
            lastProcessingTime = currentTime;
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
    
    // Disconnect and clean up audio processing
    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }
    
    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        microphoneStream = null;
    }
    
    isRecording = false;
    
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
