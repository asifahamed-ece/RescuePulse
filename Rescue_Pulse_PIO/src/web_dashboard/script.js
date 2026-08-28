// RescuePulse Dashboard JavaScript

// WebSocket connection
let socket = null;
let packetCount = 0;

// DOM elements
const directionEl = document.getElementById('direction');
const confidenceEl = document.getElementById('confidence');
const timestampEl = document.getElementById('timestamp');
const leftLevelEl = document.getElementById('left-level');
const leftValueEl = document.getElementById('left-value');
const rightLevelEl = document.getElementById('right-level');
const rightValueEl = document.getElementById('right-value');
const statusEl = document.getElementById('status');
const connectionStatusEl = document.getElementById('connectionStatus');
const packetCountEl = document.getElementById('packetCount');
const spectrumCanvas = document.getElementById('spectrumCanvas');
const spectrumCtx = spectrumCanvas.getContext('2d');

// Spectrum data (for visualization)
const spectrumData = new Uint8Array(128); // 128 frequency bins

// Initialize the dashboard
function initDashboard() {
    // Set initial state
    updateStatus('idle', 'Listening...');
    updateConnectionStatus(false);
    packetCount = 0;
    updatePacketCount();

    // Attempt to connect to WebSocket
    connectWebSocket();
}

// Update status indicator and text
function updateStatus(type, text) {
    statusEl.className = `status ${type}`;
    statusEl.textContent = text;
}

// Update connection status
function updateConnectionStatus(connected) {
    if (connected) {
        connectionStatusEl.className = 'connection connected';
        connectionStatusEl.textContent = 'Connected';
    } else {
        connectionStatusEl.className = 'connection disconnected';
        connectionStatusEl.textContent = 'Disconnected';
    }
}

// Update packet count
function updatePacketCount() {
    packetCountEl.textContent = `Packets: ${packetCount}`;
}

// Update detection information
function updateDetection(data) {
    directionEl.textContent = data.direction || '-';
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    timestampEl.textContent = new Date(data.timestamp * 1000).toLocaleTimeString();

    // Update status based on detection
    if (data.detected) {
        updateStatus('detection', `SIREN DETECTED [${data.direction}]`);
    } else {
        updateStatus('idle', 'Background Noise');
    }
}

// Update audio levels
function updateAudioLevels(data) {
    // Left mic
    const leftLevel = Math.min(data.rms_l * 500, 100); // Scale for visualization
    leftLevelEl.style.width = `${leftLevel}%`;
    leftValueEl.textContent = data.rms_l.toFixed(3);

    // Right mic
    const rightLevel = Math.min(data.rms_r * 500, 100); // Scale for visualization
    rightLevelEl.style.width = `${rightLevel}%`;
    rightValueEl.textContent = data.rms_r.toFixed(3);
}

// Update spectrum visualization
function updateSpectrum(data) {
    if (data.spectrum && data.spectrum.length) {
        // Copy spectrum data (limit to our buffer size)
        const len = Math.min(data.spectrum.length, spectrumData.length);
        spectrumData.set(data.spectrum.subarray(0, len));

        // Draw spectrum
        drawSpectrum();
    }
}

// Draw spectrum on canvas
function drawSpectrum() {
    spectrumCtx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);

    const barWidth = spectrumCanvas.width / spectrumData.length;
    let x = 0;

    for (let i = 0; i < spectrumData.length; i++) {
        const height = (spectrumData[i] / 255) * spectrumCanvas.height;
        spectrumCtx.fillStyle = '#00ff00';
        spectrumCtx.fillRect(x, spectrumCanvas.height - height, barWidth - 1, height);
        x += barWidth;
    }
}

// Handle incoming WebSocket message
function handleMessage(message) {
    try {
        const data = JSON.parse(message);
        packetCount++;
        updatePacketCount();

        // Update dashboard based on message type
        if (data.type === 'detection') {
            updateDetection(data);
            updateAudioLevels(data);
            if (data.spectrum) {
                updateSpectrum(data);
            }
        } else if (data.type === 'noise') {
            // Handle noise level updates if needed
            updateAudioLevels(data);
        } else if (data.type === 'system') {
            // Handle system status updates
            // For now, we'll just log
            console.log('System update:', data);
        }
    } catch (e) {
        console.error('Error parsing WebSocket message:', e, message);
    }
}

// WebSocket connection logic
function connectWebSocket() {
    // In a real implementation, we would get the IP from the device or use mDNS
    // For now, we'll use a placeholder - in practice, this would be configured
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${window.location.host}/ws`;

    console.log('Connecting to WebSocket:', wsUrl);

    socket = new WebSocket(wsUrl);

    socket.onopen = function(event) {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
        // Optionally send an initial message to request data
        // socket.send(JSON.stringify({type: 'request_initial_data'}));
    };

    socket.onmessage = function(event) {
        handleMessage(event.data);
    };

    socket.onclose = function(event) {
        console.log('WebSocket disconnected', event);
        updateConnectionStatus(false);
        // Attempt to reconnect after a delay
        setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
}

// Handle page visibility changes to pause/resume if needed
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden, we could reduce update frequency
    } else {
        // Page is visible, resume normal updates
    }
});

// Initialize when page loads
window.addEventListener('load', initDashboard);

// Clean up when page unloads
window.addEventListener('beforeunload', function() {
    if (socket) {
        socket.close();
    }
});