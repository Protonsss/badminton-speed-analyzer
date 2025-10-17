/*
  Badminton Smash Speed Analyzer - API Client
  Connects to Python/PyTorch backend for analysis
*/

const API_BASE = 'http://localhost:8000';

const els = {
  fileInput: document.getElementById('fileInput'),
  demoBtn: document.getElementById('demoBtn'),
  video: document.getElementById('video'),
  canvas: document.getElementById('canvas'),
  startCalib: document.getElementById('startCalib'),
  resetCalib: document.getElementById('resetCalib'),
  courtType: document.getElementById('courtType'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  peakSpeed: document.getElementById('peakSpeed'),
  avgSpeed: document.getElementById('avgSpeed'),
  initSpeed: document.getElementById('initSpeed'),
  fps: document.getElementById('fps'),
  chart: document.getElementById('chart'),
  exportCSV: document.getElementById('exportCSV'),
  exportPNG: document.getElementById('exportPNG')
};

const ctx = els.canvas.getContext('2d');
let chart;

// Help modal
const helpBtn = document.getElementById('helpBtn');
const helpModal = document.getElementById('helpModal');
const helpBackdrop = document.getElementById('helpBackdrop');
const closeHelp = document.getElementById('closeHelp');
function openHelp(){ helpModal?.setAttribute('aria-hidden','false'); }
function closeHelpFn(){ helpModal?.setAttribute('aria-hidden','true'); }
helpBtn?.addEventListener('click', openHelp);
helpBackdrop?.addEventListener('click', closeHelpFn);
closeHelp?.addEventListener('click', closeHelpFn);

// Tabs
const tabUpload = document.getElementById('tabUpload');
const tabGuide = document.getElementById('tabGuide');
const panelUpload = document.getElementById('panel-upload');
const panelGuide = document.getElementById('panel-guide');
function setTab(which){
  if (which === 'upload') {
    tabUpload?.classList.add('active'); tabGuide?.classList.remove('active');
    panelUpload?.classList.add('active'); panelGuide?.classList.remove('active');
  } else {
    tabGuide?.classList.add('active'); tabUpload?.classList.remove('active');
    panelGuide?.classList.add('active'); panelUpload?.classList.remove('active');
  }
}
tabUpload?.addEventListener('click', ()=>setTab('upload'));
tabGuide?.addEventListener('click', ()=>setTab('guide'));

// Reveal analyzer
const openAnalyzer = document.getElementById('openAnalyzer');
const advanced = document.getElementById('advanced');
openAnalyzer?.addEventListener('click', () => {
  advanced?.classList.remove('hidden');
  openAnalyzer?.closest('.panel-actions')?.remove();
});

// State
let calibrationPoints = []; // [{x,y}] in video pixels
let uploadedFileId = null;
let currentVideoPath = null;
let videoFps = 60;
let analysisResult = null;

// ============= Video Handling =============
els.fileInput?.addEventListener('change', loadVideo);
els.demoBtn?.addEventListener('click', () => {
  alert('Demo video feature coming soon. Please upload your own video.');
});

async function loadVideo(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  // Load video in browser
  const url = URL.createObjectURL(file);
  els.video.src = url;
  
  els.video.onloadedmetadata = () => {
    videoFps = els.video.playbackRate || 60;
    els.canvas.width = els.video.videoWidth;
    els.canvas.height = els.video.videoHeight;
    drawFrame();
    els.startCalib.disabled = false;
  };
  
  // Upload to backend
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    if (result.success) {
      uploadedFileId = result.file_id;
      console.log('Video uploaded:', uploadedFileId);
    } else {
      alert('Upload failed');
    }
  } catch (err) {
    console.error('Upload error:', err);
    alert('Failed to connect to backend. Make sure Python server is running on port 8000.');
  }
}

function drawFrame() {
  if (!els.video.paused && !els.video.ended) {
    ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
    requestAnimationFrame(drawFrame);
  } else {
    ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
  }
}

// ============= Calibration =============
let isCalibrating = false;

els.startCalib?.addEventListener('click', startCalibration);
els.resetCalib?.addEventListener('click', resetCalibration);

function startCalibration() {
  isCalibrating = true;
  calibrationPoints = [];
  els.startCalib.disabled = true;
  els.resetCalib.disabled = false;
  alert('Click 4 corners of the court in order: top-left, top-right, bottom-right, bottom-left');
}

function resetCalibration() {
  calibrationPoints = [];
  isCalibrating = false;
  els.startCalib.disabled = false;
  els.resetCalib.disabled = true;
  drawFrame();
  drawCalibrationPoints();
}

els.canvas.addEventListener('click', (e) => {
  if (!isCalibrating) return;
  
  const rect = els.canvas.getBoundingClientRect();
  const scaleX = els.canvas.width / rect.width;
  const scaleY = els.canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  
  calibrationPoints.push({x, y});
  drawCalibrationPoints();
  
  if (calibrationPoints.length === 4) {
    isCalibrating = false;
    sendCalibration();
  }
});

function drawCalibrationPoints() {
  drawFrame();
  ctx.fillStyle = '#ff3366';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  
  calibrationPoints.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(`${i+1}`, p.x - 5, p.y - 12);
    ctx.fillStyle = '#ff3366';
  });
  
  if (calibrationPoints.length === 4) {
    ctx.strokeStyle = '#ff3366';
    ctx.lineWidth = 2;
    ctx.beginPath();
    calibrationPoints.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.closePath();
    ctx.stroke();
  }
}

async function sendCalibration() {
  if (calibrationPoints.length !== 4) return;
  
  try {
    const response = await fetch(`${API_BASE}/api/calibrate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        points: calibrationPoints.map(p => [p.x, p.y]),
        court_type: els.courtType.value
      })
    });
    
    const result = await response.json();
    if (result.success) {
      alert('Calibration successful! You can now analyze.');
      els.analyzeBtn.disabled = false;
    } else {
      alert('Calibration failed: ' + result.message);
    }
  } catch (err) {
    console.error('Calibration error:', err);
    alert('Failed to calibrate. Check backend connection.');
  }
}

// ============= Analysis =============
els.analyzeBtn?.addEventListener('click', analyzeVideo);

async function analyzeVideo() {
  if (!uploadedFileId) {
    alert('Please upload a video first');
    return;
  }
  
  if (calibrationPoints.length !== 4) {
    alert('Please complete calibration first');
    return;
  }
  
  els.analyzeBtn.disabled = true;
  els.analyzeBtn.textContent = 'Analyzing...';
  
  try {
    const formData = new FormData();
    formData.append('file_id', uploadedFileId);
    formData.append('use_yolo', 'true');
    formData.append('start_frame', '0');
    
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    if (result.success) {
      analysisResult = result;
      displayResults(result);
    } else {
      alert('Analysis failed: ' + result.error);
    }
  } catch (err) {
    console.error('Analysis error:', err);
    alert('Analysis failed. Check backend logs.');
  } finally {
    els.analyzeBtn.disabled = false;
    els.analyzeBtn.textContent = 'Analyze Video';
  }
}

function displayResults(result) {
  // Display speeds
  els.peakSpeed.textContent = `${result.peak_speed.kmh.toFixed(1)} km/h (${result.peak_speed.mph.toFixed(1)} mph)`;
  els.avgSpeed.textContent = `${result.avg_speed.kmh.toFixed(1)} km/h (${result.avg_speed.mph.toFixed(1)} mph)`;
  
  if (result.initial_speed) {
    els.initSpeed.textContent = `${result.initial_speed.kmh.toFixed(1)} km/h (${result.initial_speed.mph.toFixed(1)} mph)`;
  } else {
    els.initSpeed.textContent = 'N/A';
  }
  
  els.fps.textContent = `${result.fps.toFixed(1)} fps`;
  
  // Draw trajectory on canvas
  drawTrajectory(result.trajectory_pixel);
  
  // Plot speed chart
  plotSpeedChart(result.times, result.speeds);
  
  // Enable exports
  els.exportCSV.disabled = false;
  els.exportPNG.disabled = false;
}

function drawTrajectory(trajectory) {
  drawFrame();
  drawCalibrationPoints();
  
  if (trajectory.length < 2) return;
  
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 3;
  ctx.beginPath();
  
  trajectory.forEach((point, i) => {
    const [t, x, y] = point;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  
  ctx.stroke();
  
  // Draw points
  ctx.fillStyle = '#00ff88';
  trajectory.forEach(point => {
    const [t, x, y] = point;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function plotSpeedChart(times, speeds) {
  if (chart) chart.destroy();
  
  const kmhSpeeds = speeds.map(s => s * 3.6);
  
  chart = new Chart(els.chart, {
    type: 'line',
    data: {
      labels: times.map(t => t.toFixed(3)),
      datasets: [{
        label: 'Speed (km/h)',
        data: kmhSpeeds,
        borderColor: '#306cff',
        backgroundColor: 'rgba(48, 108, 255, 0.1)',
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {display: false},
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.parsed.y.toFixed(1)} km/h`
          }
        }
      },
      scales: {
        x: {
          title: {display: true, text: 'Time (s)'},
          ticks: {maxTicksLimit: 10}
        },
        y: {
          title: {display: true, text: 'Speed (km/h)'},
          beginAtZero: true
        }
      }
    }
  });
}

// ============= Export =============
els.exportCSV?.addEventListener('click', exportCSV);
els.exportPNG?.addEventListener('click', exportPNG);

function exportCSV() {
  if (!analysisResult) return;
  
  let csv = 'Time (s),X (m),Y (m),Speed (km/h)\n';
  
  analysisResult.trajectory_meter.forEach((point, i) => {
    const [t, x, y] = point;
    const speed = i < analysisResult.speeds.length ? analysisResult.speeds[i] * 3.6 : 0;
    csv += `${t.toFixed(4)},${x.toFixed(3)},${y.toFixed(3)},${speed.toFixed(2)}\n`;
  });
  
  const blob = new Blob([csv], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'smash_speed_data.csv';
  a.click();
}

function exportPNG() {
  const link = document.createElement('a');
  link.download = 'smash_speed_chart.png';
  link.href = els.chart.toDataURL();
  link.click();
}

