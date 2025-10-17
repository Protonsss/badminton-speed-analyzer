/*
  Badminton Smash Speed Analyzer
  - Upload video, calibrate court via 4 points -> homography to metric coordinates
  - Detect shuttle per frame (background subtraction + white object morphology + size gating)
  - Track with Kalman filter + Hungarian association
  - Compute speed from metric displacements with frame timestamps
  - Render overlay and speed chart
*/

let cvReady = false;
function onOpenCvReady() { cvReady = true; }

const els = {
  fileInput: document.getElementById('fileInput'),
  demoBtn: document.getElementById('demoBtn'),
  video: document.getElementById('video'),
  canvas: document.getElementById('canvas'),
  startCalib: document.getElementById('startCalib'),
  resetCalib: document.getElementById('resetCalib'),
  courtType: document.getElementById('courtType'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  smoothWin: document.getElementById('smoothWin'),
  minSize: document.getElementById('minSize'),
  maxSize: document.getElementById('maxSize'),
  manualMode: document.getElementById('manualMode'),
  useYolo: document.getElementById('useYolo'),
  yoloUrl: document.getElementById('yoloUrl'),
  loadYolo: document.getElementById('loadYolo'),
  prevFrame: document.getElementById('prevFrame'),
  nextFrame: document.getElementById('nextFrame'),
  addMark: document.getElementById('addMark'),
  finishManual: document.getElementById('finishManual'),
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
// Help modal controls
const helpBtn = document.getElementById('helpBtn');
const helpModal = document.getElementById('helpModal');
const helpBackdrop = document.getElementById('helpBackdrop');
const closeHelp = document.getElementById('closeHelp');
function openHelp(){ helpModal?.setAttribute('aria-hidden','false'); }
function closeHelpFn(){ helpModal?.setAttribute('aria-hidden','true'); }
helpBtn?.addEventListener('click', openHelp);
helpBackdrop?.addEventListener('click', closeHelpFn);
closeHelp?.addEventListener('click', closeHelpFn);

// Tabs behavior
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

// Reveal advanced analyzer
const openAnalyzer = document.getElementById('openAnalyzer');
const advanced = document.getElementById('advanced');
openAnalyzer?.addEventListener('click', () => {
  advanced?.classList.remove('hidden');
  openAnalyzer?.closest('.panel-actions')?.remove();
});

// State
let calibrationPoints = []; // [{x,y}] in canvas pixels
let H; // homography 3x3
let metersPerUnit = 1; // pixel->meter after homography (should be 1 if homography maps to meters)
let frameTimes = []; // seconds per frame
let trackHistory = []; // [{t, x, y}] in meters
let overlayFrames = []; // for export
let manualMarks = []; // [{t, xpx, ypx}]
// YOLO session
let yoloSession = null;
let yoloInputShape = null; // [N,C,H,W]
let yoloScoreThresh = 0.3;
let yoloIouThresh = 0.4;
// Utility: order and validate calibration corner points
function orderCalibrationPoints(points) {
  if (points.length !== 4) return points;
  const cx = points.reduce((a,p)=>a+p.x,0)/4, cy = points.reduce((a,p)=>a+p.y,0)/4;
  const withAngle = points.map(p => ({...p, angle: Math.atan2(p.y - cy, p.x - cx)}));
  withAngle.sort((a,b)=>a.angle-b.angle); // counterclockwise
  // Reorder to start at top-left (smallest y then x)
  let startIdx = 0;
  let best = {y: Infinity, x: Infinity};
  for (let i=0;i<withAngle.length;i++){ const p=withAngle[i]; if (p.y < best.y - 1e-6 || (Math.abs(p.y-best.y)<1e-6 && p.x < best.x)){ best=p; startIdx=i; } }
  return [withAngle[startIdx], withAngle[(startIdx+1)%4], withAngle[(startIdx+2)%4], withAngle[(startIdx+3)%4]].map(({x,y})=>({x,y}));
}
function validateCalibrationQuad(points) {
  if (points.length !== 4) throw new Error('Need 4 points');
  const area = Math.abs(0.5*((points[0].x*points[1].y - points[1].x*points[0].y) + (points[1].x*points[2].y - points[2].x*points[1].y) + (points[2].x*points[3].y - points[3].x*points[2].y) + (points[3].x*points[0].y - points[0].x*points[3].y)));
  if (area < 1000) throw new Error('Calibration area too small');
  // Check non-collinearity via cross products
  function cross(ax,ay,bx,by){return ax*by - ay*bx;}
  for (let i=0;i<4;i++){
    const a=points[i], b=points[(i+1)%4], c=points[(i+2)%4];
    const v1={x:b.x-a.x,y:b.y-a.y}, v2={x:c.x-b.x,y:c.y-b.y};
    const cr = Math.abs(cross(v1.x,v1.y,v2.x,v2.y));
    if (cr < 1e-2) throw new Error('Calibration points nearly collinear');
  }
}

const COURT = {
  singles: { length: 13.40, width: 5.18 },
  doubles: { length: 13.40, width: 6.10 }
};

function getCourtRectMeters(courtType) {
  const c = COURT[courtType || 'singles'];
  // Define rectangle corners in meters (clockwise starting top-left)
  return [
    { x: 0, y: 0 },
    { x: c.width, y: 0 },
    { x: c.width, y: c.length },
    { x: 0, y: c.length }
  ];
}

function drawCalibrationPoints() {
  ctx.save();
  ctx.strokeStyle = '#64d2ff';
  ctx.fillStyle = '#64d2ff';
  ctx.lineWidth = 2;
  
  // Convert video coords back to canvas coords for drawing
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  const cw = els.canvas.width, ch = els.canvas.height;
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale, dh = vh * scale;
  const dx = (cw - dw) / 2, dy = (ch - dh) / 2;
  
  const canvasPoints = calibrationPoints.map(p => ({
    x: p.x * scale + dx,
    y: p.y * scale + dy
  }));
  
  for (let i = 0; i < canvasPoints.length; i++) {
    const p = canvasPoints[i];
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(String(i + 1), p.x + 8, p.y - 8);
  }
  if (canvasPoints.length === 4) {
    ctx.beginPath();
    ctx.moveTo(canvasPoints[0].x, canvasPoints[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(canvasPoints[i].x, canvasPoints[i].y);
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

function computeHomography() {
  const ordered = orderCalibrationPoints(calibrationPoints);
  validateCalibrationQuad(ordered);
  const dst = getCourtRectMeters(els.courtType.value); // in meters
  // Build OpenCV Mats
  const srcMat = cv.matFromArray(4, 1, cv.CV_32FC2, new Float32Array(ordered.flatMap(p => [p.x, p.y])));
  const dstMat = cv.matFromArray(4, 1, cv.CV_32FC2, new Float32Array(dst.flatMap(p => [p.x, p.y])));
  const Hmat = cv.getPerspectiveTransform(srcMat, dstMat);
  srcMat.delete(); dstMat.delete();
  H = Hmat; // keep as cv.Mat
  metersPerUnit = 1; // destination is meters
}

function applyHomography(x, y) {
  if (!H) return null;
  const src = cv.matFromArray(1, 1, cv.CV_32FC2, new Float32Array([x, y]));
  const dst = new cv.Mat();
  cv.perspectiveTransform(src, dst, H);
  const arr = dst.data32F;
  const point = { x: arr[0], y: arr[1] };
  src.delete(); dst.delete();
  return point;
}

function movingAverage(data, windowSize) {
  if (windowSize <= 1) return data.slice();
  const half = Math.floor(windowSize / 2);
  const result = data.map((v, i) => {
    const start = Math.max(0, i - half);
    const end = Math.min(data.length, i + half + 1);
    const slice = data.slice(start, end);
    const sum = slice.reduce((a, b) => a + b, 0);
    return sum / slice.length;
  });
  return result;
}

// Savitzky–Golay smoothing for evenly spaced data; precomputed symmetric coefficients for common windows (poly=3)
function savitzkyGolay(data, windowSize) {
  const coeffsByWindow = {
    5: [-3, 12, 17, 12, -3].map(v => v / 35),
    7: [-2, 3, 6, 7, 6, 3, -2].map(v => v / 21),
    9: [-21, 14, 39, 54, 59, 54, 39, 14, -21].map(v => v / 231),
    11: [-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36].map(v => v / 429)
  };
  const coeffs = coeffsByWindow[windowSize];
  if (!coeffs) return movingAverage(data, Math.min(5, data.length|0));
  const half = Math.floor(windowSize / 2);
  const out = new Array(data.length).fill(0);
  for (let i = 0; i < data.length; i++) {
    let acc = 0, wsum = 0;
    for (let k = -half; k <= half; k++) {
      const idx = i + k;
      const c = coeffs[k + half];
      if (idx >= 0 && idx < data.length) { acc += data[idx] * c; wsum += c; }
    }
    out[i] = wsum ? acc / wsum : data[i];
  }
  return out;
}

function computeSpeedsMetersPerSecond(history) {
  if (history.length < 2) return [];
  const speeds = [];
  for (let i = 1; i < history.length; i++) {
    const a = history[i - 1];
    const b = history[i];
    const dt = b.t - a.t;
    if (dt <= 0) { speeds.push(0); continue; }
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.hypot(dx, dy);
    speeds.push(dist / dt);
  }
  return speeds;
}

// Physics model: dv/dt = -k v^2; v(t) = v0 / (1 + k v0 t)
function estimateInitialSpeedWithQuadraticDrag(history) {
  if (history.length < 5) return null;
  const t0 = history[0].t;
  const t = history.map(h => h.t - t0);
  const segT = [], segV = [];
  for (let i = 1; i < history.length; i++) {
    const dt = t[i] - t[i-1]; if (dt <= 0) continue;
    const dx = history[i].x - history[i-1].x;
    const dy = history[i].y - history[i-1].y;
    const v = Math.hypot(dx, dy) / dt;
    segT.push(t[i]);
    segV.push(v);
  }
  if (segV.length < 4) return null;
  const v0guess = Math.max(...segV);
  let best = { err: Infinity, v0: v0guess, k: 0.02 };
  for (let k0 of [0.005, 0.01, 0.02, 0.04, 0.08]) {
    for (let v0 of [v0guess*0.8, v0guess, v0guess*1.2, v0guess*1.5]) {
      let err = 0;
      for (let i = 0; i < segT.length; i++) {
        const pred = v0 / (1 + k0 * v0 * segT[i]);
        const e = pred - segV[i];
        err += e*e;
      }
      if (err < best.err) best = { err, v0, k: k0 };
    }
  }
  // refine v0
  const k = best.k; let v0 = best.v0;
  for (let iter = 0; iter < 10; iter++) {
    let num = 0, den = 0;
    for (let i = 0; i < segT.length; i++) {
      const ti = segT[i];
      const denom = (1 + k * v0 * ti);
      const pred = v0 / denom;
      const d = (1/denom) - (k * ti * v0) / (denom*denom);
      const r = segV[i] - pred;
      num += r * d; den += d*d;
    }
    if (den <= 1e-9) break;
    const step = num / den;
    v0 += step;
    if (Math.abs(step) < 1e-4) break;
  }
  return { v0, k };
}

function formatSpeed(valueMps) {
  const kmh = valueMps * 3.6;
  const mph = valueMps * 2.23694;
  return `${kmh.toFixed(1)} km/h (${mph.toFixed(1)} mph)`;
}

function setupChart() {
  if (chart) chart.destroy();
  chart = new Chart(els.chart.getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Speed (m/s)', data: [], borderColor: '#64d2ff', borderWidth: 2, pointRadius: 0, tension: .2 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Time (s)' }, grid: { color: 'rgba(255,255,255,.05)' } },
        y: { title: { display: true, text: 'Speed (m/s)' }, grid: { color: 'rgba(255,255,255,.05)' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function updateChart(times, speeds) {
  setupChart();
  chart.data.labels = times.slice(1).map(t => t.toFixed(3));
  chart.data.datasets[0].data = speeds;
  chart.update();
}

// --- YOLO helpers (ONNX Runtime Web) ---
async function loadYoloModel(url) {
  if (!url) throw new Error('Provide YOLO ONNX URL');
  if (typeof ort === 'undefined') throw new Error('ONNX Runtime Web not loaded');
  yoloSession = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  // Infer input shape from first input
  const inputName = yoloSession.inputNames[0];
  const inMeta = yoloSession.inputMetadata[inputName];
  yoloInputShape = inMeta.dimensions.map(d => (typeof d === 'number' ? d : 1));
}

function letterboxResize(imageData, dstW, dstH) {
  // Draw into a temp canvas with letterboxing, return {data:Float32Array, scale, dx, dy}
  const srcW = imageData.width, srcH = imageData.height;
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const newW = Math.round(srcW * scale), newH = Math.round(srcH * scale);
  const dx = Math.floor((dstW - newW) / 2), dy = Math.floor((dstH - newH) / 2);
  const tmp = document.createElement('canvas'); tmp.width = dstW; tmp.height = dstH;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = '#000'; tctx.fillRect(0, 0, dstW, dstH);
  // Put imageData into an intermediate canvas to drawImage
  const srcCanvas = document.createElement('canvas'); srcCanvas.width = srcW; srcCanvas.height = srcH;
  srcCanvas.getContext('2d').putImageData(imageData, 0, 0);
  tctx.drawImage(srcCanvas, 0, 0, srcW, srcH, dx, dy, newW, newH);
  const resized = tctx.getImageData(0, 0, dstW, dstH);
  // Convert to NCHW float32 normalized 0..1
  const chw = new Float32Array(3 * dstH * dstW);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const si = (y * dstW + x) * 4;
      const r = resized.data[si] / 255, g = resized.data[si + 1] / 255, b = resized.data[si + 2] / 255;
      const di = y * dstW + x;
      chw[di] = r; chw[dstH * dstW + di] = g; chw[2 * dstH * dstW + di] = b;
    }
  }
  return { data: chw, scale, dx, dy, dstW, dstH };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const uni = areaA + areaB - inter;
  return uni <= 0 ? 0 : inter / uni;
}

function nms(boxes, iouThresh) {
  boxes.sort((a,b)=>b.score-a.score);
  const out = [];
  for (const b of boxes) {
    let keep = true;
    for (const o of out) { if (iou(b, o) > iouThresh) { keep = false; break; } }
    if (keep) out.push(b);
  }
  return out;
}

async function yoloDetectCenter(imageData) {
  if (!yoloSession || !els.useYolo || !els.useYolo.checked) return null;
  const inputName = yoloSession.inputNames[0];
  const outputName = yoloSession.outputNames[0];
  const [N, C, H, W] = yoloInputShape || [1,3,416,416];
  const prep = letterboxResize(imageData, W, H);
  const feeds = {}; feeds[inputName] = new ort.Tensor('float32', prep.data, [1, 3, H, W]);
  const results = await yoloSession.run(feeds);
  const out = results[outputName];
  const data = out.data; // assume [num, 6+] x,y,w,h,score,class
  const num = out.dims[0];
  const stride = out.dims[1];
  const boxes = [];
  for (let i = 0; i < num; i++) {
    const off = i * stride;
    const cx = data[off + 0];
    const cy = data[off + 1];
    const w = data[off + 2];
    const h = data[off + 3];
    const score = data[off + 4];
    if (score < yoloScoreThresh) continue;
    // Map back to canvas image coords (undo letterbox)
    const x1n = cx - w/2, y1n = cy - h/2;
    const x2n = cx + w/2, y2n = cy + h/2;
    const x1 = (x1n - prep.dx) / prep.scale;
    const y1 = (y1n - prep.dy) / prep.scale;
    const x2 = (x2n - prep.dx) / prep.scale;
    const y2 = (y2n - prep.dy) / prep.scale;
    boxes.push({ x1, y1, x2, y2, score });
  }
  const kept = nms(boxes, yoloIouThresh);
  if (!kept.length) return null;
  const b = kept[0];
  return { x: (b.x1 + b.x2)/2, y: (b.y1 + b.y2)/2 };
}

function drawOverlayFrame(pointPx, timeSeconds) {
  const w = els.canvas.width, h = els.canvas.height;
  const overlay = document.createElement('canvas');
  overlay.width = w; overlay.height = h;
  const octx = overlay.getContext('2d');
  octx.drawImage(els.canvas, 0, 0); // canvas already has video frame
  if (pointPx) {
    octx.save();
    octx.fillStyle = '#34c759';
    octx.strokeStyle = 'rgba(52,199,89,.6)';
    octx.lineWidth = 2;
    octx.beginPath();
    octx.arc(pointPx.x, pointPx.y, 6, 0, Math.PI * 2);
    octx.fill();
    octx.stroke();
    octx.fillStyle = '#fff';
    octx.font = '12px Inter, sans-serif';
    octx.fillText(timeSeconds.toFixed(3) + 's', pointPx.x + 10, pointPx.y - 10);
    octx.restore();
  }
  overlayFrames.push(overlay);
}

function exportCSV(times, speeds) {
  const lines = ['time_s,speed_mps,speed_kmh,speed_mph'];
  for (let i = 1; i < times.length; i++) {
    const t = times[i];
    const v = speeds[i - 1] || 0;
    lines.push(`${t.toFixed(6)},${v.toFixed(6)},${(v*3.6).toFixed(6)},${(v*2.23694).toFixed(6)}`);
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'smash_speed.csv'; a.click();
  URL.revokeObjectURL(url);
}

function exportOverlayPNG() {
  if (!overlayFrames.length) return;
  const strip = document.createElement('canvas');
  const frame = overlayFrames[overlayFrames.length - 1];
  strip.width = frame.width;
  strip.height = frame.height;
  const sctx = strip.getContext('2d');
  sctx.drawImage(frame, 0, 0);
  strip.toBlob(b => {
    const url = URL.createObjectURL(b);
    const a = document.createElement('a');
    a.href = url; a.download = 'overlay.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

// Video loading
els.fileInput.addEventListener('change', e => {
  const file = e.target.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  els.video.src = url;
  els.video.play();
});

els.demoBtn.addEventListener('click', () => {
  // Provide a demo sample if hosted with demo.mp4 alongside
  fetch('demo.mp4').then(r => {
    if (!r.ok) throw new Error('No demo video found');
    els.video.src = 'demo.mp4';
    els.video.play();
  }).catch(() => alert('Place a demo.mp4 next to the page to use demo.'));
});

// Draw video into canvas continuously
let rafId;
function drawLoop() {
  if (els.video.readyState >= 2) {
    const vw = els.video.videoWidth, vh = els.video.videoHeight;
    // Fit video into canvas while preserving aspect
    const cw = els.canvas.width, ch = els.canvas.height;
    const scale = Math.min(cw / vw, ch / vh);
    const dw = vw * scale, dh = vh * scale;
    const dx = (cw - dw) / 2, dy = (ch - dh) / 2;
    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(els.video, dx, dy, dw, dh);
    drawCalibrationPoints();
  }
  rafId = requestAnimationFrame(drawLoop);
}
drawLoop();

// Calibration interactions
let calibrating = false;
els.startCalib.addEventListener('click', () => {
  if (!cvReady) { alert('OpenCV is still loading.'); return; }
  calibrating = true;
  calibrationPoints = [];
});
els.resetCalib.addEventListener('click', () => {
  calibrationPoints = [];
  calibrating = false;
  if (H) { H.delete(); H = null; }
});

// Helper: map canvas click to video coordinates (accounting for letterboxing)
function canvasToVideoCoords(canvasX, canvasY) {
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  const cw = els.canvas.width, ch = els.canvas.height;
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale, dh = vh * scale;
  const dx = (cw - dw) / 2, dy = (ch - dh) / 2;
  // Invert letterbox transform
  const videoX = (canvasX - dx) / scale;
  const videoY = (canvasY - dy) / scale;
  return { x: videoX, y: videoY };
}

els.canvas.addEventListener('click', e => {
  const rect = els.canvas.getBoundingClientRect();
  const canvasX = e.clientX - rect.left;
  const canvasY = e.clientY - rect.top;
  
  if (els.manualMode.checked) {
    const videoCoords = canvasToVideoCoords(canvasX, canvasY);
    manualMarks.push({ t: els.video.currentTime, xpx: videoCoords.x, ypx: videoCoords.y });
    // quick visual (draw at canvas coords)
    ctx.save();
    ctx.fillStyle = '#ffcc00';
    ctx.beginPath(); ctx.arc(canvasX, canvasY, 5, 0, Math.PI*2); ctx.fill();
    ctx.restore();
    return;
  }
  if (!calibrating) return;
  
  const videoCoords = canvasToVideoCoords(canvasX, canvasY);
  calibrationPoints.push(videoCoords);
  if (calibrationPoints.length === 4) {
    calibrating = false;
    try {
      computeHomography();
    } catch (err) {
      console.error(err);
      alert('Calibration failed. Try again with clearer corner clicks.');
      calibrationPoints = [];
    }
  }
});

// Analysis
els.analyzeBtn.addEventListener('click', async () => {
  if (!cvReady) { alert('OpenCV is still loading.'); return; }
  if (!H) { alert('Please calibrate the court first.'); return; }
  const video = els.video;
  if (!video.duration || video.videoWidth === 0) { alert('Load a valid video first.'); return; }

  overlayFrames = [];
  trackHistory = [];
  frameTimes = [];
  manualMarks = [];

  const minSize = parseInt(els.minSize.value, 10);
  const maxSize = parseInt(els.maxSize.value, 10);
  const smoothWin = Math.max(5, parseInt(els.smoothWin.value, 10) | 0);

  // Prepare OpenCV Mats
  const frame = new cv.Mat(els.canvas.height, els.canvas.width, cv.CV_8UC4);
  const frameGray = new cv.Mat();
  const prevGray = new cv.Mat();
  const diff = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));

  // Optical flow buffers
  let prevPtMat = null; // cv.Mat Nx1 CV_32FC2
  let havePrevPoint = false;
  let lastTime = null;
  let gh = null; // g–h filter state

  function ghCreate(x, y) {
    return { x, y, vx: 0, vy: 0 };
  }
  function ghUpdate(state, meas, dt, alpha = 0.7, beta = 0.2) {
    // Predict
    const px = state.x + state.vx * dt;
    const py = state.y + state.vy * dt;
    // Innovation
    const rx = meas.x - px;
    const ry = meas.y - py;
    // Update
    state.x = px + alpha * rx;
    state.y = py + alpha * ry;
    state.vx = state.vx + beta * (rx / Math.max(dt, 1e-3));
    state.vy = state.vy + beta * (ry / Math.max(dt, 1e-3));
    return state;
  }

  function detectShuttle(centerFromFlow) {
    // Motion-based detection vs previous frame
    cv.absdiff(frameGray, prevGray, diff);
    cv.threshold(diff, diff, 35, 255, cv.THRESH_BINARY);
    cv.morphologyEx(diff, diff, cv.MORPH_OPEN, kernel);
    cv.morphologyEx(diff, diff, cv.MORPH_DILATE, kernel);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(diff, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let best = null; let bestScore = -1;
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const rect = cv.boundingRect(cnt);
      const area = rect.width * rect.height;
      if (area < minSize * minSize || area > maxSize * maxSize) { cnt.delete(); continue; }
      const cx = rect.x + rect.width / 2;
      const cy = rect.y + rect.height / 2;
      const intensity = frameGray.ucharPtr(Math.max(0, Math.min(frameGray.rows - 1, cy|0)), Math.max(0, Math.min(frameGray.cols - 1, cx|0)))[0];
      let score = intensity + Math.min(4000, area);
      if (centerFromFlow) {
        const dx = cx - centerFromFlow.x, dy = cy - centerFromFlow.y;
        const distPenalty = Math.min(200, Math.hypot(dx, dy));
        score -= distPenalty; // prefer near flow prediction
      }
      if (score > bestScore) { bestScore = score; best = { x: cx, y: cy, rect }; }
      cnt.delete();
    }
    hierarchy.delete(); contours.delete();
    return best; // in pixels on canvas
  }

  // Start
  await video.play().catch(() => {});
  const useRVFC = typeof video.requestVideoFrameCallback === 'function';
  let tsForFps = [];

  const handle = (now, meta) => {
    if (video.ended) { return finalize(); }
    // Draw current frame to canvas
    const vw = video.videoWidth, vh = video.videoHeight;
    const cw = els.canvas.width, ch = els.canvas.height;
    const scale = Math.min(cw / vw, ch / vh);
    const dw = vw * scale, dh = vh * scale;
    const dx = (cw - dw) / 2, dy = (ch - dh) / 2;
    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(video, dx, dy, dw, dh);
    drawCalibrationPoints();

    const imageData = ctx.getImageData(0, 0, cw, ch);
    frame.data.set(imageData.data);
    cv.cvtColor(frame, frameGray, cv.COLOR_RGBA2GRAY);

    const t = meta?.mediaTime ?? video.currentTime;
    frameTimes.push(t);
    tsForFps.push(t);

    let candidate = null; // {x,y} px
    if (!prevGray.empty()) {
      // Optical flow if previous point exists
      if (havePrevPoint && prevPtMat && !prevPtMat.isDeleted()) {
        const nextPts = new cv.Mat();
        const status = new cv.Mat();
        const err = new cv.Mat();
        cv.calcOpticalFlowPyrLK(prevGray, frameGray, prevPtMat, nextPts, status, err);
        const ok = status.data && status.data[0] === 1;
        let flowPt = null;
        if (ok) {
          const arr = nextPts.data32F;
          flowPt = { x: arr[0], y: arr[1] };
        }
        // Re-detect if flow failed or error too large
        const reDetect = !ok || (err.data && err.data[0] > 25);
        if (reDetect) {
          candidate = detectShuttle(flowPt);
        } else {
          candidate = flowPt;
        }
        nextPts.delete(); status.delete(); err.delete();
      } else {
        candidate = detectShuttle(null);
      }
    }

    // YOLO re-lock periodically or when no candidate
    const needYolo = (typeof els.useYolo !== 'undefined') && els.useYolo && els.useYolo.checked && (!candidate || (trackHistory.length && (trackHistory.length % 8 === 0)));
    const afterYoloPromise = (yoloSession && needYolo) ? yoloDetectCenter(imageData).then(pt => pt || candidate) : Promise.resolve(candidate);

    Promise.resolve(afterYoloPromise).then(candidate2 => {
      if (candidate2) {
        if (prevPtMat && !prevPtMat.isDeleted()) prevPtMat.delete();
        prevPtMat = cv.matFromArray(1, 1, cv.CV_32FC2, new Float32Array([candidate2.x, candidate2.y]));
        havePrevPoint = true;

        const metric = applyHomography(candidate2.x, candidate2.y);
        if (metric) {
          if (!gh) gh = ghCreate(metric.x, metric.y);
          const dt = lastTime != null ? Math.max(1e-3, t - lastTime) : 1/60;
          ghUpdate(gh, metric, dt);
          trackHistory.push({ t, x: gh.x, y: gh.y });
        }
        drawOverlayFrame({ x: candidate2.x, y: candidate2.y }, t);
      } else {
        drawOverlayFrame(null, t);
      }

      frameGray.copyTo(prevGray);
      lastTime = t;
      if (useRVFC) video.requestVideoFrameCallback(handle); else setTimeout(()=>handle(performance.now(), { mediaTime: video.currentTime }), 0);
    });
  };

  const finalize = () => {
    // Cleanup mats
    frame.delete(); frameGray.delete(); prevGray.delete(); diff.delete(); kernel.delete();
    if (prevPtMat && !prevPtMat.isDeleted()) prevPtMat.delete();

    // FPS estimate and display
    if (tsForFps.length > 1) {
      const fps = (tsForFps.length - 1) / (tsForFps[tsForFps.length - 1] - tsForFps[0]);
      els.fps.textContent = isFinite(fps) ? fps.toFixed(1) + ' fps' : '—';
    }

    // Compute speeds and smooth
    const speeds = computeSpeedsMetersPerSecond(trackHistory);
    const win = [11,9,7,5].find(w => w <= speeds.length && (smoothWin%2?true:false));
    const smoothed = speeds.length ? (win ? savitzkyGolay(speeds, win) : movingAverage(speeds, Math.min(5, speeds.length))) : [];

    const peak = smoothed.length ? Math.max(...smoothed) : 0;
    const avg = smoothed.length ? smoothed.reduce((a,b)=>a+b,0) / smoothed.length : 0;
    els.peakSpeed.textContent = formatSpeed(peak);
    els.avgSpeed.textContent = formatSpeed(avg);

    // Physics-based initial speed (v0) estimate
    const phys = estimateInitialSpeedWithQuadraticDrag(trackHistory);
    if (els.initSpeed) {
      if (phys && isFinite(phys.v0)) els.initSpeed.textContent = formatSpeed(phys.v0);
      else els.initSpeed.textContent = '—';
    }

    updateChart(frameTimes, smoothed);

    cancelAnimationFrame(rafId);
    drawLoop();
  };

  if (typeof video.requestVideoFrameCallback === 'function') {
    video.requestVideoFrameCallback(handle);
  } else {
    // Fallback loop
    const loop = () => {
      if (video.ended) return finalize();
      handle(performance.now(), { mediaTime: video.currentTime });
    };
    setTimeout(loop, 0);
  }
});

els.exportCSV.addEventListener('click', () => {
  if (!frameTimes.length) return;
  const speeds = trackHistory.length ? computeSpeedsMetersPerSecond(trackHistory) : [];
  exportCSV(frameTimes, speeds);
});

els.exportPNG.addEventListener('click', exportOverlayPNG);

// Manual controls
function stepFrame(direction) {
  const fps = Math.max(24, Math.min(240, parseFloat((els.fps.textContent||'').split(' ')[0]) || 60));
  const step = 1 / fps;
  els.video.currentTime = Math.max(0, Math.min(els.video.duration, els.video.currentTime + (direction < 0 ? -step : step)));
  // redraw
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  const cw = els.canvas.width, ch = els.canvas.height;
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale, dh = vh * scale;
  const dx = (cw - dw) / 2, dy = (ch - dh) / 2;
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(els.video, dx, dy, dw, dh);
  drawCalibrationPoints();
}

els.prevFrame.addEventListener('click', () => stepFrame(-1));
els.nextFrame.addEventListener('click', () => stepFrame(1));

els.addMark.addEventListener('click', () => {
  // enable click to mark
  els.manualMode.checked = true;
});

els.finishManual.addEventListener('click', () => {
  if (!H) { alert('Calibrate first'); return; }
  if (!manualMarks.length) { alert('Add marks by clicking on the shuttle each frame.'); return; }
  // Convert marks to metric and compute speeds
  trackHistory = manualMarks.map(m => {
    const pt = applyHomography(m.xpx, m.ypx);
    return { t: m.t, x: pt?.x ?? 0, y: pt?.y ?? 0 };
  }).filter(h => h.t != null);
  const speeds = computeSpeedsMetersPerSecond(trackHistory);
  const smoothWin = Math.max(1, parseInt(els.smoothWin.value, 10) | 0);
  const smoothed = movingAverage(speeds, smoothWin);
  const peak = smoothed.length ? Math.max(...smoothed) : 0;
  const avg = smoothed.length ? smoothed.reduce((a,b)=>a+b,0) / smoothed.length : 0;
  els.peakSpeed.textContent = formatSpeed(peak);
  els.avgSpeed.textContent = formatSpeed(avg);
  updateChart(frameTimes.length ? frameTimes : manualMarks.map(m=>m.t), smoothed);
});

// Load YOLO model button
els.loadYolo && els.loadYolo.addEventListener('click', async () => {
  try {
    await loadYoloModel(els.yoloUrl.value.trim());
    alert('YOLO model loaded');
  } catch (e) {
    console.error(e);
    alert('Failed to load YOLO model');
  }
});


