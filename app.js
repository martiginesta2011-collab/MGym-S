// AI Gym Coach - Prototipo (español)
// Requiere: tf.min.js y @tensorflow-models/pose-detection (incluir via CDN en index.html)

let video = document.getElementById('video');
let canvas = document.getElementById('overlay');
let ctx = canvas.getContext('2d');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const exerciseSelect = document.getElementById('exercise');

const statusEl = document.getElementById('status');
const anglesEl = document.getElementById('angles');
const repsEl = document.getElementById('reps');
const feedbackEl = document.getElementById('feedback');

let detector = null;
let rafId = null;
let running = false;
let detectionInterval = 100; // ms entre inferencias
let lastDetection = 0;

// Estado para conteo de sentadillas
let squatState = 'top'; // 'top' | 'bottom'
let squatReps = 0;

// Ajustes/umbrales (puedes modificar)
const SQUAT = {
  kneeTop: 150,   // ángulo rodilla cuando está arriba (> = top)
  kneeBottom: 100 // ángulo rodilla cuando está abajo (<= bottom)
};
const PLANK = {
  torsoMinAngle: 160 // hombro-cadera-tobillo (cerca de 180 = recto)
};

// Conexiones para dibujar esqueleto (COCO-ish)
const connections = [
  [0,1],[0,2],[1,3],[2,4],
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [11,12],[5,11],[6,12],[11,13],[13,15],[12,14],[14,16]
];

// Cargar la cámara
async function setupCamera(){
  console.log('setupCamera: solicitando cámara');
  const stream = await navigator.mediaDevices.getUserMedia({
    audio:false,
    video:{facingMode:'user'}
  });
  video.srcObject = stream;
  await video.play();

  // ajustar canvas al video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  video.width = video.videoWidth;
  video.height = video.videoHeight;
  console.log('setupCamera: cámara lista', video.videoWidth, video.videoHeight);
}

// Crear detector MoveNet
async function createDetector(){
  statusEl.textContent = 'Cargando modelo...';
  console.log('createDetector: cargando modelo MoveNet');
  const model = poseDetection.SupportedModels.MoveNet;
  const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
  detector = await poseDetection.createDetector(model, detectorConfig);
  statusEl.textContent = 'Modelo cargado';
  console.log('createDetector: detector listo');
}

// Calcular ángulo entre tres puntos A-B-C (B articulación central). Coordenadas en pixel.
function angleBetween(A,B,C){
  const BAx = A.x - B.x, BAy = A.y - B.y;
  const BCx = C.x - B.x, BCy = C.y - B.y;
  const dot = BAx*BCx + BAy*BCy;
  const magBA = Math.hypot(BAx, BAy);
  const magBC = Math.hypot(BCx, BCy);
  if(magBA === 0 || magBC === 0) return null;
  const cos = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
  return Math.acos(cos) * (180/Math.PI);
}

// Dibuja keypoints y líneas
function drawPose(keypoints){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // Dibujar conexiones
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#00C853';
  for(const [i,j] of connections){
    const a = keypoints[i], b = keypoints[j];
    if(a && b && a.score > 0.25 && b.score > 0.25){
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }

  // Dibujar puntos
  for(const p of keypoints){
    if(!p) continue;
    if(p.score > 0.25){
      ctx.beginPath();
      ctx.fillStyle = '#00C853';
      ctx.arc(p.x, p.y, 4, 0, Math.PI*2);
      ctx.fill();
    }
  }
}

// Obtener keypoint por nombre si existe (pose.keypoints tiene 'name' en MoveNet)
function kpByName(keypoints, name){
  return keypoints.find(k=>k.name===name) || null;
}

// Procesar pose y calcular ángulos/feedback
function processPose(pose){
  if(!pose || !pose.keypoints) return;

  const k = pose.keypoints;

  const leftShoulder = kpByName(k,'left_shoulder');
  const rightShoulder = kpByName(k,'right_shoulder');
  const leftHip = kpByName(k,'left_hip');
  const rightHip = kpByName(k,'right_hip');
  const leftKnee = kpByName(k,'left_knee');
  const rightKnee = kpByName(k,'right_knee');
  const leftAnkle = kpByName(k,'left_ankle');
  const rightAnkle = kpByName(k,'right_ankle');

  // Calcular ángulos
  const leftKneeAngle = (leftHip && leftKnee && leftAnkle) ? angleBetween(leftHip,leftKnee,leftAnkle) : null;
  const rightKneeAngle = (rightHip && rightKnee && rightAnkle) ? angleBetween(rightHip,rightKnee,rightAnkle) : null;

  const leftHipAngle = (leftShoulder && leftHip && leftKnee) ? angleBetween(leftShoulder,leftHip,leftKnee) : null;
  const rightHipAngle = (rightShoulder && rightHip && rightKnee) ? angleBetween(rightShoulder,rightHip,rightKnee) : null;

  // Torso angle for plank: shoulder - hip - ankle (average sides)
  let torsoAngle = null;
  if(leftShoulder && leftHip && leftAnkle){
    torsoAngle = angleBetween(leftShoulder,leftHip,leftAnkle);
  } else if(rightShoulder && rightHip && rightAnkle){
    torsoAngle = angleBetween(rightShoulder,rightHip,rightAnkle);
  }

  // Actualizar UI de ángulos
  anglesEl.children[0].textContent = `Rodilla izq: ${leftKneeAngle ? leftKneeAngle.toFixed(0)+'°' : '—'}`;
  anglesEl.children[1].textContent = `Rodilla der: ${rightKneeAngle ? rightKneeAngle.toFixed(0)+'°' : '—'}`;
  anglesEl.children[2].textContent = `Cadera izq: ${leftHipAngle ? leftHipAngle.toFixed(0)+'°' : '—'}`;
  anglesEl.children[3].textContent = `Cadera der: ${rightHipAngle ? rightHipAngle.toFixed(0)+'°' : '—'}`;
  anglesEl.children[4].textContent = `Torso (plank): ${torsoAngle ? torsoAngle.toFixed(0)+'°' : '—'}`;

  // Lógica por ejercicio
  const exercise = exerciseSelect.value;
  if(exercise === 'squat'){
    const kneeAvg = (() => {
      if(leftKneeAngle && rightKneeAngle) return (leftKneeAngle + rightKneeAngle)/2;
      return leftKneeAngle || rightKneeAngle || null;
    })();

    if(kneeAvg){
      if(squatState === 'top' && kneeAvg <= SQUAT.kneeBottom){
        squatState = 'bottom';
        statusEl.textContent = 'Abajo (bottom)';
      } else if(squatState === 'bottom' && kneeAvg >= SQUAT.kneeTop){
        squatState = 'top';
        squatReps += 1;
        repsEl.textContent = squatReps;
        statusEl.textContent = 'Arriba (top)';
      }
    }

    let fb = [];
    if(kneeAvg){
      if(kneeAvg > 160) fb.push('Extiende piernas completamente (arriba).');
      else if(kneeAvg < 80) fb.push('Profundidad alta — controla la espalda.');
    } else {
      fb.push('Posición no detectada con suficiente confianza.');
    }

    const hipAvg = (leftHipAngle && rightHipAngle) ? (leftHipAngle + rightHipAngle)/2 : (leftHipAngle || rightHipAngle || null);
    if(hipAvg && hipAvg < 70) fb.push('Flexión de cadera excesiva: cuida la espalda.');

    feedbackEl.textContent = fb.join(' ');
  } else if(exercise === 'plank'){
    let fb = [];
    if(torsoAngle){
      if(torsoAngle >= PLANK.torsoMinAngle){
        statusEl.textContent = 'Buena línea';
        fb.push('Mantén línea del cuerpo recta.');
      } else {
        statusEl.textContent = 'Cadera baja/alta';
        fb.push('Ajusta la cadera para alinear hombros-cadera-tobillos.');
      }
    } else {
      statusEl.textContent = 'Posición no detectada';
      fb.push('Acércate a la cámara y asegúrate de que se vean hombros, caderas y tobillos.');
    }
    feedbackEl.textContent = fb.join(' ');
  }
}

// Loop principal: detectar y dibujar
async function renderLoop(){
  if(!running) return;
  const now = performance.now();
  if(now - lastDetection >= detectionInterval){
    lastDetection = now;
    try{
      const poses = await detector.estimatePoses(video, {flipHorizontal: true});
      const pose = poses && poses[0] ? poses[0] : null;
      if(pose) drawPose(pose.keypoints);
      processPose(pose);
    }catch(e){
      console.error('Error en estimación de poses', e);
    }
  }
  rafId = requestAnimationFrame(renderLoop);
}

// Start / Stop lógica
async function start(){
  startBtn.disabled = true;
  stopBtn.disabled = false;
  try{
    if(!video.srcObject){
      await setupCamera();
    }
    if(!detector){
      await createDetector();
    }
    running = true;
    squatReps = 0;
    repsEl.textContent = '0';
    statusEl.textContent = 'En ejecución';
    lastDetection = 0;
    console.log('start: iniciando bucle');
    renderLoop();
  }catch(err){
    console.error('Error al iniciar:', err);
    statusEl.textContent = 'Error al iniciar, mira la consola';
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function stop(){
  startBtn.disabled = false;
  stopBtn.disabled = true;
  running = false;
  statusEl.textContent = 'Detenido';
  if(rafId) cancelAnimationFrame(rafId);
  if(video.srcObject){
    const tracks = video.srcObject.getTracks();
    tracks.forEach(t=>t.stop());
    video.srcObject = null;
  }
  console.log('stop: detenido');
}

// Eventos UI
startBtn.addEventListener('click', ()=>{ start().catch(console.error); });
stopBtn.addEventListener('click', stop);

// ajustar canvas si cambia tamaño del video
video.addEventListener('loadeddata', ()=>{
  if(video.videoWidth){
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
});

window.addEventListener('resize', ()=>{
  if(video.videoWidth){
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
});
