const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const planoInfo = document.getElementById('plano');
const imagemProcessada = document.getElementById('frame-processado');

// Inicializa a câmera
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();

    video.addEventListener('loadeddata', () => {
      processarFrameContinuamente(video);
    });
  })
  .catch(err => {
    console.error("Erro ao acessar a câmera:", err);
  });

// Envia 1 frame por vez, aguarda resposta antes de enviar o próximo
async function processarFrameContinuamente(video) {
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!blob) return;

    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    try {
      const res = await fetch('/video_feed', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (data.plano) {
        planoInfo.innerText = "Plano: " + data.plano;
      } else {
        planoInfo.innerText = "Plano: não detectado";
      }

      if (data.frame_base64) {
        imagemProcessada.src = 'data:image/jpeg;base64,' + data.frame_base64;
      }
    } catch (err) {
      console.error("Erro ao enviar frame:", err);
    }

    // Aguarda antes de continuar
    setTimeout(() => processarFrameContinuamente(video), 200); // ~5 FPS
  }, 'image/jpeg', 0.7); // compressão média
}
