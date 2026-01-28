const upload = document.getElementById("upload");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

const MODEL_PATH = "/public/model/u2net.onnx";
const SIZE = 512;

let session;

async function loadModel() {
  try {
    statusText.innerText = "Loading AI model...";
    session = await ort.InferenceSession.create(MODEL_PATH);
    statusText.innerText = "Model loaded ✔";
    console.log("Inputs:", session.inputNames);
    console.log("Outputs:", session.outputNames);
  } catch (e) {
    console.error(e);
    statusText.innerText = "Model load failed ❌";
  }
}
loadModel();

upload.addEventListener("change", async (e) => {
  if (!session) return;

  const file = e.target.files[0];
  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    canvas.width = SIZE;
    canvas.height = SIZE;
    ctx.drawImage(img, 0, 0, SIZE, SIZE);

    const imageData = ctx.getImageData(0, 0, SIZE, SIZE);

    // ===== PREPROCESS (channel-first RGB) =====
    const input = new Float32Array(1 * 3 * SIZE * SIZE);
    let r = 0, g = SIZE * SIZE, b = SIZE * SIZE * 2;

    for (let i = 0; i < imageData.data.length; i += 4) {
      input[r++] = imageData.data[i] / 255;
      input[g++] = imageData.data[i + 1] / 255;
      input[b++] = imageData.data[i + 2] / 255;
    }

    const tensor = new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
    statusText.innerText = "Removing background...";

    const result = await session.run({ input: tensor });
    const matte = result.output.data;

    // ===== POST PROCESS (SMOOTH + THRESHOLD) =====
    for (let i = 0; i < imageData.data.length; i += 4) {
      let alpha = matte[i / 4];

      // smooth
      alpha = Math.pow(alpha, 1.5);

      // threshold
      if (alpha < 0.2) alpha = 0;
      if (alpha > 0.95) alpha = 1;

      imageData.data[i + 3] = alpha * 255;
    }

    ctx.putImageData(imageData, 0, 0);
    statusText.innerText = "Done ✔ Right-click → Save image";
  };
});
