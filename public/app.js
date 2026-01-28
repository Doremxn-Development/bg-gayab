const upload = document.getElementById("upload");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

// IMPORTANT: vercel path (no /public)
const MODEL_PATH = "/model/u2net.onnx";
const SIZE = 320;

let session;

// Load U2Net model
(async () => {
  statusText.innerText = "Loading AI model...";
  session = await ort.InferenceSession.create(MODEL_PATH);
  statusText.innerText = "Model loaded ✔";

  console.log("Inputs:", session.inputNames);
  console.log("Outputs:", session.outputNames);
})();

upload.addEventListener("change", async (e) => {
  if (!session) return;

  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();

  canvas.width = SIZE;
  canvas.height = SIZE;
  ctx.drawImage(img, 0, 0, SIZE, SIZE);

  const imageData = ctx.getImageData(0, 0, SIZE, SIZE);

  // Preprocess (RGB, CHW)
  const input = new Float32Array(1 * 3 * SIZE * SIZE);
  let r = 0;
  let g = SIZE * SIZE;
  let b = SIZE * SIZE * 2;

  for (let i = 0; i < imageData.data.length; i += 4) {
    input[r++] = imageData.data[i] / 255;
    input[g++] = imageData.data[i + 1] / 255;
    input[b++] = imageData.data[i + 2] / 255;
  }

  statusText.innerText = "Removing background...";

  const tensor = new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);

  // Correct input name (dynamic)
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;

  const result = await session.run(feeds);
  const mask = result[session.outputNames[0]].data;

  // Apply alpha mask
  for (let i = 0; i < imageData.data.length; i += 4) {
    let alpha = mask[i / 4];
    alpha = Math.min(1, Math.max(0, alpha));
    alpha = Math.pow(alpha, 1.2); // smoothing
    imageData.data[i + 3] = alpha * 255;
  }

  ctx.putImageData(imageData, 0, 0);
  statusText.innerText = "Done ✔ Right-click → Save image";
});
