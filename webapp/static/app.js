const $ = (id) => document.getElementById(id);
const state = { denoise: null, anc: null, demo: { denoise: null, primary: null, reference: null } };



function decodeBase64(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function audioBlobUrl(base64) {
  return URL.createObjectURL(new Blob([decodeBase64(base64)], { type: "audio/wav" }));
}

function db(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)} dB`;
}

function seconds(value) {
  if (!value && value !== 0) return "–";
  return `${value.toFixed(1)} s`;
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 4200);
}

function setBusy(button, busy, label) {
  if (busy) {
    button.dataset.label = button.textContent;
    button.disabled = true;
    button.innerHTML = `<span class="spinner"></span>${label}`;
  } else {
    button.disabled = false;
    button.textContent = button.dataset.label || button.textContent;
  }
}

function showError(id, message) {
  const el = $(id);
  el.textContent = message;
  el.hidden = false;
}

function clearError(id) {
  const el = $(id);
  el.textContent = "";
  el.hidden = true;
}

async function fetchDemo(kind, filename) {
  const response = await fetch(`/api/demo/${kind}`);
  if (!response.ok) throw new Error("Demo clip unavailable.");
  const blob = await response.blob();
  return new File([blob], filename, { type: "audio/wav" });
}

async function loadDemoDenoise() {
  state.demo.denoise = await fetchDemo("noisy", "demo-noisy.wav");
  $("audio-name").textContent = state.demo.denoise.name;
  await runDenoise();
}

async function loadDemoAnc() {
  const [primary, reference] = await Promise.all([
    fetchDemo("anc_primary", "demo-primary.wav"),
    fetchDemo("anc_reference", "demo-reference.wav"),
  ]);
  state.demo.primary = primary;
  state.demo.reference = reference;
  $("primary-name").textContent = primary.name;
  $("reference-name").textContent = reference.name;
  await runAnc();
}

async function postForm(url, formData) {
  const response = await fetch(url, { method: "POST", body: formData });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload;
}



function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
  const height = canvas.clientHeight || parseInt(canvas.getAttribute("height"), 10) || 200;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

const RAMP = [
  [255, 255, 255], [223, 232, 246], [166, 195, 234], [92, 144, 214],
  [29, 95, 191], [17, 55, 110], [8, 22, 46],
];

const LUT = (() => {
  const table = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const x = (i / 255) * (RAMP.length - 1);
    const low = Math.floor(x);
    const high = Math.min(low + 1, RAMP.length - 1);
    const f = x - low;
    for (let c = 0; c < 3; c += 1) {
      table[i * 3 + c] = RAMP[low][c] + (RAMP[high][c] - RAMP[low][c]) * f;
    }
  }
  return table;
})();

const INK = "#0a0a0a";
const MUTED = "#6b6b6b";
const RULE = "#f0f0f0";
const EDGE = "#e6e6e6";

function drawSpectrogram(canvas, payload) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  if (!payload) return;

  const padL = 30;
  const padB = 18;
  const padT = 6;
  const padR = 4;
  const plotW = Math.max(1, width - padL - padR);
  const plotH = Math.max(1, height - padT - padB);

  const bytes = decodeBase64(payload.data);
  const offscreen = document.createElement("canvas");
  offscreen.width = payload.width;
  offscreen.height = payload.height;
  const offCtx = offscreen.getContext("2d");
  const image = offCtx.createImageData(payload.width, payload.height);

  for (let y = 0; y < payload.height; y += 1) {
    const sourceRow = payload.height - 1 - y;
    for (let x = 0; x < payload.width; x += 1) {
      const value = bytes[sourceRow * payload.width + x];
      const out = (y * payload.width + x) * 4;
      image.data[out] = LUT[value * 3];
      image.data[out + 1] = LUT[value * 3 + 1];
      image.data[out + 2] = LUT[value * 3 + 2];
      image.data[out + 3] = 255;
    }
  }
  offCtx.putImageData(image, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(offscreen, padL, padT, plotW, plotH);
  ctx.strokeStyle = EDGE;
  ctx.lineWidth = 1;
  ctx.strokeRect(padL + 0.5, padT + 0.5, plotW - 1, plotH - 1);

  ctx.font = '11px "Geist", system-ui, sans-serif';
  ctx.fillStyle = MUTED;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const maxKhz = payload.max_frequency / 1000;
  [0, 0.5, 1].forEach((frac) => {
    const label = frac === 0 ? "0" : `${Math.round(maxKhz * frac)}k`;
    ctx.fillText(label, padL - 6, padT + plotH * (1 - frac));
  });

  ctx.textBaseline = "top";
  [0, 0.5, 1].forEach((frac, index) => {
    const x = padL + plotW * frac;
    const label = `${(payload.duration * frac).toFixed(frac === 0 ? 0 : 1)}s`;
    ctx.textAlign = index === 0 ? "left" : index === 1 ? "center" : "right";
    ctx.fillText(label, index === 2 ? width - 2 : x, padT + plotH + 5);
  });
  ctx.textAlign = "start";
}

function drawWaveform(canvas, values, color) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const mid = height / 2;
  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid + 0.5);
  ctx.lineTo(width, mid + 0.5);
  ctx.stroke();

  if (!values || values.length === 0) return;

  ctx.beginPath();
  ctx.moveTo(0, mid);
  for (let i = 0; i < values.length; i += 1) {
    const x = (i / (values.length - 1 || 1)) * width;
    ctx.lineTo(x, mid - values[i] * (mid - 3));
  }
  for (let i = values.length - 1; i >= 0; i -= 1) {
    const x = (i / (values.length - 1 || 1)) * width;
    ctx.lineTo(x, mid + values[i] * (mid - 3));
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function drawErleChart(canvas, series) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const pad = { left: 40, right: 18, top: 10, bottom: 26 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  let maxLen = 1;
  let minY = 0;
  let maxY = 30;
  series.forEach((s) => {
    maxLen = Math.max(maxLen, s.values.length);
    const numeric = s.values.filter((v) => v !== null && v !== undefined);
    if (numeric.length) {
      minY = Math.min(minY, Math.min(...numeric));
      maxY = Math.max(maxY, Math.max(...numeric));
    }
  });
  minY = Math.floor(Math.max(-40, minY) / 10) * 10;
  maxY = Math.ceil(Math.min(70, maxY) / 10) * 10;

  const xOf = (i) => pad.left + (i / Math.max(1, maxLen - 1)) * plotW;
  const yOf = (v) => pad.top + (1 - (v - minY) / (maxY - minY || 1)) * plotH;

  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.font = '11px "Geist", system-ui, sans-serif';
  ctx.fillStyle = MUTED;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let value = minY; value <= maxY; value += 10) {
    const y = Math.round(yOf(value)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(`${value}`, pad.left - 6, y);
  }

  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#d4d4d4";
  const refY = Math.round(yOf(20)) + 0.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, refY);
  ctx.lineTo(width - pad.right, refY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText("20 dB", width - pad.right, refY - 3);

  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("0", pad.left, pad.top + plotH + 6);
  ctx.textAlign = "right";
  ctx.fillText(`${maxLen} blocks`, width - pad.right, pad.top + plotH + 6);

  series.forEach((s) => {
    if (!s.values.length) return;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let pen = false;
    s.values.forEach((value, i) => {
      if (value === null || value === undefined) {
        pen = false;
        return;
      }
      const x = xOf(i);
      const y = yOf(value);
      if (!pen) {
        ctx.moveTo(x, y);
        pen = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

function bindDropzone(zoneId, inputId, nameId) {
  const zone = $(zoneId);
  const input = $(inputId);

  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dragover");
    if (event.dataTransfer.files.length) {
      input.files = event.dataTransfer.files;
      input.dispatchEvent(new Event("change"));
    }
  });
  input.addEventListener("change", () => {
    const file = input.files[0];
    $(nameId).textContent = file ? file.name : "no file selected";
  });
}

function bindSlider(id, outId, digits = 2) {
  const input = $(id);
  const out = $(outId);
  const update = () => {
    const value = parseFloat(input.value);
    out.textContent = digits === 0 ? String(Math.round(value)) : value.toFixed(digits);
  };
  input.addEventListener("input", update);
  update();
}

const PANEL_COPY = {
  denoise: {
    title: "Single-channel denoise",
    sub: "Remove noise from one recording. The noise spectrum is learned from the pauses between words.",
  },
  anc: {
    title: "Two-channel adaptive noise cancellation",
    sub: "One microphone hears the noise, the other hears speech and noise. The filter learns the path between them.",
  },
};

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  });

  const copy = PANEL_COPY[name];
  if (copy) {
    $("page-title").textContent = copy.title;
    $("page-sub").textContent = copy.sub;
  }
  $("run-denoise").hidden = name !== "denoise";
  $("run-anc").hidden = name !== "anc";
  requestAnimationFrame(redraw);
}

let resizeTimer = null;
function redraw() {
  if (state.denoise) {
    const key = state.denoise.waveform.current;
    drawWaveform($("waveform"), state.denoise.waveform[key], state.denoise.waveform.color);
    drawSpectrogram($("spec-input"), state.denoise.spectrogram.input);
    drawSpectrogram($("spec-output"), state.denoise.spectrogram.output);
  }
  if (state.anc) drawErleChart($("erle-chart"), state.anc.series);
}

window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(redraw, 120);
});



function setStat(id, text, tone) {
  const el = $(id);
  el.textContent = text;
  el.classList.remove("good", "bad");
  el.classList.toggle("blank", text === "–" || text === "Needs reference");
  if (tone) el.classList.add(tone);
}

function playDenoise(source) {
  const audio = $("denoise-audio");
  const wasPlaying = !audio.paused;
  audio.src = source === "input" ? state.denoise.inputUrl : state.denoise.outputUrl;
  if (wasPlaying) audio.play();
  document.querySelectorAll("#panel-denoise .seg-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.source === source);
  });
  state.denoise.current = source;
  drawWaveform(
    $("waveform"),
    source === "input" ? state.denoise.waveform.input : state.denoise.waveform.output,
    source === "input" ? "#9a9a9a" : "#0a0a0a",
  );
}

async function runDenoise() {
  const file = $("file-audio").files[0] || state.demo.denoise;
  clearError("denoise-error");
  if (!file) {
    showError("denoise-error", "Choose an audio file first.");
    return;
  }

  const button = $("run-denoise");
  setBusy(button, true, "Filtering…");

  const form = new FormData();
  form.append("audio", file);
  const reference = $("file-reference").files[0];
  if (reference) form.append("reference", reference);
  form.append("settings", JSON.stringify({
    method: $("method").value,
    gain_floor: parseFloat($("gain-floor").value),
    dd: parseFloat($("dd").value),
    noise_alpha: parseFloat($("alpha").value),
    use_notch: $("use-notch").checked,
    notch_harmonics: parseInt($("harmonics").value, 10),
    use_ale: $("use-ale").checked,
  }));

  try {
    const data = await postForm("/api/denoise", form);
    if (state.denoise) {
      URL.revokeObjectURL(state.denoise.inputUrl);
      URL.revokeObjectURL(state.denoise.outputUrl);
    }

    state.denoise = {
      ...data,
      inputUrl: audioBlobUrl(data.input_audio),
      outputUrl: audioBlobUrl(data.output_audio),
      waveform: { input: data.waveform.input, output: data.waveform.output, current: "output", color: "#0a0a0a" },
      current: "output",
    };

    $("denoise-empty").hidden = true;
    $("denoise-stats").hidden = false;
    $("denoise-player").hidden = false;
    $("denoise-viz").hidden = false;

    const audio = $("denoise-audio");
    audio.src = state.denoise.outputUrl;
    $("download-denoised").href = state.denoise.outputUrl;

    document.querySelectorAll("#panel-denoise .seg-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.source === "output");
    });

    const floor = data.metrics.noise_floor_reduction_db;
    setStat("stat-floor", db(floor), floor > 3 ? "good" : null);

    if (data.metrics.has_reference) {
      const delta = data.metrics.delta_si_sdr;
      setStat("stat-sisdr", db(delta), delta > 0 ? "good" : "bad");
      setStat("stat-segsnr", `${data.metrics.denoised_seg_snr.toFixed(1)} dB`, null);
    } else {
      setStat("stat-sisdr", "Needs reference", null);
      setStat("stat-segsnr", "Needs reference", null);
    }

    setStat("stat-duration", seconds(data.duration), null);

    const notes = [...data.messages];
    if (data.config.notch_applied) notes.push("Mains hum detected and notched.");
    if (notes.length) toast(notes.join(" "));

    requestAnimationFrame(redraw);
  } catch (error) {
    showError("denoise-error", error.message);
  } finally {
    setBusy(button, false);
  }
}



function playAnc(source) {
  const audio = $("anc-audio");
  const wasPlaying = !audio.paused;
  const filter = $("anc-filter").value;
  const result = state.anc.results[filter];

  if (source === "input") {
    audio.src = state.anc.inputUrl;
  } else if (result && result.audio) {
    audio.src = audioBlobUrl(result.audio);
  }
  if (wasPlaying) audio.play();

  document.querySelectorAll("#panel-anc .seg-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.source === source);
  });
}

async function runAnc() {
  const primary = $("file-primary").files[0] || state.demo.primary;
  const reference = $("file-reference-anc").files[0] || state.demo.reference;
  clearError("anc-error");

  if (!primary || !reference) {
    showError("anc-error", "Both a primary and a reference recording are required.");
    return;
  }

  const filters = [];
  if ($("run-lms").checked) filters.push("lms");
  if ($("run-nlms").checked) filters.push("nlms");
  if ($("run-rls").checked) filters.push("rls");
  if (!filters.length) {
    showError("anc-error", "Select at least one adaptive filter.");
    return;
  }

  const button = $("run-anc");
  setBusy(button, true, "Adapting…");

  const form = new FormData();
  form.append("primary", primary);
  form.append("reference", reference);
  form.append("settings", JSON.stringify({
    order: parseInt($("order").value, 10),
    mu_lms: parseFloat($("mu-lms").value),
    mu_nlms: parseFloat($("mu-nlms").value),
    lambda_rls: parseFloat($("lambda").value),
    align: $("align").checked,
    filters,
  }));

  try {
    const data = await postForm("/api/anc", form);
    if (state.anc) URL.revokeObjectURL(state.anc.inputUrl);

    const urls = {};
    Object.entries(data.results).forEach(([label, result]) => {
      if (result.audio) urls[label] = audioBlobUrl(result.audio);
    });

    state.anc = {
      ...data,
      inputUrl: audioBlobUrl(data.input_audio),
      urls,
      series: [],
    };

    $("anc-empty").hidden = true;
    $("anc-player").hidden = false;
    $("anc-viz").hidden = false;

    const select = $("anc-filter");
    select.innerHTML = "";
    const colours = { LMS: "#141414", NLMS: "#8a8a8a", RLS: "#1d5fbf" };
    const series = [];

    Object.entries(data.results).forEach(([label, result]) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = result.error ? `${label} (failed)` : label;
      option.disabled = Boolean(result.error);
      select.appendChild(option);

      if (!result.error) {
        series.push({
          name: label,
          color: colours[label] || "#1d5fbf",
          values: result.erle_curve || [],
        });
      }
    });

    const preferred = data.best;
    if (data.results[preferred] && data.results[preferred].audio) select.value = preferred;

    state.anc.series = series;
    state.anc.results = data.results;

    const audio = $("anc-audio");
    const chosen = select.value;
    const chosenResult = data.results[chosen];
    if (chosenResult && urls[chosen]) audio.src = urls[chosen];

    $("anc-lag").textContent = `${data.lag_ms.toFixed(1)} ms`;
    setStat("anc-order", `${data.order} taps`, null);
    if (chosenResult) {
      setStat("anc-erle", db(chosenResult.erle_db), chosenResult.erle_db > 3 ? "good" : "bad");
      setStat(
        "anc-conv",
        chosenResult.conv_block === null || chosenResult.conv_block === undefined
          ? "not reached"
          : `block ${chosenResult.conv_block}`,
        null,
      );
    }

    document.querySelectorAll("#panel-anc .seg-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.source === "output");
    });

    const failed = Object.entries(data.results)
      .filter(([, result]) => result.error)
      .map(([label, result]) => `${label}: ${result.error}`);
    if (failed.length) toast(failed.join(" "));

    requestAnimationFrame(() => drawErleChart($("erle-chart"), series));
  } catch (error) {
    showError("anc-error", error.message);
  } finally {
    setBusy(button, false);
  }
}



document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

bindDropzone("drop-audio", "file-audio", "audio-name");
bindDropzone("drop-primary", "file-primary", "primary-name");
bindDropzone("drop-reference", "file-reference-anc", "reference-name");

bindSlider("gain-floor", "gain-floor-out", 2);
bindSlider("dd", "dd-out", 2);
bindSlider("alpha", "alpha-out", 3);
bindSlider("harmonics", "harmonics-out", 0);
bindSlider("order", "order-out", 0);
bindSlider("mu-lms", "mu-lms-out", 4);
bindSlider("mu-nlms", "mu-nlms-out", 2);
bindSlider("lambda", "lambda-out", 3);

$("run-denoise").addEventListener("click", runDenoise);
$("run-anc").addEventListener("click", runAnc);

$("demo-denoise").addEventListener("click", () => {
  loadDemoDenoise().catch((error) => showError("denoise-error", error.message));
});
$("demo-anc").addEventListener("click", () => {
  loadDemoAnc().catch((error) => showError("anc-error", error.message));
});

document.querySelectorAll("#panel-denoise .seg-btn").forEach((button) => {
  button.addEventListener("click", () => playDenoise(button.dataset.source));
});
document.querySelectorAll("#panel-anc .seg-btn").forEach((button) => {
  button.addEventListener("click", () => playAnc(button.dataset.source));
});

$("anc-filter").addEventListener("change", () => {
  const filter = $("anc-filter").value;
  const result = state.anc && state.anc.results[filter];
  if (!result || result.error) return;
  if (state.anc.urls[filter]) $("anc-audio").src = state.anc.urls[filter];
  setStat("anc-erle", db(result.erle_db), result.erle_db > 3 ? "good" : "bad");
  setStat("anc-conv", result.conv_block === null ? "not reached" : `block ${result.conv_block}`, null);
});

const params = new URLSearchParams(window.location.search);
if (params.get("demo") === "anc") {
  switchTab("anc");
  loadDemoAnc().catch((error) => showError("anc-error", error.message));
} else if (params.get("demo") === "1") {
  loadDemoDenoise().catch((error) => showError("denoise-error", error.message));
}
