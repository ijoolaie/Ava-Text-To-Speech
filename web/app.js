// آوا — رابط کاربری دموی مرورگر؛ کد اپلیکیشن متعلق به همین پروژه است.
// مدل در همین صفحه با رانتایم WebAssembly «phonon-tts» اجرا می‌شود؛ گام‌های فارسی متن
// (نرمال‌سازی، جمله‌بندی، شمارش توکن، زمان‌بندی کلمات) از ./persian.js می‌آیند.
import * as P from "./persian.js";

const $ = (id) => document.getElementById(id);
const SR = 24000;
const SAMPLES = [
  ["خبر", "بر اساس گزارش سازمان هواشناسی، بارش باران و برف در بیشتر استان‌های شمالی و غربی کشور تا پایان هفته ادامه خواهد داشت. از رانندگان خواسته شده است که در جاده‌های کوهستانی با احتیاط بیشتری حرکت کنند."],
  ["داستان", "پیرمرد از پنجره‌ی کوچک اتاق به کوچه نگاه کرد. باران ریزی می‌بارید و چراغ‌های زرد خیابان روی آسفالت خیس می‌درخشیدند."],
  ["اعداد", "قیمت این کتاب ۲۵۰ هزار تومان است و ۱۲ درصد تخفیف دارد."],
  ["پرسش", "آیا می‌دانید امروز هوا چگونه خواهد بود؟"],
];
const VOICE_LABELS = { nava: "پاریسا", shirin: "shirin (زن)", arash: "arash (مرد)" };
const MODEL_CACHE = "phonon-tts-v1"; // نام حافظهٔ پکیج phonon-tts (fetch.js)

window.__avaStarted = true;
document.getElementById("loadfail")?.setAttribute("hidden", "");

// شمارش درخواست‌های صفحه برای کارت «شبکه» (ورکرِ مدل حین تولید چیزی نمی‌گیرد)
let netCount = 0;
const _fetch = window.fetch.bind(window);
window.fetch = (...a) => { netCount++; return _fetch(...a); };

let busy = false, abort = false, tts = null, ttsLoad = null, audioCtx = null, counter = null, objectUrl = null, timingModel = null;

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* حالت خصوصی */ } },
};

function setStatus(msg, isErr = false) { $("status").textContent = msg; $("status").classList.toggle("err", isErr); }
function setBusy(b) { busy = b; $("stop").hidden = !b; updateGo(); }
// «بگو» تا مدل آماده نشود غیرفعال است؛ مدل هیچ‌وقت بدون کلیک دانلود نمی‌شود
function updateGo() { $("go").disabled = busy || !tts; }
function stat(label, value, caption) {
  const d = document.createElement("div"); d.className = "stat";
  d.innerHTML = "<span class=\"l\"></span><b></b><span class=\"c\"></span>";
  d.children[0].textContent = label; d.children[1].textContent = value; d.children[2].textContent = caption || "";
  return d;
}
function showStats(items) { $("stats").replaceChildren(...items.map(([l, v, c]) => stat(l, v, c))); }
function showAudio(blob) {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(blob);
  $("player").src = objectUrl; $("dl").href = objectUrl; $("dl").hidden = false; $("out").hidden = false;
}
const ms = (x) => (x < 1 ? `${Math.round(x * 1000)} میلی‌ثانیه` : `${x.toFixed(2)} ثانیه`);
const sec = (x) => `${x.toFixed(x < 10 ? 2 : 1)} ثانیه`;

// ---- زیرنویس: جمله‌های گفته‌شده زیر پخش‌کننده، با هایلایت کلمهٔ جاری ----
// زمان شروع کلمه‌ها تخمینی است (persian.js: wordTimings). دو ساعت داریم:
// «live» هنگام پخش قطعه‌به‌قطعه از AudioContext می‌خواند، «final» بعد از اتمام از پخش‌کننده.
class Captions {
  constructor(root) { this.root = root; this.words = []; this.mode = "final"; this.idx = -2; this.sent = -1; this.endLive = 0; this.endFinal = 0; this.loop = 0; }
  render(texts) {
    this.root.replaceChildren(); this.words = []; this.idx = -2; this.sent = -1; this.endLive = 0; this.endFinal = 0;
    texts.forEach((t, si) => {
      const p = document.createElement("span"); p.className = "s";
      t.split(/\s+/).filter(Boolean).forEach((w) => {
        const el = document.createElement("span"); el.className = "w"; el.textContent = w;
        const rec = { el, si, live: NaN, final: NaN };
        el.addEventListener("click", () => this.seek(rec));
        p.append(el, " "); this.words.push(rec);
      });
      this.root.append(p);
    });
    this.root.hidden = false; $("capnote").hidden = false; this.root.scrollTop = 0;
  }
  // starts: ثانیه از ابتدای صدای قطعه؛ base: جای صدای قطعه روی ساعت
  set(si, starts, base, key, end) {
    const mine = this.words.filter((w) => w.si === si);
    mine.forEach((w, i) => { w[key] = base + (starts[i] ?? starts[starts.length - 1] ?? 0); });
    if (key === "live") this.endLive = Math.max(this.endLive, base + end); else this.endFinal = Math.max(this.endFinal, base + end);
  }
  seek(rec) {
    if (this.mode !== "final" || Number.isNaN(rec.final)) return;
    const player = $("player");
    player.currentTime = Math.max(0, rec.final - 0.05); player.play().catch(() => {});
  }
  time() {
    if (this.mode === "live") return audioCtx && audioCtx.state === "running" ? audioCtx.currentTime : -1;
    return $("player").currentTime;
  }
  tick() {
    const key = this.mode, t = this.time(), end = key === "live" ? this.endLive : this.endFinal;
    let idx = -1;
    this.words.forEach((w, i) => { if (!Number.isNaN(w[key]) && w[key] <= t) idx = i; });
    if (idx === this.words.length - 1 && t > end + 0.1) idx = this.words.length; // همه گفته شده
    if (idx === this.idx) return;
    this.idx = idx;
    this.words.forEach((w, i) => { w.el.classList.toggle("done", i < idx); w.el.classList.toggle("now", i === idx); });
    const si = idx >= 0 && idx < this.words.length ? this.words[idx].si : -1;
    if (si !== this.sent) {
      this.sent = si;
      [...this.root.children].forEach((c, k) => c.classList.toggle("cur", k === si));
      if (si >= 0) { const c = this.root.children[si]; this.root.scrollTop = Math.max(0, c.offsetTop - this.root.offsetTop - this.root.clientHeight / 3); }
    }
  }
  run() { cancelAnimationFrame(this.loop); const step = () => { this.tick(); this.loop = requestAnimationFrame(step); }; step(); }
  clear() { cancelAnimationFrame(this.loop); this.root.hidden = true; $("capnote").hidden = true; this.root.replaceChildren(); this.words = []; }
}
const caps = new Captions($("caps"));

// ---- نمودار موج: میله‌های گفتار، بخش پخش‌شده پررنگ، کلیک برای پرش ----
class Waveform {
  constructor(canvas) {
    this.c = canvas; this.ctx = canvas.getContext("2d"); this.pcm = null; this.sr = SR; this.fine = null; this.max = 1; this.progress = 0;
    canvas.addEventListener("pointerdown", (e) => this.seek(e));
    new ResizeObserver(() => this.draw()).observe(canvas);
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => this.draw());
  }
  set(pcm, sr) {
    this.pcm = pcm; this.sr = sr;
    const M = 2400, seg = Math.max(1, Math.ceil(pcm.length / M)), fine = new Float32Array(Math.ceil(pcm.length / seg));
    for (let i = 0; i < fine.length; i++) { let m = 0; for (let j = i * seg, e = Math.min(pcm.length, j + seg); j < e; j++) { const a = Math.abs(pcm[j]); if (a > m) m = a; } fine[i] = m; }
    this.fine = fine; this.max = Math.max(1e-6, ...fine.subarray(0, Math.min(fine.length, 100000)));
    this.draw();
  }
  clear() { this.pcm = null; this.fine = null; this.progress = 0; this.draw(); }
  setProgress(p) { if (Math.abs(p - this.progress) > 0.0005) { this.progress = p; this.draw(); } }
  seek(e) {
    const player = $("player");
    if (caps.mode !== "final" || !player.duration) return;
    const r = this.c.getBoundingClientRect();
    player.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * player.duration;
    player.play().catch(() => {});
  }
  draw() {
    const dpr = window.devicePixelRatio || 1, w = this.c.clientWidth, h = this.c.clientHeight;
    if (!w || !h) return;
    if (this.c.width !== Math.round(w * dpr) || this.c.height !== Math.round(h * dpr)) { this.c.width = Math.round(w * dpr); this.c.height = Math.round(h * dpr); }
    const g = this.ctx; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    if (!this.fine) return;
    const color = getComputedStyle(this.c).getPropertyValue("--accent").trim() || "#b4532a";
    const bar = 3, gap = 2, n = Math.max(1, Math.floor((w + gap) / (bar + gap))), M = this.fine.length;
    g.fillStyle = color;
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (let k = Math.floor((i * M) / n), e = Math.max(k + 1, Math.floor(((i + 1) * M) / n)); k < e && k < M; k++) if (this.fine[k] > m) m = this.fine[k];
      const amp = Math.pow(m / this.max, 0.6), bh = Math.max(3, amp * (h - 10));
      // موج مثل پخش‌کنندهٔ صدا از چپ به راست رشد می‌کند، حتی در صفحهٔ RTL
      const x = i * (bar + gap), y = (h - bh) / 2;
      g.globalAlpha = (i + 0.5) / n <= this.progress ? 1 : 0.3;
      if (g.roundRect) { g.beginPath(); g.roundRect(x, y, bar, bh, bar / 2); g.fill(); } else g.fillRect(x, y, bar, bh);
    }
    g.globalAlpha = 1;
  }
}
const wave = new Waveform($("wave"));
let liveMap = [], loopId = 0, liveTimer = 0;
// زمان AudioContext پخش زنده، بر صدای نهایی نگاشت می‌شود (قطعه‌ها وقتی آماده‌اند زمان‌بندی می‌شوند و بین‌شان فاصله می‌افتد)
function liveToFinal(ct) { let t = 0; for (const m of liveMap) if (ct >= m.liveStart) t = m.finalStart + Math.min(ct - m.liveStart, m.dur); return t; }
function waveTick() {
  if (!wave.pcm) return;
  const total = wave.pcm.length / wave.sr;
  const t = caps.mode === "live" ? liveToFinal(audioCtx && audioCtx.state === "running" ? audioCtx.currentTime : -1) : $("player").currentTime || 0;
  wave.setProgress(total > 0 ? Math.min(1, t / total) : 0);
}
function startLoop() { cancelAnimationFrame(loopId); const step = () => { caps.tick(); waveTick(); loopId = requestAnimationFrame(step); }; step(); }

async function loadTimingModel() {
  if (timingModel !== null) return timingModel;
  try { timingModel = await (await fetch("timing_model.json")).json(); } catch { timingModel = false; }
  return timingModel;
}

const cat = (...parts) => { const out = new Float32Array(parts.reduce((n, a) => n + a.length, 0)); let o = 0; for (const a of parts) { out.set(a, o); o += a.length; } return out; };
// پرصداترین پنجرهٔ ۲۰ میلی‌ثانیه‌ای قطعه (همان آزمون سکوتِ trimSilence، قطعه‌به‌قطعه)
function loudest(c) {
  const win = Math.floor(0.02 * SR); let m = 0;
  for (let i = 0; i < c.length; i += win) { let q = 0; const e = Math.min(i + win, c.length); for (let j = i; j < e; j++) q += c[j] * c[j]; m = Math.max(m, Math.sqrt(q / (e - i))); }
  return m;
}

async function ensureBrowserModel() {
  if (tts) return tts;
  if (ttsLoad) return ttsLoad;
  ttsLoad = (async () => {
    if (!crossOriginIsolated) console.warn("صفحه cross-origin ایزوله نیست: تک‌رشته‌ای اجرا می‌شود");
    const { PhononTTS } = await import("./vendor/phonon-tts/index.js");
    const tok = await (await fetch("model/tokenizer.json")).json();
    counter = P.makeTokenCounter(tok);
    $("load").hidden = false; $("load").value = 0;
    const t0 = performance.now(), files = {};
    let fromCache = true;
    const mb = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0);
    setStatus("در حال بارگذاری مدل…");
    const voices = {};
    for (const v of (AVA.voices || ["nava", "shirin", "arash"])) voices[v] = `model/voices/${v}.safetensors`;
    const m = await PhononTTS.load({
      lang: "none", quant: "f32", device: "cpu", threads: "auto",
      model: {
        weights: { f32: "model/model.safetensors" },
        tokenizer: "model/tokenizer.json",
        config: "model/config.json",
        voices, defaultVoice: "shirin",
      },
      onProgress: ({ file, loaded, total, cached }) => {
        files[file] = { loaded, total: total || loaded };
        if (!cached) fromCache = false;
        let L = 0, T = 0;
        for (const f of Object.values(files)) { L += f.loaded; T += f.total; }
        $("load").value = T ? L / T : 0;
        setStatus(fromCache ? `بارگذاری مدل از حافظهٔ مرورگر: ${mb(L)} از ${mb(T)} مگابایت`
                            : `دانلود مدل: ${mb(L)} از ${mb(T)} مگابایت (فقط بار اول؛ مرورگر پس از آن نگهش می‌دارد)`);
      },
    });
    $("load").hidden = true;
    const size = Object.values(files).reduce((a, f) => a + f.total, 0);
    m.loadSeconds = (performance.now() - t0) / 1000;
    m.sizeMB = size / 1048576; m.fromCache = fromCache;
    setStatus(`مدل آماده است: ${mb(size)} مگابایت ${fromCache ? "از حافظهٔ مرورگر" : "دانلود شد و در مرورگر ماند"}، بارگذاری در ${sec(m.loadSeconds)}.`);
    tts = m;
    return m;
  })();
  try { return await ttsLoad; } catch (e) { ttsLoad = null; throw e; }
}

async function speakBrowser(text, voice, temp, seed) {
  const model = await ensureBrowserModel();
  const pieces = P.textToPieces(text, counter);
  if (!pieces.length) throw new Error("چیزی برای گفتن نیست: متن حروف فارسی ندارد (کلمات انگلیسی و نویسه‌های بیگانه حذف می‌شوند)");
  const model2 = await loadTimingModel();
  $("out").hidden = false; $("dl").hidden = true; $("player").removeAttribute("src"); $("player").load(); $("stats").replaceChildren();
  caps.mode = "live"; caps.render(pieces); wave.clear(); liveMap = []; startLoop();
  let finalPos = 0, gaps = 0;
  const netBase = netCount;
  clearTimeout(liveTimer); // تایمر تحویل اجرای قبلی نباید در این اجرا بیفاید
  audioCtx?.close().catch(() => {});
  audioCtx = new AudioContext({ sampleRate: SR });
  let at = audioCtx.currentTime + 0.05, firstAudio = null, genSec = 0, audioSec = 0;
  const t0 = performance.now(), done = [];
  const streamOn = $("stream").checked;
  // استریم: هر قطعهٔ ۸۰ میلی‌ثانیه‌ای همان لحظهٔ ساخته‌شدن زمان‌بندی می‌شود. پاک‌سازی جمله‌به‌جمله قطعه‌به‌قطعه انجام می‌شود:
  // سکوت پیش از آغاز حرف حذف می‌شود (۱۲۰ میلی‌ثانیه نگه داشته می‌شود و ۷۰ میلی‌ثانیهٔ اولش بی‌صداست)،
  // سکوتِ داخل جمله تا رسیدن حرف بعدی نگه داشته می‌شود، و سکوت پس از آخرین حرف تا ۱۲۰ میلی‌ثانیه بریده می‌شود.
  async function streamSentence(i, fp) {
    const keep = Math.floor(0.12 * SR), out = [];
    let started = false, lead = new Float32Array(0), held = [], sumsq = 0, n = 0, raw = 0, entry = null;
    const play = (a) => {
      if (!a.length) return;
      const buf = audioCtx.createBuffer(1, a.length, SR); buf.getChannelData(0).set(a);
      const src = audioCtx.createBufferSource(); src.buffer = buf; src.connect(audioCtx.destination);
      const now = audioCtx.currentTime;
      if (firstAudio === null) at = Math.max(at, now + 0.1);           // کمی سبقت برای اولین قطعهٔ اجرا
      else if (now > at) { gaps++; at = now + 0.02; }                  // پخش به تولید رسید
      src.start(at);
      if (!entry) { entry = { liveStart: at, finalStart: fp, dur: 0 }; liveMap.push(entry); }
      entry.dur += buf.duration; at += buf.duration; out.push(a);
      if (firstAudio === null) firstAudio = (performance.now() - t0) / 1000;
    };
    const g0 = performance.now();
    const stream = model.stream(pieces[i], { voice, temperature: temp, seed: seed + i });
    for await (const c of stream) {
      if (abort) { stream.cancel(); break; }
      raw += c.length; for (let j = 0; j < c.length; j++) sumsq += c[j] * c[j]; n += c.length;
      const speech = loudest(c) > Math.max(0.001, 0.04 * Math.sqrt(sumsq / n));
      if (!started) {
        if (!speech) { lead = cat(lead, c); if (lead.length > keep) lead = lead.slice(lead.length - keep); continue; }
        started = true;
        play(P.cleanStart(cat(lead, c), SR));
      } else if (speech) { held.forEach(play); held = []; play(c); }
      else held.push(c);
    }
    if (started && held.length) play(cat(...held).slice(0, keep));
    const genS = (performance.now() - g0) / 1000;
    at += 0.15;                                                         // مکث پیش از جملهٔ بعد
    return { piece: cat(...out), genS, rawS: raw / SR, at0: entry ? entry.liveStart : at };
  }
  for (let i = 0; i < pieces.length; i++) {
    if (abort) break;
    setStatus(`${streamOn ? "استریم جمله" : "جمله"} ${i + 1} از ${pieces.length}…`);
    const fp = finalPos;
    let piece, at0;
    if (streamOn) {
      const r = await streamSentence(i, fp);
      piece = r.piece; at0 = r.at0; genSec += r.genS; audioSec += r.rawS;
      if (!piece.length) continue;
    } else {
      const g0 = performance.now();
      const pcm = await model.synth(pieces[i], { voice, temperature: temp, seed: seed + i });
      genSec += (performance.now() - g0) / 1000; audioSec += pcm.length / SR;
      piece = P.cleanStart(P.trimSilence(pcm, SR), SR);
    }
    done.push(piece);
    const starts = P.wordTimings(pieces[i], piece, SR, { model: model2 || null }).map((w) => w.start);
    caps.set(i, starts, finalPos, "final", piece.length / SR);
    finalPos += piece.length / SR + 0.15;
    wave.set(P.joinPieces(done, SR), SR);
    if (!streamOn) {
      const buf = audioCtx.createBuffer(1, piece.length, SR);
      buf.getChannelData(0).set(piece);
      const src = audioCtx.createBufferSource();
      src.buffer = buf; src.connect(audioCtx.destination);
      if (i > 0 && audioCtx.currentTime > at) gaps++; // جملهٔ قبل پیش از آماده‌شدن این یکی تمام شد
      at = Math.max(at, audioCtx.currentTime + 0.02);
      src.start(at);
      liveMap.push({ liveStart: at, finalStart: fp, dur: buf.duration });
      at0 = at;
      at += buf.duration + 0.15;
      if (firstAudio === null) firstAudio = (performance.now() - t0) / 1000;
    }
    caps.set(i, starts, at0, "live", piece.length / SR);
  }
  if (!done.length) { setStatus("متوقف شد."); return; }
  const { encodeWav } = await import("./vendor/phonon-tts/index.js");
  const joined = P.joinPieces(done, SR);
  showAudio(encodeWav(joined, SR));
  // وقتی پخش زنده تمام شد، زیرنویس از پخش‌کننده دنبال می‌شود (بازپخش، پرش، کلیک روی کلمه)
  const waitMs = Math.max(0, (at - audioCtx.currentTime) * 1000) + 250;
  liveTimer = setTimeout(() => { if (caps.mode === "live") { caps.mode = "final"; caps.idx = -2; } }, waitMs);
  const frames = Math.round(audioSec * 12.5), finalLen = joined.length / SR;
  const thr = model.threads ?? 1;
  showStats([
    ["اولین صدا", ms(firstAudio), (streamOn ? "اولین قطعهٔ ۸۰ میلی‌ثانیه‌ای، استریم‌شده" : "اولین جمله آماده شد") + (model.loadSeconds ? ` · مدل در ${sec(model.loadSeconds)} بارگذاری شد` : "")],
    ["RTF", (genSec / audioSec).toFixed(3), `${(audioSec / Math.max(genSec, 1e-6)).toFixed(1)}× سریع‌تر از زمان واقعی`],
    ["هر فریم", `${(1000 * genSec / frames).toFixed(1)} میلی‌ثانیه`, `بودجهٔ ۸۰ میلی‌ثانیه · ${thr} رشته`],
    ["صدا", sec(finalLen), `${frames} فریم · ${pieces.length} جمله · ${gaps ? `${gaps} وقفه` : "بدون وقفه"}`],
    ["شبکه", String(netCount - netBase), "درخواست از زمان بارگذاری مدل؛ همه‌چیز روی دستگاه"],
  ]);
  setStatus(abort ? "متوقف شد." : streamOn ? "همزمان با تولید پخش شد. پخش‌کنندهٔ زیر دوباره پخشش می‌کند." : "در همان حینِ تولید پخش شد. پخش‌کنندهٔ زیر دوباره پخشش می‌کند.");
}

// ---- اتصال‌ها ----
async function go() {
  if (!tts) { setStatus("اول مدل را دانلود کنید (دکمهٔ بالا).", true); return; }
  const text = $("text").value.trim();
  if (!text) { setStatus("اول کمی متن فارسی بنویسید.", true); return; }
  abort = false; setBusy(true); $("out").hidden = true; caps.clear(); wave.clear(); liveMap = []; $("player").pause();
  const voice = $("voice").value, temp = Number($("temp").value), seed = Math.max(0, Math.floor(Number($("seed").value) || 0));
  try {
    await speakBrowser(text, voice, temp, seed);
  } catch (e) {
    console.error(e);
    setStatus(String(e.message || e), true);
  } finally { setBusy(false); $("load").hidden = true; }
}

// ---- مدل در مرورگر: تا کلیک نکنید چیزی دانلود نمی‌شود ----
const MODEL_URL = "model/model.safetensors";
async function modelStored() { try { return !!(await caches.match(MODEL_URL, { cacheName: MODEL_CACHE })); } catch { return false; } }
async function modelBytes() { try { const r = await fetch(MODEL_URL, { method: "HEAD" }); return r.ok ? Number(r.headers.get("content-length")) || 0 : 0; } catch { return 0; } }
function startLoad() {
  const done = () => refreshModelBox();
  ensureBrowserModel().then(done, (err) => { done(); if (!busy) setStatus(String(err.message || err), true); });
  refreshModelBox();
}
async function refreshModelBox() {
  const box = $("modelbox"), text = $("modeltext"), dl = $("dlmodel"), rm = $("rmmodel");
  box.hidden = false; dl.hidden = true; rm.hidden = true;
  if (tts) {
    text.textContent = `مدل آماده است (${Math.round(tts.sizeMB)} مگابایت، در این مرورگر نگه داشته شده).`; rm.hidden = false;
    // داخل قابِ صفحهٔ دیگر مرورگر فقط یک رشته می‌دهد: همان صفحه در تب خودش سریع‌تر است
    if (window.top !== window && !crossOriginIsolated) text.textContent += " تا وقتی داخل قاب نشان داده می‌شود فقط روی یک رشته اجرا می‌شود؛ صفحه را در تب جداگانه باز کنید.";
  }
  else if (ttsLoad) text.textContent = "در حال بارگذاری مدل…";
  else {
    text.textContent = "بررسی می‌کنیم که مدل از قبل روی این دستگاه هست یا نه…";
    const stored = await modelStored();
    if (tts || ttsLoad) return refreshModelBox();
    if (stored) { startLoad(); return; }
    const n = await modelBytes();
    if (tts || ttsLoad) return refreshModelBox();
    text.textContent = `مدل هنوز روی این دستگاه نیست. حجمش ${n ? "" : "حدود "}${n ? Math.round(n / 1048576) : 107} مگابایت است؛ یک بار دانلود می‌شود و مرورگر همان را نگه می‌دارد.`;
    dl.hidden = false;
  }
  updateGo();
}
$("dlmodel").addEventListener("click", () => startLoad());
$("rmmodel").addEventListener("click", async (e) => { e.preventDefault(); try { await caches.delete(MODEL_CACHE); } catch {} location.reload(); });

for (const [label, text] of SAMPLES) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "chip"; b.textContent = label;
  b.onclick = () => { $("text").value = text; };
  $("samples").append(b);
}
$("stream").checked = store.get("stream") === "1";
$("stream").addEventListener("change", () => store.set("stream", $("stream").checked ? "1" : "0"));
$("temp").addEventListener("input", () => { $("tempVal").textContent = Number($("temp").value).toFixed(2).replace(/0$/, ""); });
$("go").addEventListener("click", go);
$("stop").addEventListener("click", () => { abort = true; audioCtx?.close().catch(() => {}); setStatus("در حال توقف…"); });
$("text").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") go(); });

const voices = AVA.voices || ["nava", "shirin", "arash"];
$("voice").replaceChildren(...voices.map((v) => new Option(VOICE_LABELS[v] || v, v)));
$("voice").value = voices.includes("shirin") ? "shirin" : voices[0];
refreshModelBox();
