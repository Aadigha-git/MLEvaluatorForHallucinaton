// Dashboard v1 (docs/SPEC.md, "Dashboard UI"). Display only: sliders re-band calibrated P in the
// browser for illustration; the scoring service and its frozen policy never change.
const CONFIG = {
  bands: { accept: "accept", flag: "flag", escalate: "escalate" },
  maxRows: 300,
};
const state = { verdicts: [], frozen: null, tLow: null, tHigh: null, judgeCost: {}, selected: null };
const $ = (id) => document.getElementById(id);
const fmtUsd = (x) => "$" + x.toFixed(4);

function bandOf(p) {
  if (p == null) return "error";
  return p < state.tLow ? "accept" : p < state.tHigh ? "escalate" : "flag";
}

function setPolicy(policy) {
  state.frozen = policy;
  state.tLow = policy.t_low;
  state.tHigh = policy.t_high;
  syncSliders();
}

function syncSliders() {
  $("tLow").value = state.tLow; $("tHigh").value = state.tHigh;
  $("tLowOut").textContent = state.tLow.toFixed(3); $("tHighOut").textContent = state.tHigh.toFixed(3);
  const moved = state.frozen && (state.tLow !== state.frozen.t_low || state.tHigh !== state.frozen.t_high);
  $("policyNote").textContent = !state.frozen ? "waiting for the first verdict"
    : moved ? "illustration: not the frozen policy" : "frozen Step 7 policy";
  render();
}

function render() {
  const counts = { accept: 0, flag: 0, escalate: 0 };
  for (const v of state.verdicts) { const b = bandOf(v.p_hallucinated); if (b in counts) counts[b]++; }
  const n = state.verdicts.length;
  $("nVerdicts").textContent = n;
  const g = state.judgeCost.gemini || 0, m = state.judgeCost.nemotron || 0;
  $("judgeAll").textContent = fmtUsd(n * g);
  $("netCost").textContent = fmtUsd(counts.escalate * g);
  $("nemotron").textContent = `Nemotron: ${fmtUsd(n * m)} on everything, ${fmtUsd(counts.escalate * m)} net. ` +
    "Classifier: $0 incremental. Judge costs measured in Step 9.";
  for (const b of ["accept", "flag", "escalate"]) {
    $("c" + b[0].toUpperCase() + b.slice(1)).textContent = counts[b];
    $("b" + b[0].toUpperCase() + b.slice(1)).style.width = n ? (100 * counts[b] / n) + "%" : "0";
  }
  document.querySelectorAll(".row").forEach((row) => {
    const v = state.verdicts.find((x) => x.question_id === row.dataset.id);
    const pill = row.querySelector(".pill"), b = bandOf(v.p_hallucinated);
    pill.className = "pill " + b; pill.textContent = b;
  });
}

function addRow(v) {
  const row = document.createElement("div");
  row.className = "row"; row.dataset.id = v.question_id;
  const trap = v.source === "trap" ? '<span class="trap">trap</span>' : "";
  row.innerHTML = `<div class="q"></div><div class="a"></div><div><span class="pill"></span></div><div class="e"></div>`;
  row.querySelector(".q").textContent = v.question; row.querySelector(".q").insertAdjacentHTML("beforeend", trap);
  row.querySelector(".a").textContent = (v.answer || "").slice(0, 120) + ((v.answer || "").length > 120 ? "…" : "");
  row.querySelector(".e").textContent = v.error ? "error: " + v.error : v.explanation;
  row.onclick = () => { document.querySelectorAll(".row.sel").forEach((r) => r.classList.remove("sel")); row.classList.add("sel"); showDetail(v); };
  $("feed").prepend(row);
  while ($("feed").children.length > CONFIG.maxRows) $("feed").lastChild.remove();
}

function showDetail(v) {
  const d = $("detail"); d.innerHTML = "";
  const h = (tag, text, cls) => { const e = document.createElement(tag); e.textContent = text; if (cls) e.className = cls; d.appendChild(e); return e; };
  h("h3", v.question);
  if (v.reference_answer) h("p", "Reference answer: " + v.reference_answer, "muted");
  h("p", v.answer || "(no answer)");
  if (v.p_hallucinated != null) h("p", `P(hallucinated) ${v.p_hallucinated.toFixed(3)} · band ${bandOf(v.p_hallucinated)} · frozen band ${v.band}`, "muted");
  if (v.explanation) h("p", v.explanation);
  if (v.features) {
    h("h4", "Features (bar) against val_id median (tick)");
    const med = v.classifier.feature_medians_val_id || {};
    const vals = Object.values(v.features).concat(Object.values(med));
    const lo = Math.min(0, ...vals), hi = Math.max(1, ...vals), pct = (x) => 100 * (x - lo) / (hi - lo);
    for (const [f, x] of Object.entries(v.features)) {
      const r = document.createElement("div"); r.className = "feat";
      r.innerHTML = `<span></span><div class="track"><div class="val"></div><div class="med"></div></div><span></span>`;
      r.children[0].textContent = f; r.children[2].textContent = x.toFixed(2);
      r.querySelector(".val").style.width = pct(x) + "%"; r.querySelector(".med").style.left = pct(med[f] ?? 0) + "%";
      d.appendChild(r);
    }
  }
  if (v.evidence) {
    h("h4", "Retrieved evidence (top 5, BM25 over Wiki-DPR)");
    for (const p of v.evidence) h("div", `${p.title}: ${p.text}`, "passage");
  }
  const ts = v.timestamps || {};
  h("p", `produced ${ts.produced} · answered ${ts.answered} · scored ${ts.scored} · displayed ${ts.displayed}`, "muted");
}

function connect() {
  const es = new EventSource("/api/verdicts/stream");
  es.addEventListener("verdict", (msg) => {
    const v = JSON.parse(msg.data);
    v.timestamps = { ...(v.timestamps || {}), displayed: new Date().toISOString() };
    if (!state.frozen && v.policy) setPolicy(v.policy);
    state.verdicts.push(v);
    addRow(v); render();
  });
  es.onerror = () => { $("conn").className = "conn off"; };
}

async function poll() {
  try {
    const s = await (await fetch("/api/status")).json();
    state.judgeCost = s.judge_cost_usd || {};
    $("conn").className = "conn " + (s.kafka_connected ? "on" : "off");
    $("conn").textContent = "Kafka: " + (s.kafka_connected ? "connected" : "disconnected") + (s.streaming ? ` · streaming ${s.sent}/${s.total}` : "");
  } catch { $("conn").className = "conn off"; $("conn").textContent = "Kafka: unreachable"; }
}

$("tLow").oninput = (e) => { state.tLow = Math.min(+e.target.value, state.tHigh); syncSliders(); };
$("tHigh").oninput = (e) => { state.tHigh = Math.max(+e.target.value, state.tLow); syncSliders(); };
$("reset").onclick = () => state.frozen && setPolicy(state.frozen);
$("start").onclick = () => fetch("/api/stream/start", { method: "POST" }).then(poll);
$("stop").onclick = () => fetch("/api/stream/stop", { method: "POST" }).then(poll);
state.tLow = 0.5; state.tHigh = 0.9; syncSliders();
connect(); poll(); setInterval(poll, 3000);
