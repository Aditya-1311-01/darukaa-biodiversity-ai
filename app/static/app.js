const API = "";
let sessionId = "s-" + Math.random().toString(36).slice(2, 10);

const el = (id) => document.getElementById(id);
const messages = el("messages");

const LABELS = {
  soil_organic_carbon_pct: "Soil organic carbon",
  soil_ph: "Soil pH",
  soil_moisture_pct: "Soil moisture",
  soil_texture: "Texture",
  land_use: "Land use",
  current_crop: "Crop",
  area_hectares: "Area (ha)",
  climate_zone: "Climate",
  rainfall_regime: "Rainfall",
  annual_rainfall_mm: "Rainfall (mm)",
  mean_temperature_c: "Mean temp (C)",
  tree_cover_pct: "Tree cover",
  species_richness_note: "Species note",
  habitat_diversity_note: "Habitat note",
  irrigation_source: "Irrigation",
  fertiliser_use: "Fertiliser",
  pesticide_use: "Pesticide",
  grazing_pressure: "Grazing",
  pollution_note: "Pollution",
  latitude: "Latitude",
  longitude: "Longitude",
  region_name: "Region",
};

const escape = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const pretty = (v) => String(v).replace(/_/g, " ");

/* ---------------- status ---------------- */
async function loadStatus() {
  try {
    const r = await fetch(API + "/api/health");
    const d = await r.json();
    el("st-knowledge").textContent = `${d.interventions_loaded} interventions · ${d.corpus_chunks} chunks`;
    el("st-vector").textContent = d.vector_store === "chromadb" ? "ChromaDB" : "keyword fallback";
    el("st-model").textContent = d.groq_configured ? "Groq connected" : "no API key";
  } catch {
    el("st-knowledge").textContent = "backend unreachable";
    el("st-vector").textContent = "—";
    el("st-model").textContent = "—";
  }
}

/* ---------------- rendering ---------------- */
function addUser(text) {
  const node = document.createElement("article");
  node.className = "msg user";
  node.innerHTML = `<p>${escape(text)}</p>`;
  messages.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

function addPending() {
  const node = document.createElement("article");
  node.className = "msg assistant";
  node.innerHTML = `<p class="thinking">Matching site conditions against the knowledge base…</p>`;
  messages.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

function paragraphs(text) {
  return text
    .split(/\n{2,}|\n/)
    .filter((p) => p.trim())
    .map((p) => `<p>${escape(p.trim())}</p>`)
    .join("");
}

function metricRows(effects) {
  return effects
    .map(
      (e) => `<tr>
        <td>${escape(pretty(e.metric))}</td>
        <td class="${e.direction === "increase" ? "up" : "down"}">
          ${e.direction === "increase" ? "▲" : "▼"} ${escape(e.magnitude)}
        </td>
        <td>${escape(e.horizon)}</td>
        <td>${escape(e.confidence)}</td>
      </tr>`
    )
    .join("");
}

function citationItems(cites) {
  return cites
    .map((c) => {
      const label = `${escape(c.organisation)} — ${escape(c.title)}${c.year ? " (" + c.year + ")" : ""}${
        c.locator ? ", " + escape(c.locator) : ""
      }`;
      const link = c.url ? `<a href="${escape(c.url)}" target="_blank" rel="noopener">source</a>` : "";
      const flag = c.verified ? "" : `<span class="unverified">figure not yet verified against primary source</span>`;
      return `<li>${label} ${link} ${flag}</li>`;
    })
    .join("");
}

function recommendationCard(r) {
  return `<section class="rec">
    <div class="rec-head">
      <h3>${escape(r.title)}</h3>
      <div class="chips">
        <span class="chip">${escape(r.horizon)} term</span>
        <span class="chip">confidence: ${escape(r.confidence)}</span>
        <span class="chip">fit ${(r.fit_score * 100).toFixed(0)}%</span>
        ${r.preconditions_met.map((p) => `<span class="chip flag">${escape(pretty(p))}</span>`).join("")}
      </div>
    </div>
    <div class="rec-body">
      <h4>What to do</h4>
      <p>${escape(r.what_to_do)}</p>

      <h4>Why it works</h4>
      <p>${escape(r.why_it_works)}</p>

      <h4>Causal chain across variables</h4>
      <ul class="chain">${r.causal_chain.map((c) => `<li>${escape(c)}</li>`).join("")}</ul>

      <h4>Metrics affected</h4>
      <table class="metrics">
        <thead><tr><th>Metric</th><th>Expected change</th><th>Horizon</th><th>Confidence</th></tr></thead>
        <tbody>${metricRows(r.metric_effects)}</tbody>
      </table>

      ${r.trade_offs ? `<div class="tradeoff"><strong>Trade-off.</strong> ${escape(r.trade_offs)}</div>` : ""}

      <h4>Sources</h4>
      <ul class="cites">${citationItems(r.citations)}</ul>
    </div>
  </section>`;
}

function renderTurn(node, data) {
  let html = `<div class="narrative">${paragraphs(data.message)}</div>`;

  if (data.clarifying_questions?.length) {
    html += `<ul class="questions">${data.clarifying_questions
      .map((q) => `<li>${escape(q)}</li>`)
      .join("")}</ul>`;
  }
  (data.recommendations || []).forEach((r) => (html += recommendationCard(r)));

  node.className = "msg assistant";
  node.innerHTML = html;
  node.scrollIntoView({ behavior: "smooth", block: "start" });

  renderProfile(data.profile);
  renderEvidence(data.evidence_used || []);
}

function renderProfile(profile) {
  const list = el("profile-list");
  const entries = Object.entries(profile || {}).filter(([, v]) => v !== null && v !== undefined);
  el("profile-hint").textContent = entries.length
    ? `${entries.length} variable${entries.length === 1 ? "" : "s"} recorded and carried into every following turn.`
    : "Nothing recorded yet. Values persist across turns.";
  list.innerHTML = entries
    .map(
      ([k, v]) =>
        `<div><dt>${escape(LABELS[k] || pretty(k))}</dt><dd>${escape(pretty(v))}</dd></div>`
    )
    .join("");
}

function renderEvidence(items) {
  const list = el("evidence-list");
  if (!items.length) return;
  el("evidence-hint").textContent = `${items.length} passages retrieved for the last recommendation.`;
  list.innerHTML = items
    .map(
      (e) => `<li>
        <span class="src">${escape(e.source_org || "unknown")} · ${escape(e.chunk_id)} · score ${e.score}</span>
        <span class="ex">${escape(e.excerpt)}…</span>
      </li>`
    )
    .join("");
}

/* ---------------- sending ---------------- */
async function send(text, structured) {
  const btn = el("send-btn");
  btn.disabled = true;
  if (text) addUser(text);
  const pending = addPending();

  try {
    const res = await fetch(API + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text, structured_input: structured }),
    });
    if (!res.ok) throw new Error(`Request failed (${res.status}). ${await res.text()}`);
    renderTurn(pending, await res.json());
  } catch (err) {
    pending.className = "msg error";
    pending.innerHTML = `<p>Could not reach the reasoning service. ${escape(err.message)}</p>
      <p>Check that the backend is running and that GROQ_API_KEY is set.</p>`;
  } finally {
    btn.disabled = false;
  }
}

/* ---------------- wiring ---------------- */
el("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = el("input").value.trim();
  const panel = el("json-panel");
  let structured = null;

  if (!panel.hidden) {
    const raw = el("json-input").value.trim();
    if (raw) {
      try {
        structured = JSON.parse(raw);
      } catch {
        alert("Structured input is not valid JSON. Fix it or close the panel.");
        return;
      }
    }
  }
  if (!text && !structured) return;
  el("input").value = "";
  send(text, structured);
});

el("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    el("composer").requestSubmit();
  }
});

el("json-toggle").addEventListener("click", () => {
  const panel = el("json-panel");
  panel.hidden = !panel.hidden;
  el("json-toggle").textContent = panel.hidden ? "Structured input" : "Hide structured input";
});

el("example-btn").addEventListener("click", () => {
  el("input").value =
    "Biodiversity is declining on my land. It is 4 ha of monoculture wheat in a semi-arid region, " +
    "soil organic carbon is 0.3%, rainfall is low, and I broadcast all the urea at sowing.";
});

el("reset-btn").addEventListener("click", async () => {
  await fetch(API + `/api/session/${sessionId}/reset`, { method: "POST" }).catch(() => {});
  sessionId = "s-" + Math.random().toString(36).slice(2, 10);
  messages.innerHTML = `<article class="msg assistant"><p>New session. The site profile has been cleared.</p></article>`;
  renderProfile({});
  el("evidence-list").innerHTML = "";
  el("evidence-hint").textContent = "Passages pulled from the vector store on the last recommendation turn.";
});

loadStatus();
renderProfile({});
