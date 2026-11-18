// PathSnap evaluator UI. Talks only to the API; it never sees reference routes.

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = "pathsnap_api_url";

// ---- API address: saved value in this browser, else config.js
function savedUrl() {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}
function saveUrl(url) {
  try { localStorage.setItem(STORAGE_KEY, url); } catch { /* storage blocked: ignore */ }
}
let apiUrl = (savedUrl() || window.PATHSNAP_API_URL || "http://localhost:8000").replace(/\/+$/, "");
$("apiUrl").value = apiUrl;

// ---- map
const map = L.map("map").setView([20, 0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const gpsLayer = L.layerGroup().addTo(map);
const routeLayer = L.layerGroup().addTo(map);
L.control.layers(null, { "GPS observations": gpsLayer, "Predicted route": routeLayer },
                 { collapsed: false }).addTo(map);
const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const div = L.DomUtil.create("div", "legend");
  div.innerHTML = '<i class="dot" style="background:#1f6feb"></i>GPS observation<br>' +
                  '<i style="background:#d62728"></i>Predicted route';
  return div;
};
legend.addTo(map);

// ---- helpers
function setStatus(text, cls) {
  const el = $("apiStatus");
  el.textContent = text;
  el.className = "status " + (cls || "");
}
async function api(path, options) {
  const res = await fetch(apiUrl + path, options);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}
const pct = (x) => (x === null || x === undefined ? "n/a" : (100 * x).toFixed(2) + " %");

// ---- connect + list trips
async function connect() {
  apiUrl = $("apiUrl").value.trim().replace(/\/+$/, "");
  saveUrl(apiUrl);
  setStatus("Connecting…");
  try {
    const h = await api("/health");
    setStatus(`Connected · ${h.model}`, "ok");
    const { trips } = await api("/trips");
    const list = $("tripList");
    list.innerHTML = "";
    if (!trips.length) list.innerHTML = '<p class="muted">The server has no trips.</p>';
    trips.forEach((id) => {
      const b = document.createElement("button");
      b.textContent = id;
      b.onclick = () => runTrip(id, b);
      list.appendChild(b);
    });
  } catch (e) {
    setStatus("Cannot reach API: " + e.message, "err");
  }
}

// ---- run one trip
async function runTrip(id, button) {
  document.querySelectorAll(".trip-list button").forEach((b) => b.classList.remove("active"));
  button.classList.add("active");
  $("busy").classList.remove("hidden");
  $("results").innerHTML = '<p class="muted">Waiting for the server…</p>';
  try {
    const r = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: id }),
    });
    draw(r);
    showResults(r);
  } catch (e) {
    $("results").innerHTML = `<p class="status err">Error: ${e.message}</p>`;
  } finally {
    $("busy").classList.add("hidden");
  }
}

function draw(r) {
  gpsLayer.clearLayers();
  routeLayer.clearLayers();
  r.gps_points.forEach((p) =>
    L.circleMarker(p, { radius: 2, color: "#1f6feb", weight: 0, fillOpacity: 0.8 }).addTo(gpsLayer));
  L.polyline(r.predicted_route, { color: "#d62728", weight: 4, opacity: 0.85 }).addTo(routeLayer);
  map.invalidateSize();          // the map panel may have changed size
  map.fitBounds(L.latLngBounds(r.gps_points), { padding: [20, 20] });
}

function showResults(r) {
  const m = r.metrics, t = r.timing_s;
  $("results").innerHTML = `
    <div class="muted">Point accuracy (trip ${r.trip_id})</div>
    <div class="big">${pct(m.point_accuracy)}</div>
    <table>
      <tr><td>GPS points</td><td>${r.n_gps_points.toLocaleString()}</td></tr>
      <tr><td>Points matched to a road</td><td>${r.n_points_matched.toLocaleString()}</td></tr>
      <tr><td>Points scored (label ≤ 50 m)</td><td>${m.scored_points.toLocaleString()}</td></tr>
      <tr><td>Route continuity</td><td>${pct(m.route_continuity)}</td></tr>
      <tr><td>Route segment F1</td><td>${pct(m.route_arc_f1)}</td></tr>
      <tr><td>Predicted road segments</td><td>${r.n_route_segments.toLocaleString()}</td></tr>
      <tr><td>HMM chains (breaks + 1)</td><td>${r.n_hmm_chains}</td></tr>
      <tr><td>Server time</td><td>${t.request_total.toFixed(2)} s</td></tr>
      <tr><td>&nbsp;· features / classifier / HMM</td>
          <td>${t.features.toFixed(2)} / ${t.classifier.toFixed(2)} / ${t.hmm.toFixed(2)} s</td></tr>
    </table>`;
}

$("connectBtn").onclick = connect;
$("apiUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
connect();
