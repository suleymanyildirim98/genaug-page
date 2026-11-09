(() => {
  const runControls = document.querySelector("#run-controls");
  const summary = document.querySelector("#experiment-summary");
  const assets = document.querySelector("#experiment-assets");
  const state = { run: null, tab: "metrics" };
  let data = null;

  const pairs = (items) => items.map(([k, v]) => `<div class="recipe-item"><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  const table = (t) => `<section class="comparison-section"><div class="comparison-heading"><h3>${t.title}</h3>${t.note ? `<p>${t.note}</p>` : ""}</div>`
    + `<div class="metric-table-wrap"><table><thead><tr>${t.headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`
    + `<tbody>${t.rows.map((r) => `<tr>${r.map((c, i) => `<td${i === 0 ? ' class="metric-name"' : ""}>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div></section>`;
  const figure = (f, alt) => `<figure class="result-figure"><a href="${f.src}" target="_blank" rel="noopener"><img src="${f.src}" alt="${alt}: ${f.caption}" loading="lazy"></a><figcaption>${f.caption}</figcaption></figure>`;
  const run = () => data.runs.find((r) => r.id === state.run);

  function card(r) {
    return `<section class="experiment-card"><div class="experiment-card-heading"><div><p class="card-eyebrow">${r.name} · experiment description</p><h3>${r.title}</h3></div><p>${r.summary}</p></div>`
      + `<div class="experiment-purpose"><span>Why this run?</span><p>${r.purpose}</p></div>`
      + `<div class="recipe-columns"><div><h4>Run-specific settings</h4><dl class="recipe-list">${pairs(r.variables)}${pairs([["Compared to", r.compared_to], ["Checkpoints", `<code>${r.checkpoint}</code>`]])}</dl></div>`
      + `<div><h4>Shared training protocol</h4><dl class="recipe-list">${pairs(data.shared)}</dl></div></div>`
      + `<div class="lrgen-result"><h4>Findings</h4><ul class="lrgen-findings">${r.findings.map((f) => `<li>${f}</li>`).join("")}</ul><p class="lrgen-verdict"><strong>Conclusion.</strong> ${r.verdict}</p></div></section>`;
  }
  function visuals(r) {
    const v = r.visuals;
    const sweep = `<section class="comparison-section"><div class="comparison-heading"><h3>Reference sweep (DRealSR test, textured crops)</h3><p>${v.sweep_note}</p></div>`
      + (v.sweep.length ? `<div class="lrgen-figures">${v.sweep.map((f) => figure(f, `${r.name} reference sweep`)).join("")}</div>` : "") + `</section>`;
    const hand = `<section class="comparison-section"><div class="comparison-heading"><h3>Hand-cropped reference sweep</h3><p>${v.hand_note}</p></div>`
      + (v.hand.length ? `<div class="lrgen-figures wide">${v.hand.map((f) => figure(f, `${r.name} hand-crop sweep`)).join("")}</div>` : "") + `</section>`;
    return sweep + hand;
  }
  function render() {
    const r = run();
    runControls.querySelectorAll(".chip").forEach((c) => c.classList.toggle("selected", c.dataset.id === r.id));
    try { history.replaceState(null, "", `#${r.id}`); } catch (_) { /* file:// or sandboxed */ }
    summary.innerHTML = `${card(r)}<p class="result-kicker">${r.name} · ${r.title}</p>`
      + `<div class="result-tabs"><button class="result-tab ${state.tab === "metrics" ? "active" : ""}" type="button" data-tab="metrics">Metrics</button>`
      + `<button class="result-tab ${state.tab === "visuals" ? "active" : ""}" type="button" data-tab="visuals">Visuals</button></div>`;
    summary.querySelectorAll("[data-tab]").forEach((t) => t.addEventListener("click", () => { state.tab = t.dataset.tab; render(); }));
    assets.innerHTML = state.tab === "metrics" ? r.tables.map(table).join("") : visuals(r);
  }
  function select(id) {
    state.run = data.runs.some((r) => r.id === id) ? id : data.runs[data.runs.length - 1].id;
    render();
  }

  fetch(`assets/lrgen/data.json?updated=${Date.now()}`, { cache: "no-store" })
    .then((r) => { if (!r.ok) throw new Error("The run log is unavailable."); return r.json(); })
    .then((d) => {
      data = d;
      document.querySelector("#generated").textContent = `Updated ${d.generated} · ${d.runs.length} runs`;
      document.querySelector("#overview").innerHTML = table(d.overview);
      document.querySelector("#glossary").innerHTML = d.glossary.map(([k, v]) => `<tr><td><strong>${k}</strong></td><td>${v}</td></tr>`).join("");
      runControls.innerHTML = d.runs.map((r) => `<button type="button" class="chip" data-id="${r.id}" title="${r.title}">${r.name}</button>`).join("");
      runControls.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => select(c.dataset.id)));
      select(location.hash.slice(1));
      window.addEventListener("hashchange", () => { if (location.hash.slice(1) !== state.run) select(location.hash.slice(1)); });
    })
    .catch((err) => { summary.innerHTML = `<p class="asset-note">${err.message}</p>`; });
})();
