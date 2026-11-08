(() => {
  const nav = document.querySelector("#experiment-nav");
  const view = document.querySelector("#experiment");
  let experiments = [];

  const list = (items) => items.map(([k, v]) => `<div class="recipe-item"><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  const table = (t) => `<section class="comparison-section"><div class="comparison-heading"><h3>${t.title}</h3>${t.note ? `<p>${t.note}</p>` : ""}</div>`
    + `<div class="metric-table-wrap"><table><thead><tr>${t.headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`
    + `<tbody>${t.rows.map((r) => `<tr>${r.map((c, i) => `<td${i === 0 ? ' class="metric-name"' : ""}>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div></section>`;
  const figure = (f) => `<figure class="result-figure"><a href="${f.src}" target="_blank" rel="noopener"><img src="${f.src}" alt="${f.caption}" loading="lazy"></a><figcaption>${f.caption}</figcaption></figure>`;

  function show(id) {
    const e = experiments.find((x) => x.id === id) || experiments[0];
    nav.querySelectorAll(".chip").forEach((c) => c.classList.toggle("selected", c.dataset.id === e.id));
    try { history.replaceState(null, "", `#${e.id}`); } catch (_) { /* file:// or sandboxed */ }
    view.innerHTML = `<section class="experiment-card"><div class="experiment-card-heading"><div><p class="card-eyebrow">${e.date}</p><h3>${e.title}</h3></div><p>${e.summary}</p></div>`
      + `<div class="experiment-purpose"><span>Why this experiment?</span><p>${e.purpose}</p></div>`
      + `<div class="recipe-columns"><div><h4>Setup</h4><dl class="recipe-list">${list(e.setup)}</dl></div>`
      + `<div><h4>Findings</h4><ul class="lrgen-findings">${e.findings.map((f) => `<li>${f}</li>`).join("")}</ul><p class="lrgen-verdict"><strong>Conclusion.</strong> ${e.verdict}</p></div></div></section>`
      + (e.tables || []).map(table).join("")
      + ((e.figures || []).length ? `<div class="lrgen-figures">${e.figures.map(figure).join("")}</div>` : "");
  }

  fetch(`assets/lrgen/data.json?updated=${Date.now()}`, { cache: "no-store" })
    .then((r) => { if (!r.ok) throw new Error("The experiment log is unavailable."); return r.json(); })
    .then((data) => {
      experiments = data.experiments;
      document.querySelector("#generated").textContent = `Updated ${data.generated}. ${experiments.length} experiments.`;
      document.querySelector("#glossary").innerHTML = data.glossary.map(([k, v]) => `<tr><td><strong>${k}</strong></td><td>${v}</td></tr>`).join("");
      nav.innerHTML = experiments.map((e, i) => `<button type="button" class="chip" role="tab" data-id="${e.id}">${i + 1}. ${e.title}</button>`).join("");
      nav.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => show(c.dataset.id)));
      show(location.hash.slice(1));
    })
    .catch((err) => { view.innerHTML = `<p class="asset-note">${err.message}</p>`; });
})();
