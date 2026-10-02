(() => {
  const prefix = "assets/results/";
  const state = { run: null, epoch: null, evaluation: null, tab: "metrics" };
  const runControls = document.querySelector("#run-controls");
  const epochSelect = document.querySelector("#epoch-select");
  const evaluationControls = document.querySelector("#evaluation-controls");
  const summary = document.querySelector("#experiment-summary");
  const assets = document.querySelector("#experiment-assets");
  let runs = {};

  const button = (text, selected, onClick) => {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = text;
    element.className = selected ? "chip selected" : "chip";
    element.addEventListener("click", onClick);
    return element;
  };
  const number = (value, digits = 4) => Number(value).toFixed(digits);
  const name = (value) => value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  const sortEpochs = (left, right) => left === "last" ? 1 : right === "last" ? -1 : Number(left) - Number(right);
  const epochText = (epoch) => epoch === "last" ? "Latest checkpoint" : `Epoch ${epoch}`;
  const runText = (run) => run.replace("run", "Run ");
  const current = () => runs[state.run][state.epoch];
  const label = () => `${runText(state.run)} · ${epochText(state.epoch).toLowerCase()}`;
  const assetRoot = () => `${prefix}${state.run}-epoch${state.epoch}/${state.evaluation}`;
  const isScalar = (value) => typeof value === "number" || typeof value === "string" || typeof value === "boolean";
  const scalar = (value) => typeof value === "number" ? number(value) : String(value);

  function scalarRows(value, prefixName = "") {
    if (isScalar(value)) return [[prefixName || "Value", scalar(value)]];
    return Object.entries(value).flatMap(([key, child]) => scalarRows(child, prefixName ? `${prefixName} · ${name(key)}` : name(key)));
  }
  function scalarTable(title, value) {
    const rows = scalarRows(value).map(([metricName, metricValue]) => `<tr><td>${metricName}</td><td>${metricValue}</td></tr>`).join("");
    return `<section class="metric-section"><h3>${title}</h3><div class="metric-table-wrap"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
  }
  function statisticsTable(title, metrics) {
    const columns = [...new Set(Object.values(metrics).flatMap((value) => isScalar(value) ? ["value"] : Object.keys(value)))];
    const header = columns.map((column) => `<th>${name(column)}</th>`).join("");
    const rows = Object.entries(metrics).map(([metricName, values]) => {
      const record = isScalar(values) ? { value: values } : values;
      return `<tr><td>${name(metricName)}</td>${columns.map((column) => `<td>${record[column] === undefined ? "—" : scalar(record[column])}</td>`).join("")}</tr>`;
    }).join("");
    return `<section class="metric-section"><h3>${title}</h3><div class="metric-table-wrap"><table><thead><tr><th>Metric</th>${header}</tr></thead><tbody>${rows}</tbody></table></div></section>`;
  }
  function contentMetrics(report) {
    const sources = report.metrics_by_condition_source;
    return `<div class="metric-layout">${statisticsTable("Cross-capture condition", sources.cross_capture)}${statisticsTable("Same-capture condition", sources.same_capture)}${statisticsTable("Cross-capture − same-capture", report.cross_capture_minus_same_capture)}</div>`;
  }
  function transferMetrics(report) {
    return `<div class="metric-layout">${scalarTable("Reference conditions", { degradation_distance_low_vs_high: report.reference_degradation_distance_low_vs_high, detail: report.reference_condition_detail })}${statisticsTable("Low degradation condition", report.conditions.low)}${statisticsTable("High degradation condition", report.conditions.high)}${statisticsTable("Null-condition control", report.conditions.null)}${statisticsTable("Paired high − low", report.paired_high_minus_low)}</div>`;
  }
  function visuals(root, visualizations) {
    const image = state.evaluation === "content-fidelity" ? "sample_grid.svg" : "generated_pairs_grid.svg";
    const caption = state.evaluation === "content-fidelity" ? "Content-fidelity sample grid" : "Generated pairs under low and high degradation conditions";
    if (!visualizations.includes(image)) return `<p class="asset-note">This evaluation has a numeric summary but no saved qualitative grid.</p>`;
    return `<figure class="result-figure qualitative"><img src="${root}/visualizations/${image}" alt="${caption} for ${label()}"><figcaption>${caption}</figcaption></figure>`;
  }
  function renderTabs() {
    return `<div class="result-tabs"><button class="result-tab ${state.tab === "metrics" ? "active" : ""}" type="button" data-tab="metrics">Metrics</button><button class="result-tab ${state.tab === "visuals" ? "active" : ""}" type="button" data-tab="visuals">Visuals</button></div>`;
  }
  function bindTabs() {
    summary.querySelectorAll("[data-tab]").forEach((tab) => tab.addEventListener("click", () => {
      state.tab = tab.dataset.tab;
      renderResult();
    }));
  }

  async function renderResult() {
    const root = assetRoot();
    const viewName = state.evaluation === "content-fidelity" ? "Content fidelity" : "Degradation transfer";
    summary.innerHTML = `<p class="result-kicker">${label()} · ${viewName}</p><p class="loading">Loading summary…</p>`;
    assets.innerHTML = "";
    try {
      const report = await fetch(`${root}/metrics_summary.json`).then((response) => {
        if (!response.ok) throw new Error("The selected summary is unavailable.");
        return response.json();
      });
      const visualizations = current()[state.evaluation].visualizations;
      const context = state.evaluation === "content-fidelity"
        ? `${report.num_examples} examples × ${report.samples_per_input} samples · ${report.condition.replace(/_/g, " ")} condition`
        : `${report.conditions.low.assigned_degradation_distance.count} generated samples per low/high condition · ${report.degradation_encoder}`;
      summary.innerHTML = `<p class="result-kicker">${label()} · ${viewName} · ${context}</p>${renderTabs()}`;
      bindTabs();
      assets.innerHTML = state.tab === "metrics"
        ? (state.evaluation === "content-fidelity" ? contentMetrics(report) : transferMetrics(report))
        : visuals(root, visualizations);
    } catch (error) {
      summary.innerHTML = `<p class="asset-note">${error.message}</p>`;
    }
  }

  function renderControls() {
    runControls.replaceChildren(...Object.keys(runs).sort().map((run) => button(runText(run), state.run === run, () => {
      state.run = run;
      state.epoch = Object.keys(runs[run]).sort(sortEpochs).pop();
      state.evaluation = Object.keys(current())[0];
      state.tab = "metrics";
      render();
    })));
    epochSelect.replaceChildren(...Object.keys(runs[state.run]).sort(sortEpochs).map((epoch) => {
      const option = document.createElement("option");
      option.value = epoch;
      option.textContent = epochText(epoch);
      option.selected = epoch === state.epoch;
      return option;
    }));
    epochSelect.onchange = () => {
      state.epoch = epochSelect.value;
      state.evaluation = Object.keys(current())[0];
      state.tab = "metrics";
      render();
    };
    evaluationControls.replaceChildren(...Object.keys(current()).map((evaluation) => button(evaluation === "content-fidelity" ? "Content fidelity" : "Degradation transfer", state.evaluation === evaluation, () => {
      state.evaluation = evaluation;
      state.tab = "metrics";
      render();
    })));
  }
  function render() { renderControls(); renderResult(); }

  fetch(`${prefix}manifest.json`)
    .then((response) => {
      if (!response.ok) throw new Error("Result manifest is unavailable.");
      return response.json();
    })
    .then((manifest) => {
      runs = manifest.runs;
      const availableRuns = Object.keys(runs).sort();
      if (!availableRuns.length) throw new Error("No evaluation results are published yet.");
      state.run = availableRuns.includes("run5") ? "run5" : availableRuns[availableRuns.length - 1];
      state.epoch = Object.keys(runs[state.run]).sort(sortEpochs).pop();
      state.evaluation = Object.keys(current())[0];
      render();
    })
    .catch((error) => { summary.innerHTML = `<p class="asset-note">${error.message}</p>`; });
})();
