(() => {
  const prefix = "assets/results/";
  const state = { run: null, epoch: null, evaluation: null, tab: "metrics" };
  const runControls = document.querySelector("#run-controls");
  const epochSelect = document.querySelector("#epoch-select");
  const evaluationControls = document.querySelector("#evaluation-controls");
  const summary = document.querySelector("#experiment-summary");
  const assets = document.querySelector("#experiment-assets");
  let runs = {};
  let runDescriptions = {};
  let sharedTrainingRecipe = [];

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
  function comparisonTable(title, headers, rows, note = "") {
    return `<section class="comparison-section"><div class="comparison-heading"><h3>${title}</h3>${note ? `<p>${note}</p>` : ""}</div><div class="metric-table-wrap"><table><thead><tr>${headers.map((header) => `<th>${header}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell, index) => `<td${index === 0 ? " class=\"metric-name\"" : ""}>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div></section>`;
  }
  function detailsPanel(content) {
    return `<details class="metric-details"><summary>Complete metric details</summary><div class="detail-layout">${content}</div></details>`;
  }
  function experimentCard() {
    const description = runDescriptions[state.run];
    if (!description) return "";
    const recipe = description.recipe.map(([key, value]) => `<div class="recipe-item"><dt>${key}</dt><dd>${value}</dd></div>`).join("");
    const shared = sharedTrainingRecipe.map(([key, value]) => `<div class="recipe-item"><dt>${key}</dt><dd>${value}</dd></div>`).join("");
    return `<section class="experiment-card"><div class="experiment-card-heading"><div><p class="card-eyebrow">Experiment description</p><h3>${description.title}</h3></div><p>${description.summary}</p></div><div class="experiment-purpose"><span>Why this run?</span><p>${description.purpose}</p></div><div class="recipe-columns"><div><h4>Run-specific settings</h4><dl class="recipe-list">${recipe}</dl></div><div><h4>Shared training protocol</h4><dl class="recipe-list">${shared}</dl></div></div></section>`;
  }
  function contentMetrics(report) {
    const sources = report.metrics_by_condition_source;
    const delta = report.cross_capture_minus_same_capture;
    const metrics = ["hr_lowpass_l1", "hr_lowpass_psnr", "hr_lowpass_ssim", "hr_gradient_cosine", "target_lowpass_l1", "target_lowpass_ssim", "variance_pixel_l1", "variance_pixel_std"];
    const rows = metrics.map((key) => [
      name(key),
      number(sources.cross_capture[key].mean),
      number(sources.same_capture[key].mean),
      number(delta[key].mean_cross_minus_same),
    ]);
    return `${comparisonTable("Content preservation comparison", ["Metric", "Cross capture", "Same capture", "Cross − same"], rows, "Cross-capture values measure whether content remains stable when the degradation condition comes from another capture.")}${detailsPanel(`${statisticsTable("Cross-capture condition", sources.cross_capture)}${statisticsTable("Same-capture condition", sources.same_capture)}${statisticsTable("Cross-capture − same-capture", delta)}`)}`;
  }
  function transferMetrics(report) {
    const low = report.conditions.low;
    const high = report.conditions.high;
    const reference = report.reference_condition_detail;
    const targetRows = [["Low", low, reference.low], ["High", high, reference.high]].map(([condition, values, target]) => {
      const assigned = values.assigned_degradation_distance.mean;
      const margin = values.degradation_margin_other_minus_assigned.mean;
      return [
        `${condition} target`,
        number(target.gradient_energy),
        `${number(values.gradient_energy.mean)} ± ${number(values.gradient_energy.std)}`,
        number(target.laplacian_variance),
        `${number(values.laplacian_variance.mean)} ± ${number(values.laplacian_variance.std)}`,
        number(assigned),
        number(assigned + margin),
        number(margin),
        `${number(values.condition_accuracy * 100, 0)}%`,
      ];
    });
    const separation = report.paired_high_minus_low.degradation_distance_between_generated_conditions;
    const separationRows = [
      ["Reference low ↔ high targets", number(report.reference_degradation_distance_low_vs_high), "—"],
      ["Generated low ↔ high outputs", `${number(separation.mean)} ± ${number(separation.std)}`, separation.count],
    ];
    return `${comparisonTable("Embedding alignment with target conditions", ["Requested target", "Target gradient energy", "Generated gradient energy", "Target Laplacian variance", "Generated Laplacian variance", "Distance to assigned target ↓", "Distance to other target", "Other − assigned margin ↑", "Assignment accuracy ↑"], targetRows, "Lower assigned distance and a positive margin indicate that generated degradations embed nearer their requested target than the opposite condition.")}${comparisonTable("Low/high separation", ["Comparison", "Embedding distance", "Samples"], separationRows, "The generated low/high separation can be compared directly with the reference target separation.")}${detailsPanel(`${scalarTable("Reference conditions", { degradation_distance_low_vs_high: report.reference_degradation_distance_low_vs_high, detail: reference })}${statisticsTable("Low degradation condition", low)}${statisticsTable("High degradation condition", high)}${statisticsTable("Null-condition control", report.conditions.null)}${statisticsTable("Paired high − low", report.paired_high_minus_low)}`)}`;
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
      summary.innerHTML = `${experimentCard()}<p class="result-kicker">${label()} · ${viewName} · ${context}</p>${renderTabs()}`;
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
      runDescriptions = manifest.run_descriptions || {};
      sharedTrainingRecipe = manifest.shared_training_recipe || [];
      const availableRuns = Object.keys(runs).sort();
      if (!availableRuns.length) throw new Error("No evaluation results are published yet.");
      state.run = availableRuns.includes("run5") ? "run5" : availableRuns[availableRuns.length - 1];
      state.epoch = Object.keys(runs[state.run]).sort(sortEpochs).pop();
      state.evaluation = Object.keys(current())[0];
      render();
    })
    .catch((error) => { summary.innerHTML = `<p class="asset-note">${error.message}</p>`; });
})();
