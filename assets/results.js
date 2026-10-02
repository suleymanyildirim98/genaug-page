(() => {
  const runs = {
    run3: [60, 80, 480, 800],
    run4: [200, 300, 400],
    run5: [200, 300, 400, 800],
  };
  const hasContentVisuals = new Set(["run4-200", "run4-300", "run4-400", "run5-300", "run5-400"]);
  const hasContent = new Set(["run3-60", "run3-80", "run3-480", "run3-800", ...hasContentVisuals]);
  const prefix = "assets/results/";
  const state = { run: "run4", epoch: 400, evaluation: "content" };
  const runControls = document.querySelector("#run-controls");
  const epochSelect = document.querySelector("#epoch-select");
  const evaluationControls = document.querySelector("#evaluation-controls");
  const summary = document.querySelector("#experiment-summary");
  const assets = document.querySelector("#experiment-assets");

  const key = () => `${state.run}-${state.epoch}`;
  const label = () => `${state.run.replace("run", "Run ")} · epoch ${state.epoch}`;
  const assetRoot = () => `${prefix}${state.run}-epoch${state.epoch}/${state.evaluation === "content" ? "content-fidelity" : "degradation-transfer"}`;
  const button = (text, selected, onClick) => {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = text;
    element.className = selected ? "chip selected" : "chip";
    element.addEventListener("click", onClick);
    return element;
  };
  const metric = (name, value) => `<div class="metric"><span>${name}</span><strong>${value}</strong></div>`;
  const number = (value, digits = 3) => Number(value).toFixed(digits);

  async function renderResult() {
    const root = assetRoot();
    summary.innerHTML = `<p class="result-kicker">${label()} · ${state.evaluation === "content" ? "Content fidelity" : "Degradation transfer"}</p><p class="loading">Loading summary…</p>`;
    assets.innerHTML = "";
    try {
      const report = await fetch(`${root}/metrics_summary.json`).then((response) => {
        if (!response.ok) throw new Error("The selected summary is unavailable.");
        return response.json();
      });
      if (state.evaluation === "content") {
        const metrics = report.metrics_by_condition_source.cross_capture;
        summary.innerHTML = `<p class="result-kicker">${label()} · Content fidelity · ${report.num_examples} examples × ${report.samples_per_input} samples</p><div class="metrics">${metric("HR low-pass PSNR", number(metrics.hr_lowpass_psnr.mean, 2))}${metric("HR low-pass SSIM", number(metrics.hr_lowpass_ssim.mean))}${metric("Gradient cosine", number(metrics.hr_gradient_cosine.mean))}${metric("Pixel diversity", number(metrics.variance_pixel_std.mean))}</div>`;
        if (hasContentVisuals.has(key())) {
          assets.innerHTML = `<figure class="result-figure wide"><img src="${root}/visualizations/sample_grid.svg" alt="Content-fidelity samples for ${label()}"><figcaption>Sample grid</figcaption></figure><figure class="result-figure"><img src="${root}/visualizations/dashboard.svg" alt="Content-fidelity metric dashboard for ${label()}"><figcaption>Metric dashboard</figcaption></figure>`;
        } else {
          assets.innerHTML = `<p class="asset-note">This evaluation has a summary report but no saved visualizations. Run 3 qualitative exports were not generated in the source experiment.</p>`;
        }
      } else {
        const low = report.conditions.low;
        const high = report.conditions.high;
        const paired = report.paired_high_minus_low;
        summary.innerHTML = `<p class="result-kicker">${label()} · Degradation transfer · ${low.assigned_degradation_distance.count} generated samples per condition</p><div class="metrics">${metric("Low condition accuracy", `${number(low.condition_accuracy * 100, 0)}%`)}${metric("High condition accuracy", `${number(high.condition_accuracy * 100, 0)}%`)}${metric("Low assigned distance", number(low.assigned_degradation_distance.mean))}${metric("High − low distance", number(paired.degradation_distance_between_generated_conditions.mean))}</div>`;
        assets.innerHTML = `<figure class="result-figure wide"><img src="${root}/visualizations/generated_pairs_grid.svg" alt="Degradation-transfer generations for ${label()}"><figcaption>Generated pairs under low and high degradation conditions</figcaption></figure><figure class="result-figure"><img src="${root}/visualizations/dashboard.svg" alt="Degradation-transfer metric dashboard for ${label()}"><figcaption>Metric dashboard</figcaption></figure>`;
      }
    } catch (error) {
      summary.innerHTML = `<p class="asset-note">${error.message}</p>`;
    }
  }

  function renderControls() {
    runControls.replaceChildren(...Object.keys(runs).map((run) => button(run.replace("run", "Run "), state.run === run, () => {
      state.run = run;
      state.epoch = runs[run][runs[run].length - 1];
      state.evaluation = hasContent.has(key()) ? "content" : "transfer";
      render();
    })));
    epochSelect.replaceChildren(...runs[state.run].map((epoch) => {
      const option = document.createElement("option");
      option.value = epoch;
      option.textContent = `Epoch ${epoch}`;
      option.selected = epoch === state.epoch;
      return option;
    }));
    epochSelect.onchange = () => {
      state.epoch = Number(epochSelect.value);
      state.evaluation = hasContent.has(key()) ? "content" : "transfer";
      render();
    };
    const available = hasContent.has(key()) ? ["content", "transfer"] : ["transfer"];
    evaluationControls.replaceChildren(...available.map((evaluation) => button(evaluation === "content" ? "Content fidelity" : "Degradation transfer", state.evaluation === evaluation, () => {
      state.evaluation = evaluation;
      render();
    })));
  }
  function render() { renderControls(); renderResult(); }
  render();
})();
