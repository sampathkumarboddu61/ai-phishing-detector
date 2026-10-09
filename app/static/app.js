// Phishing URL Detector front end.
// SECURITY: every piece of server/user supplied text is inserted with textContent
// (never innerHTML), so a hostile URL such as "<img src=x onerror=...>" is shown
// as inert text. URLs are also never rendered as links.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const form = $("scan-form");
  const input = $("url-input");
  const button = $("scan-btn");
  const errorBox = $("error");

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = !message;
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    let body = null;
    try { body = await response.json(); } catch (_) { /* non-JSON error page */ }
    if (!response.ok) {
      throw new Error((body && body.error) || `Request failed (${response.status})`);
    }
    return body;
  }

  // ---- rendering ----------------------------------------------------------
  const VERDICT_TEXT = {
    low: ["Likely legitimate", "Looks legitimate"],
    medium: ["Suspicious", "Likely phishing"],
    high: ["Phishing", "Likely phishing"],
  };

  function renderResult(r) {
    const [badgeText, title] = VERDICT_TEXT[r.risk_level];
    const badge = $("verdict-badge");
    badge.textContent = badgeText;
    badge.className = `badge ${r.risk_level}`;
    $("verdict-title").textContent = r.low_confidence ? `${title} (low confidence)` : title;
    $("result-url").textContent = r.url;

    $("score-text").textContent = `${r.risk_score} / 100`;
    const meter = $("meter");
    meter.className = `meter ${r.risk_level}`;
    meter.setAttribute("aria-label", `Risk score ${r.risk_score} out of 100`);
    $("meter-fill").style.width = `${Math.max(2, Math.min(100, r.risk_score))}%`;
    $("meter-threshold").style.left = `${r.decision_threshold}%`;
    $("threshold-text").textContent =
      `Scores at or above ${r.decision_threshold} are flagged. The threshold is set low on purpose, ` +
      `because missing a phishing link is worse than raising a false alarm.`;

    const warnings = $("warnings");
    warnings.replaceChildren(...r.warnings.map((w) => el("li", "", w)));
    warnings.hidden = r.warnings.length === 0;

    const list = $("indicators");
    if (r.indicators.length === 0) {
      list.replaceChildren(el("li", "indicator-none", "Nothing unusual was found in the address itself."));
    } else {
      list.replaceChildren(...r.indicators.map((i) => {
        const li = el("li");
        li.append(el("span", `sev ${i.severity}`, i.severity), el("span", "", i.message));
        return li;
      }));
    }

    const liveBlock = $("live-block");
    const live = r.live_checks;
    liveBlock.hidden = !live;
    if (live) {
      const dl = $("live-list");
      const rows = [];
      if (live.note && live.redirect_count === undefined) {
        rows.push(["Note", live.note]);
      } else {
        rows.push(["Redirects", live.redirect_count === null ? "unknown" : String(live.redirect_count)]);
        if (live.final_host) rows.push(["Final host", live.final_host]);
        rows.push(["Domain age", live.domain_age_days === null || live.domain_age_days === undefined
          ? "unknown" : `${live.domain_age_days} days`]);
        if (live.note) rows.push(["Note", live.note]);
      }
      dl.replaceChildren(...rows.flatMap(([k, v]) => [el("dt", "", k), el("dd", "", v)]));
    }

    $("feature-rows").replaceChildren(...Object.entries(r.features).map(([k, v]) => {
      const tr = el("tr");
      tr.append(el("td", "", k), el("td", "num", String(v)));
      return tr;
    }));

    $("result").hidden = false;
  }

  function renderHistory(rows) {
    $("history-empty").hidden = rows.length > 0;
    $("history-rows").replaceChildren(...rows.map((row) => {
      const tr = el("tr");
      const when = new Date(row.created_at);
      const urlCell = el("td", "cell-url", row.url);
      urlCell.title = row.url;
      const tag = el("span", `tag ${row.label}`, row.label === "phishing" ? "Phishing" : "Legitimate");
      const verdict = el("td");
      verdict.append(tag);
      if (row.low_confidence) verdict.append(" ", el("span", "muted small", "(low conf.)"));
      tr.append(el("td", "muted small", isNaN(when) ? row.created_at : when.toLocaleString()),
                urlCell, verdict, el("td", "num", String(row.risk_score)));
      return tr;
    }));
  }

  async function refreshSideData() {
    try {
      const [stats, history] = await Promise.all([api("/api/stats"), api("/api/history?limit=10")]);
      $("stat-total").textContent = stats.total;
      $("stat-phish").textContent = stats.phishing;
      $("stat-legit").textContent = stats.legitimate;
      renderHistory(history);
    } catch (_) { /* side panels are non-essential */ }
  }

  // ---- events -------------------------------------------------------------
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const url = input.value.trim();
    if (!url) { showError("Please enter a URL."); input.focus(); return; }
    showError("");
    button.disabled = true;
    button.textContent = "Analyzing…";
    try {
      const result = await api("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, live_checks: $("live-input").checked }),
      });
      renderResult(result);
      refreshSideData();
    } catch (err) {
      $("result").hidden = true;
      showError(err.message);
    } finally {
      button.disabled = false;
      button.textContent = "Analyze";
    }
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.url;
      form.requestSubmit();
    });
  });

  // ---- startup ------------------------------------------------------------
  (async () => {
    try {
      const health = await api("/api/health");
      if (!health.model_loaded) showError("The model has not been trained yet. Run: python -m src.train");
      $("live-wrap").hidden = !health.live_checks_enabled;
      const info = await api("/api/model-info");
      const badge = $("model-badge");
      badge.textContent = info.model;
      badge.hidden = false;
    } catch (_) { /* handled by the other calls */ }
    refreshSideData();
  })();
})();
