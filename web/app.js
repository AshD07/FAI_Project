/* Live demo driver: loads the project's Python code into Pyodide, steps the
 * simulation in real time and draws each snapshot on the canvas. */
(() => {
  "use strict";

  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  const DT = 1 / 30;
  const MAX_STEPS_PER_FRAME = 8;

  const $ = (id) => document.getElementById(id);
  const canvas = $("arena");
  const ctx = canvas.getContext("2d");

  const ui = {
    loading: $("loading"), loadingText: $("loading-text"), progress: $("progress-bar"),
    gameover: $("gameover"), goTitle: $("go-title"), goText: $("go-text"), goAuto: $("go-auto"),
    play: $("btn-play"), restart: $("btn-restart"), seedBtn: $("btn-seed"),
    mode: $("sel-mode"), params: $("sel-params"), seed: $("inp-seed"), speed: $("sel-speed"),
    tgGrid: $("tg-grid"), tgPred: $("tg-pred"), tgRadar: $("tg-radar"), tgAuto: $("tg-auto"),
  };

  let bridge = null;
  let state = null;
  let lastPlan = null;
  let running = false;
  let ended = false;
  let acc = 0;
  let lastTs = 0;
  let autoTimer = null;

  // ------------------------------------------------------------------ loading
  function setProgress(pct, text) {
    ui.progress.style.width = pct + "%";
    if (text) ui.loadingText.textContent = text;
  }

  async function boot() {
    try {
      setProgress(5, "Starting Python in your browser…");
      const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
      setProgress(35, "Loading NumPy and SciPy…");
      await pyodide.loadPackage(["numpy", "scipy"]);

      setProgress(70, "Loading the project's source code…");
      const manifest = await (await fetch("web/manifest.json", { cache: "no-cache" })).json();
      const root = "/home/pyodide/app";
      const made = new Set();
      const mkdirs = (dir) => {
        const parts = dir.split("/").filter(Boolean);
        let cur = "";
        for (const p of parts) {
          cur += "/" + p;
          if (made.has(cur)) continue;
          try { pyodide.FS.mkdir(cur); } catch (e) { /* exists */ }
          made.add(cur);
        }
      };
      let done = 0;
      await Promise.all(manifest.files.map(async (rel) => {
        const res = await fetch(rel, { cache: "no-cache" });
        if (!res.ok) throw new Error("Could not load " + rel + " (" + res.status + ")");
        const text = await res.text();
        const full = root + "/" + rel;
        mkdirs(full.slice(0, full.lastIndexOf("/")));
        pyodide.FS.writeFile(full, text);
        done += 1;
        setProgress(70 + Math.round((done / manifest.files.length) * 25));
      }));

      setProgress(97, "Starting the pilot…");
      pyodide.runPython(
        "import os, sys\n" +
        "os.chdir('" + root + "')\n" +
        "sys.path.insert(0, '" + root + "')\n" +
        "sys.path.insert(0, '" + root + "/web')\n" +
        "import sim_bridge\n"
      );
      bridge = pyodide.pyimport("sim_bridge");
      setProgress(100);
      ui.loading.classList.add("hidden");
      [ui.play, ui.restart, ui.seedBtn].forEach((b) => (b.disabled = false));
      startEpisode();
      requestAnimationFrame(frame);
    } catch (err) {
      console.error(err);
      ui.loadingText.textContent = "Could not start the demo: " + (err && err.message ? err.message : err);
      document.querySelector("#loading .spinner").style.display = "none";
    }
  }

  // ------------------------------------------------------------------ episode control
  function startEpisode() {
    clearTimeout(autoTimer);
    const seed = Math.max(0, parseInt(ui.seed.value, 10) || 0);
    ui.seed.value = seed;
    state = JSON.parse(bridge.new_episode(seed, ui.mode.value, ui.params.value));
    lastPlan = null;
    ended = false;
    acc = 0;
    running = true;
    ui.play.textContent = "Pause";
    ui.gameover.classList.add("hidden");
    render();
    updatePanel();
  }

  function newSeed() {
    ui.seed.value = Math.floor(Math.random() * 10000);
    startEpisode();
  }

  function togglePlay() {
    if (ended) { startEpisode(); return; }
    running = !running;
    ui.play.textContent = running ? "Pause" : "Play";
  }

  function onEnd() {
    ended = true;
    running = false;
    ui.play.textContent = "Play";
    const destroyed = !state.alive;
    ui.goTitle.textContent = destroyed ? "Ship destroyed" : (state.asteroids.length === 0 ? "Field cleared" : "Time limit reached");
    ui.goText.textContent = "Survived " + state.t.toFixed(1) + " s · score " + state.score +
      " · " + state.hits + "/" + state.shots + " hits" + (state.shots ? " (" + state.acc.toFixed(1) + "%)" : "");
    ui.goAuto.textContent = ui.tgAuto.checked ? "Next seed starts in 4 s…" : "";
    ui.gameover.classList.remove("hidden");
    if (ui.tgAuto.checked) autoTimer = setTimeout(newSeed, 4000);
  }

  // ------------------------------------------------------------------ main loop
  function frame(ts) {
    const elapsed = lastTs ? Math.min(0.25, (ts - lastTs) / 1000) : 0;
    lastTs = ts;
    if (running && bridge) {
      acc += elapsed * parseFloat(ui.speed.value);
      let n = Math.floor(acc / DT);
      if (n > 0) {
        acc -= n * DT;
        n = Math.min(n, MAX_STEPS_PER_FRAME);
        state = JSON.parse(bridge.step(n));
        if (state.decision && state.decision.astar) lastPlan = state.decision.astar;
        updatePanel();
        if (state.over) onEnd();
      }
    }
    render();
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------------ drawing
  const C = {
    bg: "#0b1220", gridLine: "rgba(148,163,184,0.06)", wall: "#334155",
    asteroid: "#94a3b8", asteroidFill: "rgba(148,163,184,0.12)",
    ship: "#38bdf8", rival: "#f97316", bulletOwn: "#fde047", bulletRival: "#fb923c",
    blocked: "rgba(239,68,68,0.22)", path: "#22c55e", pred: "rgba(167,139,250,0.75)",
    radar: "rgba(250,204,21,0.9)", aim: "#f43f5e", hazard: "#ef4444",
  };

  function render() {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = C.gridLine;
    ctx.lineWidth = 1;
    for (let x = 0; x <= canvas.width; x += 50) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
    for (let y = 0; y <= canvas.height; y += 50) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }
    ctx.strokeStyle = C.wall;
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
    if (!state) return;

    const dec = state.decision || {};

    // A* occupancy grid (20 x 20 cells of 20 px, centred on the ship at planning time)
    if (ui.tgGrid.checked && lastPlan && lastPlan.grid) {
      const ox = lastPlan.origin[0] - 200, oy = lastPlan.origin[1] - 200;
      ctx.fillStyle = C.blocked;
      lastPlan.grid.forEach((row, r) => {
        for (let c = 0; c < row.length; c++) if (row[c] === "1") ctx.fillRect(ox + c * 20, oy + r * 20, 19, 19);
      });
      ctx.strokeStyle = "rgba(148,163,184,0.25)";
      ctx.lineWidth = 1;
      ctx.strokeRect(ox, oy, 400, 400);
    }

    // Kalman predicted trajectories
    if (ui.tgPred.checked && state.tracks) {
      ctx.strokeStyle = C.pred;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 5]);
      state.tracks.forEach((t) => {
        if (!t.ok || !t.traj || !t.traj.length) return;
        ctx.beginPath();
        ctx.moveTo(t.x, t.y);
        t.traj.forEach((p) => ctx.lineTo(p[0], p[1]));
        ctx.stroke();
      });
      ctx.setLineDash([]);
    }

    // A* path and minimax waypoint
    if (lastPlan && lastPlan.path && lastPlan.path.length) {
      ctx.strokeStyle = C.path;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(lastPlan.origin[0], lastPlan.origin[1]);
      lastPlan.path.forEach((p) => ctx.lineTo(p[0], p[1]));
      ctx.stroke();
    }
    if (dec.mm && dec.mm.wp) {
      ctx.strokeStyle = C.path;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(dec.mm.wp[0], dec.mm.wp[1], 9, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.arc(dec.mm.wp[0], dec.mm.wp[1], 2.5, 0, Math.PI * 2); ctx.fillStyle = C.path; ctx.fill();
    }

    // Asteroids
    state.asteroids.forEach((a) => {
      ctx.beginPath();
      const k = 9;
      for (let i = 0; i <= k; i++) {
        const ang = (i / k) * Math.PI * 2 + a.id;
        const rr = a.r * (0.86 + 0.14 * Math.sin(a.id * 7 + i * 2.3));
        const px = a.x + Math.cos(ang) * rr, py = a.y + Math.sin(ang) * rr;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.fillStyle = C.asteroidFill; ctx.fill();
      ctx.strokeStyle = C.asteroid; ctx.lineWidth = 1.6; ctx.stroke();
    });

    // Imminent hazard ring
    if (dec.facts && dec.facts.imminent && dec.facts.hazard) {
      ctx.strokeStyle = C.hazard;
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(dec.facts.hazard[0], dec.facts.hazard[1], 40, 0, Math.PI * 2); ctx.stroke();
    }

    // Noisy radar returns
    if (ui.tgRadar.checked && state.radar) {
      ctx.fillStyle = C.radar;
      state.radar.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, Math.PI * 2); ctx.fill(); });
    }

    // Bullets
    state.bullets.forEach((b) => {
      ctx.fillStyle = b.own === 1 ? C.bulletOwn : C.bulletRival;
      ctx.beginPath(); ctx.arc(b.x, b.y, 2.5, 0, Math.PI * 2); ctx.fill();
    });

    // CSP aim point
    if (dec.csp && dec.csp.aim) {
      const [ax, ay] = dec.csp.aim;
      ctx.strokeStyle = C.aim;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(ax, ay, 7, 0, Math.PI * 2);
      ctx.moveTo(ax - 11, ay); ctx.lineTo(ax + 11, ay); ctx.moveTo(ax, ay - 11); ctx.lineTo(ax, ay + 11);
      ctx.stroke();
    }

    // Ships
    state.ships.forEach((s) => {
      if (!s.alive) return;
      const h = (s.heading * Math.PI) / 180;
      const fx = Math.sin(h), fy = -Math.cos(h);
      const rx = Math.cos(h), ry = Math.sin(h);
      const r = s.r;
      ctx.beginPath();
      ctx.moveTo(s.x + fx * r * 1.3, s.y + fy * r * 1.3);
      ctx.lineTo(s.x - fx * r * 0.9 + rx * r * 0.85, s.y - fy * r * 0.9 + ry * r * 0.85);
      ctx.lineTo(s.x - fx * r * 0.45, s.y - fy * r * 0.45);
      ctx.lineTo(s.x - fx * r * 0.9 - rx * r * 0.85, s.y - fy * r * 0.9 - ry * r * 0.85);
      ctx.closePath();
      const col = s.id === 1 ? C.ship : C.rival;
      ctx.fillStyle = col + "33"; ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
      if (s.id === 1 && dec.thrust > 0.05) {
        ctx.strokeStyle = "rgba(251,146,60,0.9)";
        ctx.beginPath();
        ctx.moveTo(s.x - fx * r * 0.6, s.y - fy * r * 0.6);
        ctx.lineTo(s.x - fx * r * (0.9 + dec.thrust), s.y - fy * r * (0.9 + dec.thrust));
        ctx.stroke();
      }
      if (s.inv) {
        ctx.strokeStyle = "rgba(56,189,248,0.4)";
        ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.arc(s.x, s.y, r + 9, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Explosions
    (state.explosions || []).forEach((e) => {
      ctx.strokeStyle = "rgba(251,191,36," + Math.max(0, e.a) + ")";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(e.x, e.y, Math.max(1, e.r), 0, Math.PI * 2); ctx.stroke();
    });

    // Clock
    ctx.fillStyle = "rgba(226,232,240,0.85)";
    ctx.font = "600 16px 'JetBrains Mono', monospace";
    ctx.fillText("t = " + state.t.toFixed(1) + " s", 16, 28);
  }

  // ------------------------------------------------------------------ explanation panel
  const RULE_LABELS = [
    ["Emergency", "Emergency collision avoidance", "emergency"],
    ["Wall", "Wall boundary repulsion", "wall"],
    ["Proximity", "Proximity hazard caution", "caution"],
    ["Caution", "Proximity hazard caution", "caution"],
    ["Mine", "Defensive mine drop", "emergency"],
    ["Cruise", "Default cruise", "cruise"],
  ];
  const CSP_NAMES = { C1: "C1 cooldown", C2: "C2 ammo", C3: "C3 aim", C4: "C4 reach", C5: "C5 standoff" };

  function fmt(v, unit, nd = 1) {
    return v === null || v === undefined ? "–" : Number(v).toFixed(nd) + unit;
  }

  function updatePanel() {
    if (!state) return;
    $("st-time").textContent = state.t.toFixed(1) + " s";
    $("st-score").textContent = state.score;
    $("st-acc").textContent = state.shots ? state.acc.toFixed(1) + "%" : "–";
    $("st-speed").textContent = Math.round(state.speed);

    const d = state.decision;
    if (!d) return;

    const rule = d.rule || "";
    const match = RULE_LABELS.find(([k]) => rule.includes(k));
    const badge = $("rule-badge");
    badge.textContent = match ? match[1] : rule;
    badge.className = "badge " + (match ? match[2] : "");
    $("rule-why").textContent = d.why || "";

    if (d.facts) {
      $("f-tti").textContent = fmt(d.facts.tti, " s", 2);
      $("f-closure").textContent = fmt(d.facts.closure, " px/s", 0);
      $("f-near").textContent = fmt(d.facts.nearest, " px", 0);
    }

    if (d.mm) {
      $("mm-tactic").textContent = d.mm.tactic.replace(/_/g, " ");
      $("mm-u").textContent = d.mm.searched ? "U = " + d.mm.u : "";
      $("mm-why").textContent = d.mm.searched
        ? (d.mm.why || "")
        : "No rival on radar: a rule picks retreat or contest; " + (d.mm.target || "");
      $("mm-search").textContent = d.mm.searched
        ? "Search: " + d.mm.nodes + " leaves evaluated, " + d.mm.pruned + " cut-offs. Rival reply assumed: " + d.mm.rival.replace(/_/g, " ")
        : "Minimax searches only when a rival ship is present (benchmarks 3 and 4).";
    }

    if (d.csp) {
      const chips = $("csp-chips");
      chips.textContent = "";
      Object.keys(CSP_NAMES).forEach((k) => {
        const key = Object.keys(d.csp.audit).find((a) => a === k || a.startsWith(k + "_"));
        if (!key) return;
        const el = document.createElement("span");
        el.className = "chip " + (d.csp.audit[key] ? "pass" : "fail");
        el.textContent = CSP_NAMES[k];
        chips.appendChild(el);
      });
      $("csp-why").textContent = d.csp.why || "";
    }

    const turn = Math.max(-180, Math.min(180, d.turn || 0));
    const mt = $("m-turn");
    mt.style.left = turn >= 0 ? "50%" : (50 + (turn / 180) * 50) + "%";
    mt.style.width = Math.abs(turn / 180) * 50 + "%";
    $("v-turn").textContent = Math.round(turn) + "°/s";
    $("m-thrust").style.width = Math.max(0, Math.min(1, d.thrust || 0)) * 100 + "%";
    $("v-thrust").textContent = (d.thrust || 0).toFixed(2);
    const fire = $("v-fire");
    fire.textContent = d.shoot ? "FIRE" : "hold";
    fire.className = "fire" + (d.shoot ? " on" : "");
  }

  // ------------------------------------------------------------------ inputs
  ui.play.addEventListener("click", togglePlay);
  ui.restart.addEventListener("click", startEpisode);
  ui.seedBtn.addEventListener("click", newSeed);
  $("go-replay").addEventListener("click", startEpisode);
  $("go-next").addEventListener("click", newSeed);
  [ui.mode, ui.params].forEach((el) => el.addEventListener("change", () => bridge && startEpisode()));
  ui.seed.addEventListener("change", () => bridge && startEpisode());
  [ui.tgGrid, ui.tgPred, ui.tgRadar].forEach((el) => el.addEventListener("change", render));
  ui.tgAuto.addEventListener("change", () => { if (!ui.tgAuto.checked) clearTimeout(autoTimer); });

  window.addEventListener("keydown", (e) => {
    if (!bridge) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.key === "r" || e.key === "R") startEpisode();
    else if (e.key === "n" || e.key === "N") newSeed();
    else if (e.key === "g" || e.key === "G") { ui.tgGrid.checked = !ui.tgGrid.checked; render(); }
  });

  render();
  boot();
})();
