// ============================================================================
// Identifying Normative Categories — 3D Gaussian-mixture clustering demo
//
// Data model: a "base state" is sampled with k clusters as if the
// separation and variance sliders were both maxed out — cluster means
// placed via a separation band, points drawn from each cluster's diagonal
// Gaussian. The separation and variance sliders then don't resample
// anything; they linearly rescale that fixed base state (mean distance
// from the origin, and each point's offset from its own cluster mean,
// respectively) as a percentage. This keeps their effect on the plot
// exactly predictable and lets points move continuously instead of
// jumping between independent random draws. Only the cluster-count slider
// and the Resample button actually draw fresh randomness.
//
// Rendering: Three.js (vendored locally in lib/three.min.js, loaded as a
// classic global script — not an ES module — so this works from a flash
// drive over file:// without any server or internet access).
// ============================================================================

(function () {
  'use strict';

  // ---------------------------------------------------------------------
  // Configuration constants
  // ---------------------------------------------------------------------
  const CUBE_HALF = 10;              // reference scale for axis-line length, label placement, and Position color normalization
  const TOTAL_POINTS = 300;          // fixed total point count, split evenly across clusters
  const MAX_SEPARATION = 14;         // data-space units at separation slider = 100
  // Cluster means must land not just at least minSeparation from their
  // nearest neighbor, but at most that plus this band width — otherwise a
  // low slider value could still occasionally place means very far apart
  // by chance, which made the slider's effect hard to predict.
  const SEPARATION_BAND_WIDTH = MAX_SEPARATION * 0.4;
  const MAX_VARIANCE_MEAN = 6;       // per-axis variance mean at slider = 100
  const VARIANCE_SPREAD_FRAC = 0.3;  // sd of the per-axis variance draw, as a fraction of its mean
  const MIN_VARIANCE = 0.05;         // floor so a cluster can never collapse to a single point
  const TRANSITION_MS = 450;         // duration of the position/color tween on resample
  const RESAMPLE_THROTTLE_MS = 130;  // minimum time between resamples while dragging a slider
  // Axis labels, arbitrarily assigned to x/y/z (spec says "in no particular order")
  const AXIS_LABELS = ['Seriousness', 'Universality', 'Authority Independence'];
  const AXIS_COLORS = [0xff6b6b, 0x6bff8f, 0x6b9bff]; // tint each axis to match the Position color mapping

  // ---------------------------------------------------------------------
  // Random sampling helpers
  // ---------------------------------------------------------------------
  function randUniform(min, max) {
    return min + Math.random() * (max - min);
  }

  // Box-Muller transform: turns two uniform draws into one standard-normal draw.
  function randStandardNormal() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function randGaussian(mean, sd) {
    return mean + sd * randStandardNormal();
  }

  // ---------------------------------------------------------------------
  // Dataset generation
  // ---------------------------------------------------------------------

  // Sample k cluster means uniformly in the reference cube. Each new
  // candidate's distance to its *nearest* already-placed mean must fall
  // within [minSeparation, maxSeparation] — not just "at least," since an
  // unbounded upper end let a low separation slider occasionally place
  // means far apart purely by chance, making the slider's effect
  // unpredictable. If placement keeps failing (can happen for large k,
  // since you can't pack too many means into a narrow distance band within
  // a fixed volume), the band is widened from both ends so this never hangs.
  function sampleClusterMeans(k, minSeparation, maxSeparation) {
    const means = [];
    const maxAttemptsBeforeRelax = 500;
    let requiredMin = minSeparation;
    let requiredMax = maxSeparation;

    for (let i = 0; i < k; i++) {
      let attempts = 0;
      let placed = false;
      while (!placed) {
        const candidate = [
          randUniform(-CUBE_HALF, CUBE_HALF),
          randUniform(-CUBE_HALF, CUBE_HALF),
          randUniform(-CUBE_HALF, CUBE_HALF)
        ];
        let nearestDist = Infinity;
        for (let j = 0; j < means.length; j++) {
          const dx = candidate[0] - means[j][0];
          const dy = candidate[1] - means[j][1];
          const dz = candidate[2] - means[j][2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < nearestDist) nearestDist = dist;
        }
        const inBand = means.length === 0 || (nearestDist >= requiredMin && nearestDist <= requiredMax);
        attempts++;
        if (inBand) {
          means.push(candidate);
          placed = true;
        } else if (attempts >= maxAttemptsBeforeRelax) {
          requiredMin *= 0.85;
          requiredMax *= 1.15;
          attempts = 0;
        }
      }
    }
    return means;
  }

  // Split TOTAL_POINTS as evenly as possible across k clusters.
  function splitCounts(total, k) {
    const base = Math.floor(total / k);
    const remainder = total - base * k;
    const counts = new Array(k).fill(base);
    for (let i = 0; i < remainder; i++) counts[i]++;
    return counts;
  }

  // Builds a fresh "base state" for k clusters, sampled as if both sliders
  // were maxed out: means placed at the maximum separation band, points
  // drawn from each cluster's diagonal Gaussian at the maximum variance.
  // The separation/variance sliders never resample — they just rescale
  // this fixed base state (see computeDisplayedDataset) — so this is the
  // only place randomness enters, besides the Resample button.
  //
  // Each point is stored as a residual (its offset from its own cluster's
  // *empirical* mean, not the nominal sampled mean) together with its
  // cluster index. Two exact-centering steps happen here, once, so that
  // no matter how the sliders later rescale things, the whole cloud stays
  // centered on the origin:
  //   1. baseMeans are shifted so their point-count-weighted average is
  //      exactly the origin.
  //   2. Each cluster's residuals are taken relative to that cluster's own
  //      empirical mean, so they average to exactly zero — removing the
  //      finite-sample noise that would otherwise leave a tiny residual
  //      offset (this also makes each cluster's true centroid land exactly
  //      on its scaled mean at every separation setting, which is what
  //      Cluster-mode color relies on).
  function generateBaseState(k) {
    const rawMeans = sampleClusterMeans(k, MAX_SEPARATION, MAX_SEPARATION + SEPARATION_BAND_WIDTH);
    const counts = splitCounts(TOTAL_POINTS, k);

    const weightedMean = [0, 0, 0];
    for (let c = 0; c < k; c++) {
      for (let ax = 0; ax < 3; ax++) weightedMean[ax] += rawMeans[c][ax] * counts[c];
    }
    for (let ax = 0; ax < 3; ax++) weightedMean[ax] /= TOTAL_POINTS;
    const baseMeans = rawMeans.map(function (m) {
      return [m[0] - weightedMean[0], m[1] - weightedMean[1], m[2] - weightedMean[2]];
    });

    const basePoints = [];
    for (let c = 0; c < k; c++) {
      // Each cluster's per-axis variance is itself a random draw centered
      // on the maximum, per spec ("variance along each dimension should be
      // randomly sampled from a gaussian with the mean set by this slider").
      const variance = [0, 1, 2].map(function () {
        const sd = MAX_VARIANCE_MEAN * VARIANCE_SPREAD_FRAC;
        return Math.max(MIN_VARIANCE, randGaussian(MAX_VARIANCE_MEAN, sd));
      });

      const rawPoints = [];
      for (let n = 0; n < counts[c]; n++) {
        rawPoints.push([0, 1, 2].map(function (ax) {
          return randGaussian(baseMeans[c][ax], Math.sqrt(variance[ax]));
        }));
      }
      const empiricalMean = [0, 0, 0];
      for (let n = 0; n < rawPoints.length; n++) {
        for (let ax = 0; ax < 3; ax++) empiricalMean[ax] += rawPoints[n][ax];
      }
      for (let ax = 0; ax < 3; ax++) empiricalMean[ax] /= rawPoints.length;

      for (let n = 0; n < rawPoints.length; n++) {
        basePoints.push({
          residual: [
            rawPoints[n][0] - empiricalMean[0],
            rawPoints[n][1] - empiricalMean[1],
            rawPoints[n][2] - empiricalMean[2]
          ],
          cluster: c
        });
      }
    }

    return { baseMeans: baseMeans, basePoints: basePoints };
  }

  // Rescales a base state by the current slider percentages: cluster means
  // move radially toward/away from the origin by sepPct, and each point's
  // offset from its (rescaled) cluster mean is scaled by varPct. No new
  // randomness is introduced — every point's identity and relative
  // position within its cluster stays exactly fixed as the sliders move.
  function computeDisplayedDataset(baseState, sepPct, varPct) {
    const means = baseState.baseMeans.map(function (m) {
      return [m[0] * sepPct, m[1] * sepPct, m[2] * sepPct];
    });
    const points = baseState.basePoints.map(function (bp) {
      const mean = means[bp.cluster];
      return {
        pos: [
          mean[0] + bp.residual[0] * varPct,
          mean[1] + bp.residual[1] * varPct,
          mean[2] + bp.residual[2] * varPct
        ],
        cluster: bp.cluster
      };
    });
    return { means: means, points: points };
  }

  // ---------------------------------------------------------------------
  // Color mapping
  // ---------------------------------------------------------------------

  // "Position" mode: map each axis linearly from [-CUBE_HALF, CUBE_HALF]
  // to [0, 1] and use those as RGB. Axis tint colors (see AXIS_COLORS)
  // are chosen to match this (x -> red, y -> green, z -> blue).
  function positionColor(pos) {
    return [
      (pos[0] + CUBE_HALF) / (2 * CUBE_HALF),
      (pos[1] + CUBE_HALF) / (2 * CUBE_HALF),
      (pos[2] + CUBE_HALF) / (2 * CUBE_HALF)
    ];
  }

  const NONE_COLOR = [0.55, 0.65, 0.95];

  // Returns an array of [r,g,b] (0-1), one per point in dataset.points,
  // in the same order as dataset.points.
  function computeColors(dataset, colorMode) {
    if (colorMode === 'position') {
      return dataset.points.map(function (p) { return positionColor(p.pos); });
    }
    if (colorMode === 'cluster') {
      // Cluster color = the Position-mode color of that cluster's mean,
      // so switching between Cluster/Position reveals the same categories.
      const clusterColors = dataset.means.map(positionColor);
      return dataset.points.map(function (p) { return clusterColors[p.cluster]; });
    }
    return dataset.points.map(function () { return NONE_COLOR; });
  }

  // ---------------------------------------------------------------------
  // Greedy farthest-first matching between old and new point sets.
  //
  // Goal: pair each of the n "old" (currently displayed) points with one
  // of the n "new" (freshly sampled) points, approximately minimizing the
  // *maximum* distance any pair has to travel, so the resample animation
  // reads as a smooth deformation rather than points darting past each
  // other. This is a greedy heuristic, not an exact bottleneck-matching
  // solver: repeatedly take the still-unassigned old point whose nearest
  // still-unassigned new point is farthest away (the "hardest to place"
  // point) and commit it to that nearest neighbor.
  //
  // Cost is roughly O(n^3) in the worst case; at n = TOTAL_POINTS = 150
  // that's a few million flops, done well within one animation frame.
  // ---------------------------------------------------------------------
  function computeMatching(oldPositions, newPositions, n) {
    const unassignedOld = [];
    const unassignedNew = [];
    for (let i = 0; i < n; i++) {
      unassignedOld.push(i);
      unassignedNew.push(i);
    }
    const mapping = new Array(n); // mapping[oldIndex] = newIndex

    function dist2(a, aIdx, b, bIdx) {
      const dx = a[aIdx * 3] - b[bIdx * 3];
      const dy = a[aIdx * 3 + 1] - b[bIdx * 3 + 1];
      const dz = a[aIdx * 3 + 2] - b[bIdx * 3 + 2];
      return dx * dx + dy * dy + dz * dz;
    }

    while (unassignedOld.length > 0) {
      let worstOldSlot = -1;
      let worstOldNearestNewSlot = -1;
      let worstOldNearestDist = -Infinity;

      for (let oi = 0; oi < unassignedOld.length; oi++) {
        const oIdx = unassignedOld[oi];
        let nearestDist = Infinity;
        let nearestNewSlot = -1;
        for (let ni = 0; ni < unassignedNew.length; ni++) {
          const nIdx = unassignedNew[ni];
          const d = dist2(oldPositions, oIdx, newPositions, nIdx);
          if (d < nearestDist) {
            nearestDist = d;
            nearestNewSlot = ni;
          }
        }
        if (nearestDist > worstOldNearestDist) {
          worstOldNearestDist = nearestDist;
          worstOldSlot = oi;
          worstOldNearestNewSlot = nearestNewSlot;
        }
      }

      const oIdx = unassignedOld[worstOldSlot];
      const nIdx = unassignedNew[worstOldNearestNewSlot];
      mapping[oIdx] = nIdx;
      unassignedOld.splice(worstOldSlot, 1);
      unassignedNew.splice(worstOldNearestNewSlot, 1);
    }
    return mapping;
  }

  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  // ---------------------------------------------------------------------
  // DOM references
  // ---------------------------------------------------------------------
  const container = document.getElementById('plot-container');
  const plotWrap = document.getElementById('plot-wrap');
  const kSlider = document.getElementById('k-slider');
  const kValueLabel = document.getElementById('k-value');
  const sepSlider = document.getElementById('sep-slider');
  const varSlider = document.getElementById('var-slider');
  const colorModeSelect = document.getElementById('color-mode');
  const resampleBtn = document.getElementById('resample-btn');

  // ---------------------------------------------------------------------
  // Three.js scene setup
  // ---------------------------------------------------------------------
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(0x101114, 1);
  container.appendChild(renderer.domElement);

  // Colored axis lines through the origin — no bounding box, since points
  // aren't confined to a fixed volume (variance/separation are unbounded).
  const axisDirs = [
    { x: 1, y: 0, z: 0 },
    { x: 0, y: 1, z: 0 },
    { x: 0, y: 0, z: 1 }
  ];
  axisDirs.forEach(function (dir, i) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-dir.x * CUBE_HALF, -dir.y * CUBE_HALF, -dir.z * CUBE_HALF),
      new THREE.Vector3(dir.x * CUBE_HALF, dir.y * CUBE_HALF, dir.z * CUBE_HALF)
    ]);
    const mat = new THREE.LineBasicMaterial({ color: AXIS_COLORS[i], transparent: true, opacity: 0.6 });
    scene.add(new THREE.Line(geo, mat));
  });

  // Point cloud. `displayPositions`/`displayColors` are the live buffers:
  // the BufferAttributes below wrap these arrays directly (no copying), so
  // mutating them in place and flagging needsUpdate is all rendering needs.
  const displayPositions = new Float32Array(TOTAL_POINTS * 3);
  const displayColors = new Float32Array(TOTAL_POINTS * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(displayPositions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(displayColors, 3));
  const pointsMaterial = new THREE.PointsMaterial({ size: 0.35, vertexColors: true, sizeAttenuation: true });
  scene.add(new THREE.Points(geometry, pointsMaterial));

  // HTML overlay labels for the three axes, projected from 3D each frame.
  const axisLabelPoints = [
    new THREE.Vector3(CUBE_HALF * 1.3, 0, 0),
    new THREE.Vector3(0, CUBE_HALF * 1.3, 0),
    new THREE.Vector3(0, 0, CUBE_HALF * 1.3)
  ];
  const axisLabelEls = AXIS_LABELS.map(function (text, i) {
    const el = document.createElement('div');
    el.className = 'axis-label';
    el.textContent = text;
    el.style.color = '#' + AXIS_COLORS[i].toString(16).padStart(6, '0');
    container.appendChild(el);
    return el;
  });

  // Labels sit just beyond the cube in world space, but since the camera
  // orbits freely, that projected point can land outside the container in
  // some orientations. Clamp to a margin so a label slides along the edge
  // instead of being clipped by the container's rounded-corner overflow.
  const LABEL_MARGIN_X = 70;
  const LABEL_MARGIN_Y = 14;
  function updateLabelPositions() {
    const w = container.clientWidth, h = container.clientHeight;
    for (let i = 0; i < axisLabelEls.length; i++) {
      const p = axisLabelPoints[i].clone().project(camera);
      const x = (p.x * 0.5 + 0.5) * w;
      const y = (-p.y * 0.5 + 0.5) * h;
      axisLabelEls[i].style.left = Math.max(LABEL_MARGIN_X, Math.min(w - LABEL_MARGIN_X, x)) + 'px';
      axisLabelEls[i].style.top = Math.max(LABEL_MARGIN_Y, Math.min(h - LABEL_MARGIN_Y, y)) + 'px';
    }
  }

  // ---------------------------------------------------------------------
  // Custom orbit camera (drag to rotate, wheel to zoom) — hand-rolled
  // instead of Three.js's OrbitControls.js to avoid vendoring a second file.
  // ---------------------------------------------------------------------
  let theta = Math.PI / 4;   // azimuth
  let phi = Math.PI / 3;     // polar angle, clamped away from the poles
  let radius = CUBE_HALF * 3.2;
  const MIN_RADIUS = CUBE_HALF * 1.6;
  const MAX_RADIUS = CUBE_HALF * 6;
  let isDragging = false;
  let lastPointerX = 0, lastPointerY = 0;

  container.addEventListener('pointerdown', function (e) {
    e.preventDefault(); // belt-and-suspenders alongside touch-action:none in CSS
    isDragging = true;
    lastPointerX = e.clientX;
    lastPointerY = e.clientY;
    container.classList.add('dragging');
  });
  window.addEventListener('pointerup', function () {
    isDragging = false;
    container.classList.remove('dragging');
  });
  window.addEventListener('pointermove', function (e) {
    if (!isDragging) return;
    const dx = e.clientX - lastPointerX;
    const dy = e.clientY - lastPointerY;
    lastPointerX = e.clientX;
    lastPointerY = e.clientY;
    theta -= dx * 0.005;
    phi -= dy * 0.005;
    phi = Math.max(0.15, Math.min(Math.PI - 0.15, phi));
  });
  container.addEventListener('wheel', function (e) {
    e.preventDefault();
    radius *= (1 + e.deltaY * 0.001);
    radius = Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, radius));
  }, { passive: false });

  function updateCamera() {
    camera.position.set(
      radius * Math.sin(phi) * Math.sin(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.cos(theta)
    );
    camera.lookAt(0, 0, 0);
  }

  // ---------------------------------------------------------------------
  // Sizing: the plot area is a square that fills whatever vertical space
  // #plot-wrap has left after the title and controls.
  // ---------------------------------------------------------------------
  function resize() {
    const size = Math.floor(Math.min(plotWrap.clientWidth, plotWrap.clientHeight));
    container.style.width = size + 'px';
    container.style.height = size + 'px';
    renderer.setSize(size, size, true);
    camera.aspect = 1;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);

  // ---------------------------------------------------------------------
  // Resample / rescale / recolor / animation state
  // ---------------------------------------------------------------------
  let baseState = null;      // { baseMeans, basePoints } — the fixed random draw;
                              // only replaced by the k slider or the Resample button
  let currentDataset = null; // { means, points } — the currently *displayed* dataset
                              // (baseState rescaled by the current slider percentages),
                              // points in the same order as the display slots (0..TOTAL_POINTS-1)
  let colorMode = colorModeSelect.value;
  let animating = false;
  let animStart = 0;
  let animFrom = { positions: null, colors: null };
  let animTo = { positions: null, colors: null };

  function sepPct() { return parseFloat(sepSlider.value) / 100; }
  function varPct() { return parseFloat(varSlider.value) / 100; }

  function flattenPositions(dataset) {
    const arr = new Float32Array(TOTAL_POINTS * 3);
    for (let i = 0; i < TOTAL_POINTS; i++) {
      arr[i * 3] = dataset.points[i].pos[0];
      arr[i * 3 + 1] = dataset.points[i].pos[1];
      arr[i * 3 + 2] = dataset.points[i].pos[2];
    }
    return arr;
  }

  // If a resample/recolor tween is still in flight, snap it straight to its
  // end state. Called before a direct rescale update so the two update
  // paths (animated tween vs. direct write) never fight over the display
  // buffers on the same frame.
  function finishAnimation() {
    if (!animating) return;
    displayPositions.set(animTo.positions);
    displayColors.set(animTo.colors);
    animating = false;
  }

  // Begin an animated transition to a freshly generated dataset: matches
  // old points to new ones (see computeMatching), then reorders the new
  // dataset into the matched slot order so `currentDataset` stays aligned
  // with the display buffers for any future recolor-only transition.
  function startResampleTransition(newDataset) {
    const fromPositions = displayPositions.slice();
    const fromColors = displayColors.slice();
    const newPositionsFlat = flattenPositions(newDataset);
    const newColorsFlat = computeColors(newDataset, colorMode);

    const mapping = computeMatching(fromPositions, newPositionsFlat, TOTAL_POINTS);

    const toPositions = new Float32Array(TOTAL_POINTS * 3);
    const toColors = new Float32Array(TOTAL_POINTS * 3);
    const reorderedPoints = new Array(TOTAL_POINTS);
    for (let i = 0; i < TOTAL_POINTS; i++) {
      const j = mapping[i];
      toPositions[i * 3] = newPositionsFlat[j * 3];
      toPositions[i * 3 + 1] = newPositionsFlat[j * 3 + 1];
      toPositions[i * 3 + 2] = newPositionsFlat[j * 3 + 2];
      const col = newColorsFlat[j];
      toColors[i * 3] = col[0];
      toColors[i * 3 + 1] = col[1];
      toColors[i * 3 + 2] = col[2];
      reorderedPoints[i] = newDataset.points[j];
    }
    currentDataset = { means: newDataset.means, points: reorderedPoints };

    animFrom = { positions: fromPositions, colors: fromColors };
    animTo = { positions: toPositions, colors: toColors };
    animStart = performance.now();
    animating = true;
  }

  // Recompute colors only (positions unchanged) — used when the color
  // mode dropdown changes without resampling the underlying points.
  function startRecolorTransition() {
    if (!currentDataset) return;
    const newColors = computeColors(currentDataset, colorMode);
    const toColors = new Float32Array(TOTAL_POINTS * 3);
    for (let i = 0; i < TOTAL_POINTS; i++) {
      const col = newColors[i];
      toColors[i * 3] = col[0];
      toColors[i * 3 + 1] = col[1];
      toColors[i * 3 + 2] = col[2];
    }
    animFrom = { positions: displayPositions.slice(), colors: displayColors.slice() };
    animTo = { positions: displayPositions.slice(), colors: toColors };
    animStart = performance.now();
    animating = true;
  }

  // Directly writes the display buffers from a rescaled dataset — no
  // matching, no tween. Rescaling a base state never changes point
  // identity or order, so there's nothing to animate: setting the buffers
  // on every slider 'input' event already reads as smooth, continuous
  // motion as the user drags.
  function applyDatasetDirect(dataset) {
    finishAnimation();
    currentDataset = dataset;
    const colors = computeColors(dataset, colorMode);
    for (let i = 0; i < TOTAL_POINTS; i++) {
      displayPositions[i * 3] = dataset.points[i].pos[0];
      displayPositions[i * 3 + 1] = dataset.points[i].pos[1];
      displayPositions[i * 3 + 2] = dataset.points[i].pos[2];
      const c = colors[i];
      displayColors[i * 3] = c[0];
      displayColors[i * 3 + 1] = c[1];
      displayColors[i * 3 + 2] = c[2];
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  }

  // Triggered by the k slider and the Resample button — the only two
  // controls that actually draw fresh randomness.
  function triggerFullResample() {
    const k = parseInt(kSlider.value, 10);
    baseState = generateBaseState(k);
    startResampleTransition(computeDisplayedDataset(baseState, sepPct(), varPct()));
  }

  // Triggered by the separation/variance sliders: rescales the existing
  // base state in place, no resampling.
  function applyPctRescale() {
    if (!baseState) return;
    applyDatasetDirect(computeDisplayedDataset(baseState, sepPct(), varPct()));
  }

  // Throttle full resamples while the k slider is being dragged: run at
  // most once per RESAMPLE_THROTTLE_MS, but always schedule a trailing
  // call so the final slider position (e.g. on release) is never dropped.
  // Separation/variance don't need this — see applyPctRescale.
  let throttleTimer = null;
  let lastResampleTime = 0;
  function throttledCall(fn) {
    const now = performance.now();
    const elapsed = now - lastResampleTime;
    if (elapsed >= RESAMPLE_THROTTLE_MS) {
      lastResampleTime = now;
      fn();
    } else {
      clearTimeout(throttleTimer);
      throttleTimer = setTimeout(function () {
        lastResampleTime = performance.now();
        fn();
      }, RESAMPLE_THROTTLE_MS - elapsed);
    }
  }

  kSlider.addEventListener('input', function () {
    kValueLabel.textContent = kSlider.value;
    throttledCall(triggerFullResample);
  });
  sepSlider.addEventListener('input', applyPctRescale);
  varSlider.addEventListener('input', applyPctRescale);
  colorModeSelect.addEventListener('change', function () {
    colorMode = colorModeSelect.value;
    startRecolorTransition();
  });
  resampleBtn.addEventListener('click', triggerFullResample);

  // ---------------------------------------------------------------------
  // Initial dataset (no animation — set the display buffers directly)
  // ---------------------------------------------------------------------
  function initDataset() {
    const k = parseInt(kSlider.value, 10);
    baseState = generateBaseState(k);
    applyDatasetDirect(computeDisplayedDataset(baseState, sepPct(), varPct()));
  }

  // ---------------------------------------------------------------------
  // Main loop
  // ---------------------------------------------------------------------
  function animate() {
    requestAnimationFrame(animate);

    if (animating) {
      const t = Math.min(1, (performance.now() - animStart) / TRANSITION_MS);
      const e = easeInOutCubic(t);
      for (let idx = 0; idx < TOTAL_POINTS * 3; idx++) {
        displayPositions[idx] = animFrom.positions[idx] + (animTo.positions[idx] - animFrom.positions[idx]) * e;
        displayColors[idx] = animFrom.colors[idx] + (animTo.colors[idx] - animFrom.colors[idx]) * e;
      }
      geometry.attributes.position.needsUpdate = true;
      geometry.attributes.color.needsUpdate = true;
      if (t >= 1) animating = false;
    }

    updateCamera();
    updateLabelPositions();
    renderer.render(scene, camera);
  }

  resize();
  initDataset();
  animate();
})();
