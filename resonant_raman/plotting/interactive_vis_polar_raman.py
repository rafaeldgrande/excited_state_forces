"""
RETIRED (2026-08-05): this standalone viewer has been superseded by
`interactive_vis_resonant_map.py --plot-polar-plots`, which adds the same
polar panel (plus a new helicity panel for sigma_+/sigma_-) to the main
viewer, wired to its existing map click, instead of duplicating the guide
map in a second file. Kept here unmodified for reference; not deployed or
regenerated going forward -- see resonant_raman/README.md's entry for
interactive_vis_resonant_map.py.

Generate a self-contained interactive HTML viewer for angle-resolved
polarized resonant Raman intensities I_parallel(theta), I_perp(theta).

Reads, per flavor N:
  - resonant_raman_data_flavor{N}.h5   (existing, full-resolution Cartesian
    output) -- only raman_map_unpolarized is used here, as a "guide map"
    background exactly like interactive_vis_resonant_map.py's left panel.
  - resonant_raman_polar_flavor{N}.h5  (from resonant_raman.py --polarized
    --polar-store-maps) -- raman_map_parallel/perpendicular, downsampled and
    embedded so the browser can look up the nearest (Omega_exc, omega_ph)
    point and render its polar curve live.

Left panel   -- 2-D unpolarized guide map (click anywhere to set both
                 Omega_exc and omega_ph, exactly like interactive_vis_resonant_map.py)
Right panel  -- polar plot: I_parallel(theta), I_perp(theta) at the nearest
                 embedded (Omega_exc, omega_ph) grid point

Usage:
  python interactive_vis_polar_raman.py
  python interactive_vis_polar_raman.py --data-dir /path/to/run --output polar_viewer.html
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import FLAVOR_DESC

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Generate interactive HTML polarized-Raman viewer')
parser.add_argument('--data-dir', default='.',
                    help='Directory with resonant_raman_data_flavor*.h5 and '
                         'resonant_raman_polar_flavor*.h5  (default: .)')
parser.add_argument('--output', default='raman_polar_interactive.html',
                    help='Output HTML file  (default: raman_polar_interactive.html)')
parser.add_argument('--max-eexc-points', type=int, default=40,
                    help='Max Eexc-axis points (polar data) after down-sampling  (default: 40 -- '
                         'kept small since JSON text-encodes each float and this axis is '
                         'multiplied by both --max-ph-points and --max-theta-points)')
parser.add_argument('--max-ph-points', type=int, default=40,
                    help='Max omega_ph-axis points (polar data) after down-sampling  (default: 40)')
parser.add_argument('--max-theta-points', type=int, default=48,
                    help='Max theta-axis points after down-sampling  (default: 48 -- still smooth '
                         'for a closed polar curve; the (Eexc,omega_ph) SELECTION is guided '
                         'visually by the full-resolution guide map on the left, so this grid '
                         'only needs to be fine enough for a reasonable nearest-point snap)')
parser.add_argument('--guide-max-eexc-points', type=int, default=200,
                    help='Max Eexc-axis points for the guide map  (default: 200)')
parser.add_argument('--guide-max-ph-points', type=int, default=300,
                    help='Max omega_ph-axis points for the guide map  (default: 300)')
args = parser.parse_args()


def _downsample_idx(n_full, n_target):
    n_target = min(n_full, n_target)
    return np.round(np.linspace(0, n_full - 1, n_target)).astype(int)


# ── load all available flavors ─────────────────────────────────────────────────
all_data = {}
for flavor in range(9):
    guide_path = Path(args.data_dir) / f'resonant_raman_data_flavor{flavor}.h5'
    polar_path = Path(args.data_dir) / f'resonant_raman_polar_flavor{flavor}.h5'
    if not (guide_path.exists() and polar_path.exists()):
        print(f'  Flavor {flavor}: missing {"guide" if not guide_path.exists() else "polar"} file, skipping')
        continue
    print(f'  Loading flavor {flavor} …', end=' ', flush=True)

    # -- guide map (unpolarized, existing full-resolution output) --
    with h5py.File(guide_path, 'r') as hf:
        g_exc  = hf['excitation_energies'][:]
        g_ph   = hf['freq_axis_cm'][:]
        g_map  = hf['raman_map_unpolarized'][:]
        flavor_label = str(hf.attrs.get('flavor_label', FLAVOR_DESC.get(flavor, f'flavor {flavor}')))
        T = float(hf.attrs.get('temperature_K', 300))
    ie_g  = _downsample_idx(len(g_exc), args.guide_max_eexc_points)
    iph_g = _downsample_idx(len(g_ph),  args.guide_max_ph_points)
    guide = {
        'excitation_energies': g_exc[ie_g].tolist(),
        'freq_axis_cm':        g_ph[iph_g].tolist(),
        'map':                 g_map[np.ix_(ie_g, iph_g)].tolist(),
    }

    # -- polarized maps (theta, Omega_exc, omega_ph) --
    with h5py.File(polar_path, 'r') as hf:
        if 'raman_map_parallel' not in hf:
            print('no stored maps (rerun with --polar-store-maps), skipping')
            continue
        theta  = hf['theta'][:]
        n_hat  = hf['n_hat'][:]
        e1     = hf['e1'][:]
        p_exc  = hf['excitation_energies'][:]
        p_ph   = hf['freq_axis_cm'][:]
        map_par  = hf['raman_map_parallel'][:]        # (Ntheta, Nfreq, Nfreq_ph) float32
        map_perp = hf['raman_map_perpendicular'][:]

    it  = _downsample_idx(len(theta), args.max_theta_points)
    ie  = _downsample_idx(len(p_exc), args.max_eexc_points)
    iph = _downsample_idx(len(p_ph),  args.max_ph_points)

    # reorder to (Eexc, omega_ph, theta) for compact per-point JS lookup
    par_ds  = map_par[np.ix_(it, ie, iph)].transpose(1, 2, 0)   # (Nexc, Nph, Ntheta)
    perp_ds = map_perp[np.ix_(it, ie, iph)].transpose(1, 2, 0)

    all_data[str(flavor)] = {
        'flavor_label':  flavor_label,
        'temperature_K': T,
        'n_hat': n_hat.tolist(),
        'e1':    e1.tolist(),
        'guide': guide,
        'polar': {
            'theta':                theta[it].tolist(),
            'excitation_energies':  p_exc[ie].tolist(),
            'freq_axis_cm':         p_ph[iph].tolist(),
            'I_parallel':           par_ds.tolist(),
            'I_perp':               perp_ds.tolist(),
        },
    }
    print('done')

if not all_data:
    raise SystemExit('No matching resonant_raman_data_flavor*.h5 / '
                     f'resonant_raman_polar_flavor*.h5 pairs found in "{args.data_dir}".')

# ── serialise data ────────────────────────────────────────────────────────────
data_json = json.dumps(all_data, separators=(',', ':'))

# ── HTML template (raw string — backslashes kept for JS unicode escapes) ──────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Polarized Resonant Raman Viewer</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js" charset="utf-8"></script>
<style>
  body  { font-family: Arial, sans-serif; margin: 12px; background: #f7f7f7; }
  h2    { margin: 4px 0 10px; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-end;
    background: #ebebeb; padding: 10px 14px; border-radius: 6px;
    margin-bottom: 12px;
  }
  .ctrl-group > label { display: block; font-weight: bold; font-size: 13px; margin-bottom: 3px; }
  select, input[type=number] {
    font-size: 13px; padding: 4px 6px;
    border: 1px solid #bbb; border-radius: 4px; background: #fff;
  }
  .radio-row label { font-weight: normal; margin-right: 10px; cursor: pointer; }
  .plots { display: flex; gap: 8px; }
  .plot  { flex: 1; min-width: 0; }
</style>
</head>
<body>

<h2>Polarized Resonant Raman Viewer</h2>
<p style="max-width:900px; color:#555; font-size:13px;">
  Left: unpolarized guide map (click to pick &Omega;<sub>exc</sub> and &omega;<sub>ph</sub>).
  Right: I<sub>&parallel;</sub>(&theta;), I<sub>&perp;</sub>(&theta;) at the nearest embedded point
  &mdash; a purely geometric projection of the Raman tensor (no optical propagation effects; see
  resonant_raman/README.md's "Polarized Raman" section).
</p>

<div class="controls">
  <div class="ctrl-group">
    <label>Flavor</label>
    <select id="flavor-sel"></select>
  </div>
  <div class="ctrl-group">
    <label>&Omega;<sub>exc</sub> (eV) &nbsp;<small style="font-weight:normal">(or click map)</small></label>
    <input id="eexc-input" type="number" step="0.001" style="width:90px">
  </div>
  <div class="ctrl-group">
    <label>&omega;<sub>ph</sub> (cm&#8315;&#185;) &nbsp;<small style="font-weight:normal">(or click map)</small></label>
    <input id="rshift-input" type="number" step="1" style="width:100px">
  </div>
  <div class="ctrl-group">
    <label>Guide map scale</label>
    <div class="radio-row">
      <label><input type="radio" name="scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="scale" value="log"> Log</label>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Polar radial scale</label>
    <div class="radio-row">
      <label><input type="radio" name="polar-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="polar-scale" value="log"> Log</label>
    </div>
  </div>
</div>

<div class="plots">
  <div class="plot" id="map-div"></div>
  <div class="plot" id="polar-div"></div>
</div>

<script>
const DATA = __DATA__;

// ── populate flavor dropdown ───────────────────────────────────────────────────
const flavorSel = document.getElementById('flavor-sel');
for (const [k, v] of Object.entries(DATA)) {
  const o = document.createElement('option');
  o.value = k;
  o.text  = 'Flavor ' + k + ': ' + v.flavor_label;
  flavorSel.appendChild(o);
}

// ── global log scale bounds for the guide map (computed once over ALL flavors) ─
var GLOBAL_LOG_FLOOR = Infinity;
var GLOBAL_LOG_MAX   = -Infinity;
(function() {
  for (const fdata of Object.values(DATA)) {
    for (const row of fdata.guide.map) {
      for (const v of row) {
        if (v > 0 && v < GLOBAL_LOG_FLOOR) GLOBAL_LOG_FLOOR = v;
        if (v > GLOBAL_LOG_MAX)            GLOBAL_LOG_MAX   = v;
      }
    }
  }
  if (!isFinite(GLOBAL_LOG_FLOOR)) GLOBAL_LOG_FLOOR = 1e-30;
  if (!isFinite(GLOBAL_LOG_MAX))   GLOBAL_LOG_MAX   = 1.0;
})();

// ── helpers ───────────────────────────────────────────────────────────────────
function curData() { return DATA[flavorSel.value]; }
function isLogMap()   {
  return document.querySelector('input[name="scale"]:checked').value === 'log';
}
function isLogPolar() {
  return document.querySelector('input[name="polar-scale"]:checked').value === 'log';
}

function applyLog(z) {
  return z.map(function(row) {
    return row.map(function(v) { return Math.log10(Math.max(v, GLOBAL_LOG_FLOOR)); });
  });
}

function nearestIdx(arr, val) {
  let best = 0, d0 = Math.abs(arr[0] - val);
  for (let i = 1; i < arr.length; i++) {
    const d = Math.abs(arr[i] - val);
    if (d < d0) { d0 = d; best = i; }
  }
  return best;
}

function eexcVal()   { return parseFloat(document.getElementById('eexc-input').value); }
function setEexcInput(val)   { document.getElementById('eexc-input').value = val.toFixed(4); }
function rshiftVal() { return parseFloat(document.getElementById('rshift-input').value); }
function setRshiftInput(val) { document.getElementById('rshift-input').value = val.toFixed(1); }

// ── figure builders ───────────────────────────────────────────────────────────
function buildMapTrace() {
  const d        = curData();
  const raw      = d.guide.map;
  const logScale = isLogMap();
  const z        = logScale ? applyLog(raw) : raw;
  const trace = {
    type: 'heatmap',
    x: d.guide.freq_axis_cm,
    y: d.guide.excitation_energies,
    z: z,
    colorscale: 'Viridis',
    colorbar: { title: { text: logScale ? 'log₁₀(I)' : 'I (a.u.)', side: 'right' }, thickness: 14 },
    hovertemplate: 'ω: %{x:.1f} cm⁻¹<br>Ω<sub>exc</sub>: %{y:.4f} eV<br>I: %{z:.3e}<extra></extra>',
  };
  if (logScale) {
    trace.zmin = Math.log10(GLOBAL_LOG_FLOOR);
    trace.zmax = Math.log10(GLOBAL_LOG_MAX);
  }
  return [trace];
}

function buildMapLayout(eexc, rshift) {
  const d  = curData();
  const x0 = d.guide.freq_axis_cm[0], x1 = d.guide.freq_axis_cm[d.guide.freq_axis_cm.length - 1];
  const y0 = d.guide.excitation_energies[0], y1 = d.guide.excitation_energies[d.guide.excitation_energies.length - 1];
  return {
    title: { text: 'Guide map (unpolarized) — Flavor ' + flavorSel.value + ' — T = ' + d.temperature_K + ' K', font: { size: 13 } },
    xaxis: { title: 'ω<sub>ph</sub> (cm⁻¹)' },
    yaxis: { title: 'Ω<sub>exc</sub> (eV)' },
    shapes: [
      { type: 'line', x0: x0, x1: x1, y0: eexc, y1: eexc, line: { color: 'red', width: 1.5, dash: 'dash' } },
      { type: 'line', x0: rshift, x1: rshift, y0: y0, y1: y1, line: { color: '#ff7f0e', width: 1.5, dash: 'dash' } },
    ],
    margin: { l: 60, r: 10, t: 55, b: 55 },
  };
}

function buildPolarTraces(eexc, rshift) {
  const d = curData();
  const p = d.polar;
  const iE  = nearestIdx(p.excitation_energies, eexc);
  const iPh = nearestIdx(p.freq_axis_cm, rshift);
  var Ipar  = p.I_parallel[iE][iPh];   // (Ntheta,)
  var Iperp = p.I_perp[iE][iPh];
  const logPolar = isLogPolar();
  if (logPolar) {
    const floor = Math.max(1e-30, Math.min(...Ipar.concat(Iperp).filter(function(v){return v>0;})) || 1e-30);
    Ipar  = Ipar.map(function(v)  { return Math.log10(Math.max(v, floor)); });
    Iperp = Iperp.map(function(v) { return Math.log10(Math.max(v, floor)); });
  }
  // close the curve
  const theta_deg = p.theta.map(function(t) { return t * 180 / Math.PI; });
  const theta_c = theta_deg.concat([theta_deg[0]]);
  const Ipar_c  = Ipar.concat([Ipar[0]]);
  const Iperp_c = Iperp.concat([Iperp[0]]);
  return [
    { type: 'scatterpolar', mode: 'lines', name: 'I_parallel', r: Ipar_c,  theta: theta_c, line: { color: 'black', width: 2.5 } },
    { type: 'scatterpolar', mode: 'lines', name: 'I_perp',     r: Iperp_c, theta: theta_c, line: { color: '#d62728', width: 2.5 } },
  ];
}

function buildPolarLayout(eexc_actual, ph_actual) {
  const d = curData();
  return {
    title: {
      text: 'Flavor ' + flavorSel.value + ': ' + d.flavor_label
          + ' — Ω<sub>exc</sub> ≈ ' + eexc_actual.toFixed(4) + ' eV'
          + ', ω ≈ ' + ph_actual.toFixed(1) + ' cm⁻¹'
          + '<br><sub>n̂=[' + d.n_hat.map(function(v){return v.toFixed(2);}).join(',')
          + ']  ê₁=[' + d.e1.map(function(v){return v.toFixed(2);}).join(',') + ']</sub>',
      font: { size: 12 },
    },
    polar: { angularaxis: { direction: 'counterclockwise', rotation: 0 } },
    showlegend: true,
    legend: { font: { size: 10 }, x: 1, xanchor: 'right', y: 1.15 },
    margin: { l: 40, r: 40, t: 80, b: 40 },
  };
}

// ── update functions ──────────────────────────────────────────────────────────
function updateMap() {
  const d      = curData();
  const eexc   = isNaN(eexcVal())   ? d.guide.excitation_energies[0] : eexcVal();
  const rshift = isNaN(rshiftVal()) ? d.guide.freq_axis_cm[0]        : rshiftVal();
  return Plotly.react('map-div', buildMapTrace(), buildMapLayout(eexc, rshift));
}

function updatePolar() {
  const d      = curData();
  const eexc   = isNaN(eexcVal())   ? d.guide.excitation_energies[0] : eexcVal();
  const rshift = isNaN(rshiftVal()) ? d.guide.freq_axis_cm[0]        : rshiftVal();
  const iE     = nearestIdx(d.polar.excitation_energies, eexc);
  const iPh    = nearestIdx(d.polar.freq_axis_cm, rshift);
  return Plotly.react('polar-div',
                      buildPolarTraces(eexc, rshift),
                      buildPolarLayout(d.polar.excitation_energies[iE], d.polar.freq_axis_cm[iPh]));
}

function updateAll() { updateMap(); updatePolar(); }

// ── event wiring ──────────────────────────────────────────────────────────────
flavorSel.addEventListener('change', updateAll);

document.getElementById('eexc-input').addEventListener('change', updateAll);
document.getElementById('rshift-input').addEventListener('change', updateAll);

document.querySelectorAll('input[name="scale"]')
        .forEach(function(r) { r.addEventListener('change', updateMap); });
document.querySelectorAll('input[name="polar-scale"]')
        .forEach(function(r) { r.addEventListener('change', updatePolar); });

// ── initialise ────────────────────────────────────────────────────────────────
(async function() {
  const d0 = curData();
  setEexcInput(d0.guide.excitation_energies[Math.floor(d0.guide.excitation_energies.length / 2)]);
  setRshiftInput(d0.guide.freq_axis_cm[Math.floor(d0.guide.freq_axis_cm.length / 2)]);

  await updateMap();
  await updatePolar();

  document.getElementById('map-div').on('plotly_click', function(evtData) {
    setEexcInput(evtData.points[0].y);
    setRshiftInput(evtData.points[0].x);
    updateAll();
  });
})();
</script>
</body>
</html>
"""

html_out = HTML_TEMPLATE.replace('__DATA__', data_json)

out_path = Path(args.output)
out_path.write_text(html_out, encoding='utf-8')
size_mb = out_path.stat().st_size / 1e6
print(f'\nSaved: {out_path}  ({size_mb:.1f} MB)')
print('Open in any browser — no server required.')
