"""
Generate a self-contained interactive HTML viewer for resonant Raman maps.

Reads  resonant_raman_data_flavor{0..5}.h5  and embeds all data into a
single HTML file backed by Plotly.js (loaded from CDN).

Left panel   — 2-D Raman map (click anywhere to set both Omega_exc and Raman shift)
Middle panel — Raman spectrum at the selected Omega_exc
Right panel  — Excitation profile (intensity vs Omega_exc) at the selected Raman shift

Controls:
  Flavor dropdown  |  Polarization dropdown
  Omega_exc input  |  Raman shift input  |  Linear / Log toggles for each panel

Usage:
  python interactive_vis_resonant_map.py
  python interactive_vis_resonant_map.py --data-dir /path/to/run --output viewer.html
  python interactive_vis_resonant_map.py --max-eexc-points 100 --max-ph-points 200
"""

import argparse
import json
from pathlib import Path
import numpy as np
import h5py

from common import FLAVOR_DESC
CART       = ['x', 'y', 'z']
POL_LABELS = ['unpolarized'] + [a + b for a in CART for b in CART]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Generate interactive HTML Raman viewer')
parser.add_argument('--data-dir', default='.',
                    help='Directory with resonant_raman_data_flavor*.h5  (default: .)')
parser.add_argument('--output', default='raman_interactive.html',
                    help='Output HTML file  (default: raman_interactive.html)')
parser.add_argument('--max-eexc-points', type=int, default=200,
                    help='Max Eexc-axis points after down-sampling  (default: 200)')
parser.add_argument('--max-ph-points', type=int, default=300,
                    help='Max phonon-freq-axis points after down-sampling  (default: 300)')
args = parser.parse_args()

# ── load all available flavor files ──────────────────────────────────────────
all_data = {}
for flavor in range(6):
    path = Path(args.data_dir) / f'resonant_raman_data_flavor{flavor}.h5'
    if not path.exists():
        print(f'  Flavor {flavor}: not found, skipping')
        continue
    print(f'  Loading flavor {flavor} from {path} …', end=' ', flush=True)
    with h5py.File(path, 'r') as hf:
        exc_en      = hf['excitation_energies'][:]       # (Nfreq,)
        freq_ax     = hf['freq_axis_cm'][:]              # (Nfreq_ph,)
        raman_maps  = hf['raman_maps'][:]                # (3,3,Nfreq,Nfreq_ph)
        raman_unpol = hf['raman_map_unpolarized'][:]     # (Nfreq,Nfreq_ph)
        flavor_label = str(hf.attrs.get(
            'flavor_label', FLAVOR_DESC.get(flavor, f'flavor {flavor}')))
        T = float(hf.attrs.get('temperature_K', 300))

    # Down-sample to reduce HTML size
    ne, nph     = len(exc_en), len(freq_ax)
    ne_out      = min(ne,  args.max_eexc_points)
    nph_out     = min(nph, args.max_ph_points)
    ie          = np.round(np.linspace(0, ne  - 1, ne_out )).astype(int)
    iph         = np.round(np.linspace(0, nph - 1, nph_out)).astype(int)

    maps = {'unpolarized': raman_unpol[np.ix_(ie, iph)].tolist()}
    for ia, a in enumerate(CART):
        for ib, b in enumerate(CART):
            maps[f'{a}{b}'] = raman_maps[ia, ib][np.ix_(ie, iph)].tolist()

    all_data[str(flavor)] = {
        'flavor_label':       flavor_label,
        'temperature_K':      T,
        'excitation_energies': exc_en[ie].tolist(),
        'freq_axis_cm':        freq_ax[iph].tolist(),
        'maps':                maps,
    }
    print('done')

if not all_data:
    raise SystemExit('No resonant_raman_data_flavor*.h5 files found in '
                     f'"{args.data_dir}".')

# ── serialise data ────────────────────────────────────────────────────────────
data_json = json.dumps(all_data,    separators=(',', ':'))
pol_json  = json.dumps(POL_LABELS,  separators=(',', ':'))

# ── HTML template (raw string — backslashes kept for JS unicode escapes) ──────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Resonant Raman Viewer</title>
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

<h2>Resonant Raman Interactive Viewer</h2>

<div class="controls">
  <div class="ctrl-group">
    <label>Flavor</label>
    <select id="flavor-sel"></select>
  </div>
  <div class="ctrl-group">
    <label>Polarization</label>
    <select id="pol-sel"></select>
  </div>
  <div class="ctrl-group">
    <label>&Omega;<sub>exc</sub> (eV) &nbsp;<small style="font-weight:normal">(or click map)</small></label>
    <input id="eexc-input" type="number" step="0.001" style="width:90px">
  </div>
  <div class="ctrl-group">
    <label>Raman shift (cm&#8315;&#185;) &nbsp;<small style="font-weight:normal">(or click map)</small></label>
    <input id="rshift-input" type="number" step="1" style="width:100px">
  </div>
  <div class="ctrl-group">
    <label>Map scale</label>
    <div class="radio-row">
      <label><input type="radio" name="scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="scale" value="log"> Log</label>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Spectrum scale</label>
    <div class="radio-row">
      <label><input type="radio" name="spec-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="spec-scale" value="log"> Log</label>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Excitation profile scale</label>
    <div class="radio-row">
      <label><input type="radio" name="exc-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="exc-scale" value="log"> Log</label>
    </div>
  </div>
</div>

<div class="plots">
  <div class="plot" id="map-div"></div>
  <div class="plot" id="spec-div"></div>
  <div class="plot" id="exc-div"></div>
</div>

<script>
const DATA       = __DATA__;
const POL_LABELS = __POL_LABELS__;

// ── populate dropdowns ────────────────────────────────────────────────────────
const flavorSel = document.getElementById('flavor-sel');
for (const [k, v] of Object.entries(DATA)) {
  const o = document.createElement('option');
  o.value = k;
  o.text  = 'Flavor ' + k + ': ' + v.flavor_label;
  flavorSel.appendChild(o);
}

const polSel = document.getElementById('pol-sel');
for (const p of POL_LABELS) {
  const o = document.createElement('option');
  o.value = p; o.text = p;
  polSel.appendChild(o);
}

// ── global log scale bounds (computed once over ALL maps / flavors / pols) ────
var GLOBAL_LOG_FLOOR = Infinity;   // smallest positive value seen
var GLOBAL_LOG_MAX   = -Infinity;  // largest value seen
(function() {
  for (const fdata of Object.values(DATA)) {
    for (const rows of Object.values(fdata.maps)) {
      for (const row of rows) {
        for (const v of row) {
          if (v > 0 && v < GLOBAL_LOG_FLOOR) GLOBAL_LOG_FLOOR = v;
          if (v > GLOBAL_LOG_MAX)            GLOBAL_LOG_MAX   = v;
        }
      }
    }
  }
  // Fallback guards
  if (!isFinite(GLOBAL_LOG_FLOOR)) GLOBAL_LOG_FLOOR = 1e-30;
  if (!isFinite(GLOBAL_LOG_MAX))   GLOBAL_LOG_MAX   = 1.0;
})();

// ── helpers ───────────────────────────────────────────────────────────────────
function curData() { return DATA[flavorSel.value]; }
function curPol()  { return polSel.value; }
function isLog()   {
  return document.querySelector('input[name="scale"]:checked').value === 'log';
}
function isLogSpec() {
  return document.querySelector('input[name="spec-scale"]:checked').value === 'log';
}
function isLogExc() {
  return document.querySelector('input[name="exc-scale"]:checked').value === 'log';
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

function eexcVal() {
  return parseFloat(document.getElementById('eexc-input').value);
}
function setEexcInput(val) {
  document.getElementById('eexc-input').value = val.toFixed(4);
}

function rshiftVal() {
  return parseFloat(document.getElementById('rshift-input').value);
}
function setRshiftInput(val) {
  document.getElementById('rshift-input').value = val.toFixed(1);
}

// ── figure builders ───────────────────────────────────────────────────────────
function buildMapTrace() {
  const d        = curData();
  const raw      = d.maps[curPol()];
  const logScale = isLog();
  const z        = logScale ? applyLog(raw) : raw;

  const trace = {
    type:          'heatmap',
    x:             d.freq_axis_cm,
    y:             d.excitation_energies,
    z:             z,
    colorscale:    'Viridis',
    colorbar: {
      title: { text: logScale ? 'log\u2081\u2080(I)' : 'I (a.u.)', side: 'right' },
      thickness: 14,
    },
    hovertemplate: '\u03c9: %{x:.1f} cm\u207b\u00b9<br>'
                 + '\u03a9<sub>exc</sub>: %{y:.4f} eV<br>'
                 + 'I: %{z:.3e}<extra></extra>',
  };

  if (logScale) {
    // Pin colorbar to the true global min/max across ALL maps so the scale
    // is consistent when switching flavor or polarization.
    trace.zmin = Math.log10(GLOBAL_LOG_FLOOR);
    trace.zmax = Math.log10(GLOBAL_LOG_MAX);
  }

  return [trace];
}

function buildMapLayout(eexc, rshift) {
  const d  = curData();
  const x0 = d.freq_axis_cm[0];
  const x1 = d.freq_axis_cm[d.freq_axis_cm.length - 1];
  const y0 = d.excitation_energies[0];
  const y1 = d.excitation_energies[d.excitation_energies.length - 1];
  return {
    title: {
      text: 'Raman Map \u2014 ' + curPol()
          + ' \u2014 Flavor ' + flavorSel.value
          + ' \u2014 T = ' + d.temperature_K + ' K',
      font: { size: 13 },
    },
    xaxis: { title: '\u03c9<sub>ph</sub> (cm\u207b\u00b9)' },
    yaxis: { title: '\u03a9<sub>exc</sub> (eV)' },
    shapes: [
      {
        // horizontal line — fixed excitation energy (red)
        type: 'line',
        x0: x0, x1: x1, y0: eexc, y1: eexc,
        line: { color: 'red', width: 1.5, dash: 'dash' },
      },
      {
        // vertical line — fixed Raman shift (orange)
        type: 'line',
        x0: rshift, x1: rshift, y0: y0, y1: y1,
        line: { color: '#ff7f0e', width: 1.5, dash: 'dash' },
      },
    ],
    margin: { l: 60, r: 10, t: 55, b: 55 },
  };
}

function buildSpectrumTrace(iE) {
  const d = curData();
  return [{
    type: 'scatter',
    x:    d.freq_axis_cm,
    y:    d.maps[curPol()][iE],
    mode: 'lines',
    line: { color: '#1f77b4', width: 1.5 },
    hovertemplate: '\u03c9: %{x:.1f} cm\u207b\u00b9<br>I: %{y:.3e}<extra></extra>',
  }];
}

function buildSpectrumLayout(eexc_actual, rshift) {
  const logSpec = isLogSpec();
  return {
    title: {
      text: 'Raman Spectrum \u2014 ' + curPol()
          + ' \u2014 \u03a9<sub>exc</sub> = ' + eexc_actual.toFixed(4) + ' eV',
      font: { size: 13 },
    },
    xaxis: { title: 'Raman shift (cm\u207b\u00b9)' },
    yaxis: {
      title: logSpec ? 'log\u2081\u2080(I)' : 'Raman Intensity (a.u.)',
      type:  logSpec ? 'log' : 'linear',
    },
    // vertical marker at the pinned Raman shift (orange, matches map crosshair)
    shapes: [{
      type: 'line',
      x0: rshift, x1: rshift, y0: 0, y1: 1, yref: 'paper',
      line: { color: '#ff7f0e', width: 1.5, dash: 'dash' },
    }],
    margin: { l: 65, r: 10, t: 55, b: 55 },
  };
}

function buildExcProfileTrace(iPh) {
  const d   = curData();
  const map = d.maps[curPol()];
  // Extract the column at index iPh across all excitation energies
  const y = map.map(function(row) { return row[iPh]; });
  return [{
    type: 'scatter',
    x:    d.excitation_energies,
    y:    y,
    mode: 'lines',
    line: { color: '#d62728', width: 1.5 },
    hovertemplate: '\u03a9<sub>exc</sub>: %{x:.4f} eV<br>I: %{y:.3e}<extra></extra>',
  }];
}

function buildExcProfileLayout(ph_actual, eexc) {
  const logExc = isLogExc();
  return {
    title: {
      text: 'Excitation Profile \u2014 ' + curPol()
          + ' \u2014 \u03c9 = ' + ph_actual.toFixed(1) + ' cm\u207b\u00b9',
      font: { size: 13 },
    },
    xaxis: { title: '\u03a9<sub>exc</sub> (eV)' },
    yaxis: {
      title: logExc ? 'log\u2081\u2080(I)' : 'Raman Intensity (a.u.)',
      type:  logExc ? 'log' : 'linear',
    },
    // vertical marker at the current Eexc (red, matches map crosshair)
    shapes: [{
      type: 'line',
      x0: eexc, x1: eexc, y0: 0, y1: 1, yref: 'paper',
      line: { color: 'red', width: 1.5, dash: 'dash' },
    }],
    margin: { l: 65, r: 10, t: 55, b: 55 },
  };
}

// ── update functions ──────────────────────────────────────────────────────────
function updateMap() {
  const d      = curData();
  const eexc   = isNaN(eexcVal())   ? d.excitation_energies[0] : eexcVal();
  const rshift = isNaN(rshiftVal()) ? d.freq_axis_cm[0]        : rshiftVal();
  return Plotly.react('map-div', buildMapTrace(), buildMapLayout(eexc, rshift));
}

function updateSpectrum() {
  const d      = curData();
  const eexc   = isNaN(eexcVal())   ? d.excitation_energies[0] : eexcVal();
  const rshift = isNaN(rshiftVal()) ? d.freq_axis_cm[0]        : rshiftVal();
  const iE     = nearestIdx(d.excitation_energies, eexc);
  return Plotly.react('spec-div',
                      buildSpectrumTrace(iE),
                      buildSpectrumLayout(d.excitation_energies[iE], rshift));
}

function updateExcProfile() {
  const d      = curData();
  const eexc   = isNaN(eexcVal())   ? d.excitation_energies[0] : eexcVal();
  const rshift = isNaN(rshiftVal()) ? d.freq_axis_cm[0]        : rshiftVal();
  const iPh    = nearestIdx(d.freq_axis_cm, rshift);
  return Plotly.react('exc-div',
                      buildExcProfileTrace(iPh),
                      buildExcProfileLayout(d.freq_axis_cm[iPh], eexc));
}

function updateAll() { updateMap(); updateSpectrum(); updateExcProfile(); }

// ── event wiring ──────────────────────────────────────────────────────────────
flavorSel.addEventListener('change', function() {
  // reset both cursors to mid-point of the new flavor's grids
  const d = curData();
  setEexcInput(d.excitation_energies[Math.floor(d.excitation_energies.length / 2)]);
  setRshiftInput(d.freq_axis_cm[Math.floor(d.freq_axis_cm.length / 2)]);
  updateAll();
});
polSel.addEventListener('change', updateAll);

document.getElementById('eexc-input').addEventListener('change', function() {
  updateMap(); updateSpectrum(); updateExcProfile();
});
document.getElementById('rshift-input').addEventListener('change', function() {
  updateMap(); updateSpectrum(); updateExcProfile();
});

document.querySelectorAll('input[name="scale"]')
        .forEach(function(r) { r.addEventListener('change', updateMap); });
document.querySelectorAll('input[name="spec-scale"]')
        .forEach(function(r) { r.addEventListener('change', updateSpectrum); });
document.querySelectorAll('input[name="exc-scale"]')
        .forEach(function(r) { r.addEventListener('change', updateExcProfile); });

// ── initialise ────────────────────────────────────────────────────────────────
(async function() {
  const d0 = curData();
  setEexcInput(d0.excitation_energies[Math.floor(d0.excitation_energies.length / 2)]);
  setRshiftInput(d0.freq_axis_cm[Math.floor(d0.freq_axis_cm.length / 2)]);

  // First render — must await so Plotly attaches .on() to the element
  await updateMap();
  await updateSpectrum();
  await updateExcProfile();

  // Map click → set BOTH Eexc (y) and Raman shift (x) → refresh all panels
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

html_out = (HTML_TEMPLATE
            .replace('__DATA__',       data_json)
            .replace('__POL_LABELS__', pol_json))

out_path = Path(args.output)
out_path.write_text(html_out, encoding='utf-8')
size_mb = out_path.stat().st_size / 1e6
print(f'\nSaved: {out_path}  ({size_mb:.1f} MB)')
print('Open in any browser — no server required.')
