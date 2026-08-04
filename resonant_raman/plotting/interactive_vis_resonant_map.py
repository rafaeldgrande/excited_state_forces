"""
Generate a self-contained interactive HTML viewer for resonant Raman maps.

Reads  resonant_raman_data_flavor{0..8}.h5  and embeds all data into a
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

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import FLAVOR_DESC, cartesian_from_bvec_basis
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
parser.add_argument('--gwbse-qdir', default='../GWBSE/5.2-absorption_Q_shift_comensurable',
                    help='Base dir with per-Q-index eigenvalues_b{1,2,3}.dat subdirs '
                         '(default: ../GWBSE/5.2-absorption_Q_shift_comensurable)')
parser.add_argument('--qpoints-file', default='../RESONANT_RAMAN_2nd_ORDER_all_q/qpoints_crystal.dat',
                    help='BZ q-points + weights file, 7 rows matching the finite-Q '
                         'susceptibility files (default: ../RESONANT_RAMAN_2nd_ORDER_all_q/qpoints_crystal.dat)')
parser.add_argument('--phonon-susc-dir', default='../RESONANT_RAMAN_2nd_ORDER_all_q',
                    help='Dir with susceptibility_tensors_second_order_q_{iq}.h5 files, '
                         'source of phonon_frequencies_cm per q (default: ../RESONANT_RAMAN_2nd_ORDER_all_q)')
parser.add_argument('--dos-gamma-lor', type=float, default=10.0,
                    help='Lorentzian broadening for phonon DOS/JDOS, cm^-1 (default: 10.0, '
                         'matches resonant_raman.py --gamma-lor default)')
parser.add_argument('--dos-gamma-ev', type=float, default=0.01,
                    help='Lorentzian broadening for exciton DOS/absorption, eV (default: 0.01, '
                         'matches susceptibility_tensors_*.py --gamma default)')
parser.add_argument('--dos-npts', type=int, default=400,
                    help='Number of points on each new DOS/absorption axis (default: 400)')
parser.add_argument('--dos-emin', type=float, default=1.0,
                    help='Lower bound of the exciton DOS/absorption energy axis, eV (default: 1.0)')
parser.add_argument('--dos-emax', type=float, default=4.0,
                    help='Upper bound of the exciton DOS/absorption energy axis, eV (default: 4.0)')
args = parser.parse_args()

# Q-shift indices matching phonon q-points in elph.h5 (Gamma + 6 finite-Q,
# same order as --qpoints-file's 7 rows / the *_q_{iq}.h5 files' iq index)
QIDX_LIST = [0, 2, 4, 6, 12, 14, 18]

# ── load all available flavor files ──────────────────────────────────────────
all_data = {}
for flavor in range(9):
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

# ── phonon DOS / two-phonon JDOS + exciton DOS / optical absorption ───────────
# All four are precomputed once here (peak-height Lorentzian broadening,
# same L(x) = gamma^2 / ((x-x0)^2 + gamma^2) form used throughout
# resonant_raman.py) and embedded as static curves -- no live recomputation
# in JS. BZ q-average uses the same 7 q-points / weights already established
# for the finite-Q resonant-Raman pipeline (Gamma + 6 matching Q-shift
# indices); optical absorption uses Q=0 only, since photons carry ~0
# momentum and only Gamma-point excitons couple to light.
def _lorentzian_sum(axis, centers, weights, gamma):
    """sum_i weights[i] * gamma^2 / ((axis - centers[i])^2 + gamma^2)"""
    diff = axis[np.newaxis, :] - centers[:, np.newaxis]
    return (weights[:, np.newaxis] * (gamma**2 / (diff**2 + gamma**2))).sum(axis=0)

dos_data = {'has_dos': False}
base_dir = Path(args.data_dir)
try:
    qpts = np.loadtxt(base_dir / args.qpoints_file)
    q_weights = qpts[:, -1]
    if len(q_weights) != len(QIDX_LIST):
        raise ValueError(f'{args.qpoints_file} has {len(q_weights)} rows, '
                         f'expected {len(QIDX_LIST)}')

    # -- phonon frequencies per q, from the already-computed per-q susceptibility files --
    ph_freqs = []
    for iq in range(len(QIDX_LIST)):
        with h5py.File(base_dir / args.phonon_susc_dir /
                       f'susceptibility_tensors_second_order_q_{iq}.h5', 'r') as hf:
            ph_freqs.append(hf['phonon_frequencies_cm'][:])
    ph_freqs = np.array(ph_freqs)                      # (Nq, Nmodes)
    Nq, Nmodes = ph_freqs.shape

    max_freq = ph_freqs.max()
    ph_axis  = np.linspace(0.0, 1.05 * max_freq, args.dos_npts)
    ph_centers  = ph_freqs.reshape(-1)                  # (Nq*Nmodes,)
    ph_weights  = np.repeat(q_weights, Nmodes)
    phonon_dos  = _lorentzian_sum(ph_axis, ph_centers, ph_weights, args.dos_gamma_lor)

    # two-phonon joint DOS: full double sum over mode pairs (i,j) per q
    pair_sums    = (ph_freqs[:, :, np.newaxis] + ph_freqs[:, np.newaxis, :]).reshape(Nq, -1)  # (Nq, Nmodes^2)
    jdos_axis    = np.linspace(0.0, 1.05 * 2 * max_freq, args.dos_npts)
    jdos_centers = pair_sums.reshape(-1)
    jdos_weights = np.repeat(q_weights, Nmodes * Nmodes)
    phonon_jdos  = _lorentzian_sum(jdos_axis, jdos_centers, jdos_weights, args.dos_gamma_lor)

    # -- exciton DOS: same BZ q-average, over exciton energies at each q --
    exc_axis = np.linspace(args.dos_emin, args.dos_emax, args.dos_npts)
    exc_dos  = np.zeros_like(exc_axis)
    for w, qidx in zip(q_weights, QIDX_LIST):
        edat = np.loadtxt(base_dir / args.gwbse_qdir / str(qidx) / 'eigenvalues_b1.dat')
        energies = edat[:, 0]
        exc_dos += w * _lorentzian_sum(exc_axis, energies,
                                       np.ones_like(energies), args.dos_gamma_ev)

    # -- optical absorption: Q=0 only. The b1/b2/b3 dipole moments are
    #    projections onto the (generally non-orthogonal) reciprocal lattice
    #    unit vectors -- BerkeleyGW's default polarization basis, per
    #    BSE/vmtxel.f90 -- not Cartesian x/y/z, so Sum_i|d_bi|^2 != |P|^2
    #    unless those axes happen to be orthonormal. Convert to Cartesian
    #    first, then take the genuinely isotropic |dx|^2+|dy|^2+|dz|^2.
    dip_b = []
    for b in (1, 2, 3):
        edat = np.loadtxt(base_dir / args.gwbse_qdir / '0' / f'eigenvalues_b{b}.dat')
        dip_b.append(edat[:, 2] + 1j * edat[:, 3])       # complex dipole moment
    energies_q0 = edat[:, 0]
    with h5py.File(base_dir / args.gwbse_qdir / '0' / 'eigenvectors.h5', 'r') as hf:
        bvec = hf['mf_header/crystal/bvec'][:]
    dip_x, dip_y, dip_z = cartesian_from_bvec_basis(*dip_b, bvec)
    dip2_iso   = np.abs(dip_x)**2 + np.abs(dip_y)**2 + np.abs(dip_z)**2
    absorption = _lorentzian_sum(exc_axis, energies_q0, dip2_iso, args.dos_gamma_ev)

    dos_data = {
        'has_dos':      True,
        'ph_axis':      ph_axis.tolist(),
        'phonon_dos':   phonon_dos.tolist(),
        'jdos_axis':    jdos_axis.tolist(),
        'phonon_jdos':  phonon_jdos.tolist(),
        'exc_axis':     exc_axis.tolist(),
        'exciton_dos':  exc_dos.tolist(),
        'absorption':   absorption.tolist(),
    }
    print('  DOS/JDOS/exciton-DOS/absorption panels: computed')
except (FileNotFoundError, OSError, ValueError) as e:
    print(f'  DOS/JDOS/exciton-DOS/absorption panels: skipped ({e})')

# ── serialise data ────────────────────────────────────────────────────────────
data_json = json.dumps(all_data,    separators=(',', ':'))
pol_json  = json.dumps(POL_LABELS,  separators=(',', ':'))
dos_json  = json.dumps(dos_data,    separators=(',', ':'))

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
  .plots-dos { display: flex; gap: 8px; margin-top: 14px; }
  .plot  { flex: 1; min-width: 0; }
  .overlay-panel {
    background: #ebebeb; padding: 10px 14px; border-radius: 6px;
    margin-bottom: 12px;
  }
  .overlay-panel > .title { font-weight: bold; font-size: 13px; margin-bottom: 6px; }
  .overlay-row {
    display: flex; align-items: center; gap: 10px; font-size: 13px;
    padding: 2px 0;
  }
  .overlay-row label.flavor-label { min-width: 260px; font-weight: normal; cursor: pointer; }
  .overlay-row select, .overlay-row input[type=number] {
    font-size: 12px; padding: 2px 4px;
    border: 1px solid #bbb; border-radius: 4px; background: #fff;
  }
  .overlay-row input[type=number] { width: 55px; }
  .dos-placeholder { color: #999; font-size: 13px; padding: 30px; text-align: center; }
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
  <div class="ctrl-group">
    <label>DOS / absorption scale</label>
    <div class="radio-row">
      <label><input type="radio" name="dos-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="dos-scale" value="log"> Log</label>
    </div>
  </div>
</div>

<div class="overlay-panel">
  <div class="title">Overlay additional flavors (Spectrum &amp; Excitation Profile panels only)</div>
  <div id="overlay-rows"></div>
</div>

<div class="plots">
  <div class="plot" id="map-div"></div>
  <div class="plot" id="spec-div"></div>
  <div class="plot" id="exc-div"></div>
</div>

<div class="plots-dos">
  <div class="plot" id="phdos-div"></div>
  <div class="plot" id="phjdos-div"></div>
  <div class="plot" id="excdos-div"></div>
  <div class="plot" id="abs-div"></div>
</div>

<script>
const DATA       = __DATA__;
const POL_LABELS = __POL_LABELS__;
const DOS_DATA   = __DOS_DATA__;

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

// ── overlay panel: one row per flavor (color/linestyle/linewidth pickers) ─────
const OVERLAY_COLORS = ['red', 'blue', 'green', 'orange', 'purple',
                         'brown', 'magenta', 'cyan', 'gray'];
const LINESTYLES = [['solid', 'Solid'], ['dash', 'Dashed'], ['dot', 'Dotted']];
const overlayRowsDiv = document.getElementById('overlay-rows');
const overlayState = {};   // flavor key -> {checkbox, colorSel, styleSel, widthInput}

Object.keys(DATA).forEach(function(k, i) {
  const row = document.createElement('div');
  row.className = 'overlay-row';

  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.id = 'overlay-cb-' + k;

  const label = document.createElement('label');
  label.className = 'flavor-label';
  label.htmlFor = cb.id;
  label.textContent = 'Flavor ' + k + ': ' + DATA[k].flavor_label;

  const colorSel = document.createElement('select');
  OVERLAY_COLORS.forEach(function(c) {
    const o = document.createElement('option'); o.value = c; o.text = c;
    colorSel.appendChild(o);
  });
  colorSel.value = OVERLAY_COLORS[i % OVERLAY_COLORS.length];

  const styleSel = document.createElement('select');
  LINESTYLES.forEach(function(s) {
    const o = document.createElement('option'); o.value = s[0]; o.text = s[1];
    styleSel.appendChild(o);
  });

  const widthInput = document.createElement('input');
  widthInput.type = 'number'; widthInput.step = '0.5'; widthInput.min = '0.5';
  widthInput.value = '1.5';

  row.appendChild(cb);
  row.appendChild(label);
  row.appendChild(colorSel);
  row.appendChild(styleSel);
  row.appendChild(widthInput);
  overlayRowsDiv.appendChild(row);

  overlayState[k] = { checkbox: cb, colorSel: colorSel, styleSel: styleSel, widthInput: widthInput };

  [cb, colorSel, styleSel, widthInput].forEach(function(el) {
    el.addEventListener('change', function() { updateSpectrum(); updateExcProfile(); });
  });
});

function activeOverlays() {
  // list of {key, color, dash, width} for checked flavors, excluding the
  // primary (map) flavor to avoid a redundant duplicate legend entry
  const out = [];
  for (const [k, st] of Object.entries(overlayState)) {
    if (st.checkbox.checked && k !== flavorSel.value) {
      out.push({
        key:   k,
        color: st.colorSel.value,
        dash:  st.styleSel.value,
        width: parseFloat(st.widthInput.value) || 1.5,
      });
    }
  }
  return out;
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
function isLogDos() {
  return document.querySelector('input[name="dos-scale"]:checked').value === 'log';
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

function buildSpectrumTraces(eexc) {
  const d  = curData();
  const iE = nearestIdx(d.excitation_energies, eexc);
  const traces = [{
    type: 'scatter',
    name: 'Flavor ' + flavorSel.value + ' (map)',
    x:    d.freq_axis_cm,
    y:    d.maps[curPol()][iE],
    mode: 'lines',
    line: { color: 'black', width: 2.5 },
    hovertemplate: '\u03c9: %{x:.1f} cm\u207b\u00b9<br>I: %{y:.3e}<extra>Flavor ' + flavorSel.value + '</extra>',
  }];
  for (const ov of activeOverlays()) {
    const od  = DATA[ov.key];
    const oiE = nearestIdx(od.excitation_energies, eexc);
    traces.push({
      type: 'scatter',
      name: 'Flavor ' + ov.key + ': ' + od.flavor_label,
      x:    od.freq_axis_cm,
      y:    od.maps[curPol()][oiE],
      mode: 'lines',
      line: { color: ov.color, width: ov.width, dash: ov.dash },
      hovertemplate: '\u03c9: %{x:.1f} cm\u207b\u00b9<br>I: %{y:.3e}<extra>Flavor ' + ov.key + '</extra>',
    });
  }
  return traces;
}

function buildSpectrumLayout(eexc_actual, rshift) {
  const logSpec = isLogSpec();
  return {
    title: {
      text: 'Raman Spectrum \u2014 ' + curPol()
          + ' \u2014 \u03a9<sub>exc</sub> \u2248 ' + eexc_actual.toFixed(4) + ' eV',
      font: { size: 13 },
    },
    xaxis: { title: 'Raman shift (cm\u207b\u00b9)' },
    yaxis: {
      title: logSpec ? 'log\u2081\u2080(I)' : 'Raman Intensity (a.u.)',
      type:  logSpec ? 'log' : 'linear',
    },
    showlegend: true,
    legend: { font: { size: 10 }, x: 1, xanchor: 'right', y: 1 },
    // vertical marker at the pinned Raman shift (orange, matches map crosshair)
    shapes: [{
      type: 'line',
      x0: rshift, x1: rshift, y0: 0, y1: 1, yref: 'paper',
      line: { color: '#ff7f0e', width: 1.5, dash: 'dash' },
    }],
    margin: { l: 65, r: 10, t: 55, b: 55 },
  };
}

function buildExcProfileTraces(rshift) {
  const d    = curData();
  const iPh  = nearestIdx(d.freq_axis_cm, rshift);
  const traces = [{
    type: 'scatter',
    name: 'Flavor ' + flavorSel.value + ' (map)',
    x:    d.excitation_energies,
    y:    d.maps[curPol()].map(function(row) { return row[iPh]; }),
    mode: 'lines',
    line: { color: 'black', width: 2.5 },
    hovertemplate: '\u03a9<sub>exc</sub>: %{x:.4f} eV<br>I: %{y:.3e}<extra>Flavor ' + flavorSel.value + '</extra>',
  }];
  for (const ov of activeOverlays()) {
    const od   = DATA[ov.key];
    const oiPh = nearestIdx(od.freq_axis_cm, rshift);
    traces.push({
      type: 'scatter',
      name: 'Flavor ' + ov.key + ': ' + od.flavor_label,
      x:    od.excitation_energies,
      y:    od.maps[curPol()].map(function(row) { return row[oiPh]; }),
      mode: 'lines',
      line: { color: ov.color, width: ov.width, dash: ov.dash },
      hovertemplate: '\u03a9<sub>exc</sub>: %{x:.4f} eV<br>I: %{y:.3e}<extra>Flavor ' + ov.key + '</extra>',
    });
  }
  return traces;
}

function buildExcProfileLayout(ph_actual, eexc) {
  const logExc = isLogExc();
  return {
    title: {
      text: 'Excitation Profile \u2014 ' + curPol()
          + ' \u2014 \u03c9 \u2248 ' + ph_actual.toFixed(1) + ' cm\u207b\u00b9',
      font: { size: 13 },
    },
    xaxis: { title: '\u03a9<sub>exc</sub> (eV)' },
    yaxis: {
      title: logExc ? 'log\u2081\u2080(I)' : 'Raman Intensity (a.u.)',
      type:  logExc ? 'log' : 'linear',
    },
    showlegend: true,
    legend: { font: { size: 10 }, x: 1, xanchor: 'right', y: 1 },
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
                      buildSpectrumTraces(eexc),
                      buildSpectrumLayout(d.excitation_energies[iE], rshift));
}

function updateExcProfile() {
  const d      = curData();
  const eexc   = isNaN(eexcVal())   ? d.excitation_energies[0] : eexcVal();
  const rshift = isNaN(rshiftVal()) ? d.freq_axis_cm[0]        : rshiftVal();
  const iPh    = nearestIdx(d.freq_axis_cm, rshift);
  return Plotly.react('exc-div',
                      buildExcProfileTraces(rshift),
                      buildExcProfileLayout(d.freq_axis_cm[iPh], eexc));
}

function updateAll() { updateMap(); updateSpectrum(); updateExcProfile(); }

// ── static DOS / JDOS / exciton-DOS / absorption panels ───────────────────────
function dosLineLayout(title, xtitle, ytitle) {
  const logDos = isLogDos();
  return {
    title: { text: title, font: { size: 13 } },
    xaxis: { title: xtitle },
    yaxis: {
      title: logDos ? 'log\u2081\u2080(' + ytitle + ')' : ytitle,
      type:  logDos ? 'log' : 'linear',
    },
    margin: { l: 55, r: 10, t: 45, b: 50 },
  };
}

function renderDosPlaceholder(divId, label) {
  document.getElementById(divId).innerHTML =
    '<div class="dos-placeholder">' + label + ' data not available<br>'
    + '(run with --gwbse-qdir / --qpoints-file / --phonon-susc-dir pointing '
    + 'at the resonant-Raman pipeline directories)</div>';
}

function updateDosPanels() {
  if (!DOS_DATA.has_dos) {
    renderDosPlaceholder('phdos-div',  'Phonon DOS');
    renderDosPlaceholder('phjdos-div', 'Phonon joint DOS');
    renderDosPlaceholder('excdos-div', 'Exciton DOS');
    renderDosPlaceholder('abs-div',    'Optical absorption');
    return;
  }
  Plotly.react('phdos-div', [{
    type: 'scatter', mode: 'lines', x: DOS_DATA.ph_axis, y: DOS_DATA.phonon_dos,
    line: { color: '#1f77b4', width: 1.5 },
  }], dosLineLayout('Phonon DOS', '\u03c9 (cm\u207b\u00b9)', 'DOS (a.u.)'));

  Plotly.react('phjdos-div', [{
    type: 'scatter', mode: 'lines', x: DOS_DATA.jdos_axis, y: DOS_DATA.phonon_jdos,
    line: { color: '#9467bd', width: 1.5 },
  }], dosLineLayout('Two-Phonon Joint DOS', '\u03c9<sub>1</sub>+\u03c9<sub>2</sub> (cm\u207b\u00b9)', 'JDOS (a.u.)'));

  Plotly.react('excdos-div', [{
    type: 'scatter', mode: 'lines', x: DOS_DATA.exc_axis, y: DOS_DATA.exciton_dos,
    line: { color: '#2ca02c', width: 1.5 },
  }], dosLineLayout('Exciton DOS (BZ-averaged)', '\u03a9 (eV)', 'DOS (a.u.)'));

  Plotly.react('abs-div', [{
    type: 'scatter', mode: 'lines', x: DOS_DATA.exc_axis, y: DOS_DATA.absorption,
    line: { color: '#d62728', width: 1.5 },
  }], dosLineLayout('Optical Absorption (Q=0)', '\u03a9 (eV)', 'Absorption (a.u.)'));
}

// ── event wiring ──────────────────────────────────────────────────────────────
flavorSel.addEventListener('change', function() {
  // Keep the current Omega_exc / Raman shift cursor values across a flavor
  // switch (nearestIdx() already clamps gracefully if they fall outside the
  // new flavor's grid) so comparing the same point across flavors is easy.
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
document.querySelectorAll('input[name="dos-scale"]')
        .forEach(function(r) { r.addEventListener('change', updateDosPanels); });

// ── initialise ────────────────────────────────────────────────────────────────
(async function() {
  const d0 = curData();
  setEexcInput(d0.excitation_energies[Math.floor(d0.excitation_energies.length / 2)]);
  setRshiftInput(d0.freq_axis_cm[Math.floor(d0.freq_axis_cm.length / 2)]);

  // First render — must await so Plotly attaches .on() to the element
  await updateMap();
  await updateSpectrum();
  await updateExcProfile();
  updateDosPanels();

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
            .replace('__POL_LABELS__', pol_json)
            .replace('__DOS_DATA__',   dos_json))

out_path = Path(args.output)
out_path.write_text(html_out, encoding='utf-8')
size_mb = out_path.stat().st_size / 1e6
print(f'\nSaved: {out_path}  ({size_mb:.1f} MB)')
print('Open in any browser — no server required.')
