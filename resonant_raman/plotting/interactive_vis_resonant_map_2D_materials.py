"""
Interactive BZ q-contribution map for second-order resonant Raman in 2D materials.

For each q-point in q_points.dat, loads susceptibility_tensors_second_order_q_{iq}.h5,
computes the phonon-weighted Raman intensity I_q(pol, Omega_exc), and renders an
interactive HTML showing which q-points in the first Brillouin zone dominate the
second-order Raman signal at each excitation energy.

Physics
-------
At each q-point the intensity contribution is:
    I_q[pol, Omega_exc] = sum_{nu,mu valid} (w_nu * w_mu)^2
                          * |alpha_tr[pol, nu, mu, Omega_exc]|^2
where  w_nu = sqrt((n_nu + 1) * hbar / (2 * omega_nu))  is the phonon weight.
The q-averaged total equals what resonant_raman.py computes before Lorentzian
broadening.

Usage
-----
  # From BGW WFN.h5 (reads lattice vectors automatically):
  python interactive_vis_resonant_map_2D_materials.py \\
      --q-points-file q_points.dat --wfn WFN_fi.h5

  # From explicit lattice vectors (graphene, a=2.46 Å):
  python interactive_vis_resonant_map_2D_materials.py \\
      --q-points-file q_points.dat \\
      --a1 2.46 0.0 --a2 1.23 2.132 \\
      --output bz_raman.html
"""

from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import (rec_cm_to_eV, k_B, hbar, ignore_0_freq_modes,
                    unpolarized_invariant, bohr2A)

CART       = ['x', 'y', 'z']
POL_LABELS = ['unpolarized'] + [a + b for a in CART for b in CART]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='BZ q-contribution heatmap for second-order resonant Raman (2D materials)')
parser.add_argument('--q-points-file', required=True,
                    help='"qx qy qz weight" in crystal coords, one q-point per row')
parser.add_argument('--data-dir', default='.',
                    help='Directory with susceptibility_tensors_second_order_q_{iq}.h5')
parser.add_argument('--wfn', default=None,
                    help='BerkeleyGW WFN.h5 or WFN_fi.h5 — reads direct lattice vectors '
                         '(mf_header/crystal/avec and /alat)')
parser.add_argument('--a1', nargs=2, type=float, metavar=('AX', 'AY'), default=None,
                    help='In-plane direct lattice vector a1 in Ångström (x y)')
parser.add_argument('--a2', nargs=2, type=float, metavar=('AX', 'AY'), default=None,
                    help='In-plane direct lattice vector a2 in Ångström (x y)')
parser.add_argument('--temperature', type=float, default=300,
                    help='Temperature in K for Bose factors (default: 300)')
parser.add_argument('--output', default='bz_raman_map.html',
                    help='Output HTML file (default: bz_raman_map.html)')
parser.add_argument('--max-exc-points', type=int, default=150,
                    help='Max excitation-energy axis points after downsampling (default: 150)')
args = parser.parse_args()
T = args.temperature

# ── Reciprocal lattice vectors (2D) ──────────────────────────────────────────
if args.wfn is not None:
    print(f'Reading lattice vectors from {args.wfn} ...')
    with h5py.File(args.wfn, 'r') as fh:
        alat = float(fh['mf_header/crystal/alat'][()])   # bohr
        avec = fh['mf_header/crystal/avec'][:]            # (3,3) in units of alat
    a1 = avec[0, :2] * alat * bohr2A
    a2 = avec[1, :2] * alat * bohr2A
    print(f'  alat = {alat:.6f} bohr')
elif args.a1 is not None and args.a2 is not None:
    a1 = np.array(args.a1)
    a2 = np.array(args.a2)
else:
    sys.exit('ERROR: provide --wfn or both --a1 and --a2.')

area = a1[0]*a2[1] - a1[1]*a2[0]   # Å²
b1   = (2*np.pi / area) * np.array([ a2[1], -a2[0]])   # Å⁻¹
b2   = (2*np.pi / area) * np.array([-a1[1],  a1[0]])
print(f'  a1 = [{a1[0]:.4f}, {a1[1]:.4f}] Å   a2 = [{a2[0]:.4f}, {a2[1]:.4f}] Å')
print(f'  b1 = [{b1[0]:.4f}, {b1[1]:.4f}] Å⁻¹  b2 = [{b2[0]:.4f}, {b2[1]:.4f}] Å⁻¹')
print(f'  BZ area = {abs(b1[0]*b2[1] - b1[1]*b2[0]):.6f} Å⁻²')

# ── Wigner–Seitz BZ boundary polygon ─────────────────────────────────────────
def _bz_polygon(b1, b2, n_shell=3):
    """Return BZ boundary vertices sorted by angle (Wigner–Seitz construction)."""
    try:
        from scipy.spatial import Voronoi
    except ImportError:
        print('WARNING: scipy not found; falling back to parallelogram BZ boundary.')
        return None
    pts = np.array([n*b1 + m*b2
                    for n in range(-n_shell, n_shell+1)
                    for m in range(-n_shell, n_shell+1)])
    vor = Voronoi(pts)
    ig  = int(np.argmin(np.linalg.norm(pts, axis=1)))
    reg = vor.regions[vor.point_region[ig]]
    if -1 in reg:
        return None
    verts = vor.vertices[reg]
    return verts[np.argsort(np.arctan2(verts[:, 1], verts[:, 0]))]

bz_verts = _bz_polygon(b1, b2)
if bz_verts is None:
    bz_verts = np.array([0.5*b1+0.5*b2, -0.5*b1+0.5*b2,
                         -0.5*b1-0.5*b2, 0.5*b1-0.5*b2])
bz_closed = np.vstack([bz_verts, bz_verts[:1]])   # close the polygon for plotting

# ── High-symmetry markers: Γ, BZ vertices (K-type), edge midpoints (M-type) ──
hs_labels = [{'label': 'Γ', 'x': 0.0, 'y': 0.0}]
Nv = len(bz_verts)
for i, v in enumerate(bz_verts):
    hs_labels.append({'label': f'K{i}', 'x': float(v[0]), 'y': float(v[1])})
for i in range(Nv):
    mid = 0.5 * (bz_verts[i] + bz_verts[(i+1) % Nv])
    hs_labels.append({'label': f'M{i}', 'x': float(mid[0]), 'y': float(mid[1])})

# ── q-point grid ──────────────────────────────────────────────────────────────
q_raw = np.loadtxt(args.q_points_file)
if q_raw.ndim == 1:
    q_raw = q_raw[np.newaxis, :]
Nq        = len(q_raw)
q_crys    = q_raw[:, :3]
q_weights = q_raw[:, 3]

# fold crystal coordinates to [-0.5, 0.5) → nearest BZ image
q_crys_fold = q_crys - np.round(q_crys)
# crystal → Cartesian 2D (Å⁻¹)
B      = np.vstack([b1, b2])           # (2, 2)
q_cart = q_crys_fold[:, :2] @ B        # (Nq, 2)
print(f'\nLoaded {Nq} q-points from {args.q_points_file}')
for iq in range(Nq):
    print(f'  iq={iq}: crys=[{q_crys_fold[iq,0]:.4f},{q_crys_fold[iq,1]:.4f}]  '
          f'cart=[{q_cart[iq,0]:.4f},{q_cart[iq,1]:.4f}] Å⁻¹  w={q_weights[iq]:.4f}')

# ── Per-q Raman intensity ─────────────────────────────────────────────────────
print('\nComputing per-q Raman intensities ...')
exc_en_ref = None
ie_ref     = None
q_items    = []

for iq in range(Nq):
    fname = Path(args.data_dir) / f'susceptibility_tensors_second_order_q_{iq}.h5'
    print(f'  iq={iq}: {fname.name}', end=' ... ', flush=True)
    with h5py.File(fname, 'r') as fh:
        exc_en   = fh['excitation_energies'][:]             # (Nfreq,)
        alpha_tr = fh['alpha_tensor_triple_resonance'][:]   # (3,3,Nm,Nm,Nfreq)
        alpha_db = fh['alpha_tensor_double_resonance'][:]   # (3,3,Nm,Nfreq)
        freqs_cm = fh['phonon_frequencies_cm'][:]           # (Nm,)

    # fold double-resonance diagonal into triple-resonance
    for _im in range(alpha_tr.shape[2]):
        alpha_tr[:, :, _im, _im, :] += alpha_db[:, :, _im, :]

    Nm = alpha_tr.shape[2]

    # establish common downsampled excitation energy axis from first q-point
    if exc_en_ref is None:
        n_full = len(exc_en)
        n_out  = min(n_full, args.max_exc_points)
        ie_ref = np.round(np.linspace(0, n_full - 1, n_out)).astype(int)
        exc_en_ref = exc_en[ie_ref].tolist()
        Nout = n_out

    # phonon Bose weights  w_nu = sqrt((n_nu+1) * hbar / (2*omega_nu))
    freqs_eV = freqs_cm * rec_cm_to_eV
    safe_eV  = np.maximum(freqs_eV, 1e-8)
    bose     = 1.0 / (np.exp(safe_eV / (k_B * T)) - 1)
    w_ph     = np.sqrt((bose + 1) * hbar / (2 * safe_eV))   # (Nm,)

    # valid phonon-pair mask
    vm = (freqs_cm > 1e-2) if ignore_0_freq_modes else np.ones(Nm, dtype=bool)
    vp = np.outer(vm, vm)                                    # (Nm, Nm)

    # w2[nu,mu] = (w_nu * w_mu)^2  (zero for invalid pairs)
    w2 = np.where(vp, (w_ph[:, None] * w_ph[None, :])**2, 0.0)   # (Nm, Nm)

    # downsample excitation axis
    alpha_ds = alpha_tr[:, :, :, :, ie_ref]                 # (3,3,Nm,Nm,Nout)

    int_pol = {}

    # per-polarization:  I[f] = sum_{nu,mu} w2[nu,mu] * |alpha[nu,mu,f]|^2
    for ia, ac in enumerate(CART):
        for ib, bc in enumerate(CART):
            a_ab = alpha_ds[ia, ib]                          # (Nm, Nm, Nout)
            int_pol[ac + bc] = np.einsum('nm,nmf->f',
                                         w2, np.abs(a_ab)**2).tolist()

    # unpolarized: use full rotational invariant (45|ᾱ|² + 7γ² + 5δ²)
    # alpha_w[3,3,Nm,Nm,Nout]:  w_nu*w_mu * alpha_ds
    alpha_w = alpha_ds * np.sqrt(w2)[np.newaxis, np.newaxis, :, :, np.newaxis]
    # flatten pairs → (3,3,Nm²,Nout), keep only valid pairs
    alpha_w_flat = alpha_w.reshape(3, 3, Nm**2, Nout)[:, :, vp.ravel(), :]
    i_unp = unpolarized_invariant(alpha_w_flat).sum(axis=0)  # (Nout,)
    int_pol['unpolarized'] = i_unp.tolist()

    q_items.append({
        'iq':          iq,
        'qx':          float(q_cart[iq, 0]),
        'qy':          float(q_cart[iq, 1]),
        'q_crys':      [round(float(q_crys_fold[iq, 0]), 4),
                        round(float(q_crys_fold[iq, 1]), 4)],
        'weight':      float(q_weights[iq]),
        'label':       (f'q<sub>{iq}</sub>: [{q_crys_fold[iq,0]:.3f},'
                        f'{q_crys_fold[iq,1]:.3f}]'),
        'freqs_cm':    freqs_cm.tolist(),
        'intensities': int_pol,
    })
    print(f'done  (Nm={Nm}, Nout={Nout})')

# ── Serialize ─────────────────────────────────────────────────────────────────
all_json = json.dumps({
    'q_items':   q_items,
    'exc_en':    exc_en_ref,
    'bz_x':      bz_closed[:, 0].tolist(),
    'bz_y':      bz_closed[:, 1].tolist(),
    'hs_points': hs_labels,
    'b1':        b1.tolist(),
    'b2':        b2.tolist(),
}, separators=(',', ':'))

pol_json = json.dumps(POL_LABELS, separators=(',', ':'))

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BZ Raman Contribution Map</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js" charset="utf-8"></script>
<style>
  body { font-family: Arial,sans-serif; margin:12px; background:#f4f4f4; }
  h2   { margin:4px 0 10px; }
  .controls {
    display:flex; flex-wrap:wrap; gap:18px; align-items:flex-end;
    background:#e8e8e8; padding:10px 14px; border-radius:6px; margin-bottom:10px;
  }
  .ctrl-group > label { display:block; font-weight:bold; font-size:13px; margin-bottom:3px; }
  select, input[type=number] {
    font-size:13px; padding:4px 6px; border:1px solid #bbb;
    border-radius:4px; background:#fff;
  }
  input[type=range] { width:180px; vertical-align:middle; }
  .radio-row label  { font-weight:normal; margin-right:10px; cursor:pointer; }
  .panels { display:flex; gap:8px; }
  .panel  { flex:1; min-width:0; }
  .info-box {
    background:#fff; border:1px solid #ccc; border-radius:4px;
    padding:8px 14px; font-size:12px; margin-top:8px; min-height:50px;
    color:#333;
  }
</style>
</head>
<body>
<h2>Brillouin Zone q-contribution Map &mdash; Second-order Resonant Raman</h2>

<div class="controls">
  <div class="ctrl-group">
    <label>Polarization</label>
    <select id="pol-sel"></select>
  </div>
  <div class="ctrl-group">
    <label>&Omega;<sub>exc</sub> (eV)</label>
    <input id="eexc-input" type="number" step="0.001" style="width:90px">
  </div>
  <div class="ctrl-group">
    <label>&Omega;<sub>exc</sub> slider &nbsp;<small id="eexc-label" style="font-weight:normal"></small></label>
    <input id="eexc-slider" type="range" min="0" step="1">
  </div>
  <div class="ctrl-group">
    <label>BZ map scale</label>
    <div class="radio-row">
      <label><input type="radio" name="bz-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="bz-scale" value="log"> Log</label>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Profile scale</label>
    <div class="radio-row">
      <label><input type="radio" name="exc-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="exc-scale" value="log"> Log</label>
    </div>
  </div>
</div>

<div class="panels">
  <div class="panel" id="bz-div"  style="max-width:55%"></div>
  <div class="panel" id="exc-div"></div>
</div>
<div class="info-box" id="info-box">
  Click a q-point marker in the BZ map to inspect its excitation profile.<br>
  Click the excitation profile to update &Omega;<sub>exc</sub> and refresh the BZ map.
</div>

<script>
const DATA      = __DATA__;
const POL_LABELS = __POL_LABELS__;

const excEn   = DATA.exc_en;         // eV, (Nfreq,)
const qItems  = DATA.q_items;
const bzX     = DATA.bz_x;
const bzY     = DATA.bz_y;
const hsPoints= DATA.hs_points;
const Nq      = qItems.length;
const Nfreq   = excEn.length;

// ── Populate polarization dropdown ────────────────────────────────────────────
const polSel = document.getElementById('pol-sel');
for (const p of POL_LABELS) {
  const o = document.createElement('option');
  o.value = p; o.text = p;
  polSel.appendChild(o);
}

// ── Slider init ───────────────────────────────────────────────────────────────
const slider = document.getElementById('eexc-slider');
slider.max   = Nfreq - 1;
let iExc = Math.floor(Nfreq / 2);
slider.value = iExc;
document.getElementById('eexc-input').value = excEn[iExc].toFixed(4);
document.getElementById('eexc-label').textContent = excEn[iExc].toFixed(4) + ' eV';

let selectedQ = null;   // currently highlighted q-point index, or null

// ── Helpers ───────────────────────────────────────────────────────────────────
function curPol() { return polSel.value; }

function bzLogScale()  {
  return document.querySelector('input[name="bz-scale"]:checked').value === 'log';
}
function excLogScale() {
  return document.querySelector('input[name="exc-scale"]:checked').value === 'log';
}

function nearestIdx(arr, val) {
  let bi = 0, bd = Math.abs(arr[0] - val);
  for (let i = 1; i < arr.length; i++) {
    const d = Math.abs(arr[i] - val);
    if (d < bd) { bd = d; bi = i; }
  }
  return bi;
}

// per-q intensity at the current Omega_exc, current polarization
function qIntAt(iq, pol, ie) {
  return qItems[iq].intensities[pol][ie];
}

// global min/max of positive intensities (for log scale floor)
function globalLogFloor(pol) {
  let floor = Infinity;
  for (const qi of qItems)
    for (const v of qi.intensities[pol])
      if (v > 0 && v < floor) floor = v;
  return isFinite(floor) ? floor : 1e-30;
}

// ── BZ scatter plot ───────────────────────────────────────────────────────────
function buildBZPlot() {
  const pol   = curPol();
  const logSc = bzLogScale();
  const floor = logSc ? globalLogFloor(pol) : 0;

  const colors = qItems.map(function(qi, iq) {
    const v = qIntAt(iq, pol, iExc);
    return logSc ? Math.log10(Math.max(v, floor)) : v;
  });

  const wts    = qItems.map(qi => qi.weight);
  const maxWt  = Math.max(...wts);
  const szBase = qItems.map(w => 10 + 20 * (w / maxWt));
  const szSel  = szBase.map((s, i) => i === selectedQ ? s * 1.5 : s);
  const outCol = qItems.map((_, i) => i === selectedQ ? 'white' : '#333');
  const outWid = qItems.map((_, i) => i === selectedQ ? 2.5 : 1.0);

  // BZ boundary
  const traceBZ = {
    type: 'scatter', x: bzX, y: bzY,
    mode: 'lines',
    line: { color: '#dddddd', width: 2.5 },
    hoverinfo: 'skip', showlegend: false,
  };

  // q-point markers
  const traceQ = {
    type: 'scatter',
    x: qItems.map(qi => qi.qx),
    y: qItems.map(qi => qi.qy),
    mode: 'markers',
    marker: {
      color: colors,
      colorscale: 'Viridis',
      showscale: true,
      size: szSel,
      line: { color: outCol, width: outWid },
      colorbar: {
        title: { text: logSc ? 'log₁₀(I)' : 'I (a.u.)', side: 'right' },
        thickness: 14,
      },
    },
    text: qItems.map(qi => qi.label),
    hovertemplate: '%{text}<br>q = (%{x:.4f}, %{y:.4f}) Å⁻¹'
                 + '<br>I = %{marker.color:.3e}<extra></extra>',
    customdata: qItems.map((_, i) => i),
    showlegend: false,
  };

  // High-symmetry markers (Γ at origin)
  const gammaHs = hsPoints.filter(h => h.label === 'Γ');
  const otherHs = hsPoints.filter(h => h.label !== 'Γ');

  const traceGamma = {
    type: 'scatter',
    x: gammaHs.map(h => h.x), y: gammaHs.map(h => h.y),
    mode: 'markers+text',
    text: gammaHs.map(h => h.label),
    textposition: 'top center',
    textfont: { size: 13, color: '#fff' },
    marker: { symbol: 'cross', size: 12, color: '#fff',
              line: { color:'#aaa', width:1.5 } },
    hoverinfo: 'skip', showlegend: false,
  };

  const title = 'BZ q-contributions — ' + pol
              + ' — Ω<sub>exc</sub> = ' + excEn[iExc].toFixed(4) + ' eV';

  Plotly.react('bz-div', [traceBZ, traceQ, traceGamma], {
    title: { text: title, font: { size: 12 } },
    xaxis: { title: 'k<sub>x</sub> (Å⁻¹)',
             scaleanchor: 'y', scaleratio: 1,
             zeroline: false, gridcolor: '#333' },
    yaxis: { title: 'k<sub>y</sub> (Å⁻¹)',
             zeroline: false, gridcolor: '#333' },
    plot_bgcolor:  '#12121f',
    paper_bgcolor: '#f4f4f4',
    margin: { l:65, r:10, t:55, b:55 },
  });
}

// ── Excitation profile ────────────────────────────────────────────────────────
function buildExcProfile() {
  const pol    = curPol();
  const logSc  = excLogScale();
  const qNorm  = qItems.reduce((s, qi) => s + qi.weight, 0);

  // total (weighted sum over all q)
  const total = excEn.map((_, f) =>
    qItems.reduce((s, qi) => s + (qi.weight / qNorm) * qi.intensities[pol][f], 0));

  const traces = [{
    type: 'scatter',
    x: excEn, y: total,
    mode: 'lines',
    line: { color: '#ddd', width: 2.5 },
    name: 'Total',
    hovertemplate: 'Ω<sub>exc</sub>: %{x:.4f} eV<br>I: %{y:.3e}<extra>Total</extra>',
  }];

  // individual q-point contributions (all, semi-transparent)
  const palette = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
                   '#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4'];
  for (let iq = 0; iq < Nq; iq++) {
    const qi      = qItems[iq];
    const isSelec = (iq === selectedQ);
    traces.push({
      type: 'scatter',
      x: excEn,
      y: qi.intensities[pol],
      mode: 'lines',
      opacity: isSelec ? 1.0 : 0.35,
      line: {
        color:  isSelec ? '#ff4444' : (palette[iq % palette.length]),
        width:  isSelec ? 2.5 : 1.0,
        dash:   isSelec ? 'solid' : 'dot',
      },
      name: 'q' + iq + (isSelec ? ' ★' : ''),
      hovertemplate: qi.label + '<br>Ω<sub>exc</sub>: %{x:.4f} eV'
                   + '<br>I: %{y:.3e}<extra>q' + iq + '</extra>',
    });
  }

  // vertical marker at current Omega_exc
  const shapes = [{
    type: 'line',
    x0: excEn[iExc], x1: excEn[iExc], y0: 0, y1: 1, yref: 'paper',
    line: { color: 'red', width: 1.5, dash: 'dot' },
  }];

  Plotly.react('exc-div', traces, {
    title: { text: 'Excitation profiles — ' + pol, font: { size: 12 } },
    xaxis: { title: 'Ω<sub>exc</sub> (eV)' },
    yaxis: {
      title: logSc ? 'log₁₀(I)' : 'I (a.u.)',
      type:  logSc ? 'log' : 'linear',
    },
    shapes: shapes,
    legend: { x: 0.7, y: 0.98, bgcolor: 'rgba(0,0,0,0)' },
    plot_bgcolor:  '#fff',
    paper_bgcolor: '#f4f4f4',
    margin: { l:65, r:10, t:55, b:55 },
  });
}

function updateAll() { buildBZPlot(); buildExcProfile(); }

// ── Event wiring ──────────────────────────────────────────────────────────────
polSel.addEventListener('change', updateAll);

slider.addEventListener('input', function() {
  iExc = parseInt(this.value);
  const ev = excEn[iExc];
  document.getElementById('eexc-input').value = ev.toFixed(4);
  document.getElementById('eexc-label').textContent = ev.toFixed(4) + ' eV';
  updateAll();
});

document.getElementById('eexc-input').addEventListener('change', function() {
  const val = parseFloat(this.value);
  if (isNaN(val)) return;
  iExc = nearestIdx(excEn, val);
  slider.value = iExc;
  document.getElementById('eexc-label').textContent = excEn[iExc].toFixed(4) + ' eV';
  updateAll();
});

document.querySelectorAll('input[name="bz-scale"]')
        .forEach(r => r.addEventListener('change', buildBZPlot));
document.querySelectorAll('input[name="exc-scale"]')
        .forEach(r => r.addEventListener('change', buildExcProfile));

// ── Init + click handlers ─────────────────────────────────────────────────────
(async function() {
  updateAll();

  // Click on BZ map → select q-point
  document.getElementById('bz-div').on('plotly_click', function(evt) {
    const pt = evt.points[0];
    if (pt.curveNumber !== 1) return;   // only the q-point scatter (trace index 1)
    selectedQ = pt.customdata;
    const qi  = qItems[selectedQ];
    document.getElementById('info-box').innerHTML =
      '<b>Selected q-point:</b> ' + qi.label
      + ' &nbsp;|&nbsp; weight = ' + qi.weight.toFixed(4)
      + ' &nbsp;|&nbsp; I(Ω<sub>exc</sub>) = '
      + qIntAt(selectedQ, curPol(), iExc).toExponential(3)
      + '<br><small>Crystal coords (folded): ['
      + qi.q_crys[0].toFixed(4) + ', ' + qi.q_crys[1].toFixed(4) + ']'
      + ' &nbsp;&nbsp; Cartesian: ['
      + qi.qx.toFixed(4) + ', ' + qi.qy.toFixed(4) + '] Å⁻¹</small>';
    updateAll();
  });

  // Click on excitation profile → update Omega_exc
  document.getElementById('exc-div').on('plotly_click', function(evt) {
    if (!evt.points.length) return;
    const xval = evt.points[0].x;
    iExc = nearestIdx(excEn, xval);
    slider.value = iExc;
    const ev = excEn[iExc];
    document.getElementById('eexc-input').value = ev.toFixed(4);
    document.getElementById('eexc-label').textContent = ev.toFixed(4) + ' eV';
    updateAll();
  });
})();
</script>
</body>
</html>
"""

html_out = (HTML
            .replace('__DATA__',       all_json)
            .replace('__POL_LABELS__', pol_json))

out_path = Path(args.output)
out_path.write_text(html_out, encoding='utf-8')
size_mb = out_path.stat().st_size / 1e6
print(f'\nSaved: {out_path}  ({size_mb:.1f} MB)')
print('Open in any browser — no server required.')
print('\nTip: click a q-point in the BZ map to highlight its excitation profile.')
print('     Click the excitation profile to move the Omega_exc cursor.')
