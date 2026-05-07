"""
Readers for BerkeleyGW unformatted binary files produced by absorption.x:

  * ``vmtxel`` / ``vmtxel_b1, b2, b3`` : optical matrix elements
  * ``dtmat``                           : dcc / dvv coarse->fine transformation
                                          matrices

Both readers follow the binary layout written in
``BerkeleyGW/BSE/vmtxel.f90`` (write_vmtxel_bin) and
``BerkeleyGW/BSE/intwfn.f90`` (the dtmat write loop).

Functions
---------
read_vmtxel(filename, complex_flavor=True)
    -> dict with keys: nk, nband, mband, ns, opr, s1
read_dtmat(filename, complex_flavor=True)
    -> dict with keys: idimensions, is_periodic, npts_intp_kernel,
                       nkpt_co, n2b_co, n1b_co, nkpt_fi, ncb_fi, nvb_fi,
                       nspin, kco, dcn, dvn, intp_coefs, fi2co_wfn

vmtxel_to_hdf5(in_path, out_path, ...)
dtmat_to_hdf5(in_path, out_path, ...)
    Convenience wrappers that dump everything into a self-describing .h5.

Notes
-----
* BerkeleyGW is built in either "Real" or "Complex" flavor. ``SCALAR``
  is COMPLEX(DPC) (16 B) in the complex flavor and REAL(DP) (8 B) in
  the real flavor. The ``MoS2`` test data uses the complex flavor
  (see line 3 of absorption.out: "Complex version"), which is the
  default below.
* The vmtxel format is identical between flavors apart from SCALAR.
* The dtmat format here matches BerkeleyGW >= 1.1.0 (r5961). The very
  first record holds 6 ints: idimensions, is_periodic(1:3),
  npts_intp_kernel, nkpt_co.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
from scipy.io import FortranFile


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

def _scalar_dtype(complex_flavor: bool) -> np.dtype:
    """Return the numpy dtype used by BGW's SCALAR for a given flavor."""
    return np.dtype(np.complex128) if complex_flavor else np.dtype(np.float64)


def _read_one_record_struct(f: FortranFile, ndt_int: int, scalar_dtype: np.dtype):
    """
    Read one Fortran record made of ``ndt_int`` int32 followed by one SCALAR.

    Useful for the ``write(13) ik_fi, ic_fi, ik_co, ic_co, is, dcn(...)``
    style records used in dtmat. Returns a tuple (ints, scalar_value).
    """
    raw = f.read_record(np.dtype([
        ('ints', np.int32, (ndt_int,)),
        ('val', scalar_dtype),
    ]))
    return raw['ints'][0], raw['val'][0]


# --------------------------------------------------------------------------
# vmtxel
# --------------------------------------------------------------------------

def read_vmtxel(filename: str, complex_flavor: bool = True) -> dict:
    """
    Read a BerkeleyGW ``vmtxel`` (single polarization) binary file.

    File layout (Fortran unformatted, 4-byte record markers):

        record 1: int32 x 5  -- nk, nband, mband, ns, opr
        record 2: SCALAR x (nk*nband*mband*ns) -- s1 (flat index)

    The flat index follows BerkeleyGW's ``bse_index(ik, ic, iv, is)``.

    Parameters
    ----------
    filename : str
        Path to ``vmtxel`` or ``vmtxel_bX``.
    complex_flavor : bool
        Whether BerkeleyGW was compiled in complex flavor.

    Returns
    -------
    dict with keys:
        nk, nband, mband, ns, opr, s1
        s1 has shape (nk, nband, mband, ns) (Fortran-order reshaped to C).
        ``opr`` : 0 = velocity, 1 = momentum.
    """
    sdtype = _scalar_dtype(complex_flavor)

    with FortranFile(filename, 'r') as f:
        header = f.read_ints(np.int32)
        if header.size != 5:
            raise ValueError(
                f"vmtxel header has {header.size} ints, expected 5")
        nk, nband, mband, ns, opr = (int(x) for x in header)

        nmat = nk * nband * mband * ns
        s1_flat = f.read_record(sdtype)
        if s1_flat.size != nmat:
            raise ValueError(
                f"vmtxel s1 record has {s1_flat.size} elements, "
                f"expected {nmat}")

    # Fortran storage order is (ik, ic, iv, is) with ik fastest.
    s1 = np.reshape(s1_flat, (ns, mband, nband, nk), order='C').transpose()
    # -> shape (nk, nband, mband, ns)

    return {
        'nk': nk,
        'nband': nband,
        'mband': mband,
        'ns': ns,
        'opr': opr,
        's1': s1,
    }


def vmtxel_to_hdf5(in_path: str,
                   out_path: str,
                   complex_flavor: bool = True,
                   compress: bool = True) -> None:
    """Read a vmtxel binary and write a self-describing .h5 file."""
    import h5py

    data = read_vmtxel(in_path, complex_flavor=complex_flavor)

    kw = dict(compression='gzip', compression_opts=4) if compress else {}
    with h5py.File(out_path, 'w') as h5:
        hdr = h5.create_group('vmtxel_header')
        for k in ('nk', 'nband', 'mband', 'ns', 'opr'):
            hdr.attrs[k] = data[k]
        hdr.attrs['opr_meaning'] = 'velocity' if data['opr'] == 0 else 'momentum'
        hdr.attrs['source_file'] = os.path.basename(in_path)
        hdr.attrs['complex_flavor'] = complex_flavor

        grp = h5.create_group('vmtxel_data')
        grp.create_dataset('s1', data=data['s1'], **kw)
        grp['s1'].attrs['axes'] = '(nk, nband, mband, ns)'


# --------------------------------------------------------------------------
# dtmat
# --------------------------------------------------------------------------

def read_dtmat(filename: str, complex_flavor: bool = True) -> dict:
    """
    Read a BerkeleyGW ``dtmat`` binary file (post-r5961 format).

    Layout (matching ``intwfn.f90``):

        record 1: int32 x 6  -- idimensions, is_periodic(1:3),
                                npts_intp_kernel, nkpt_co
        record 2: int32 x 8  -- nkpt_co, n2b_co, n1b_co,
                                nkpt_fi, ncb_fi, nvb_fi,
                                nspin, npts_intp_kernel
        nkpt_co records: real(dp) x 3 -- coarse-grid kco(1:3, jj)

        for ivert in 1..npts_intp_kernel:
            (nkpt_fi*ncb_fi*n2b_co*nspin) records, each:
                int32 x 5 + SCALAR x 1  --
                ik_fi, ic_fi, ik_co, ic_co, is, dcn(ic_fi,ic_co,is,ik_fi,ivert)

        for ivert in 1..npts_intp_kernel:
            (nkpt_fi*nvb_fi*n1b_co*nspin) records, each:
                int32 x 5 + SCALAR x 1  --
                ik_fi, iv_fi, ik_co, iv_co, is, dvn(iv_fi,iv_co,is,ik_fi,ivert)

        if npts_intp_kernel > 1:
            1 record: real(dp) x (npts_intp_kernel*nkpt_fi) -- intp_coefs

    Parameters
    ----------
    filename : str
    complex_flavor : bool

    Returns
    -------
    dict with the listed scalars + arrays:
        kco          (nkpt_co, 3)
        dcn          (ncb_fi, n2b_co, nspin, nkpt_fi, npts_intp_kernel)
        dvn          (nvb_fi, n1b_co, nspin, nkpt_fi, npts_intp_kernel)
        fi2co_wfn    (npts_intp_kernel, nkpt_fi)
        intp_coefs   (npts_intp_kernel, nkpt_fi)  (1.0 if npts_intp_kernel==1)
    """
    sdtype = _scalar_dtype(complex_flavor)

    with FortranFile(filename, 'r') as f:
        h1 = f.read_ints(np.int32)
        if h1.size != 6:
            raise ValueError(
                f"dtmat header 1 has {h1.size} ints, expected 6 "
                "(post-r5961 dtmat)")
        idimensions, ip1, ip2, ip3, npts_intp_kernel, nkpt_co_h1 = h1

        h2 = f.read_ints(np.int32)
        if h2.size != 8:
            raise ValueError(
                f"dtmat header 2 has {h2.size} ints, expected 8")
        (nkpt_co, n2b_co, n1b_co,
         nkpt_fi, ncb_fi, nvb_fi,
         nspin, npts_intp_kernel_h2) = h2

        if nkpt_co_h1 != nkpt_co:
            raise ValueError(
                f"nkpt_co inconsistent in dtmat headers: "
                f"{nkpt_co_h1} vs {nkpt_co}")
        if npts_intp_kernel != npts_intp_kernel_h2:
            raise ValueError(
                f"npts_intp_kernel inconsistent in dtmat headers: "
                f"{npts_intp_kernel} vs {npts_intp_kernel_h2}")

        # Coarse k-points: nkpt_co records of 3 real(dp).
        kco = np.empty((nkpt_co, 3), dtype=np.float64)
        for jj in range(nkpt_co):
            r = f.read_reals(np.float64)
            if r.size != 3:
                raise ValueError(
                    f"dtmat kco record {jj} has {r.size} reals, expected 3")
            kco[jj, :] = r

        dcn = np.zeros((ncb_fi, n2b_co, nspin, nkpt_fi, npts_intp_kernel),
                       dtype=sdtype)
        # shape (nc_fi, nc_co, Nspin, nk_fi, )
        fi2co_wfn = np.zeros((npts_intp_kernel, nkpt_fi), dtype=np.int32)

        for ivert in range(npts_intp_kernel):
            n_records = nkpt_fi * ncb_fi * n2b_co * nspin
            for _ in range(n_records):
                ints, val = _read_one_record_struct(f, 5, sdtype)
                ik_fi, ic_fi, ik_co, ic_co, is_ = ints
                # Fortran -> 0-based
                dcn[ic_fi - 1, ic_co - 1, is_ - 1,
                    ik_fi - 1, ivert] = val
                # fi2co_wfn is overwritten with the same value many times,
                # but mirrors what intwfn.f90 reads back.
                fi2co_wfn[ivert, ik_fi - 1] = ik_co

        # dvn block
        dvn = np.zeros((nvb_fi, n1b_co, nspin, nkpt_fi, npts_intp_kernel),
                       dtype=sdtype)
        for ivert in range(npts_intp_kernel):
            n_records = nkpt_fi * nvb_fi * n1b_co * nspin
            for _ in range(n_records):
                ints, val = _read_one_record_struct(f, 5, sdtype)
                ik_fi, iv_fi, ik_co, iv_co, is_ = ints
                dvn[iv_fi - 1, iv_co - 1, is_ - 1,
                    ik_fi - 1, ivert] = val
                fi2co_wfn[ivert, ik_fi - 1] = ik_co

        if npts_intp_kernel > 1:
            r = f.read_reals(np.float64)
            expected = npts_intp_kernel * nkpt_fi
            if r.size != expected:
                raise ValueError(
                    f"dtmat intp_coefs record has {r.size} reals, "
                    f"expected {expected}")
            intp_coefs = np.reshape(r, (nkpt_fi, npts_intp_kernel),
                                    order='C').T
        else:
            intp_coefs = np.ones((1, nkpt_fi), dtype=np.float64)

    return {
        'idimensions': int(idimensions),
        'is_periodic': np.array([ip1, ip2, ip3], dtype=np.int32),
        'npts_intp_kernel': int(npts_intp_kernel),
        'nkpt_co': int(nkpt_co),
        'n2b_co': int(n2b_co),
        'n1b_co': int(n1b_co),
        'nkpt_fi': int(nkpt_fi),
        'ncb_fi': int(ncb_fi),
        'nvb_fi': int(nvb_fi),
        'nspin': int(nspin),
        'kco': kco,
        'dcn': dcn,
        'dvn': dvn,
        'fi2co_wfn': fi2co_wfn,
        'intp_coefs': intp_coefs,
    }


def dtmat_to_hdf5(in_path: str,
                  out_path: str,
                  complex_flavor: bool = True,
                  compress: bool = True) -> None:
    """Read a dtmat binary and write a self-describing .h5 file."""
    import h5py

    d = read_dtmat(in_path, complex_flavor=complex_flavor)

    kw = dict(compression='gzip', compression_opts=4) if compress else {}
    with h5py.File(out_path, 'w') as h5:
        hdr = h5.create_group('dtmat_header')
        for k in ('idimensions', 'npts_intp_kernel',
                  'nkpt_co', 'n2b_co', 'n1b_co',
                  'nkpt_fi', 'ncb_fi', 'nvb_fi', 'nspin'):
            hdr.attrs[k] = d[k]
        hdr.attrs['is_periodic'] = d['is_periodic']
        hdr.attrs['source_file'] = os.path.basename(in_path)
        hdr.attrs['complex_flavor'] = complex_flavor

        grp = h5.create_group('dtmat_data')
        grp.create_dataset('kco', data=d['kco'])
        grp['kco'].attrs['axes'] = '(nkpt_co, 3)'
        grp.create_dataset('dcn', data=d['dcn'], **kw)
        grp['dcn'].attrs['axes'] = (
            '(ncb_fi, n2b_co, nspin, nkpt_fi, npts_intp_kernel)')
        grp.create_dataset('dvn', data=d['dvn'], **kw)
        grp['dvn'].attrs['axes'] = (
            '(nvb_fi, n1b_co, nspin, nkpt_fi, npts_intp_kernel)')
        grp.create_dataset('fi2co_wfn', data=d['fi2co_wfn'])
        grp['fi2co_wfn'].attrs['axes'] = '(npts_intp_kernel, nkpt_fi)'
        grp.create_dataset('intp_coefs', data=d['intp_coefs'])
        grp['intp_coefs'].attrs['axes'] = '(npts_intp_kernel, nkpt_fi)'


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cli(argv: Optional[list[str]] = None) -> int:
    """
    Usage:
        python bgw_binary_io.py vmtxel <in> [<out.h5>] [--real]
        python bgw_binary_io.py dtmat  <in> [<out.h5>] [--real]
    """
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print(_cli.__doc__)
        return 1

    kind, in_path, *rest = argv
    complex_flavor = '--real' not in rest
    rest = [a for a in rest if a != '--real']
    out_path = rest[0] if rest else in_path + '.h5'

    if kind == 'vmtxel':
        vmtxel_to_hdf5(in_path, out_path, complex_flavor=complex_flavor)
    elif kind == 'dtmat':
        dtmat_to_hdf5(in_path, out_path, complex_flavor=complex_flavor)
    else:
        print(f"Unknown kind: {kind!r} (expected 'vmtxel' or 'dtmat')")
        return 2

    print(f"Wrote {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())
