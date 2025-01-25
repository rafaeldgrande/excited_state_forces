
# Classes
class Parameters_BSE:

    def __init__(self, Nkpoints_BSE, Kpoints_BSE, Ncbnds, Nvbnds, Nval, Ncbnds_sum, Nvbnds_sum, Ncbnds_coarse, Nvbnds_coarse, Nkpoints_coarse, rec_cell_vecs):
        self.Nkpoints_BSE = Nkpoints_BSE
        self.Kpoints_BSE = Kpoints_BSE
        self.Ncbnds = Ncbnds
        self.Nvbnds = Nvbnds
        self.Nval = Nval
        self.Ncbnds_sum = Ncbnds_sum
        self.Nvbnds_sum = Nvbnds_sum
        self.Ncbnds_coarse = Ncbnds_coarse
        self.Nvbnds_coarse = Nvbnds_coarse
        self.Nkpoints_coarse = Nkpoints_coarse
        self.rec_cell_vecs = rec_cell_vecs

class Parameters_MF:

    def __init__(self, Nat, atomic_pos, cell_vecs, cell_vol, alat):
        self.Nat = Nat
        self.atomic_pos = atomic_pos
        self.cell_vecs = cell_vecs
        self.cell_vol = cell_vol
        self.alat = alat
        self.Nmodes = 3 * Nat


class Parameters_ELPH:

    def __init__(self, Nkpoints_DPFT, Kpoints_DFPT):
        self.Nkpoints_DFPT = Nkpoints_DPFT
        self.Kpoints_DFPT = Kpoints_DFPT