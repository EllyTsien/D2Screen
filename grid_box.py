from rdkit import Chem, RDLogger
from typing import Tuple, List
import numpy as np
#############################################
# Suppress Warnings

RDLogger.DisableLog('rdApp.warning')

#############################################
# Grid Box Calculation Methods

# 定义类型别名
Ranges = Tuple[List[float], List[float], List[float]]
Coords = Tuple[float, float, float]
CenterBxSize = Tuple[Coords, Coords]

class GridBox:
    ranges: Ranges
    coords: Coords
    center_bxsize: CenterBxSize


    def __init__(self, inpt_file: str) -> None:
        self.inpt = open(inpt_file, 'r')
        self.data = self.inpt.read()
        self.cmol = Chem.MolFromPDBBlock(self.data)
        self.conf = self.cmol.GetConformer()
        self.ntom = self.cmol.GetNumAtoms()
        self.inpt.close()

    def update_gridbox(self, mol_block: str) -> None:
        self.cmol = Chem.MolFromPDBBlock(mol_block)
        self.conf = self.cmol.GetConformer()
        self.ntom = self.cmol.GetNumAtoms()

    def compute_coords(self) -> Ranges:
        x_coord = [self.conf.GetAtomPosition(c).x for c in range(self.ntom)]
        y_coord = [self.conf.GetAtomPosition(c).y for c in range(self.ntom)]
        z_coord = [self.conf.GetAtomPosition(c).z for c in range(self.ntom)]
        return x_coord, y_coord, z_coord

    def compute_ranges(self) -> Ranges:
        x, y, z = self.compute_coords()
        x_range = [min(x), max(x)]
        y_range = [min(y), max(y)]
        z_range = [min(z), max(z)]
        return x_range, y_range, z_range

    def compute_center(self, use_range: bool = True) -> Coords:
        x, y, z = self.compute_ranges() if use_range else self.compute_coords()
        x_center = round(np.mean(x), 3)
        y_center = round(np.mean(y), 3)
        z_center = round(np.mean(z), 3)
        return x_center, y_center, z_center

    def generate_res_molblock(self, residues_list: List[str]) -> str:
        res_lines = [line for line in self.data.split('\n')
                     if line[22:26].lstrip() in residues_list
                     and 'END' not in line]
        res_block = '\n'.join(res_lines)
        return res_block

    def labox(self, scale: float = 2.0) -> Coords:
        xr, yr, zr = self.compute_ranges()
        center = self.compute_center()
        bxsize = (round(abs(xr[0] - xr[1]) * scale, 3),
                  round(abs(yr[0] - yr[1]) * scale, 3),
                  round(abs(zr[0] - zr[1]) * scale, 3))
        return center, bxsize

    def eboxsize(self, gy_box_ratio: float = 0.23, modified: bool = False) -> CenterBxSize:
        xc, yc, zc = self.compute_coords()
        center = self.compute_center(modified)
        distsq = [(x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
                  for x, y, z in zip(xc, yc, zc)]
        bxsize = (round(np.sqrt(sum(distsq) / len(xc)) / gy_box_ratio, 3),) * 3
        return center, bxsize

    def autodock_grid(self) -> CenterBxSize:
        xr, yr, zr = self.compute_ranges()
        center = self.compute_center()
        bxsize = (22.5, 22.5, 22.5)
        return center, bxsize

    def defined_by_res(self, residue_number: str, scale: float = 1.25) -> CenterBxSize:
        res_list = residue_number.replace(',', ' ').split()
        res_block = self.generate_res_molblock(res_list)
        self.update_gridbox(res_block)
        return self.labox(scale=scale)
    