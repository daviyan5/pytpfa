import configparser
import os

import numpy as np
import sympy as sp

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from petsc4py import PETSc

from .analytical import analytical, initial_condition, dirichlet, neumann, source_term


@dataclass
class ReservoirDescription:
    name: str
    geom: str
    prop: str
    fluid: str
    analytical: Optional[str] = None


@dataclass
class ReservoirInput:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    kx: float
    ky: float
    kz: float
    poro: float
    rho: float
    cporo: float
    mu: float
    b: float
    cfluid: float


@dataclass
class Well:
    wellname: str
    block_coord_x: int
    block_coord_y: int
    block_coord_z: int
    radius: float
    drainage_radius: float
    skin: float
    cond: str
    value: float
    permeability: float
    h: float


@dataclass
class InitialCondition:
    pressure: Callable = field(default_factory=lambda: lambda x, y, z: np.zeros_like(x))


@dataclass
class TimeSettings:
    time_initial: float
    time_final: float
    time_step: float


@dataclass
class BoundaryCondition:
    type: str
    func: Optional[Callable] = None


@dataclass
class ReservoirConfiguration:
    description: ReservoirDescription
    input: ReservoirInput
    wells: Dict[str, Well]
    initial_condition: InitialCondition
    time_settings: TimeSettings
    boundaries: Dict[str, BoundaryCondition]
    source_term: Callable = field(default_factory=lambda: lambda x, y, z, t: np.zeros_like(x))
    analytical_functions: Dict[str, Callable] = field(default_factory=dict)


class ReservoirINIParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.config = configparser.ConfigParser()
        if isinstance(filepath, str):
            self.config.read(filepath)
        else:
            self.config.read_file(filepath)

    def parse(self) -> ReservoirConfiguration:
        description = self._parse_description()
        reservoir_input = self._parse_input()
        wells = self._parse_wells()
        initial_condition = self._parse_initial_condition(description.analytical)
        time_settings = self._parse_time_settings()

        params = self._create_params(reservoir_input, initial_condition)
        boundaries = self._parse_boundaries(description.analytical, params)

        analytical_functions = {}
        if description.analytical and description.analytical.lower() != "none":
            analytical_functions = self._create_analytical_functions(description.analytical, params)

        # At least one boundary must be dirichlet
        if not any(bc.type == "dirichlet" for bc in boundaries.values()):
            raise ValueError("At least one boundary condition must be dirichlet.")

        return ReservoirConfiguration(
            description=description,
            input=reservoir_input,
            wells=wells,
            initial_condition=initial_condition,
            time_settings=time_settings,
            boundaries=boundaries,
            source_term=analytical_functions.get("source", lambda x, y, z, t: 0.0),
            analytical_functions=analytical_functions,
        )

    def _create_params(
        self, reservoir_input: ReservoirInput, initial_condition: InitialCondition
    ) -> Dict:
        return {
            "p_ref": initial_condition.pressure,
            "fluid_compressibility": reservoir_input.cfluid,
            "porosity_compressibility": reservoir_input.cporo,
            "rho_ref": reservoir_input.rho,
            "mu_ref": reservoir_input.mu,
            "phi_ref": reservoir_input.poro,
            "permeability": (
                reservoir_input.kx,
                reservoir_input.ky,
                reservoir_input.kz,
            ),
        }

    def _parse_description(self) -> ReservoirDescription:
        section = self.config["RESERVOIR_DESCRIPTION"]
        return ReservoirDescription(
            name=section.get("NAME", "Reservoir"),
            geom=section.get("GEOM"),
            prop=section.get("PROP"),
            fluid=section.get("FLUID"),
            analytical=self._clean_string(section.get("ANALYTICAL", "None")),
        )

    def _parse_input(self) -> ReservoirInput:
        section = self.config["RESERVOIR_INPUT"]
        return ReservoirInput(
            nx=section.getint("NX"),
            ny=section.getint("NY"),
            nz=section.getint("NZ"),
            dx=section.getfloat("DX"),
            dy=section.getfloat("DY"),
            dz=section.getfloat("DZ"),
            kx=section.getfloat("KX"),
            ky=section.getfloat("KY"),
            kz=section.getfloat("KZ"),
            poro=section.getfloat("PORO"),
            rho=section.getfloat("RHO"),
            cporo=section.getfloat("CPORO"),
            mu=section.getfloat("MU"),
            b=section.getfloat("B"),
            cfluid=section.getfloat("CFLUID"),
        )

    def _parse_wells(self) -> Dict[str, Well]:
        wells = {}
        for section_name in self.config.sections():
            if section_name.startswith("WELL_"):
                section = self.config[section_name]
                well = Well(
                    wellname=section.get("WELLNAME"),
                    block_coord_x=section.getint("BLOCK_COORD_X"),
                    block_coord_y=section.getint("BLOCK_COORD_Y"),
                    block_coord_z=section.getint("BLOCK_COORD_Z"),
                    radius=section.getfloat("RADIUS"),
                    drainage_radius=section.getfloat("DRAINAGE_RADIUS"),
                    skin=section.getfloat("SKIN"),
                    cond=section.get("COND"),
                    value=section.getfloat("VALUE"),
                    permeability=section.getfloat("PERMEABILITY"),
                    h=section.getfloat("h"),
                )
                wells[section_name] = well
        return wells

    def _parse_initial_condition(self, analytical) -> InitialCondition:
        section = self.config["INITIAL_CONDITION"]
        pressure = None
        if section.get("PRESSURE") == '"None"':
            pressure = initial_condition(analytical)
        else:
            pressure_val = section.getfloat("PRESSURE", 0.0)
            pressure = lambda x, y, z, v=pressure_val: np.full_like(x, v)
        return InitialCondition(pressure=pressure)

    def _parse_time_settings(self) -> TimeSettings:
        section = self.config["TIME_SETTINGS"]
        return TimeSettings(
            time_initial=section.getfloat("TIME_INITIAL"),
            time_final=section.getfloat("TIME_FINAL"),
            time_step=section.getfloat("TIME_STEP"),
        )

    def _parse_boundaries(
        self, analytical_expr: Optional[str], params: Dict
    ) -> Dict[str, BoundaryCondition]:
        boundaries = {}
        boundary_names = ["LEFT", "RIGHT", "FRONT", "BACK", "UP", "DOWN"]

        for name in boundary_names:
            section_name = f"RESERVOIR_BOUNDARY.{name}"
            if section_name in self.config:
                section = self.config[section_name]
                bc_type = self._clean_string(section.get("TYPE")).lower()
                bc = BoundaryCondition(type=bc_type)

                if analytical_expr and analytical_expr.lower() != "none":
                    expr = self._parse_expression(analytical_expr)
                    if bc_type == "dirichlet":
                        bc.func = dirichlet(expr, params)
                    elif bc_type == "neumann":
                        bc.func = neumann(expr, params)
                else:
                    value = section.getfloat("VALUE", 0.0)
                    if bc_type == "dirichlet":
                        bc.func = lambda x, y, z, t, v=value: np.full_like(x, v)
                    elif bc_type == "neumann":
                        bc.func = lambda x, y, z, t, nx, ny, nz, v=value: np.full_like(x, v)

                boundaries[name.lower()] = bc
            else:
                boundaries[name.lower()] = BoundaryCondition(
                    type="neumann", func=lambda x, y, z, t, nx, ny, nz: np.zeros_like(x)
                )
        return boundaries

    def _create_analytical_functions(self, expr_str: str, params: Dict) -> Dict[str, Callable]:
        expr = self._parse_expression(expr_str)
        return {
            "solution": analytical(expr, params),
            "dirichlet": dirichlet(expr, params),
            "neumann": neumann(expr, params),
            "source": source_term(expr, params),
        }

    def _parse_expression(self, expr_str: str) -> sp.Expr:
        x, y, z, t = sp.symbols("x y z t")
        expr_str = expr_str.lower().replace("^", "**")
        return sp.sympify(expr_str, locals={"x": x, "y": y, "z": z, "t": t})

    def _clean_string(self, value: Optional[str]) -> Optional[str]:
        if value:
            return value.strip().strip('"').strip("'")
        return value


def parse_reservoir_config(filepath: str, cache: bool = True) -> ReservoirConfiguration:
    parser = ReservoirINIParser(filepath)
    reservoir_config = parser.parse()

    return reservoir_config
