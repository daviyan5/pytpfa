# pytpfa
**Implementation using PETSc/PETSc4Py of the Two Point Flux Approximation Model with support to GPU nodes**

---

## Governing Equation for Slightly Compressible Flow in Porous Media

The fluid pressure is modeled by the following equation:

$$\nabla \cdot \left( \frac{\beta_c k}{\mu B} \nabla p \right) + S_p = 0$$

The mass balance is expressed as:

$$S \Delta V = S_p \phi + S_u$$

Where:
* $p$: Fluid pressure.
* $k$: Rock permeability (effective permeability).
* $\mu$: Fluid dynamic viscosity.
* $B$: Formation volume factor (accounts for fluid compressibility).
* $\beta_c$: Scaling factor to adjust the transmissibility for compressibility effects.
* $S_p$: Pressure source term (includes accumulation and any production/injection effects).
* $S$: Accumulation coefficient linking pressure change to mass conservation.
* $\Delta V$: Volume of a control cell (grid block).
* $\phi$: Porosity (fraction of the cell volume that is pore space).
* $S_u$: Additional source/sink term (e.g., contributions from wells or external sources).

The variation on the density of the fluid is given by:

$$\rho = \rho_r (1 + c_\rho (p - p_r))$$

Where:
* $c_\rho$: Fluid compressibility.
* $\rho_r$: Reference fluid density.
* $p_r$: Reference fluid pressure.

Thus, $B$ shall be given by:

$$B = \frac{B_r}{1 + c_\rho (p - p_r)}$$

Where:
* $B_r$: Reference formation volume factor.

The variation on the porosity is given by:

$$\phi = \phi_r (1 + c_\phi (p - p_r))$$

Where:
* $c_\phi$: Porosity compressibility.
* $\phi_r$: Reference porosity.
* $p_r$: Reference fluid pressure.

---

### Peaceman's Well Model

The well representation is given by Peaceman's model:

$$J = \frac{2\pi k h}{\mu B (\ln(r_e / r_w) + s)}$$

Where:
* $J$: Well index, quantifying how readily fluid flows between the reservoir and the well.
* $h$: Reservoir thickness (or effective height in the well block).
* $r_w$: Wellbore radius.
* $r_e$: Effective drainage radius, representing the well’s influence in the reservoir.
* $s$: Skin factor, accounting for near-wellbore damage or stimulation.

For non-square wellblocks with anisotropic permeability, Peaceman derived an equivalent wellblock radius ($r_{eq}$):

$$r_{eq} = 0.28 \frac{\left[ \left( \frac{k_y}{k_x} \right)^{1/2} \Delta x^2 + \left( \frac{k_x}{k_y} \right)^{1/2} \Delta y^2 \right]^{1/2}}{\left( \frac{k_y}{k_x} \right)^{1/4} + \left( \frac{k_x}{k_y} \right)^{1/4}}$$

Where:
* $k_x$: Permeability in the x-direction in the wellblock.
* $k_y$: Permeability in the y-direction in the wellblock.
* $\Delta x$: Grid block width in the x-direction.
* $\Delta y$: Grid block width in the y-direction.

---

## Two-Point Flux Approximation (TPFA)

Imposing a single flux that is continuous across the interface leads to the following expression for the flux at the interface:

$$(\mathbf{v}_e \cdot \mathbf{N}_e) = - \left( \frac{2 K_L K_R}{K_L \Delta X_R + K_R \Delta X_L} \right) A (p_R - p_L)$$

Where:
* $\mathbf{v}_e$: Darcy velocity at the interface.
* $\mathbf{N}_e$: Normal vector at the interface.
* $K_L$: Permeability in the left cell.
* $K_R$: Permeability in the right cell.
* $\Delta X_L$: Distance between the interface and the left cell center.
* $\Delta X_R$: Distance between the interface and the right cell center.
* $p_L$: Pressure in the left cell.
* $p_R$: Pressure in the right cell.
* $A$: Area of the interface.

---

## Installation

To install `pytpfa`, you can use pip:

```bash
pip install .
``` 

Usage

Basic usage of the pytpfa solver should look like this:

```python
from pytpfa import TPFASolver

solver = TPFASolver("Basic Example")
reservoir_path = "<path_to_reservoir_ini_file>"
solver.solve(reservoir_path, postprocess=True, checks=True)
``` 

Examples of .ini files can be found in the examples directory.

--- 

## Examples

To run the examples, you can use GNU Make.
Running with Make

```bash
make <target> [OPTIONS]
```

Where <target> can be one of example1, example2, example_li, etc.

Options:

    MPI=<n>: Set the number of MPI processes (default: 1).

    DEBUG=<yes|no>: Enable PETSc debug flags (default: no).

    OPT=<yes|no>: Enable optimization (reuse preconditioner, skip checks) (default: yes).

    POST=<yes|no>: Enable post-processing VTK output (default: no).

    PROFILE=<yes|no>: Enable performance profiling (default: no).

    LIMIT=<yes|no>: Limit memory usage per process (default: no).

Example:

```bash 
make example_li MPI=4 POST=yes
``` 

Running Directly

You can also run the solver script directly with PETSc options:
```bash
mpirun -n 4 python3 tools/run.py -name TPFA_Example1 -reservoir tools/examples/example_1/reservoir.ini -opt -post
```

---

## Tests

The module pytest is used for testing. To run the tests with MPI, you can use:

```bash
make test MPI=<n>
``` 
Or run pytest directly:

```bash
mpirun -n <n> pytest
```
