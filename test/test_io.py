import pytest
import tempfile
import os
import numpy as np

from pytpfa.utils.io import parse_reservoir_config


@pytest.fixture
def create_test_ini_file():
    """Fixture to create temporary INI file for testing"""
    files = []

    def _create_file(content: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(content)
            files.append(f.name)
            return f.name

    yield _create_file

    for file in files:
        os.unlink(file)


ANALYTICAL_INI_CONTENT = """
[RESERVOIR_DESCRIPTION]
GEOM = LINEAR
PROP = HOMOGENEOUS
FLUID = SLIGHTLY_COMPRESSIBLE
ANALYTICAL = "exp(x**2 * t) + sin(y**0.5 * t/2) + sqrt(x*z*y) * t**2"

[RESERVOIR_INPUT]
NX = 10
NY = 12
NZ = 13
DX = 200
DY = 100
DZ = 20
KX = 0.100
KY = 0.100
KZ = 0.100
PORO = 0.20
RHO = 50
CPORO = 2.5e-5
MU = 2
B = 1
CFLUID = 2.5e-5

[INITIAL_CONDITION]
PRESSURE = 3000

[TIME_SETTINGS]
TIME_INITIAL = 0
TIME_FINAL = 20
TIME_STEP = 10

[RESERVOIR_BOUNDARY.LEFT]
TYPE = "dirichlet"

[RESERVOIR_BOUNDARY.RIGHT]
TYPE = "dirichlet"
"""


def test_configuration_with_values(create_test_ini_file):
    """Test parsing configuration with boundary values"""
    ini_content = """[RESERVOIR_DESCRIPTION]
GEOM = LINEAR
PROP = HOMOGENEOUS
FLUID = SLIGHTLY_COMPRESSIBLE
ANALYTICAL = "None"

[RESERVOIR_INPUT]
NX = 10
NY = 12
NZ = 1
DX = 200
DY = 100
DZ = 20
KX = 0.100
KY = 0.100
KZ = 0.100
PORO = 0.20
RHO = 50
CPORO = 2.5e-5
MU = 2
B = 1
CFLUID = 2.5e-5

[INITIAL_CONDITION]
PRESSURE = 3000

[TIME_SETTINGS]
TIME_INITIAL = 0
TIME_FINAL = 20
TIME_STEP = 10

[WELL_1]
WELLNAME = WELL_1
BLOCK_COORD_X = 12
BLOCK_COORD_Y = 12
BLOCK_COORD_Z = 0
RADIUS = 0.1
DRAINAGE_RADIUS = 100
SKIN = 0
COND = constant_rate
VALUE = 1.478
PERMEABILITY = 100
h = 20

[RESERVOIR_BOUNDARY.LEFT]
TYPE = "dirichlet"
VALUE = 1000

[RESERVOIR_BOUNDARY.BACK]
TYPE = "neumann"
VALUE = 0.0
"""
    filepath = create_test_ini_file(ini_content)
    config = parse_reservoir_config(filepath)

    assert config.description.geom == "LINEAR"
    assert config.description.analytical.lower() == "none"
    assert config.input.nx == 10
    assert config.initial_condition.pressure == 3000.0
    assert config.time_settings.time_final == 20.0
    assert len(config.wells) == 1
    assert config.wells["WELL_1"].value == 1.478

    x_val, y_val, z_val = np.random.rand(3) * 10
    nx_val, ny_val, nz_val = np.random.rand(3) * 2 - 1

    assert config.boundaries["left"].type == "dirichlet"
    assert config.boundaries["left"].func(x_val, y_val, z_val, 1) == 1000.0

    assert config.boundaries["back"].type == "neumann"
    assert config.boundaries["back"].func(x_val, y_val, z_val, 1, nx_val, ny_val, nz_val) == 0.0


def test_analytical_configuration(create_test_ini_file):
    """Test parsing configuration with analytical expression"""
    filepath = create_test_ini_file(ANALYTICAL_INI_CONTENT)
    config = parse_reservoir_config(filepath)

    assert config.description.analytical is not None
    assert config.boundaries["left"].type == "dirichlet"
    assert config.boundaries["left"].func is not None
    assert "solution" in config.analytical_functions
    assert "dirichlet" in config.analytical_functions
    assert "neumann" in config.analytical_functions
    assert "source" in config.analytical_functions

    left_result = config.boundaries["left"].func(1.0, 2.0, 3.0, 0.5)

    assert isinstance(left_result, (float, np.number))


def test_source_function(create_test_ini_file):
    """Tests that the analytical source term function is created and returns a number."""
    filepath = create_test_ini_file(ANALYTICAL_INI_CONTENT)
    config = parse_reservoir_config(filepath)

    assert "source" in config.analytical_functions
    source_func = config.analytical_functions["source"]

    source_result = source_func(1.0, 2.0, 3.0, 0.5)

    assert isinstance(source_result, (float, np.number))


def test_array_inputs(create_test_ini_file):
    """Tests that lambdified functions correctly handle NumPy array inputs."""
    filepath = create_test_ini_file(ANALYTICAL_INI_CONTENT)
    config = parse_reservoir_config(filepath)

    solution_func = config.analytical_functions["solution"]

    num_points = 5
    x_arr = np.linspace(0, 1, num_points)
    y_arr = np.linspace(0, 2, num_points)
    z_arr = np.linspace(0, 3, num_points)
    t_arr = np.full(num_points, 0.5)

    result_arr = solution_func(x_arr, y_arr, z_arr, t_arr)

    assert isinstance(result_arr, np.ndarray)
    assert result_arr.shape == (num_points,)
