"""
Mock library for GeoCroissant CMIP6 dataset demonstration.

This module provides mock implementations of croissant, torch, xarray and other
libraries to demonstrate environmental data workflows without requiring the
actual libraries to be installed.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Mock the core libraries and classes
class MockCroissant:
    __version__ = "1.2.3"


class MockTorch:
    __version__ = "2.1.0"
    
    class Tensor:
        def __init__(self, data, dtype=None):
            self.data = np.array(data)
            self.dtype = dtype or 'float32'
            self.shape = self.data.shape
        
        def size(self):
            return self.shape
        
        def min(self):
            return float(self.data.min())
        
        def max(self):
            return float(self.data.max())
    
    @staticmethod
    def tensor(data, dtype=None):
        return MockTorch.Tensor(data, dtype)
    
    @staticmethod
    def device(x):
        return 'cpu'  # Mock device


class MockCuda:
    @staticmethod
    def is_available():
        return False


class MockModule:
    def __init__(self):
        pass
    
    def parameters(self):
        # Return some mock parameters
        return [MockTorch.Tensor(np.random.randn(10, 10)) for _ in range(3)]
    
    def forward(self, x):
        # Mock forward pass
        return x
    
    def to(self, device):
        return self  # Mock .to() method


class MockNN:
    Module = MockModule
    
    class Conv2d:
        def __init__(self, in_channels, out_channels, kernel_size, padding=0):
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.padding = padding
    
    class LSTM:
        def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.batch_first = batch_first
    
    class Linear:
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features
    
    class ReLU:
        def __init__(self):
            pass
    
    class MSELoss:
        def __init__(self):
            pass
        
        def __call__(self, pred, target):
            return MockTorch.Tensor([0.5])  # Mock loss value


class MockOptim:
    class Adam:
        def __init__(self, params, lr=0.001):
            self.params = params
            self.lr = lr
        
        def zero_grad(self):
            pass
        
        def step(self):
            pass


class MockXArray:
    __version__ = "2023.10.1"


class MockDataset:
    def __init__(self, name, description, provider, variables, spatial_resolution, temporal_resolution):
        self.name = name
        self.description = description
        self.provider = provider
        self.variables = variables
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution


# Mock objects for detailed interrogation
class MockModel:
    def __init__(self, name, institution, resolution, experiments):
        self.name = name
        self.institution = institution
        self.nominal_resolution = resolution
        self.experiments = experiments


class MockExperiment:
    def __init__(self, exp_id, description, activity_id, models_count):
        self.experiment_id = exp_id
        self.description = description
        self.activity_id = activity_id
        # Create a mock list of models
        self.participating_models = [f"Model_{i+1}" for i in range(models_count)]


class MockVariable:
    def __init__(self, var_id, long_name, units, frequency, dimensions):
        self.variable_id = var_id
        self.long_name = long_name
        self.units = units
        self.frequency = frequency
        self.dimensions = dimensions


class MockXArrayDataset:
    """Mock xarray dataset for the to_xarray() method"""
    def __init__(self):
        # Create mock coordinate arrays
        self.lat = type('Coordinate', (), {
            'values': np.linspace(30, 70, 40),
            'diff': lambda dim: type('diff', (), {'mean': lambda: type('mean', (), {'values': 1.0})()})()
        })()
        
        self.lon = type('Coordinate', (), {
            'values': np.linspace(-130, -60, 70),
            'diff': lambda dim: type('diff', (), {'mean': lambda: type('mean', (), {'values': 1.0})()})()
        })()
        
        self.time = type('Time', (), {
            'dt': type('dt', (), {
                'strftime': lambda fmt: type('strftime', (), {'values': '2020-01'})()
            })()
        })()
        
        # Mock temperature data
        self.tas = type('Variable', (), {
            'lon': self.lon,
            'lat': self.lat,
            'values': np.random.randn(40, 70) * 10 + 280,  # Realistic temperature values
            'min': lambda: type('min', (), {'values': 265.0})(),
            'max': lambda: type('max', (), {'values': 305.0})(),
            'mean': lambda: type('mean', (), {'values': 285.0})()
        })()
    
    def isel(self, time):
        return self


class MockFilteredDataset:
    def __init__(self, original_size=1250.0, filtered_size=85.2):
        self.estimated_size_gb = filtered_size
        self.variables = ["tas"]
        self.spatial_shape = (40, 70)  # Reduced from global to North America
        self.temporal_shape = (372,)  # 31 years * 12 months
        self.n_timesteps = 372
        self.data_format = "xarray"
    
    def to_pytorch_dataset(self, target_variable, feature_variables, sequence_length, stride, normalize=True, transform=None):
        return MockPyTorchDataset(
            length=360,  # 30 years of 12-month sequences
            feature_shape=(len(feature_variables), sequence_length, *self.spatial_shape),
            target_shape=(sequence_length, *self.spatial_shape)
        )
    
    def to_xarray(self):
        return MockXArrayDataset()


class MockPyTorchDataset:
    def __init__(self, length, feature_shape, target_shape):
        self._length = length
        self.feature_shape = feature_shape
        self.target_shape = target_shape
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        # Return mock tensors with realistic shapes
        features = MockTorch.Tensor(np.random.randn(*self.feature_shape), dtype='float32')
        targets = MockTorch.Tensor(np.random.randn(*self.target_shape), dtype='float32')
        return features, targets


class MockDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self._length = len(dataset) // batch_size
    
    def __len__(self):
        return self._length
    
    def __iter__(self):
        for i in range(self._length):
            # Return a batch of samples
            batch_features = []
            batch_targets = []
            for j in range(self.batch_size):
                idx = i * self.batch_size + j
                if idx < len(self.dataset):
                    features, targets = self.dataset[idx]
                    batch_features.append(features.data)
                    batch_targets.append(targets.data)
            
            batch_features = MockTorch.Tensor(np.stack(batch_features))
            batch_targets = MockTorch.Tensor(np.stack(batch_targets))
            yield batch_features, batch_targets


class MockCMIP6Dataset:
    def __init__(self):
        self.id = "cmip6_global_climate"
        self.title = "CMIP6 Global Climate Projections"
        self.description = "Comprehensive climate model data from CMIP6 including temperature, precipitation, and atmospheric variables"
        self.license = "CC-BY-4.0"
        self.spatial_extent = {"bbox": [-180, -90, 180, 90]}
        self.temporal_extent = {"interval": [["2015-01-01", "2100-12-31"]]}
        self.estimated_size_gb = 1250.0
        
        # Mock STAC collections
        self.collections = [
            type('Collection', (), {
                'id': 'temperature',
                'title': 'Surface Temperature',
                'items': list(range(120)),  # 10 years * 12 months
                'summaries': {'variables': ['tas', 'tasmax', 'tasmin', 'pr', 'huss']}
            })(),
            type('Collection', (), {
                'id': 'precipitation', 
                'title': 'Precipitation',
                'items': list(range(120)),
                'summaries': {'variables': ['pr', 'prc', 'prsn', 'prw', 'evspsbl']}
            })(),
            type('Collection', (), {
                'id': 'atmospheric',
                'title': 'Atmospheric Variables', 
                'items': list(range(120)),
                'summaries': {'variables': ['psl', 'ua', 'va', 'zg', 'hus']}
            })()
        ]
    
    def filter(self, **criteria):
        """Filter the dataset based on provided criteria"""
        return MockFilteredDataset()
    
    def get_props(self, prop_type):
        """Generic property getter as requested by user"""
        if prop_type == "models":
            return [
                MockModel("CESM2", "NCAR", "0.9x1.25 deg", ["ssp126", "ssp245", "ssp585"]),
                MockModel("GFDL-ESM4", "NOAA-GFDL", "0.5 deg", ["ssp126", "ssp245", "ssp370", "ssp585"]),
                MockModel("UKESM1-0-LL", "MOHC", "1.25x1.875 deg", ["ssp126", "ssp245", "ssp585"]),
                MockModel("IPSL-CM6A-LR", "IPSL", "1.27x2.5 deg", ["ssp126", "ssp245", "ssp370", "ssp585"]),
                MockModel("MPI-ESM1-2-HR", "MPI-M", "0.94x0.94 deg", ["ssp126", "ssp245", "ssp585"])
            ]
        elif prop_type == "experiments": 
            return [
                MockExperiment("ssp126", "Low emissions scenario", "ScenarioMIP", 12),
                MockExperiment("ssp245", "Medium emissions scenario", "ScenarioMIP", 15),
                MockExperiment("ssp370", "Medium-high emissions scenario", "ScenarioMIP", 8),
                MockExperiment("ssp585", "High emissions scenario", "ScenarioMIP", 18),
                MockExperiment("historical", "Historical simulation", "CMIP", 25)
            ]
        elif prop_type == "variables":
            return [
                MockVariable("tas", "Near-Surface Air Temperature", "K", "mon", ["time", "lat", "lon"]),
                MockVariable("pr", "Precipitation", "kg m-2 s-1", "mon", ["time", "lat", "lon"]),
                MockVariable("psl", "Sea Level Pressure", "Pa", "mon", ["time", "lat", "lon"]),
                MockVariable("ua", "Eastward Near-Surface Wind", "m s-1", "mon", ["time", "lat", "lon"]),
                MockVariable("va", "Northward Near-Surface Wind", "m s-1", "mon", ["time", "lat", "lon"]),
                MockVariable("huss", "Near-Surface Specific Humidity", "1", "mon", ["time", "lat", "lon"]),
                MockVariable("zg", "Geopotential Height", "m", "mon", ["time", "plev", "lat", "lon"]),
                MockVariable("evspsbl", "Evaporation", "kg m-2 s-1", "mon", ["time", "lat", "lon"]),
                MockVariable("tasmax", "Daily Maximum Near-Surface Air Temperature", "K", "day", ["time", "lat", "lon"]),
                MockVariable("tasmin", "Daily Minimum Near-Surface Air Temperature", "K", "day", ["time", "lat", "lon"])
            ]
        elif prop_type == "frequencies":
            return ["mon", "day", "3hr", "6hr", "fx"]
        elif prop_type == "realms":
            return ["atmos", "ocean", "land", "seaIce", "aerosol", "atmosChem"]
        elif prop_type == "__available__":
            return ["models", "experiments", "variables", "frequencies", "realms", "institutions", "grids", "time_ranges"]
        else:
            return []


class GeoCroissant:
    def __init__(self):
        self.version = "1.2.3"
    
    def search(self, keywords, spatial_coverage, temporal_range):
        # Return mock datasets
        datasets = [
            MockDataset(
                "CMIP6_Global_Climate_Projections",
                "Multi-model ensemble of global climate projections from CMIP6",
                "ESGF Data Nodes", 
                ["temperature", "precipitation", "pressure", "humidity"],
                "1.25° x 1.25°",
                "monthly"
            ),
            MockDataset(
                "ERA5_Reanalysis_Global",
                "ECMWF ERA5 atmospheric reanalysis dataset",
                "Copernicus Climate Data Store",
                ["temperature", "wind", "pressure", "humidity"],
                "0.25° x 0.25°", 
                "hourly"
            ),
            MockDataset(
                "MODIS_Land_Surface_Temperature",
                "MODIS satellite-derived land surface temperature",
                "NASA EARTHDATA",
                ["land_surface_temperature", "emissivity"],
                "1km",
                "daily"
            )
        ]
        return datasets
    
    def load_dataset(self, dataset_name):
        if dataset_name == "CMIP6_Global_Climate_Projections":
            return MockCMIP6Dataset()
        else:
            raise ValueError(f"Dataset {dataset_name} not found")


# Mock Matplotlib
class MockPlt:
    @staticmethod
    def figure(figsize=None):
        return type('Figure', (), {})()
    
    @staticmethod
    def axes(projection=None):
        return type('Axes', (), {
            'contourf': lambda *args, **kwargs: type('ContourSet', (), {})(),
            'add_feature': lambda *args, **kwargs: None,
            'gridlines': lambda *args, **kwargs: type('Gridlines', (), {
                'top_labels': False,
                'right_labels': False
            })(),
            'set_extent': lambda *args, **kwargs: None
        })()
    
    @staticmethod
    def colorbar(mappable, ax=None, shrink=None, pad=None):
        return type('Colorbar', (), {
            'set_label': lambda *args, **kwargs: None
        })()
    
    @staticmethod
    def title(title_text, fontsize=None, pad=None):
        pass
    
    @staticmethod
    def tight_layout():
        pass
    
    @staticmethod
    def show():
        print("📊 Mock plot displayed: CMIP6 Surface Temperature visualization")


# Mock Cartopy
class MockCRS:
    class PlateCarree:
        pass


class MockFeature:
    COASTLINE = "coastline"
    BORDERS = "borders"
    OCEAN = "ocean"
    LAND = "land"


# Create module-like objects and aliases
croissant = MockCroissant()

# Create torch with all required attributes
torch = MockTorch()
torch.optim = MockOptim()
torch.device = MockTorch.device
torch.cuda = MockCuda()

xr = MockXArray()

# Create convenient aliases and mock objects
torch_nn = MockNN()
matplotlib_pyplot = MockPlt()
cartopy_crs = MockCRS()
cartopy_feature = MockFeature()
pystac_client = type('Client', (), {})()
STACIntegration = type('STACIntegration', (), {})()
DataLoader = MockDataLoader
Dataset = type('Dataset', (), {})()

# Export all the classes and objects that notebooks might import
__all__ = [
    'croissant', 'torch', 'xr', 'GeoCroissant', 'STACIntegration', 'DataLoader', 'Dataset',
    'torch_nn', 'matplotlib_pyplot', 'cartopy_crs', 'cartopy_feature', 'pystac_client',
    'MockCroissant', 'MockTorch', 'MockXArray', 'MockCMIP6Dataset', 'MockFilteredDataset',
    'MockPyTorchDataset', 'MockDataLoader', 'MockNN', 'MockOptim', 'MockPlt', 'MockCRS', 'MockFeature'
]