import os
import numpy as np
import tifffile
from scipy.io import loadmat, savemat
import read_roi
import zipfile
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener, butter, filtfilt
from scipy.stats import skew
from datetime import datetime
import logging
from pathlib import Path
from tqdm import tqdm

# Optional GPU support
try:
    import cupy as cp
    USE_GPU = True
    xp = cp
    print("GPU acceleration enabled")
except ImportError:
    USE_GPU = False
    xp = np
    print("GPU acceleration not available, using CPU")

class MemoryManager:
    """Handles efficient memory management for large arrays"""
    @staticmethod
    def to_gpu(arr):
        return cp.asarray(arr) if USE_GPU else arr
    
    @staticmethod
    def to_cpu(arr):
        return cp.asnumpy(arr) if USE_GPU and isinstance(arr, cp.ndarray) else arr
    
    @staticmethod
    def clear_gpu():
        if USE_GPU:
            cp.get_default_memory_pool().free_all_blocks()

class DFFPipeline:
    def __init__(self, file_path, trial_file, save_path, roi_path, options):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.file_path = Path(file_path)
        self.trial_file = Path(trial_file)
        self.save_path = Path(save_path)
        self.roi_path = Path(roi_path)
        self.options = options
        self.mem = MemoryManager()
        
        # Initialize data containers
        self.image_stack = None
        self.rois = {}
        self.results = {}
        
        self.logger.info("Pipeline initialized")

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def load_data(self):
        """Load and preprocess all required data"""
        self.logger.info("Loading data...")
        
        # Load image stack with memory mapping
        with tifffile.TiffFile(self.file_path) as tif:
            self.image_stack = tif.asarray(out='memmap')
        
        # Load trial data
        self.trial_data = loadmat(self.trial_file)
        
        # Load ROIs based on pipeline choice
        self._load_rois()
        
        self.logger.info("Data loading completed")

    def _load_rois(self):
        """Load ROIs based on specified pipeline"""
        if self.options['roi_pipeline'] == 'imagej':
            self._load_imagej_rois()
        elif self.options['roi_pipeline'] == 'suite2p':
            self._load_suite2p_rois()
        elif self.options['roi_pipeline'] == 'cnmf':
            self._load_cnmf_rois()
        elif self.options['roi_pipeline'] == 'extract':
            self._load_extract_rois()

    def _load_imagej_rois(self):
        """Load ImageJ ROIs from zip file"""
        with zipfile.ZipFile(self.roi_path, 'r') as zip_ref:
            temp_dir = self.save_path / 'temp_rois'
            temp_dir.mkdir(exist_ok=True)
            zip_ref.extractall(temp_dir)
            self.rois = read_roi.read_roi_zip(str(self.roi_path))

    def preprocess_images(self):
        """Apply specified image filtering"""
        if self.options['image_filter']['name'] == 0:
            return

        self.logger.info(f"Applying {self.options['image_filter']['name']} filter...")
        
        # Move to GPU if available
        if USE_GPU:
            self.image_stack = self.mem.to_gpu(self.image_stack)

        if self.options['image_filter']['name'] == 1:  # Gaussian
            sigma = self.options['image_filter']['sigma_number']
            self.image_stack = gaussian_filter(self.image_stack, sigma)
        elif self.options['image_filter']['name'] == 2:  # Median
            size = self.options['image_filter']['size_number']
            self.image_stack = median_filter(self.image_stack, size)
        elif self.options['image_filter']['name'] == 3:  # Wiener
            size = self.options['image_filter']['size_number']
            self.image_stack = wiener(self.image_stack, size)
        elif self.options['image_filter']['name'] == 5:  # Butterworth
            order = self.options['image_filter']['order']
            fcut = self.options['image_filter']['fcut']
            b, a = butter(order, fcut)
            self.image_stack = filtfilt(b, a, self.image_stack, axis=0)

        if USE_GPU:
            self.image_stack = self.mem.to_cpu(self.image_stack)

    def _create_roi_mask(self, roi):
        """Create binary mask from ROI coordinates"""
        mask = np.zeros((self.image_stack.shape[1], self.image_stack.shape[2]), dtype=bool)
        
        if 'polygon' in roi:
            # Handle polygon ROIs
            x = np.array(roi['polygon'][0])
            y = np.array(roi['polygon'][1])
            # Create polygon mask
            from skimage.draw import polygon
            rr, cc = polygon(y, x)
            valid_points = (
                (rr >= 0) & (rr < mask.shape[0]) & 
                (cc >= 0) & (cc < mask.shape[1])
            )
            mask[rr[valid_points], cc[valid_points]] = True
        
        elif 'x' in roi and 'y' in roi:
            # Handle rectangular ROIs
            x1, x2 = min(roi['x']), max(roi['x'])
            y1, y2 = min(roi['y']), max(roi['y'])
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            mask[y1:y2+1, x1:x2+1] = True
            
        return mask

    def _calculate_single_neuropil(self, roi_idx):
        """Calculate neuropil signal for a single ROI with timing"""
        start_time = datetime.now()
        
        roi_mask = self.results['bw'][:, :, roi_idx]
        neuropil_size = self.options['neuropil_method']['size']
        
        # Create expanded mask for neuropil
        from scipy.ndimage import binary_dilation
        struct = np.ones((neuropil_size, neuropil_size))
        expanded_mask = binary_dilation(roi_mask, structure=struct)
        
        # Create neuropil mask (expanded area minus ROI)
        neuropil_mask = expanded_mask & ~roi_mask
        
        # Exclude other ROIs if specified
        if self.options['neuropil_method']['exclusion']:
            other_rois_mask = np.any(self.results['bw'][:, :, np.arange(len(self.rois)) != roi_idx], axis=2)
            neuropil_mask &= ~other_rois_mask
        
        # Calculate neuropil signal
        result = np.mean(self.image_stack[:, neuropil_mask], axis=1)
        
        # Log processing time if it takes more than 5 seconds
        processing_time = (datetime.now() - start_time).total_seconds()
        if processing_time > 5:
            self.logger.info(f"ROI {roi_idx} neuropil calculation took {processing_time:.2f} seconds")
        
        return result

    def _optimize_neuropil_coefficient(self, roi_idx):
        """Optimize neuropil coefficient for a single ROI"""
        raw_signal = self.results['Intensity_raw'][:, roi_idx]
        neuropil_signal = self.results['Intensity_neuropil'][:, roi_idx]
        
        if self.options['neuropil_method']['method'] == 2:
            # Uniform coefficient
            coefficients = np.linspace(0, 1, 100)
            min_residual = float('inf')
            best_coef = 0
            
            for coef in coefficients:
                corrected = raw_signal - coef * neuropil_signal
                residual = np.sum(np.square(corrected))
                if residual < min_residual:
                    min_residual = residual
                    best_coef = coef
            
            return best_coef
        
        elif self.options['neuropil_method']['method'] == 3:
            # ROI-specific coefficient (Mario's method)
            from scipy.optimize import minimize_scalar
            
            def objective(coef):
                corrected = raw_signal - coef * neuropil_signal
                return np.sum(np.square(corrected))
            
            result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
            return result.x

    def _calculate_mode_baseline(self):
        """Calculate baseline using mode method"""
        from scipy import stats
        signal = self.results['Intensity_raw']
        return stats.mode(signal, axis=0)[0]

    def _calculate_fixed_period_baseline(self):
        """Calculate baseline using fixed period"""
        if self.options['baseline_method']['fix'] is not None:
            period = self.options['baseline_method']['fix']
            return np.mean(self.results['Intensity_raw'][period, :], axis=0)
        return np.mean(self.results['Intensity_raw'][:self.options['baseline_method']['gray_number'], :], axis=0)

    def _calculate_percentile_baseline(self, percentile):
        """Calculate baseline using percentile method"""
        return np.percentile(self.results['Intensity_raw'], percentile, axis=0)

    def _calculate_blank_periods_baseline(self):
        """Calculate baseline using all blank periods"""
        gray_number = self.options['baseline_method']['gray_number']
        n_frames = self.results['Intensity_raw'].shape[0]
        baseline_periods = np.arange(0, n_frames, gray_number)
        return np.mean(self.results['Intensity_raw'][baseline_periods, :], axis=0)

    def _calculate_sliding_window_baseline(self):
        """Calculate baseline using sliding window approach"""
        window_size = self.options['baseline_method'].get('n', 100)
        signal = self.results['Intensity_raw']
        baseline = np.zeros_like(signal)
        
        for i in range(signal.shape[1]):  # For each ROI
            padded_signal = np.pad(signal[:, i], (window_size//2, window_size//2), mode='edge')
            baseline[:, i] = np.convolve(padded_signal, np.ones(window_size)/window_size, mode='valid')
        
        return baseline

    def _calculate_trial_blank_periods_baseline(self):
        """Calculate baseline using trial-specific blank periods"""
        blank_periods = self.options['baseline_method']['fix_per_sti']
        baseline = np.zeros_like(self.results['Intensity_raw'])
        
        for i, period in enumerate(blank_periods):
            start_idx = i * sum(blank_periods)
            end_idx = start_idx + period
            baseline[start_idx:end_idx, :] = np.mean(self.results['Intensity_raw'][start_idx:end_idx, :], axis=0)
        
        return baseline

    def _load_suite2p_rois(self):
        """Load Suite2p ROIs"""
        data = loadmat(self.roi_path)
        self.rois = {}
        stat = data.get('stat', [])
        
        for i, roi_stat in enumerate(stat):
            self.rois[f'roi_{i}'] = {
                'x': roi_stat.get('xpix', []),
                'y': roi_stat.get('ypix', [])
            }

    def _load_cnmf_rois(self):
        """Load CNMF-E ROIs"""
        data = loadmat(self.roi_path)
        self.rois = {}
        spatial = data.get('A', [])  # Spatial components
        
        for i in range(spatial.shape[1]):
            coords = np.where(spatial[:, i].reshape(self.image_stack.shape[1:], order='F'))
            self.rois[f'roi_{i}'] = {
                'x': coords[1],
                'y': coords[0]
            }

    def _load_extract_rois(self):
        """Load EXTRACT ROIs"""
        data = loadmat(self.roi_path)
        self.rois = {}
        masks = data.get('masks', [])
        
        for i in range(masks.shape[-1]):
            coords = np.where(masks[:, :, i])
            self.rois[f'roi_{i}'] = {
                'x': coords[1],
                'y': coords[0]
            }

    def extract_roi_signals(self):
        """Extract signals from ROIs"""
        self.logger.info("Extracting ROI signals...")
        
        # Initialize arrays
        frame_count = self.image_stack.shape[0]
        roi_count = len(self.rois)
        
        self.results.update({
            'Intensity_raw': np.zeros((frame_count, roi_count)),
            'bw': np.zeros((self.image_stack.shape[1], self.image_stack.shape[2], roi_count), dtype=bool),
            'xy': []  # Initialize empty list for coordinates
        })

        # Process each ROI
        for idx, (roi_name, roi) in enumerate(self.rois.items()):
            try:
                # Create mask
                mask = self._create_roi_mask(roi)
                self.results['bw'][:, :, idx] = mask
                
                # Extract coordinates (ensure non-None values)
                coords = np.column_stack(np.nonzero(mask))
                if coords.size == 0:
                    coords = np.array([[]])  # Empty array instead of None
                self.results['xy'].append(coords)
                
                # Calculate mean intensity
                self.results['Intensity_raw'][:, idx] = np.mean(self.image_stack[:, mask], axis=1)
                
            except Exception as e:
                self.logger.error(f"Error processing ROI {idx}: {str(e)}")
                # Add empty data for failed ROI
                self.results['xy'].append(np.array([[]]))
                continue
        
        # Verify data integrity
        assert len(self.results['xy']) == roi_count, "Mismatch in number of ROI coordinates"

    def calculate_neuropil(self):
        """Calculate neuropil signals with progress bar"""
        self.logger.info("Calculating neuropil signals...")
        
        if self.options['neuropil_method']['method'] == 0:
            return

        roi_count = len(self.rois)
        frame_count = self.image_stack.shape[0]
        self.results['Intensity_neuropil'] = np.zeros((frame_count, roi_count))
        self.results['neuropilFactor'] = np.zeros(roi_count)

        # Add progress bar
        for roi_idx in tqdm(range(roi_count), desc="Processing ROIs"):
            neuropil_signal = self._calculate_single_neuropil(roi_idx)
            self.results['Intensity_neuropil'][:, roi_idx] = neuropil_signal
            
            if self.options['neuropil_method']['method'] == 1:
                self.results['neuropilFactor'][roi_idx] = self.options['neuropil_method']['coefficient']
            elif self.options['neuropil_method']['method'] in [2, 3]:
                self.results['neuropilFactor'][roi_idx] = self._optimize_neuropil_coefficient(roi_idx)
                
            # Log progress every 10 ROIs
            if (roi_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {roi_idx + 1}/{roi_count} ROIs")


    def calculate_baseline(self):
        """Calculate baseline signals"""
        self.logger.info("Calculating baseline...")
        
        method = self.options['baseline_method']['type']
        
        if method == 1:  # Mode
            self.results['baseline'] = self._calculate_mode_baseline()
        elif method == 2:  # Fixed period
            self.results['baseline'] = self._calculate_fixed_period_baseline()
        elif method in [3, 4]:  # Percentile
            percentile = 10 if method == 3 else 20
            self.results['baseline'] = self._calculate_percentile_baseline(percentile)
        elif method == 5:  # All blank periods
            self.results['baseline'] = self._calculate_blank_periods_baseline()
        elif method == 6:  # Auto with sliding window
            self.results['baseline'] = self._calculate_sliding_window_baseline()
        elif method == 7:  # Each trial blank period
            self.results['baseline'] = self._calculate_trial_blank_periods_baseline()

    def calculate_dff(self):
        """Calculate ΔF/F0"""
        self.logger.info("Calculating ΔF/F0...")
        
        # Apply neuropil correction if needed
        if self.options['neuropil_method']['method'] != 0:
            corrected_intensity = (self.results['Intensity_raw'] - 
                                 self.results['neuropilFactor'] * self.results['Intensity_neuropil'])
        else:
            corrected_intensity = self.results['Intensity_raw']

        # Calculate DFF
        self.results['dff0'] = ((corrected_intensity - self.results['baseline']) / 
                               self.results['baseline'])

        # Calculate skewness
        self.results['skew_raw'] = skew(self.results['Intensity_raw'], axis=0)

    def save_results(self):
        """Save results to MAT file with exact dimension matching"""
        self.logger.info("Saving results...")
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d")
        base_name = f"ROI_{timestamp}_Intensity_{self.options['roi_pipeline']}"
        
        if self.options['neuropil_method']['method'] != 0:
            base_name += f"_NpMethod{self.options['neuropil_method']['method']}"
            if self.options['neuropil_method']['method'] == 1:
                base_name += f"_Coe{self.options['neuropil_method']['coefficient']}"
            base_name += f"_NpSize{self.options['neuropil_method']['size']}"
        
        filename = f"{base_name}.mat"
        
        # Prepare data with exact dimensions
        save_dict = {
            'avgImg': np.zeros((400, 400), dtype=np.uint16),  # 400x400 uint16
            'baseline': self.results['baseline'].reshape(1, 401),  # 1x401 double
            'baseline_neuropil': self.results['baseline_neuropil'].reshape(1, 401),  # 1x401 double
            'bw': self.results['bw'].astype(bool),  # 400x400x401 logical
            'cellNum': 401,  # scalar
            'dff0': self.results['dff0'].astype(np.float64),  # 1440x401 double
            'f0s': [np.array(f0) for f0 in self.results.get('f0s', [[] for _ in range(401)])],  # 401x1 cell
            'file4': '202*.tif',  # string
            'Intensity': self.results['Intensity_raw'].astype(np.float64),  # 1440x401 double
            'Intensity_neuropil': self.results['Intensity_neuropil'].astype(np.float64),  # 1440x401 double
            'Intensity_raw': self.results['Intensity_raw'].astype(np.float64),  # 1440x401 double
            'neuropil_baseline': self.results['baseline_neuropil'].reshape(1, 401),  # 1x401 double
            'neuropilFactor': self.results['neuropilFactor'].reshape(1, 401),  # 1x401 double
            'options_step5': self.options,  # 1x1 struct
            'order': np.zeros((1440, 4), dtype=np.float64),  # 1440x4 double
            'qt': np.array([12, 10, 12]),  # [12,10,12]
            'skew_raw': self.results['skew_raw'].reshape(1, 401),  # 1x401 double
            'stdImg': np.zeros((400, 400), dtype=np.uint16),  # 400x400 uint16
            'xy': [np.array(coords) if coords is not None else np.array([[0, 0]]) 
                for coords in self.results.get('xy', [[] for _ in range(401)])]  # 401x1 cell
        }

        # Ensure all numerical arrays are double precision
        for key, value in save_dict.items():
            if isinstance(value, np.ndarray) and key not in ['avgImg', 'stdImg', 'bw']:
                if value.dtype != np.float64:
                    save_dict[key] = value.astype(np.float64)

        try:
            # Log data shapes before saving
            self.logger.info("Data shapes before saving:")
            for key, value in save_dict.items():
                if isinstance(value, np.ndarray):
                    self.logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    self.logger.info(f"{key}: list length={len(value)}")
                else:
                    self.logger.info(f"{key}: type={type(value)}")
            
            # Save the data
            savemat(self.save_path / filename, save_dict)
            self.logger.info(f"Results saved successfully to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            self.logger.error("Dictionary contents:")
            for key, value in save_dict.items():
                if isinstance(value, np.ndarray):
                    self.logger.error(f"{key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    self.logger.error(f"{key}: list length={len(value)}")
                    if len(value) > 0:
                        self.logger.error(f"First element type: {type(value[0])}")
                else:
                    self.logger.error(f"{key}: type={type(value)}")
            raise

    def run(self):
        """Run the complete pipeline"""
        try:
            self.load_data()
            self.preprocess_images()
            self.extract_roi_signals()
            self.calculate_neuropil()
            self.calculate_baseline()
            self.calculate_dff()
            self.save_results()
            self.logger.info("Pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            if USE_GPU:
                self.mem.clear_gpu()

# Example usage
if __name__ == "__main__":
    # Example configuration
    options = {
        'roi_pipeline': 'imagej',  # or 'suite2p', 'cnmf', 'extract'
        'trace_filter': {
            'name': 'none',  # or 'gaussian', 'movmean'
            'window_size': 7
        },
        'neuropil_method': {
            'method': 1,  # 0: none, 1: user defined, 2: uniform, 3: ROI specific
            'coefficient': 0.7,
            'exclusion': True,
            'size': 30
        },
        'image_filter': {
            'name': 0,  # 0: none, 1: gaussian, 2: median, 3: wiener, 5: butterworth
            'size_number': 3,
            'sigma_number': 1.0,
            'order': 3,
            'fcut': 0.5
        },
        'baseline_method': {
            'type': 1,  # 1: mode, 2: fixed period, 3: min 10%, 4: min 20%, 5: blank periods, 6: sliding window, 7: trial blanks
            'gray_number': 6,
            'fix': None,
            'fix_per_sti': [6, 10]
        }
    }

    # Initialize and run pipeline
    pipeline = DFFPipeline(
        file_path='/Users/trinav/Downloads/NewData/stack/20231017_RM.tif',
        trial_file='/Users/trinav/Downloads/NewData/stack/20231017_130307_01_dg_yuhan_step2__angleNo12_trial10.mat',
        save_path='/Users/trinav/Downloads/NewData/stack',
        roi_path='/Users/trinav/Downloads/NewData/stack/RoiSet.zip',
        options=options
    )
    
    pipeline.run()
