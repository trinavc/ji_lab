import os
import numpy as np
import tifffile as tiff
import scipy.io as sio
from tqdm import tqdm
import cv2  # For drawing and rotating lines

class Step2AddBar:
    def __init__(self, file_path, file_name, options):
        self.file_path = file_path
        self.file_name = file_name
        self.options = options
        self.image_stack = None
        self.trial_data = None
        self.sequence_angle = None
        self.grating_size = 30  # Grating size for the top-right square
        self.trialNO = None
        self.angleNO = None
        self.timepoint_blank = 6  # Hardcoded as 6 blank frames
        self.timepoint_stim = 6  # Hardcoded as 6 stimulus frames
        self.order = []  # To store angle, stim/blank, and original/reordered frame number

    def load_data(self):
        # Load the TIFF image stack
        self.image_stack = tiff.imread(os.path.join(self.file_path, self.file_name))

        # Load the experimental trial data from the .mat file
        trial_file = self.options['trial_file']
        if not os.path.isfile(trial_file):
            raise FileNotFoundError(f"MAT file not found: {trial_file}")
        else:
            print(f"MAT file found: {trial_file}")
        
        # Load MAT file
        self.trial_data = sio.loadmat(trial_file)
        self.sequence_angle = self.trial_data['sequenceAngle']  # 12x10 array of angles
        self.trialNO = int(self.trial_data['trialNO'][0][0])  # Extract number of trials (10)
        self.angleNO = int(self.trial_data['angleNO'][0][0])  # Extract number of orientations per trial (12)

        print(f"TrialNO: {self.trialNO}, AngleNO: {self.angleNO}, Blank Frames: {self.timepoint_blank}, Stim Frames: {self.timepoint_stim}, Grating Size: {self.grating_size}")

    def add_orientation_bars(self, data):
        """Add gratings to raw images based on the sequence angles."""
        grating_size = self.grating_size
        total_frames = len(data)
        frames_per_trial = self.timepoint_blank + self.timepoint_stim

        # Loop through each trial
        for trial in range(self.trialNO):
            for orientation in range(self.angleNO):
                angle = self.sequence_angle[orientation, trial]  # Get the angle for this trial and orientation
                start_frame = trial * frames_per_trial * self.angleNO + orientation * frames_per_trial
                
                # Add a grating pattern in the stimulus frames (skip the blank frames)
                for i in range(self.timepoint_blank, frames_per_trial):
                    frame_index = start_frame + i
                    if frame_index < total_frames:
                        data[frame_index] = self.overlay_grating(data[frame_index], angle)
                        self.order.append([angle, 1, frame_index])  # Stimulus frame (1)

                # Add gray square for blank frames
                for i in range(self.timepoint_blank):
                    frame_index = start_frame + i
                    if frame_index < total_frames:
                        data[frame_index] = self.overlay_blank(data[frame_index])
                        self.order.append([angle, 0, frame_index])  # Blank frame (0)

        return data

    def overlay_grating(self, frame, angle):
        """Overlay a grating pattern at the top-right corner based on the angle."""
        grating = self.create_stimulus_pattern(self.grating_size, angle)
        frame[:self.grating_size, -self.grating_size:] = grating
        return frame

    def overlay_blank(self, frame):
        """Overlay a blank gray square at the top-right corner."""
        blank_square = np.ones((self.grating_size, self.grating_size)) * 5000  # Set gray square intensity
        frame[:self.grating_size, -self.grating_size:] = blank_square
        return frame

    def create_stimulus_pattern(self, size, angle):
        """Create a grating pattern based on angle and size."""
        grating_size_padding = 4 * size  # Increased size for rotation
        x = np.linspace(0, grating_size_padding - 1, grating_size_padding)
        X = np.tile(x, (grating_size_padding, 1))
        fshift = 3 / size  # Frequency shift for the grating
        grating = np.cos(2 * np.pi * fshift * X) >= 0
        grating = grating.astype(np.uint16) * 50000

        center = (grating.shape[1] // 2, grating.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_grating = cv2.warpAffine(grating, rot_matrix, (grating.shape[1], grating.shape[0]))

        return rotated_grating[:size, :size]

    def avg_trials_bar(self, data):
        """ Averages and groups the frames by their angles, preserving the 6 blank + 6 stim structure for each angle. """
        angle_frames = {angle: {'blank': [], 'stim': []} for angle in range(0, 360, 30)}  # Angle groupings

        frames_per_trial = self.timepoint_blank + self.timepoint_stim  # 12 frames per trial (6 blank + 6 stim)

        # Group frames by their angle across all trials
        for trial in range(self.trialNO):
            for orientation in range(self.angleNO):
                angle = self.sequence_angle[orientation, trial]
                start_frame = (trial * self.angleNO * frames_per_trial) + (orientation * frames_per_trial)
                
                # Collect blank and stim frames separately for each angle
                blank_frames = data[start_frame:start_frame + self.timepoint_blank]
                stim_frames = data[start_frame + self.timepoint_blank:start_frame + frames_per_trial]
                
                angle_frames[angle]['blank'].extend(blank_frames)
                angle_frames[angle]['stim'].extend(stim_frames)

        # Compute average for each angle's blank and stim frames
        data_avg = []
        for angle, frames in angle_frames.items():
            if len(frames['blank']) > 0:
                avg_blank = np.mean(frames['blank'], axis=0)
                avg_stim = np.mean(frames['stim'], axis=0)
                
                # Append the averaged blank and stim frames (6 + 6)
                data_avg.extend([avg_blank] * self.timepoint_blank)
                data_avg.extend([avg_stim] * self.timepoint_stim)
            else:
                # Handle case where there are no frames (unlikely but safe)
                data_avg.extend([np.zeros_like(data[0])] * frames_per_trial)

        return np.array(data_avg)

    def avg_trials_avg(self, data):
        """Averages and groups the frames by their angles (for Avg_AvgNoBar, 12 frames)."""
        angle_frames = {angle: [] for angle in range(0, 360, 30)}

        frames_per_trial = self.timepoint_blank + self.timepoint_stim  # 12 frames per trial (6 blank + 6 stim)

        # Group frames by their angle across all trials
        for trial in range(self.trialNO):
            for orientation in range(self.angleNO):
                angle = self.sequence_angle[orientation, trial]
                start_frame = (trial * self.angleNO * frames_per_trial) + (orientation * frames_per_trial)
                stim_frames = data[start_frame + self.timepoint_blank:start_frame + frames_per_trial]
                angle_frames[angle].extend(stim_frames)

        # Compute average for each angle (12 frames total)
        data_avg = []
        for angle, frames in angle_frames.items():
            if len(frames) > 0:
                avg_frame = np.mean(frames, axis=0)
                data_avg.append(avg_frame)
            else:
                data_avg.append(np.zeros_like(data[0]))

        return np.array(data_avg)

    def std_trials_bar(self, data):
        """Compute the standard deviation across trials for each angle (144 frames)."""
        angle_frames = {angle: {'blank': [], 'stim': []} for angle in range(0, 360, 30)}

        frames_per_trial = self.timepoint_blank + self.timepoint_stim

        for trial in range(self.trialNO):
            for orientation in range(self.angleNO):
                angle = self.sequence_angle[orientation, trial]
                start_frame = (trial * self.angleNO * frames_per_trial) + (orientation * frames_per_trial)

                blank_frames = data[start_frame:start_frame + self.timepoint_blank]
                stim_frames = data[start_frame + self.timepoint_blank:start_frame + frames_per_trial]

                angle_frames[angle]['blank'].extend(blank_frames)
                angle_frames[angle]['stim'].extend(stim_frames)

        data_std = []
        for angle, frames in angle_frames.items():
            if len(frames['blank']) > 0:
                std_blank = np.std(frames['blank'], axis=0)
                std_stim = np.std(frames['stim'], axis=0)

                data_std.extend([std_blank] * self.timepoint_blank)
                data_std.extend([std_stim] * self.timepoint_stim)
            else:
                data_std.extend([np.zeros_like(data[0])] * frames_per_trial)

        return np.array(data_std)

    def run(self):
        self.load_data()

        # Load the image stack and order details
        data = self.image_stack
        file2 = f'bar_{self.file_name}'

        # Add gratings to raw images
        data_with_bars = self.add_orientation_bars(data)
        self.save_single_tif(file2, data_with_bars)

        # AvgNOBar: 144 frames without gratings
        data_avg_nobar = self.avg_trials_bar(data)
        self.save_single_tif(f'AvgNOBar_{self.file_name}', data_avg_nobar)

        # AvgBar: 144 frames with gratings
        data_avg_bar = self.avg_trials_bar(data_with_bars)
        self.save_single_tif(f'AvgBar_{self.file_name}', data_avg_bar)

        # Avg_AvgNoBar: 12 frames averaged across all trials
        data_avg_avg_nobar = self.avg_trials_avg(data_with_bars)
        self.save_single_tif(f'Avg_AvgNoBar_{self.file_name}', data_avg_avg_nobar)

        # StdBar: 144 frames showing the standard deviation across trials
        data_std_bar = self.std_trials_bar(data_with_bars)
        self.save_single_tif(f'StdBar_{self.file_name}', data_std_bar)

        # Save the stimulus order as a MAT file with columns [angle, stim/blank, original frame number]
        order_array = np.array(self.order)
        sio.savemat(os.path.join(self.file_path, f'Order_{self.file_name}.mat'), {'order': order_array})

    def save_single_tif(self, file_name, data):
        tiff.imwrite(os.path.join(self.file_path, file_name), data.astype(np.float32))

def get_default_options():
    options = {}
    options['trial_file'] = '/Users/trinav/Downloads/NewData/stack/20231017_130307_01_dg_yuhan_step2__angleNo12_trial10.mat'  # Path to the MAT file
    options['preMoving'] = 25  # Default pre-moving frames
    options['postMoving'] = 0  # Default post-moving frames
    options['blank'] = 1  # 0 = static grating, 1 = blank
    options['frameN'] = 1440  # Number of frames in stack (6 blank + 6 stim) * 12 orientations * 10 trials
    options['noBarSaveFlag'] = 1  # Default save flag for no bar
    return options

# File paths and options
file_path = '/Users/trinav/Downloads/NewData/stack'
file_name = '20231017_RM.tif'
options = get_default_options()

# Initialize and run the pipeline
pipeline = Step2AddBar(file_path, file_name, options)
pipeline.run()
