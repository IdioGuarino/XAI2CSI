from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from . import data_formatter as dfmt

import os
import scipy.io as sio
from os import listdir, path
from pathlib import Path
import json
from . import utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm
import h5py, joblib

def split_by_scenario(df, target_scenario=None, random_state=None, shuffle=True, val_size=0.2):
    valid_idx = None
    if target_scenario in df['scenario'].values:
        train_idx = df[df['scenario'] != target_scenario].index.to_numpy()
        test_idx = df[df['scenario'] == target_scenario].index.to_numpy()
        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)
        if 0.0 < val_size < 1.0:
            train_idx, valid_idx = train_test_split(train_idx, test_size=val_size, random_state=random_state, stratify=df.loc[train_idx, 'ENC_LABEL'])
        return train_idx, valid_idx, test_idx
    else:
        raise ValueError(f"Target scenario '{target_scenario}' not found in dataset.")
    

class DataProcessor():
    def __init__(self, dataset_name, **kwargs):

        self.dataset_name = dataset_name
        self.config = dfmt.dataset_config.get(dataset_name, {})
        if not self.config:
            raise ValueError(f"Configuration for dataset '{dataset_name}' not found.")
        
    @abstractmethod
    def _init_hook(self, **kwargs):
            """Metodo chiamato all'interno del costruttore, da implementare nella sottoclasse"""
            pass

    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def load_data(self):
        pass


# --- DATASET CLASS FOR MODEL TRAINING ---
class TrainingCSIDataset:
    """
    A memory-efficient dataset for PyTorch/TensorFlow that loads data from HDF5
    and applies pre-fitted scalers on-the-fly.
    """
    def __init__(self, metadata, h5_path, amp_scaler, ph_scaler, class_order, split='train', smode="SxF", restore_shape=True, early_fusion=True):
        self.h5_path = h5_path
        self.amp_scaler = amp_scaler
        self.ph_scaler = ph_scaler
        self.class_order = class_order
        self.restore_shape = restore_shape
        self.early_fusion = early_fusion
        # Filter metadata for the desired split (train, valid, or test)
        self.metadata = metadata[metadata['split'] == split].reset_index(drop=True)
        self.smode = smode
        # Keep HDF5 file open for performance
        self._h5_file = None

    def __len__(self):
        return len(self.metadata)

    def _open_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        self._open_h5_file()
        
        meta_row = self.metadata.iloc[idx]
        window_idx = meta_row['window_idx']
        
        # Load raw data from HDF5
        data_group = self._h5_file[f'window_{window_idx}']
        amp_raw = data_group['amp'][:] if self.amp_scaler else None
        ph_raw = data_group['ph'][:] if self.ph_scaler else None
        
        # Scale data on-the-fly
        scaled_amp, scaled_ph = None, None
        S, F, T = amp_raw.shape if amp_raw is not None else ph_raw.shape
        if amp_raw is not None:
            S, F, T = amp_raw.shape
            # Reshape for scaler: (S, F, T) -> (T, S*F)
            reshaped_amp = utils.reshape_csi_for_scaling_sample(amp_raw, S, F, T, restore_shape=False, smode=self.smode)
            scaled_reshaped = self.amp_scaler.transform(reshaped_amp)
            # Reshape back to original: (T, S*F) -> (S, F, T)
            if self.restore_shape:
                # scaled_amp = scaled_reshaped.reshape(T, S, F).transpose(1, 2, 0)
                scaled_amp = utils.reshape_csi_for_scaling_sample(scaled_reshaped, S, F, T, restore_shape=True, smode=self.smode)

        if ph_raw is not None:
            S, F, T = ph_raw.shape
            # print(f"Original phase shape: {ph_raw.shape}")
            reshaped_ph = utils.reshape_csi_for_scaling_sample(ph_raw, S, F, T, restore_shape=False, smode=self.smode)
            scaled_reshaped = self.ph_scaler.transform(reshaped_ph)
            if self.restore_shape:
                scaled_ph = utils.reshape_csi_for_scaling_sample(scaled_reshaped, S, F, T, restore_shape=True, smode=self.smode)

        # Format data and label
        if self.early_fusion and scaled_amp is not None and scaled_ph is not None:
            # Early fusion: concatenate along the channel dimension (S, F, T) -> (2*S, F, T)
            final_sample = np.concatenate([scaled_amp, scaled_ph], axis=0)
            final_sample = self._format_data(final_sample, None)  # Pass None for phase since it's already concatenated
        else:
            final_sample = self._format_data(scaled_amp, scaled_ph)
        label = self.class_order.index(meta_row['ENC_LABEL'])
        assert type(label) is int, "Label must be an integer."

        return final_sample, label

    def _format_data(self, amp, ph):
        # Combines amplitude and phase into a single array, similar to your original `format_data`
        if amp is not None and ph is not None:
            return (amp.astype('float32'), ph.astype('float32')) # Shape: (2, S, F, T)
        elif amp is not None:
            return amp.astype('float32') # Shape: (S, F, T)
        elif ph is not None:
            return ph.astype('float32') # Shape: (S, F, T)
        return None

    def close(self):
        if self._h5_file:
            self._h5_file.close()
            self._h5_file = None
    
    def set_modality(self, modality='all'):
        self.modality = modality

class AXCSILoader(DataProcessor):
    def __init__(self, dataset_name, amplitude, phase, **kwargs):
        super().__init__(dataset_name, **kwargs)
        self.dataset_name = dataset_name
        self.config = dfmt.dataset_config.get(dataset_name, {})
        self.amplitude = amplitude
        self.phase = phase
        self._init_hook(**kwargs)

    def _init_hook(self, **kwargs):

        self.wifi_type = kwargs.get('wifi', 'ac')
        assert self.wifi_type in self.config.get('wifi', ['ac']), f"Invalid wifi type '{self.wifi_type}' for dataset '{self.dataset_name}'. Available options: {self.config.get('wifi', ['ac'])}"
        print(f"Using wifi type: {self.wifi_type}")
        self.channel_bandwidth = kwargs.get('channel_bandwidth', 80)
        assert self.channel_bandwidth in self.config.get('channel_bandwidth', [20, 40, 80]), f"Invalid channel bandwidth '{self.channel_bandwidth}' for dataset '{self.dataset_name}'. Available options: {self.config.get('channel_bandwidth', [20, 40, 80])}"
        self.sampling_rate = kwargs.get('sampling_rate', 150)

        self.window_size = kwargs.get('window_size', self.config.get('window_size', 51))
        self.stride = kwargs.get('stride', 1)
        self.n_envs = kwargs.get('n_envs', 3)
        assert self.n_envs <= self.config.get('n_envs', 3), f"Number of environments '{self.n_envs}' exceeds the maximum allowed '{self.config.get('n_envs', 3)}' for dataset '{self.dataset_name}'."
        self.n_users = kwargs.get('n_users', 3)
        assert self.n_users <= self.config.get('n_users', 3), f"Number of users '{self.n_users}' exceeds the maximum allowed '{self.config.get('n_users', 3)}' for dataset '{self.dataset_name}'."


        self.path = self.config.get('path', None)
        if not self.path:
            raise ValueError(f"Path must be provided for dataset '{self.dataset_name}'.")
        
        self.tool_config = dfmt.AXCSI_config.get(self.wifi_type, None)
        if not self.tool_config:
            raise ValueError(f"Configuration for wifi type '{self.wifi_type}' not found in dataset '{self.dataset_name}'.")
        
        self.antennas = self.tool_config.get('antennas', 4)
        self.start= kwargs.get('start_idx', 0)
        print(self.path)
        with open(os.path.join(self.path, 'ground_truth.json'), 'r') as f:
            self.ground_truth = json.load(f)


    def __get_amp(self, csi_buff):    
        # Extract only amplitude and reorder the shape for the dataloader: (time, subcarriers, antennas)
        amp = np.abs(csi_buff).transpose(1, 0, 2)
        return amp
    
    def __get_ph(self, csi_full):
        """
        Estrazione della fase differenziale (Soluzione 2b: Antenna di Riferimento).
        Calcola la variazione cinematica (Doppler) delle antenne [1, 2, ..., S-1] 
        rispetto all'antenna 0.
        
        Input: 
            csi_full: array complesso con shape (Subcarriers, Time, Antennas)
        Output: 
            doppler_phase: array float32 con shape (Time-1, Subcarriers, Antennas-1)
        """
        F, T, S = csi_full.shape
        
        # Managing the SISO case separately, as it cannot remove CFO/SFO and only computes the temporal derivative on the single antenna.
        if S < 2:
            # Fallback: Compute the temporal derivative of the single antenna's phase
            D_siso = csi_full[:, 1:, :] * np.conj(csi_full[:, :-1, :])
            return np.angle(D_siso).astype(np.float32).transpose(1, 0, 2)
        
        # Spatial Filtering of Reference Antenna --> (F, T, 1)
        ref_ant = csi_full[:, :, 0:1]
        
        # Extracting the remaining antennas(from 1 to S-1)--> (F, T, S-1)
        other_ants = csi_full[:, :, 1:]
        
        # Complex conjugate multiplication: Z_i(t) = H_i(t) * conj(H_0(t))--->(F, T, S-1)
        Z = other_ants * np.conj(ref_ant)
        
        # --- STEP 2: Filtro Cinematico (Derivata Temporale) ---
        # Complex multiplication in time: D_i(t) = Z_i(t) * conj(Z_i(t-1))---> (F, T-1, S-1)
        D = Z[:, 1:, :] * np.conj(Z[:, :-1, :])
        
        # Phase extraction: doppler_phase = angle(D) ---> (F, T-1, S-1)
        doppler_phase = np.angle(D).astype(np.float32)
        
        # Transposition for the dataloader: (F, T-1, S-1) --> (Time-1, Subcarriers, Channels)
        return doppler_phase.transpose(1, 0, 2)

        
    def __segment_data_and_reshape(self, csi_data, ignore_last=True):
        # Ignoring the last segment if it doesn't fit the window size

        if ignore_last:
            # csi_data = csi_data[:-1,:,:]
            valid_len = (csi_data.shape[0] - self.window_size) // self.stride * self.stride + self.window_size
            csi_data = csi_data[:valid_len, :, :]

        assert csi_data.shape[0] >= self.window_size, f"CSI data length {csi_data.shape[0]} is less than window size {self.window_size}."
        T, subcarriers, antennas = csi_data.shape
        segments = []
        for i in range(0, T - self.window_size + 1, self.stride):
            window = csi_data[i:i+self.window_size,:, : ]
            window=window.transpose(2, 1, 0) #Riordino (antennas, subcarriers, time)
            segments.append(window)
        return np.stack(segments) 


    
    def load_data(self, apply_filter=False):
        """
        Loads and processes CSI data, saving metadata to Parquet and numerical
        data to an HDF5 file. This is memory-efficient for large datasets.
        """
        if self.amplitude and self.phase:
            pname_base = f"{self.dataset_name}_{self.wifi_type}_{self.channel_bandwidth}_{self.window_size}_{self.stride}_AMP_PH{'_FILTERED' if apply_filter else ''}"
        else:
            both_pname_base=f"{self.dataset_name}_{self.wifi_type}_{self.channel_bandwidth}_{self.window_size}_{self.stride}_AMP_PH{'_FILTERED' if apply_filter else ''}"
            if os.path.exists(os.path.join(self.path, f"{both_pname_base}_metadata.parquet")) and os.path.exists(os.path.join(self.path, f"{both_pname_base}_csi_data.hdf5")):
                pname_base=both_pname_base
            else:
                pname_base = f"{self.dataset_name}_{self.wifi_type}_{self.channel_bandwidth}_{self.window_size}_{self.stride}{'_AMP' if self.amplitude else ''}{'_PH' if self.phase else ''}{'_FILTERED' if apply_filter else ''}"

        # Define paths for the two output files
        # pname_base = f"{self.dataset_name}_{self.wifi_type}_{self.channel_bandwidth}_{self.window_size}_{self.stride}{'_AMP' if self.amplitude else ''}{'_PH' if self.phase else ''}{'_FILTERED' if apply_filter else ''}"
        
        metadata_path = os.path.join(self.path, f"{pname_base}_metadata.parquet")
        data_path = os.path.join(self.path, f"{pname_base}_csi_data.hdf5")

    
        # If both files already exist, skip processing and return them.
        if os.path.exists(metadata_path) and os.path.exists(data_path):
            print(f"Found existing processed files. Loading from:\n- {metadata_path}\n- {data_path}")
            # We return the paths so they can be used to initialize the CSIDataset
            return metadata_path, data_path

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Data path '{self.path}' does not exist for dataset '{self.dataset_name}'.")

        dirs = listdir(self.path)
        if not dirs:
            raise ValueError(f"No data directories found in '{self.path}' for dataset '{self.dataset_name}'.")

        all_metadata = []
        window_idx_counter = 0
        both = self.amplitude and self.phase
        
        with h5py.File(data_path, 'w') as hf:
            for dir_name in dirs:
                dir_path = os.path.join(self.path, dir_name)
                if not os.path.isdir(dir_path):
                    continue
                scenario = Path(dir_name).name
                print(f"\nProcessing scenario <{scenario}> | WiFi <{self.wifi_type}> | BW <{self.channel_bandwidth}>...")
                
                files = sorted([f for f in listdir(dir_path) if f.endswith('.mat')])
                for file_name in tqdm(files, desc=f"Files in {scenario}"):
                    file_path = os.path.join(dir_path, file_name)
                    
                    # CSI Processing
                    csi_buff = sio.loadmat(file_path)
                    csi_buff = csi_buff.get('csi_buff', csi_buff.get('csi'))
                    csi_buff = np.fft.fftshift(csi_buff, axes=1)

                    delete_rows = np.argwhere(np.sum(csi_buff, axis=1) == 0)[:, 0]
                    csi_buff = np.delete(csi_buff, delete_rows, axis=0)
                    
                    #Subcarrier Selection
                    offset = self.tool_config.get('offset', 0)
                    n_subcarriers = self.tool_config.get('n_subcarriers', {}).get(self.channel_bandwidth)
                    
                    if offset > 0:
                        csi_buff = csi_buff[:, ::offset,:]
                        if n_subcarriers is not None:
                            center, half = csi_buff.shape[1] // 2, n_subcarriers // 2
                            csi_buff = csi_buff[:, center-half:center+half, :]
                    elif n_subcarriers is not None:
                        center, half = csi_buff.shape[1] // 2, n_subcarriers // 2
                        csi_buff = csi_buff[:, center-half:center+half, :]
                    
                    delete_idxs = self.tool_config.get('subcarriers_filter', {}).get(self.channel_bandwidth)
                    if delete_idxs is not None:
                        csi_buff = np.delete(csi_buff, delete_idxs, axis=1)
                    
                    end = csi_buff.shape[0]
                    csi_full = np.zeros((csi_buff.shape[1], end - self.start, self.antennas), dtype=complex)
                    for s in range(self.antennas):
                        antenna = csi_buff[self.start:end, :, s]
                        antenna = antenna / np.mean(np.abs(antenna), axis=1, keepdims=True)
                        csi_full[:, :, s] = antenna.T


                    amp, ph = None, None
                    if self.amplitude:
                        print(f"Extracting AMPLITUDE...")
                        amp = self.__get_amp(csi_full)
                        if apply_filter:
                            # (time, subcarriers, antennas)
                            amp = utils.hampel_filter_optimized(amp, window_size=5, n_sigmas=3.0)
                            # amp = butterworth_bandpass_filter(amp, lowcut=1, highcut=90, fs=self.sampling_rate, order=5)

                        if both:
                            amp=amp[1:,:,:] # Temporal alignment with phase (which is T-1 due to differencing)
                            assert amp.shape[0] == csi_full.shape[1] - 1, f"Time dimension mismatch after alignment: amp {amp.shape[0]} vs expected {csi_full.shape[1] - 1}"
                        amp = self.__segment_data_and_reshape(amp, ignore_last=False)

                    if self.phase:
                        print(f"Extracting PHASE DELTA...")
                        ph = self.__get_ph(csi_full)
                        assert ph.shape[0] == csi_full.shape[1] - 1, f"Time dimension mismatch for phase: {ph.shape[0]} vs expected {csi_full.shape[1] - 1}"
                        ph = self.__segment_data_and_reshape(ph, ignore_last=False)

                    # --- NEW: Save data to HDF5 and metadata to a list ---
                    num_windows_in_file = amp.shape[0] if self.amplitude else ph.shape[0]
                    for i in range(num_windows_in_file):
                        # 1. Create metadata row (without heavy data)
                        metadata_row = {
                            'window_idx': window_idx_counter, # The crucial link!
                            'source_filename': file_name,
                            'scenario': scenario,
                            'user': self.ground_truth.get('scenarios', {}).get(scenario, {}).get('user', None),
                            'environment': self.ground_truth.get('scenarios', {}).get(scenario, {}).get('environment', None),
                            'day': self.ground_truth.get('scenarios', {}).get(scenario, {}).get('day', None),
                            'rx': self.ground_truth.get('Rx', {}).get(os.path.basename(file_name).split('.')[0].split('_')[0].replace(scenario, ''), None),
                            'activity': self.ground_truth.get('activities', {}).get(os.path.basename(file_name).split('.')[0].split('_')[1], None),
                        }
                        assert metadata_row['activity'] is not None, f"Missing activity for window {window_idx_counter}"
                        assert metadata_row['rx'] is not None, f"Missing RX for window {window_idx_counter}"
                        all_metadata.append(metadata_row)

                        # 2. Save the large numpy arrays to HDF5 file
                        group = hf.create_group(f'window_{window_idx_counter}')
                        if self.amplitude:
                            group.create_dataset('amp', data=amp[i], compression="gzip")
                        if self.phase:
                            group.create_dataset('ph', data=ph[i], compression="gzip")

                        window_idx_counter += 1
                        
        print("\nProcessing complete. Finalizing files...")
        # After all files are processed, save the collected metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_parquet(metadata_path, index=False)
        print(f"Successfully saved metadata to: {metadata_path}")
        print(f"Successfully saved numerical data to: {data_path}")

        return metadata_path, data_path

def get_data_processor(dataset_name, amplitude=False, phase=False, seed=46, 
                       split_strategy='temporal' , num_folds=10, target_fold=0, 
                       cross=None, target_scenario=None, post_scaling=False,
                       test_size=0.2, val_size=0.2, only_test=False, class_order=None, scaler="MMS",
                    #    smode="SxF",
                       smode='S', target_classes=None,
                       **kwargs ):
    shuffle_classes = False if class_order is not None else True
    assert amplitude or phase, "At least one of amplitude or phase must be True."
    both = amplitude and phase

    if dataset_name.lower() == 'cominelli23':
        loader=AXCSILoader(dataset_name, amplitude, phase,**kwargs)
        # df= loader.load_data()
        metadata_path, data_path = loader.load_data(apply_filter=True)

        wifi_type = kwargs.get('wifi', 'ac')
        channel_bandwidth = kwargs.get('channel_bandwidth', 80)
        window_size = kwargs.get('window_size', 51)
        stride = kwargs.get('stride', 1)
        string_classes = ''
        if target_classes is not None and isinstance(target_classes, list) and len(target_classes) > 0:
            print(f"Filtering dataset to include only target classes: {target_classes}")
            target_classes = np.unique(target_classes).tolist()
            target_classes = [cl.title() if isinstance(cl, str) else cl for cl in target_classes]

            string_classes = [k for k,v in loader.ground_truth.get('activities', {}).items() if v in target_classes or k in target_classes]
            string_classes = '_'+''.join(string_classes)
            if not string_classes:
                raise ValueError(f"None of the specified target classes {target_classes} were found in the dataset activities.")

        if split_strategy == 'stratified':
            assert num_folds > 1 and num_folds <= 10, f"Number of folds must be between 2 and 10, got {num_folds}."
            assert target_fold < num_folds and target_fold >= 0, f"Target fold {target_fold} must be less than number of folds {num_folds}."
            pname=f"{dataset_name}_{seed}_{target_fold}_{num_folds}F_{int(val_size*100)}_{wifi_type}_{channel_bandwidth}_{window_size}_{stride}{string_classes}"

        elif split_strategy == 'temporal':

            if cross is not None:
                assert cross in ['users', 'environments', 'rx', 'day'], f"Invalid cross option '{cross}'. Available options: None, 'users', 'environments', 'day'."
                assert target_scenario is not None, "Target scenario must be specified when cross-validation is used."

                if cross == 'users':
                    assert target_scenario in ['S1', 'S2', 'S3'], f"Invalid target scenario '{target_scenario}' for cross-validation by users. Available options: 'S1', 'S2', 'S3'."
                elif cross == 'environments':
                    assert target_scenario in ['S5', 'S6', 'S7'], f"Invalid target scenario '{target_scenario}' for cross-validation by environments. Available options: 'S5', 'S6', 'S7'."
                elif cross == 'rx':
                    assert target_scenario in ['Rx1', 'Rx2', 'Rx3'], f"Invalid target scenario '{target_scenario}' for cross-validation by Rx. Available options: 'Rx1', 'Rx2', 'Rx3'."
                elif cross == 'day':
                    assert target_scenario in ['S1', 'S5'], f"Invalid target scenario '{target_scenario}' for cross-validation by day. Available options: 'S1', 'S5'."

                pname=f"{dataset_name}_{seed}_Temporal{'_PostScaled' if post_scaling else ''}_{int(test_size*100)}_{int(val_size*100)}_{target_scenario}_{cross.capitalize()}S_{wifi_type}_{channel_bandwidth}_{window_size}_{stride}{string_classes}"
            else:
                pname=f"{dataset_name}_{seed}_Temporal_{int(test_size*100)}_{int(val_size*100)}_{wifi_type}_{channel_bandwidth}_{window_size}_{stride}{string_classes}"


        if both:
            final_meta_path = os.path.join(loader.path, f"{pname}_{scaler}_{smode}_AMP_PH_final_metadata_with_splits.parquet")
        else:
            both_meta_path=os.path.join(loader.path, f"{pname}_{scaler}_{smode}_AMP_PH_final_metadata_with_splits.parquet")
            if os.path.exists(both_meta_path):
                final_meta_path=final_meta_path = both_meta_path
            else:
                final_meta_path = os.path.join(loader.path, f"{pname}_{scaler}_{smode}_{'AMP_' if amplitude else ''}{'PH_' if phase else ''}final_metadata_with_splits.parquet")


        
        amp_scaler_path = os.path.join(loader.path, f"{pname}_{scaler}_{smode}{'SKPFIRST' if both else ''}_amp_scaler.joblib")
        ph_scaler_path = os.path.join(loader.path, f"{pname}_{scaler}_{smode}_ph_scaler.joblib")
        le_path = os.path.join(loader.path, f"{pname}_label_encoder.joblib")

        print(f"Looking for preprocessed files....")
        if os.path.exists(final_meta_path):
            print(f"Final metadata path with split: {utils.CHECKMARK}.")
        else:
            print(f"Final metadata path with split: {utils.CROSSMARK}. Will be created after processing.")
        
        if os.path.exists(le_path):
            print(f"Label encoder path: {utils.CHECKMARK}.")
        else:
            print(f"Label encoder path: {utils.CROSSMARK}. Will be created after processing.")

        if amplitude:
            if os.path.exists(amp_scaler_path):
                print(f"Amplitude scaler path: {utils.CHECKMARK}.")
            else:
                print(f"Amplitude scaler path: {utils.CROSSMARK}. Will be created after processing.")
        if phase:
            if os.path.exists(ph_scaler_path):
                print(f"Phase scaler path: {utils.CHECKMARK}.")
            else:
                print(f"Phase scaler path: {utils.CROSSMARK}. Will be created after processing.")
        
        if os.path.exists(final_meta_path) and os.path.exists(le_path) and ((amplitude and os.path.exists(amp_scaler_path)) or (phase and os.path.exists(ph_scaler_path))):
            print(f"Found existing preprocessed artifacts in '{loader.path}'. Loading them. '{final_meta_path}'")
            df_meta = pd.read_parquet(final_meta_path)
            amp_scaler = joblib.load(amp_scaler_path) if amplitude else None
            ph_scaler = joblib.load(ph_scaler_path) if phase else None
            le = joblib.load(le_path)

        else:
            # --- Preprocessing ---
            df_meta = pd.read_parquet(metadata_path)

            # --- Filter classes if requested ---
            if target_classes:
                target_classes = [cl.title() if isinstance(cl, str) else cl for cl in target_classes]
                class_mask = df_meta['activity'].isin(target_classes)
                df_meta = df_meta[class_mask]
                if df_meta.empty:
                    raise ValueError(f"No samples left after filtering by classes: {target_classes}")

            # --- Encode labels ---
            le = LabelEncoder()
            df_meta['ENC_LABEL'] = le.fit_transform(df_meta['activity'].values)
            joblib.dump(le, le_path)
            np.savetxt(os.path.join(loader.path, 'classes.txt'), le.classes_, fmt='%s')
            print(f"LabelEncoder fitted and saved. Classes: {le.classes_}")


            # per-file deterministic split (train/val/test) using original df_meta indices ---
            split_map = {}  # filename -> dict with scenario, rx, train/valid/test (lists of original indices)
            for filename, group in df_meta.groupby('source_filename'):
                n_samples = len(group)
                if n_samples == 0:
                    split_map[filename] = {'scenario': None, 'rx': None, 'train': [], 'valid': [], 'test': []}
                    continue

                test_split_point = int(n_samples * (1 - test_size))
                val_split_point = int(test_split_point * (1 - val_size))

                tr = group.index[:val_split_point].tolist()
                va = group.index[val_split_point:test_split_point].tolist()
                te = group.index[test_split_point:].tolist()

                split_map[filename] = {
                    'scenario': group['scenario'].iloc[0] if 'scenario' in group.columns else None,
                    'rx': group['rx'].iloc[0] if 'rx' in group.columns else None,
                    'train': tr,
                    'valid': va,
                    'test': te
                }

            # Combining according to cross ---
            train_idx, valid_idx, test_idx = [], [], []

            if cross is not None and target_scenario is not None:
                # define domain for the given cross
                if cross == 'users':
                    domain = ['S1', 'S2', 'S3']
                elif cross == 'environments':
                    domain = ['S5', 'S6', 'S7']
                elif cross == 'day':
                    domain = ['S1', 'S5']
                elif cross == 'rx':
                    # domain is the rx values present in the dataset (intersect with allowed)
                    # domain_all = sorted(df_meta['rx'].dropna().unique().tolist()) if 'rx' in df_meta.columns else []
                    # domain = domain_all
                    allowed_scenarios = ['S1', 'S2', 'S3']
                    domain = [sc for sc in df_meta['scenario'].unique() if sc in allowed_scenarios]
                else:
                    domain = []

                # combine only files inside domain
                for filename, parts in split_map.items():
                    scen = parts['scenario']
                    rx = parts['rx']

                    if cross == 'rx':
                        # in_domain = (rx in domain)
                        in_domain = (scen in domain)  # solo scenari ammessi
                        is_target = (rx == target_scenario)
                    else:
                        in_domain = (scen in domain)
                        is_target = (scen == target_scenario)

                    if not in_domain:
                        # skip files outside the domain completely
                        continue

                    if is_target:
                        # this file contributes only to test
                        test_idx.extend(parts['test'])
                    else:
                        # this file contributes to train/val
                        train_idx.extend(parts['train'])
                        valid_idx.extend(parts['valid'])

                # diagnostic: if test empty -> raise informative error
                if len(test_idx) == 0:
                    raise ValueError(
                        f"After applying cross='{cross}' and target_scenario='{target_scenario}', test set is empty.\n"
                        f"Domain={domain}. Available scenarios: {sorted(df_meta['scenario'].unique())}. "
                        f"Available rx: {sorted(df_meta['rx'].unique()) if 'rx' in df_meta.columns else 'N/A'}. "
                        f"Check target_classes filter: {target_classes}"
                    )

            else:
                # cross is None -> combine all files
                for parts in split_map.values():
                    train_idx.extend(parts['train'])
                    valid_idx.extend(parts['valid'])
                    test_idx.extend(parts['test'])
                    

            total = len(train_idx) + len(valid_idx) + len(test_idx)
            print(f"Data split complete: Train={len(train_idx)} [{int(len(train_idx)*100/total) if total>0 else 0 :.2f}%], "
                  f"Valid={len(valid_idx)} [{int(len(valid_idx)*100/total) if total>0 else 0 :.2f}%], "
                  f"Test={len(test_idx)} [{int(len(test_idx)*100/total) if total>0 else 0 :.2f}%]")

            # --- Fit scalers batch-wise (safe mapping using get_indexer) ---
            amp_scaler, ph_scaler = None, None
            batch_size = 256
            with h5py.File(data_path, 'r') as hf:
                

                if amplitude:
                    first_amp_shape = hf[f'window_{df_meta.iloc[0]["window_idx"]}']['amp'].shape
                    S, F, T = first_amp_shape
                    amp_scaler = MinMaxScaler((0, 1)) if scaler == "MMS" else StandardScaler()
                    for i in tqdm(range(0, len(train_idx), batch_size), desc="Fitting Amp Scaler"):
                        batch_indices = train_idx[i:i+batch_size]
                        if not batch_indices:
                            continue
                        # map original indices -> positional indices in df_meta (get_indexer)
                        pos = df_meta.index.get_indexer(batch_indices)
                        pos = pos[pos >= 0]  # drop not-found
                        if len(pos) == 0:
                            continue
                        window_idxs = df_meta.iloc[pos]['window_idx'].values
                        batch_amps = np.array([hf[f'window_{w_idx}']['amp'][:] for w_idx in window_idxs])
                        X_amp_batch = utils.reshape_csi_for_scaling_batch(batch_amps, S, F, smode=smode)
                        amp_scaler.partial_fit(X_amp_batch)
                    joblib.dump(amp_scaler, amp_scaler_path)

                if phase:
                    first_ph_shape = hf[f'window_{df_meta.iloc[0]["window_idx"]}']['ph'].shape
                    S, F, T = first_ph_shape
                    ph_scaler = MinMaxScaler((0, 1)) if scaler == "MMS" else StandardScaler()
                    for i in tqdm(range(0, len(train_idx), batch_size), desc="Fitting Phase Scaler"):
                        batch_indices = train_idx[i:i+batch_size]
                        if not batch_indices:
                            continue
                        pos = df_meta.index.get_indexer(batch_indices)
                        pos = pos[pos >= 0]
                        if len(pos) == 0:
                            continue
                        window_idxs = df_meta.iloc[pos]['window_idx'].values
                        batch_phs = np.array([hf[f'window_{w_idx}']['ph'][:] for w_idx in window_idxs])
                        X_ph_batch = utils.reshape_csi_for_scaling_batch(batch_phs, S, F, smode=smode)
                        ph_scaler.partial_fit(X_ph_batch)
                    joblib.dump(ph_scaler, ph_scaler_path)


            # --- Save final metadata ---
            df_meta['split'] = 'none'
            df_meta.loc[train_idx, 'split'] = 'train'
            df_meta.loc[valid_idx, 'split'] = 'valid'
            df_meta.loc[test_idx, 'split'] = 'test'
            df_meta.to_parquet(final_meta_path)


            print("Train scenarios:", df_meta.loc[train_idx, 'scenario'].unique())
            print("Test scenarios: ", df_meta.loc[test_idx,  'scenario'].unique())
            print("Train classes:", np.unique(df_meta.loc[train_idx, 'ENC_LABEL'], return_counts=True))
            print("Test classes: ", np.unique(df_meta.loc[test_idx,  'ENC_LABEL'], return_counts=True))
        

        # --- Post-processing: filter test based on target_scenario if requested ---
        if cross is None and target_scenario is not None:
            # I'm taking only the test set and filtering it based on the target_scenario.
            if target_scenario in df_meta['rx'].values:
                mask_test = (df_meta['split'] == 'test') & (df_meta['scenario'].isin(['S1', 'S2', 'S3'])) & (df_meta['rx'] == target_scenario)
            elif target_scenario in df_meta['scenario'].values:
                mask_test = (df_meta['split'] == 'test') & (df_meta['scenario'] == target_scenario)
            else:
                raise ValueError(f"Target scenario '{target_scenario}' not found in 'scenario' or 'rx' columns of the metadata.")
            df_meta.loc[df_meta['split'] == 'test', 'split'] = 'none'   # resetto i vecchi test
            df_meta.loc[mask_test, 'split'] = 'test'

            test_idx = df_meta.index[df_meta['split'] == 'test'].tolist()
            print(f"[Post-filter] Test set restricted to scenario '{target_scenario}'. "
                f"Remaining test samples: {len(test_idx)}")

        if class_order is None:
            num_classes = len(df_meta['ENC_LABEL'].unique())
            class_order = list(range(num_classes))
        else:
            num_classes = len(class_order)
            class_order = class_order.copy()
        if shuffle_classes:
            np.random.shuffle(class_order)

        print(f"Final class order: {class_order}")
        trn_dset, val_dset = None, None
        # 3. Create Dataset objects
        if not only_test:
            trn_dset = TrainingCSIDataset(df_meta, data_path, amp_scaler, ph_scaler, class_order, split='train', smode=smode)
            val_dset = TrainingCSIDataset(df_meta, data_path, amp_scaler, ph_scaler, class_order, split='valid', smode=smode)
        tst_dset = TrainingCSIDataset(df_meta, data_path, amp_scaler, ph_scaler, class_order, split='test', smode=smode)

        # tst_amp = [x for x, _ in tst_dset]
        # trn_amp = [x for x, _ in trn_dset]

        # def minmax_stats(amp):
        #         arrs = np.stack(amp)
        #         return arrs.min(), arrs.max(), np.percentile(arrs, [1,50,99])
        # print("AMP train stats:", minmax_stats(trn_amp))
        # print("AMP test  stats:", minmax_stats(tst_amp))
        # exit()
        print("\n--- Preprocessing Complete. Datasets are ready for training. ---")
        return trn_dset, val_dset, tst_dset, class_order
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: 'Cominelli23'.")
