import sys
import os
import torch
import random
import numpy as np
import pandas as pd
import time
import copy
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob
from matplotlib.ticker import LogLocator, FixedLocator, MultipleLocator, AutoMinorLocator, MaxNLocator, Locator

from hampel import hampel
from scipy.signal import butter, filtfilt
import scipy.signal as signal

CHECKMARK = u'\033[92m\u2714\033[0m'
CROSSMARK = u'\033[91m\u2718\033[0m'
#######################################################################################################
# Denoising functions
#######################################################################################################
def hampel_filter_csi(csi_tensor, window_size=5, n_sigmas=3.0):
        """
        Applica il filtro di Hampel a un tensore CSI (antennas, subcarriers, time).
        
        Parameters
        ----------
        csi_tensor : np.ndarray
            Array di forma (S, F, T).
        window_size : int
            Dimensione della finestra locale (default 5).
        n_sigmas : int
            Numero di deviazioni standard robuste per identificare outlier (default 3).
        
        Returns
        -------
        np.ndarray
            CSI filtrato con stessa shape (S, F, T).
        """
        S, F, T = csi_tensor.shape
        filtered = np.empty_like(csi_tensor)

        for s in range(S):
            for f in range(F):
                ts = pd.Series(csi_tensor[s, f, :])
                ts_filtered = hampel(ts, window_size=window_size, n_sigma=n_sigmas)
                filtered[s, f, :] = ts_filtered.filtered_data

        return filtered

def hampel_filter_optimized(csi_tensor, window_size=5, n_sigmas=3.0):
    """
    Applica il filtro di Hampel in modo vettorializzato al tensore CSI.

    Parameters
    ----------
    csi_tensor : np.ndarray
        Array di forma (Time, Subcarriers, Streams).
    window_size : int
        Dimensione della finestra locale (deve essere dispari).
    n_sigmas : float
        Numero di deviazioni standard robuste per identificare outlier (default 3.0).

    Returns
    -------
    np.ndarray
        CSI filtrato con stessa shape (Time, Subcarriers, Streams).
    """
    if csi_tensor.ndim != 3:
        raise ValueError("Il tensore deve avere forma (Time, Subcarriers, Streams).")
    if window_size % 2 == 0:
        raise ValueError("window_size deve essere dispari.")
        
    T, F, S = csi_tensor.shape
    
    # 1. Rimodella il tensore in (Canali, Tempo)
    # Trasponi a (Streams, Subcarriers, Time) -> (S, F, T)
    # Poi rimodella in (S*F, T) dove ogni riga è una serie temporale.
    data_2d = csi_tensor.transpose(2, 1, 0).reshape(S * F, T)
    filtered_data_2d = data_2d.copy()
    
    # Processa ogni canale (riga) indipendentemente
    for i in range(data_2d.shape[0]):
        ts = pd.Series(data_2d[i, :])
        
        # Calcola la Mediana e la MAD in modo vettoriale
        rolling_median = ts.rolling(window=window_size, center=True).median()
        mad_raw = np.abs(ts - rolling_median)
        rolling_mad = mad_raw.rolling(window=window_size, center=True).median()
        
        # Gestisci i valori NaN ai bordi (sostituisci con il valore non filtrato)
        rolling_median = rolling_median.fillna(ts)
        # Fallback per la MAD ai bordi (sostituisci con la deviazione standard)
        rolling_mad = rolling_mad.fillna(ts.std()) 

        # Definisci la maschera degli outlier (1.4826 è il fattore di scala standard)
        threshold = n_sigmas * 1.4826 * rolling_mad
        outlier_mask = (np.abs(ts - rolling_median) > threshold)
        
        # Sostituzione vettoriale dell'outlier con la mediana locale
        ts_filtered = ts.copy()
        ts_filtered[outlier_mask] = rolling_median[outlier_mask]
        filtered_data_2d[i, :] = ts_filtered.values

    # 2. Rimodella la matrice 2D filtrata alla forma originale (Time, Subcarriers, Streams)
    # (S*F, T) -> reshape -> (S, F, T) -> transpose -> (T, F, S)
    return filtered_data_2d.reshape(S, F, T).transpose(2, 1, 0)

def butterworth_filter_csi(csi_tensor, fs=150, cutoff_freq=10, order=5):
        """
        Applies a low-pass Butterworth filter to a CSI tensor.
        
        The filter is applied to the time series of each individual antenna-subcarrier pair.

        Parameters
        ----------
        csi_tensor : np.ndarray
            Input array of shape (S, F, T) where S=antennas, F=subcarriers, T=time.
        fs : float
            The sampling rate of the data in Hz.
        cutoff_freq : float
            The cutoff frequency of the filter in Hz (default 10).
        order : int
            The order of the filter (default 5).

        Returns
        -------
        np.ndarray
            The filtered CSI with the same shape (S, F, T).
        """
        if csi_tensor.ndim != 3:
            raise ValueError("Input tensor must have 3 dimensions (antennas, subcarriers, time).")

        S, F, T = csi_tensor.shape
        filtered_csi = np.zeros_like(csi_tensor)

        # Design the digital filter once
        nyquist_freq = 0.5 * fs
        normalized_cutoff = cutoff_freq / nyquist_freq
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

        # Apply the filter to each individual time series
        for s in range(S):
            for f in range(F):
                # Extract the 1D time series for the current antenna and subcarrier
                time_series = csi_tensor[s, f, :]
                
                # Apply the filter in both forward and reverse directions
                # to eliminate phase distortion.
                filtered_csi[s, f, :] = signal.filtfilt(b, a, time_series)

        return filtered_csi

def savgol_filter_csi(csi_tensor, window_length=15, polyorder=3):
    """
    Applies a Savitzky-Golay filter to a CSI tensor.
    
    The filter is applied to the time series of each individual antenna-subcarrier pair.

    Parameters
    ----------
    csi_tensor : np.ndarray
        Input array of shape (S, F, T) where S=antennas, F=subcarriers, T=time.
    window_length : int
        The length of the filter window. Must be an odd positive integer.
    polyorder : int
        The order of the polynomial used to fit the samples. Must be less than window_length.

    Returns
    -------
    np.ndarray
        The filtered CSI with the same shape (S, F, T).
    """
    if csi_tensor.ndim != 3:
        raise ValueError("Input tensor must have 3 dimensions (antennas, subcarriers, time).")

    S, F, T = csi_tensor.shape
    filtered_csi = np.zeros_like(csi_tensor)

    # Apply the filter to each individual time series
    for s in range(S):
        for f in range(F):
            time_series = csi_tensor[s, f, :]
            filtered_csi[s, f, :] = signal.savgol_filter(time_series, window_length, polyorder)

    return filtered_csi


#######################################################################################################
# Data Scaling functions
#######################################################################################################

def reshape_csi_for_scaling_batch(csi, S, F, smode="SxF"):
    if smode=='SxF':
       csi = csi.transpose(0, 3, 1, 2).reshape(-1, S*F)  
    elif smode=='S':
        csi = csi.transpose(0, 2, 3, 1).reshape(-1, S)
    else:
        raise ValueError(f"Unknown mode: {smode}")   
    return csi

def reshape_csi_for_scaling_sample(csi, S, F, T, smode="SxF", restore_shape=False):
    if smode=='SxF':
        if restore_shape:
            csi = csi.reshape(T, S, F).transpose(1,2,0)  # (S,F,T)
        else:
            csi = csi.transpose(2,0,1).reshape(T, S*F)  
    elif smode=='S':
        if restore_shape:
            csi = csi.reshape(F,T,S).transpose(2,0,1)  # (S,F,T)
        else:
            csi = csi.transpose(1,2,0).reshape(T*F, S)
    else:
        raise ValueError(f"Unknown mode: {smode}")
    return csi
#######################################################################################################
# Utility functions
#######################################################################################################
cudnn_deterministic = True
def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic

# class Logger(object):
#     def __init__(self, log_path):
#         self.terminal = sys.stdout
#         self.log = open(log_path, "a")
   
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)  

#     def flush(self):
#         # this flush method is needed for python 3 compatibility.
#         # this handles the flush command by doing nothing.
#         # you might want to specify some extra behavior here.
#         self.terminal.flush()
#         self.log.flush() # <--- THIS IS THE KEY CHANGE
#         # pass
        

class Logger(object):
    def __init__(self, log_path):
        self.log_path = log_path
        self.terminal = sys.stdout
        self.log = None

    def __enter__(self):
        """Apre il file di log e reindirizza l'output."""
        self.log = open(self.log_path, "a")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ripristina l'output standard e chiude il file."""
        sys.stdout = self.terminal
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def make_args_dict(**kwargs):
    # Default values for arguments
    default_args = {
        'device': 'cuda',
        'network': 'CNN',
        'dataset': 'UT-HAR',
    }
  
    # Update default args with provided kwargs
    args_out = default_args.copy()
    args_out.update(kwargs)
    
    return args_out    


def format_inputs(device, images, targets=None):
    if isinstance(images, list) or isinstance(images, tuple):
        if isinstance(images[0], list) or isinstance(images[0], tuple):
            images = [v.to(device) for x in images for v in x]
        else:
            images = [v.to(device) for v in images]
    else:
        images = images.to(device)
    if targets is not None:
        targets = targets.to(device)
        return images, targets
    return images


def get_elapsed(t_start, t_end=None):
    elapsed_time = time.time() - t_start if t_end is None else t_end - t_start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return hours, minutes, seconds



def wrapperScheduler(model, optimizer, scheduler_name="RoP", **kwargs):
    if scheduler_name == "RoP":
        return ReduceLROnPlateauEarlyStoppingScheduler(model, optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class ReduceLROnPlateauEarlyStoppingScheduler:
    def __init__(self, model, optimizer, mode='min', patience=10, lr_factor=0.1,
                 lr_min=1e-6, monitor='val_loss', verbose=False):
        """
        Args:
            model: il modello da monitorare (deve avere .state_dict())
            optimizer: istanza di torch.optim.*
            mode: 'min' o 'max' a seconda della metrica monitorata
            patience: numero di epoche senza miglioramenti prima della riduzione del LR
            lr_factor: fattore per ridurre il LR (nuovo_lr = lr / lr_factor)
            lr_min: LR minimo oltre il quale si ferma il training
            monitor: nome della metrica (es. 'val_loss', 'val_accuracy')
            verbose: se True, stampa i cambiamenti di LR
        """
        assert mode in ['min', 'max']
        self.model = model
        self.optimizer = optimizer
        self.mode = mode
        self.monitor = monitor
        self.patience = patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.verbose = verbose

        self.best_metric = np.inf if mode == 'min' else -np.inf
        self.best_model = None
        self.current_patience = patience
        self.current_lr = optimizer.param_groups[0]['lr']

    def update(self, metric_value):
        """
        Args:
            metric_value: valore della metrica monitorata all’epoca corrente
        Returns:
            stop (bool): True se deve fermarsi il training
            best (bool): True se questa epoca è la migliore finora
        """
        improved = (metric_value < self.best_metric) if self.mode == 'min' else (metric_value > self.best_metric)

        if improved:
            self.best_metric = metric_value
            self.best_model = self._get_model_copy()
            self.current_patience = self.patience
            return False, True  # stop, best

        self.current_patience -= 1
        if self.current_patience <= 0:
            new_lr = self.current_lr * self.lr_factor
            if new_lr < self.lr_min:
                return True, False  # early stopping
            else:
                self.current_lr = new_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                if self.verbose:
                    print(f'[Scheduler] Reducing LR to {self.current_lr:.2e}')
                self.current_patience = self.patience
        return False, False

    def _get_model_copy(self):
        # Deep copy dei pesi del modello corrente
        return copy.deepcopy(self.model.state_dict())
    

def get_class_order(stdout_file):

    class_order = None
    assert os.path.exists(stdout_file), f"File {stdout_file} non trovato"
    with open(stdout_file) as f:
        lines = f.readlines()
    for l in lines:
        if 'Final class order' in l:
            class_order = [int(elem) for elem in l.split(':')[-1].replace('[','').replace(']','').replace('','').split(',')]
            break
    assert class_order is not None, "Class order non trovato nel file di log"
    return class_order


#################################################################################################################################
# Plotting functions
#################################################################################################################################

def set_legend(ax,ncol=1,frame_alpha=0.7,loc='lower left',bbox_to_anchor=(0., 1.02, 1., .102),fontsize=10, frameon=False,mode='expand', handles=None ,borderaxespad=0.,labelspacing=0.5, columnspacing=2.0, handletextpad=0.1):
    if handles:
        ax.legend(ncol=ncol, framealpha=frame_alpha, bbox_to_anchor=bbox_to_anchor, loc=loc, mode="expand", borderaxespad= borderaxespad,fontsize=fontsize,frameon=frameon,handles=handles,labelspacing=labelspacing,columnspacing=columnspacing, handletextpad=handletextpad)
    else:
        ax.legend(ncol=ncol, framealpha=frame_alpha, bbox_to_anchor=bbox_to_anchor, loc=loc, mode="expand", borderaxespad= borderaxespad,fontsize=fontsize,frameon=frameon,labelspacing=labelspacing,columnspacing=columnspacing, handletextpad=handletextpad)

def plot_loss(exp_path, campaign_ts, normalize=True,task=0, yscale='log', show=False):
    assert yscale in ['lin','log']
    trn_chkpts_path=os.path.join(exp_path, 'epochs_training')
    val_chkpts_path=os.path.join(exp_path, 'epochs_validation')
    
    if not os.path.exists(trn_chkpts_path):
        return
    if not os.path.exists(val_chkpts_path):
        return
    
    colors=['darkseagreen', 'salmon','mediumorchid']
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    best_epoch=-1
    trn_files=glob(trn_chkpts_path+f'/task{task}-*-{campaign_ts}.tar')
    val_files=glob(val_chkpts_path+f'/task{task}-*-{campaign_ts}.tar')
    if len(trn_files)==0:
        return

    trn_files.sort()
    val_files.sort()
    trn_losses=[]
    val_losses=[]
    epochs=[]
    for trn_file, val_file in zip(trn_files, val_files):
        trn_chkpt=torch.load(trn_file,weights_only=True)
        val_chkpt=torch.load(val_file,weights_only=True)
        trn_loss=trn_chkpt['total_loss']/trn_chkpt.get('total_loss_bathes',1) if normalize else trn_chkpt['total_loss']
        trn_losses.append(trn_loss)
    
        val_loss=val_chkpt.get('val_loss',None)
        if val_loss is not None:
            val_losses.append(val_loss/val_chkpt.get('val_loss_bathes',1) if normalize else val_loss)
            epochs.append(val_chkpt['epoch'])
        if val_chkpt.get('is_best',False):
            if val_chkpt['epoch']>best_epoch:
                best_epoch=val_chkpt['epoch']
    sindex=np.argsort(epochs)
    trn_losses=[trn_losses[i] for i in sindex]
    val_losses=[val_losses[i] for i in sindex]    


    all_losses = trn_losses + val_losses
    if all_losses:
        
        max_loss = max(all_losses)
        
        if yscale == 'log':
            min_loss = min(l for l in all_losses if l > 0)
            # Calculate ymin and ymax based on powers of 10
            ymin = 10**np.floor(np.log10(min_loss))
            ymax = 10**np.ceil(np.log10(max_loss))
            ax.set_ylim(ymin, ymax)
        elif yscale == 'lin':
            min_loss = min(all_losses)
            # Calculate a buffer for the y-axis limits
            y_range = max_loss - min_loss
            y_buffer = y_range * 0.1  # 10% buffer
            ymin = max(0, min_loss - y_buffer) # ymin cannot be < 0
            ymax = max_loss + y_buffer
            ax.set_ylim(ymin, ymax)
            # Set major and minor ticks
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))


    sns.lineplot(x=range(1,len(trn_losses)+1),y=trn_losses, label=f'Train', color=colors[0], linestyle='-', ax=ax)
    if len(val_losses)>0:
        sns.lineplot(x=range(1,len(val_losses)+1),y=val_losses, label=f'Val', color=colors[1], linestyle='--', ax=ax)
        if best_epoch!=-1:
            sns.scatterplot(x=[best_epoch+1],y=val_losses[best_epoch],color=colors[0],facecolor=colors[0], marker='*',edgecolor='goldenrod', zorder=300,s=350,ax=ax)
        print(f'Best epoch: {best_epoch+1}: {trn_losses[best_epoch]} {val_losses[best_epoch]}')
    

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.set_yscale(yscale)
    if yscale=='log':
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        
    bbox_to_anchor=(0., 1.1, 1., .102)
    set_legend(ax, ncol=2, loc='upper left', frameon=False, fontsize=11, bbox_to_anchor=bbox_to_anchor)

    ax.grid(visible=True, linestyle='--', which='major', axis='y', lw=.5, zorder=0)
    ax.grid(visible=True, linestyle='--', which='minor', axis='y', lw=.1, zorder=0)
    ax.grid(visible=True, linestyle='--', which='major', axis='x', lw=.5, zorder=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f'loss-{campaign_ts}.pdf'), format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close() 



class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))