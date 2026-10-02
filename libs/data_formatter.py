import numpy as np




dataset_config = {

        'cominelli23': {
            'tool': 'AXCSI',
            'wifi': ['ac', 'ax'],
            'channel_bandwidth': [20, 40, 80, 160],
            'sampling_rate': 150, #packets per second
            'antennas': 4,
            'n_envs': 3,
            'n_users': 3,
            'win_size': 51,
            'data': 'RAW',
            'n_classes': 12,
            'path': './data/cominelli23',
        },

        'meneghello23': {
            'tool': 'Nexmon',
            'wifi': ['ac'],
            'channel_bandwidth': [80],
            'sampling_rate': 160, #packets per second
            'antennas': 4,
            'n_envs': 3,
            'n_users': 3,
            'win_size': 31,
            'data': 'RAW',
            'n_classes': 5,
        },

}



Nexmon_config={
    'ac': {
        'offset': 0,
        'n_subcarriers':{
            20: 64,
            40: 128,
            80: 256,
        },
        'subcarriers_filter': {
            20: np.array([0, 1, 2, 3, 32, 61, 62, 63]),
            40:np.array([0, 1, 2, 3, 4, 5, 63, 64, 65, 123, 124, 125, 126, 127]),
            80:np.array([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255]),
        }
    },
}

AXCSI_config={
    'ac': {
        'offset': 4,
        'n_subcarriers':{
            20: 256,
            40: 512,
            80: 1024,
            160: 2048
        },
        'subcarriers_filter': {
            20: np.array([0, 1, 2, 3, 32, 61, 62, 63]),
            40:np.array([0, 1, 2, 3, 4, 5, 63, 64, 65, 123, 124, 125, 126, 127]),
            80:np.array([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255]),
            160: np.array([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 383, 384, 385, 507, 508, 509, 510, 511])
        }
    },
    'ax': {
        'offset': 0,
        'n_subcarriers':{
            20: 256,
            40: 512,
            80: 1024,
            160: 2048
        },
        'subcarriers_filter': {
            20: np.array([0, 1, 2, 3, 4, 5, 125, 126, 127, 128, 129, 130, 131, 251, 252, 253, 254, 255]),
            40: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 254, 255, 256, 257, 258, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511]),
            80:np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 509, 510, 511, 512, 513, 514, 515, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023]),
            160: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 509, 510, 511, 512, 513, 514, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1535, 1536, 1537, 1538, 1539, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047])
        }
    }
}




def get_dataset_setup(name, wifi, channel_bandwidth):
    if name=='cominelli23':
        return {'n_subcarriers': AXCSI_config[wifi]['n_subcarriers'][channel_bandwidth], 'subcarriers_filter': AXCSI_config[wifi]['subcarriers_filter'][channel_bandwidth]}
    elif name=='meneghello23':
        return {'n_subcarriers': Nexmon_config[wifi]['n_subcarriers'][channel_bandwidth], 'subcarriers_filter': Nexmon_config[wifi]['subcarriers_filter'][channel_bandwidth]}
    return None
