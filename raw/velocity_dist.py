import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt

sys.path.append("../../../trajectron")
from environment import derivative_of




def main():
    # plt.style.use('bmh')
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:red', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:gray'])
    
    fig, ax = plt.subplots()
    ax.set_xlabel('speed (m/s)')
    ax.set_xlim(0., 7.)
    ax.set_ylim(0., 2.)
    for dataset in ['zara1', 'zara2', 'univ', 'eth', 'hotel']:
        data_dir = os.path.join(dataset, 'test')
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                full_data_path = os.path.join(data_dir, filename)
                print(full_data_path, end=' ')
                data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
  
                data['frame_id'] = data['frame_id'] // 10

                data['frame_id'] -= data['frame_id'].min()

                data['node_type'] = 'PEDESTRIAN'
                data['node_id'] = data['track_id'].astype(str)
                data.sort_values('frame_id', inplace=True)
                vs = []
                print('/ # pedestrians:', len(pd.unique(data['node_id'])), end='')
                for node_id in pd.unique(data['node_id']):

                    node_df = data[data['node_id'] == node_id]
                    assert np.all(np.diff(node_df['frame_id']) == 1)
                    node_values = node_df[['pos_x', 'pos_y']].values
                    x = node_values[:, 0]
                    y = node_values[:, 1]

                    vx = derivative_of(x, 0.4)
                    vy = derivative_of(y, 0.4)
  
                    v = (vx ** 2 + vy ** 2) ** .5
                    vs.append(v)
                v_dataset = np.concatenate(vs)
                vmax = np.max(v_dataset)
                print('/ max. speed: {:.4f}m/s'.format(vmax))
                
                ax.hist(v_dataset, bins=100, density=True, label=dataset, alpha=0.7)

    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig('velocity_dist.png')

if __name__ == '__main__':
    main()
