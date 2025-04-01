import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from baselines.linear_predictor import LinearPredictor
from baselines.koopman_predictor import KoopmanPredictor
from baselines.adaptive_koopman_predictor import AdaptiveKoopmanPredictor
import evaluation
from visualization.visualization import visualize_heatmaps, run_mpc, run_robocentric
from utils import prediction_output_to_trajectories

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--name", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":

    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    print(env)
    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes
    # print('scenes:', scenes)

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        '''
        ############### MOST LIKELY ###############
        '''
        # Most Likely (ML) -> y is generated as follows:
        # z_mode = argmax_z p(z|x)
        # y = argmax_y p(y|x, z_mode)

        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        print("-- Evaluating GMM Grid Sampled (Most Likely)")

        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            # timesteps = np.arange(scene.timesteps)
            timesteps = np.arange(scene.timesteps)

            pos_x_mean = scene.pos_x_mean
            pos_y_mean = scene.pos_y_mean

            '''
            baselines = {
                'Linear': LinearPredictor(dt=scene.dt, history_len=6, prediction_len=12),
                'Koopman': KoopmanPredictor(prediction_len=12, name=args.name, pos_x_mean=pos_x_mean, pos_y_mean=pos_y_mean),
                # 'Koopman (adaptive)': AdaptiveKoopmanPredictor(prediction_len=12, name=args.name, pos_x_mean=pos_x_mean, pos_y_mean=pos_y_mean),
            }

            baseline_colors = {
                'Linear': 'green',
                'Koopman': 'blue',
                # 'Koopman (adaptive)': 'red'
            }
            '''
            baselines = {

            }

            baseline_colors = {

            }

            predictions = eval_stg.predict(scene,
                                           timesteps,
                                           ph,
                                           num_samples=100,
                                           min_history_timesteps=1,
                                           min_future_timesteps=0,
                                           z_mode=False,
                                           gmm_mode=False,
                                           full_dist=False)  # This will trigger grid sampling

            metric = run_robocentric(predictions,
                                     pos_x_mean, pos_y_mean,
                                     scene.dt,
                                     max_hl,
                                     ph,
                                     path='{}/{}'.format(args.name, i),
                                     robot_node=None,
                                     baselines=baselines,
                                     baseline_colors=baseline_colors,
                                     map=None)

            print(args.name, metric)

            '''

            ps = []
            for timestep, single_step_prediction_dict in predictions_dict.items():
                # visualization
                single_step_prediction_dict_wrapped = {timestep: single_step_prediction_dict}
                # print(single_step_prediction_dict_wrapped)

                single_step = np.stack([np.squeeze(v) for v in single_step_prediction_dict.values()], axis=1)
                ps.append(single_step)

            filepath_to_save = os.path.join(os.path.dirname(__file__), '{}_trajectron_predictions.npy'.format(scene_id))
            np.save(filepath_to_save, ps)
            '''
            batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                   scene.dt,
                                                                   max_hl=max_hl,
                                                                   ph=ph,
                                                                   node_type_enum=env.NodeType,
                                                                   map=None,
                                                                   prune_ph_to_future=True,
                                                                   kde=False)

            eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
            eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
        '''
        print(np.mean(eval_fde_batch_errors))
        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_most_likely.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_most_likely.csv'))
        '''
