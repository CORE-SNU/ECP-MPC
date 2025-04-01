import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron

from sim_cp_mpc import run_cp_mpc
from sim_eacp_mpc import run_eacp_mpc
from sim_cc import run_cc
from sim_mpc import run_mpc

import json


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="id of dataset to evaluate on", type=str)
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--controller", help="control method to use", type=str)
parser.add_argument("--visualize", help="control method to use", action='store_true')
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
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        print("-- Evaluating GMM Grid Sampled (Most Likely)")

        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            # timesteps = np.arange(scene.timesteps)
            timesteps = np.arange(scene.timesteps)

            pos_x_mean = scene.pos_x_mean
            pos_y_mean = scene.pos_y_mean

            # RCP-MPC: robocentric CP-MPC
            # DRCP-MPC: distributional robocentric CP-MPC
            eval_stg_configs = {
                # TODO: add other baseline control methods
                'cc': {'num_samples': 1, 'z_mode': True, 'gmm_mode': True, 'full_dist': False},
                'cp-mpc': {'num_samples': 1, 'z_mode': True, 'gmm_mode': True, 'full_dist': False},
                # 'eacp-mpc':{'num_samples': 20, 'z_mode': False, 'gmm_mode': False, 'full_dist': True},
                'eacp-mpc': {'num_samples': 1, 'z_mode': True, 'gmm_mode': True, 'full_dist': False},
                'mpc': {'num_samples': 1, 'z_mode': True, 'gmm_mode': True, 'full_dist': False}
            }

            eval_functions = {
                'cc': run_cc,
                'mpc': run_mpc,
                'cp-mpc': run_cp_mpc,
                'eacp-mpc': run_eacp_mpc,
            }

            eval_task_configs = {
                # 'zara1': {'init_robot_pose': np.array([14., 5., np.pi]), 'goal_pos': np.array([3., 6.])},
                'zara1': {'init_robot_pose': np.array([12., 5., np.pi]), 'goal_pos': np.array([3., 6.])},
                'zara2': {'init_robot_pose': np.array([1., 6., 0.]), 'goal_pos': np.array([14., 5.])},
                'hotel': {'init_robot_pose': np.array([-1.5, 0., -np.pi/2]), 'goal_pos': np.array([2., -6.])},
                'eth': {'init_robot_pose': np.array([5., 1.0, np.pi/2.]), 'goal_pos': np.array([3., 10.])},
                'univ': {'init_robot_pose': np.array([3.5, 2., np.pi/4.]), 'goal_pos': np.array([11.5, 8.5])},
            }

            scenarios = {
                'zara1': [100, 200, 300],
                'zara2': [100, 200, 300],
                'eth': [100, 200, 300],
                'hotel': [100, 200, 300],
                'univ': [100]
            }

            init_frames = {
                'zara1': 0,
                'zara2': 1,
                'eth': 78,
                'hotel': 0,
                'univ': 0,
            }       # just for matching the correct frame

            max_n_steps = {
                'zara1': 100,
                'zara2': 100,
                'eth': 100,
                'hotel': 100,
                'univ': 300
            }

            # max_n_steps = {
            #     'zara1': 500,
            #     'zara2': 500,
            #     'eth': 500,
            #     'hotel': 500,
            #     'univ': 400
            # }


            prediction_kwargs = eval_stg_configs[args.controller]
            task_kwargs = eval_task_configs[args.dataset]
            eval_func = eval_functions[args.controller]

            predictions = eval_stg.predict(
                scene,
                timesteps,
                ph,
                min_history_timesteps=1,
                min_future_timesteps=0,
                **prediction_kwargs
            )  # This will trigger grid sampling

            metric_dict, trajectories = eval_func(
                dataset=args.dataset,
                scenarios=scenarios[args.dataset],
                max_linear_x=0.8,
                min_linear_x=-0.8,
                predictions=predictions,
                pos_x_mean=pos_x_mean, pos_y_mean=pos_y_mean,
                dt=scene.dt,
                init_frame=init_frames[args.dataset],
                max_hl=max_hl,
                ph=ph,
                map=None,
                visualize=args.visualize,
                max_n_steps=max_n_steps[args.dataset],
                **task_kwargs
            )

            os.makedirs('traj', exist_ok=True)
            np.save('./traj/{}_{}.npy'.format(args.dataset, args.controller), trajectories)


            dict_to_save = {
                'collision': [],
                'cost': [],
                'time': [],
                'infeasible': [],
                'miscoverage': []
            }

            print('dataset: {} / controller: {}'.format(args.dataset, args.controller))
            for scene_idx, eval_metric in metric_dict.items():
                print('-------- scene {} --------'.format(scene_idx))
                n_collisions = np.sum(eval_metric['collisions'])
                collision_ratio = n_collisions / len(eval_metric['collisions'])
                print('* collision_ratio={}'.format(collision_ratio))
                dict_to_save['collision'].append(collision_ratio)

                avg_cost = np.mean(eval_metric['costs'])
                print('* avg cost={:.4f}'.format(avg_cost))
                dict_to_save['cost'].append(avg_cost)

                exit_time = eval_metric['exit_time']
                exit_time = min(exit_time, max_n_steps[args.dataset])
                print('* exit time={}'.format(exit_time))
                dict_to_save['time'].append(exit_time)

                if 'infeasible' in eval_metric:
                    n_infeasible = np.sum(eval_metric['infeasible'])
                    infeasible_ratio = n_infeasible / len(eval_metric['infeasible'])
                    print('* infeasible_ratio={}'.format(infeasible_ratio))
                    dict_to_save['infeasible'].append(infeasible_ratio)

                if 'miscoverage' in eval_metric:
                    n_miscoverage = np.sum(eval_metric['miscoverage'])
                    miscoverage_ratio = n_miscoverage / len(eval_metric['miscoverage'])
                    print('* asymptotic miscoverage={:.4f}'.format(miscoverage_ratio))
                    dict_to_save['miscoverage'].append(miscoverage_ratio)

            save_dir = './metric'
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, '{}_{}.json'.format(args.dataset, args.controller)), 'w') as f:
                json.dump(dict_to_save, f, ensure_ascii=False, indent=4)
