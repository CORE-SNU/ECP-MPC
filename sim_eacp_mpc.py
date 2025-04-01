import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import time

from rollout_utils import render
from controllers import EgocentricACPMPC

sys.path.append("../../trajectron")

from utils import prediction_output_to_trajectories


def run_eacp_mpc(
        dataset,
        scenarios,
        init_robot_pose,
        goal_pos,
        max_linear_x,
        min_linear_x,
        predictions,
        pos_x_mean, pos_y_mean,  # dataset info (for transforming the prediction results)
        dt,  # sampling time
        init_frame,
        max_hl,
        ph,
        map,
        visualize,
        max_n_steps
):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    if visualize:
        robot_img = Image.open(os.path.join(os.path.dirname(__file__), "./assets/robot.png"))
        asset_dir = '/media/sju5379/F6340D35340CF9FF/euped_assets'
        print('dataset frames loaded from', asset_dir)

    stat_dir = os.path.join('./stats', dataset, 'eacp-mpc')
    os.makedirs(stat_dir, exist_ok=True)
    metric_dict = dict()

    trajectories = []

    for scene_idx, scenario_begin in enumerate(scenarios):
        xys = []
        eval_metrics = {
            'collisions': [],
            'costs': [],
            'exit_time': np.inf,
            'miscoverage': [],
            'infeasible': []
        }

        controller = EgocentricACPMPC(
            n_steps=12,
            dt=dt,
            min_linear_x=-min_linear_x, max_linear_x=max_linear_x,
            min_angular_z=-0.7, max_angular_z=0.7,
            n_skip=4,
            robot_rad=0.4,
            calibration_set_size=15,
            step_size=0.02,
            miscoverage_level=0.1,
            mode='deterministic'
        )

        if len(prediction_dict.keys()) == 0:
            return

        position_x, position_y, orientation_z = init_robot_pose

        if visualize:
            video_dir = os.path.join('./videos', dataset, str(scene_idx), 'eacp-mpc')
            print('path to rendered scenes:', video_dir)
            os.makedirs(video_dir, exist_ok=True)

            print('results visualized at {}'.format(video_dir))


        shifted_mean = np.array([pos_x_mean, pos_y_mean])
        # performance metrics

        count = 0
        done = False

        ts_key = scenario_begin        # t_i



        while count < max_n_steps:
            if ts_key in prediction_dict:

                print('[t={}]'.format(ts_key), end='')

                p_dict = prediction_dict[ts_key]
                h_dict = histories_dict[ts_key]
                f_dict = futures_dict[ts_key]

                p_dict = {node: p.squeeze() + shifted_mean for node, p in p_dict.items()}
                h_dict = {node: h.squeeze() + shifted_mean for node, h in h_dict.items()}
                f_dict = {node: f.squeeze() + shifted_mean for node, f in f_dict.items()}

                err = controller.update_observations(h_dict)
                eval_metrics['miscoverage'].append(np.mean(err))

                if count < 15:
                    velocity = np.array([0., 0.])
                    info = {'feasible': True,
                            'candidate_paths': np.array([]),
                            'safe_paths': np.array([]),
                            'final_path': np.tile(np.array([position_x, position_y]), (12, 1))}

                else:
                    obs_pos = np.array([o[-1] for o in h_dict.values()])  # (|V|, 2)
                    robot_pos = np.array([position_x, position_y])
                    min_obs_dist = np.min(np.sum((obs_pos - robot_pos) ** 2, -1) ** .5)

                    min_goal_dist = np.sum((robot_pos - goal_pos) ** 2, -1) ** .5
                    # print('min. goal distance={:.4f}'.format(min_goal_dist))
                    if min_goal_dist <= 0.6 and not done:
                        eval_metrics['exit_time'] = count
                        done = True

                    collision = True if min_obs_dist < 0.4 else False
                    if not done:
                        eval_metrics['collisions'].append(collision)



                    begin = time.time()
                    velocity, info = controller(pos_x=position_x,
                                                pos_y=position_y,
                                                orientation_z=orientation_z,
                                                boxes=[],  # TODO
                                                predictions=p_dict,
                                                goal=goal_pos
                                                )
                    comp_time = time.time() - begin
                    # computation_times.append(comp_time)

                    # print('computation time: {:.5f}sec /'.format(comp_time), end=' ')

                    if not done:
                        infeasible = 0. if info['feasible'] else 1.
                        eval_metrics['infeasible'].append(infeasible)

                if not info['feasible']:
                    velocity = np.array([0., 0.])
                    # print('linear_x={} / angular_z={} (infeasible)'.format(*velocity))
                else:
                    if count >= 15 and not done:
                        cost = info['cost']
                        eval_metrics['costs'].append(cost)
                    # velocity = np.array([1., 0.])
                    # print('linear_x={} / angular_z={} (feasible)'.format(*velocity))

                # if 'quantiles' in info.keys():
                    # qs = info['quantiles']
                    # print('quantiles :', np.mean(qs, axis=0))

                controller.update_predictions(p_dict)

                linear_x, angular_z = velocity
                position_x += dt * linear_x * np.cos(orientation_z)
                position_y += dt * linear_x * np.sin(orientation_z)
                orientation_z += dt * angular_z

                xys.append(np.array([position_x, position_y]))

                if visualize:
                    render(
                        dataset, ts_key, init_frame, position_x, position_y, orientation_z, robot_img, goal_pos,
                        info, h_dict, f_dict, p_dict, video_dir, asset_dir, intervals=None
                    )
                count += 1
            ts_key += 1

        trajectories.append(xys)
        metric_dict[scene_idx] = eval_metrics

    plt.clf(), plt.cla()
    xmax = -np.inf
    for scene_idx, eval_metrics in metric_dict.items():
        errors = np.array(eval_metrics['miscoverage'])   # (# simulated steps, search space size, prediction length)
        xmax = max(xmax, errors.shape[0])
        errors_cumul = np.cumsum(errors)
        errors_asymptotic = errors_cumul / (1 + np.arange(errors_cumul.size))
        # print(np.max(errors_mean, axis=0))
        plt.plot(errors_asymptotic)
    plt.axhline(y=0.1, xmin=0, xmax=xmax, color='black', linestyle='dashed')
    plt.xlabel('simulation step')
    plt.ylabel('asymptotic miscoverage')
    plt.xlim(0., xmax)
    plt.ylim(0., 1.)
    plt.grid()
    plt.savefig(os.path.join(stat_dir, 'asymptotic_error.png'))
    plt.close()

    plt.clf(), plt.cla()
    xmax = -np.inf
    for scene_idx, eval_metrics in metric_dict.items():
        collisions = np.array(eval_metrics['collisions'])
        xmax = max(xmax, len(collisions))
        collisions_cumul = np.cumsum(collisions)
        collisions_asymptotic = collisions_cumul / (1 + np.arange(collisions_cumul.size))
        plt.plot(collisions_asymptotic)
    plt.xlabel('simulation step')
    plt.ylabel('asymptotic collision rate')
    plt.xlim(0., xmax)
    plt.ylim(0.)
    plt.grid()
    plt.savefig(os.path.join(stat_dir, 'collision.png'))
    plt.close()

    plt.clf(), plt.cla()
    xmax = -np.inf
    for scene_idx, eval_metrics in metric_dict.items():
        infeas = np.array(eval_metrics['infeasible'])
        xmax = max(xmax, len(infeas))
        infeasible_cumul = np.cumsum(infeas)
        infeasible_asymptotic = infeasible_cumul / (1 + np.arange(infeasible_cumul.size))
        plt.plot(infeasible_asymptotic)
    plt.xlabel('simulation step')
    plt.ylabel('asymptotic infeasibility rate')
    plt.xlim(0., xmax)
    plt.ylim(0.)
    plt.grid()
    plt.savefig(os.path.join(stat_dir, 'infeasible.png'))
    plt.close()

    return metric_dict, trajectories
