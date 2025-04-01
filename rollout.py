import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import time

from controllers import EgocentricACPMPC
from rollout_utils import render

sys.path.append("../../trajectron")

from utils import prediction_output_to_trajectories


def simulate_episode(
        dataset,
        ts_interval,
        init_robot_pose,
        goal_pos,
        max_linear_x,
        min_linear_x,
        predictions,
        pos_x_mean, pos_y_mean,     # dataset info (for transforming the prediction results)
        dt,                         # sampling time
        ts_begin,
        max_hl,
        ph
        ):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=None)

    controller = EgocentricACPMPC(
        n_steps=12,
        dt=0.4,
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

    # setup for rendering
    robot_img = Image.open(os.path.join(os.path.dirname(__file__), "./assets/robot.png"))
    video_dir = os.path.join('./videos', dataset, 'eacp-mpc')
    asset_dir = '/media/sju5379/F6340D35340CF9FF/euped_assets'
    print('dataset frames loaded from', asset_dir)
    print('path to rendered scenes:', video_dir)
    os.makedirs(video_dir, exist_ok=True)

    shifted_mean = np.array([pos_x_mean, pos_y_mean])
    # performance metrics
    errors = []
    computation_times = []
    collisions = []
    infeasibilities = []

    count = 0
    for ts_key in range(ts_interval[0], ts_interval[1]):
        if ts_key in prediction_dict:
            # if ts_key >= 150:
            #     break

            print('[t={}]'.format(ts_key), end='')

            p_dict = prediction_dict[ts_key]
            h_dict = histories_dict[ts_key]
            f_dict = futures_dict[ts_key]

            p_dict = {node: p.squeeze() + shifted_mean for node, p in p_dict.items()}
            h_dict = {node: h.squeeze() + shifted_mean for node, h in h_dict.items()}
            f_dict = {node: f.squeeze() + shifted_mean for node, f in f_dict.items()}

            # check safety
            obs_pos = np.array([o[-1] for o in h_dict.values()])  # (|V|, 2)
            robot_pos = np.array([position_x, position_y])
            min_dist = np.min(np.sum((obs_pos - robot_pos) ** 2, -1) ** .5)

            collision = True if min_dist < 0.4 else False
            collisions.append(collision)

            err = controller.update_observations(h_dict)

            errors.append(err)

            if count < 15:
                velocity = np.array([0., 0.])
                info = {'feasible': True,
                        'candidate_paths': np.array([]),
                        'safe_paths': np.array([]),
                        'final_path': np.tile(np.array([position_x, position_y]), (12, 1))}
                info_compare = info
            else:
                begin = time.time()
                velocity, info = controller(pos_x=position_x,
                                            pos_y=position_y,
                                            orientation_z=orientation_z,
                                            boxes=[],  # TODO
                                            predictions=p_dict,
                                            goal=goal_pos
                                            )
                comp_time = time.time() - begin
                computation_times.append(comp_time)

                print('computation time: {:.5f}sec /'.format(comp_time), end=' ')

                '''
                _, info_compare = controller.run_naive_mpc(
                    pos_x=position_x,
                    pos_y=position_y,
                    orientation_z=orientation_z,
                    boxes=[],  # TODO
                    predictions=p_dict,
                    goal=goal_pos
                )
                '''
            infeasible = False if info['feasible'] else True
            infeasibilities.append(infeasible)

            if not info['feasible']:
                velocity = np.array([0., 0.])

            feasibility_flag = 'feasible' if info['feasible'] else 'infeasible'
            print('linear_x={} / angular_z={} ({})'.format(*velocity, feasibility_flag))

            if 'quantiles' in info.keys():
                qs = info['quantiles']
                print('quantiles :', np.mean(qs, axis=0))

            controller.update_predictions(p_dict)

            linear_x, angular_z = velocity

            position_x += dt * linear_x * np.cos(orientation_z)
            position_y += dt * linear_x * np.sin(orientation_z)
            orientation_z += dt * angular_z

            render(
                dataset, ts_key, ts_begin, position_x, position_y, orientation_z, robot_img, goal_pos,
                info, h_dict, f_dict, p_dict, video_dir, asset_dir
            )
            count += 1

    # metric evaluation
    errors = np.array(errors)  # (# simulated steps, search space size, prediction length)
    xmax = errors.shape[0]
    errors_cumul = np.cumsum(errors, axis=0)
    errors_asymptotic = errors_cumul / (1 + np.arange(errors.shape[0]))[:, None, None]
    errors_asymptotic = errors_asymptotic[:, :, 0]
    errors_mean = np.mean(errors_asymptotic, axis=1)
    print(np.max(errors_mean, axis=0))
    plt.clf(), plt.cla()
    plt.plot(errors_mean)
    plt.axhline(y=0.1, xmin=0, xmax=xmax, color='black', linestyle='dashed')
    plt.xlabel('simulation step')
    plt.ylabel('asymptotic miscoverage')
    plt.xlim(0., xmax)
    plt.ylim(0., 1.)
    plt.grid()
    plt.savefig(os.path.join(video_dir, 'asymptotic_error.png'))
    plt.close()

    plt.clf(), plt.cla()
    xmax = len(computation_times)
    plt.plot(computation_times)
    plt.xlabel('simulation step')
    plt.ylabel('computation time (sec)')
    plt.xlim(0., xmax)
    plt.ylim(0.)
    plt.grid()
    plt.savefig(os.path.join(video_dir, 'comp_time.png'))
    plt.close()

    plt.clf(), plt.cla()
    xmax = len(collisions)
    collisions_cumul = np.cumsum(collisions)
    collisions_asymptotic = collisions_cumul / (1 + np.arange(xmax))
    plt.plot(collisions_asymptotic)
    plt.xlabel('simulation step')
    plt.ylabel('asymptotic collision rate')
    plt.xlim(0., xmax)
    plt.ylim(0.)
    plt.grid()
    plt.savefig(os.path.join(video_dir, 'collision.png'))
    plt.close()

    plt.clf(), plt.cla()
    xmax = len(infeasibilities)
    infeasible_cumul = np.cumsum(infeasibilities)
    infeasible_asymptotic = infeasible_cumul / (1 + np.arange(xmax))
    plt.plot(infeasible_asymptotic)
    plt.xlabel('simulation step')
    plt.ylabel('asymptotic infeasibility rate')
    plt.xlim(0., xmax)
    plt.ylim(0.)
    plt.grid()
    plt.savefig(os.path.join(video_dir, 'infeasible.png'))
    plt.close()

    return
