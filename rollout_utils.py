import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2


from visualization.visualization_utils_custom import visualize_tracking_result, visualize_controller_info, \
    visualize_prediction_result, visualize_point, visualize_points, to_image_frame, visualize_sampled_prediction_result, visualize_cp_result


def render(
        dataset,
        ts_key,
        ts_begin,
        position_x,
        position_y,
        orientation_z,
        robot_img,
        goal_pos,
        info,
        h_dict,
        f_dict,
        p_dict,
        video_dir,
        asset_dir,
        intervals
):

    plt.clf(), plt.cla()
    fig, ax = plt.subplots()

    image = cv2.imread(os.path.join(asset_dir, 'frames', dataset, '{}.png'.format(ts_begin+ts_key)))
    ax.imshow(image, cmap='gray', alpha=0.6)
    h, w, _ = image.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')

    # load the homography matrix of the dataset
    homography_path = os.path.join(os.path.dirname(__file__), './assets/homographies', dataset + '.txt')
    H = np.loadtxt(homography_path, dtype=float)

    # mark the goal position
    visualize_point(goal_pos, H, ax, color='tab:red', marker='s', s=80, label='goal', zorder=500)

    # constraints for the naive MPC (for comparison)
    # visualize_controller_info(info_compare, H, ax, color='#7cabee', feasible_set=True, optimal_sol=False)
    visualize_controller_info(info, H, ax, color='#9dcda8', feasible_set=False, optimal_sol=True)

    # visualize the robot state
    draw_robot(ax, H, position_x, position_y, orientation_z, robot_img)
    visualize_tracking_result(h_dict, H, ax, color='#ffd300')

    visualize_prediction_result(p_dict, H, ax, color='#e6a8d7', linestyle='dashed', label='prediction')
    # visualize_sampled_prediction_result(p_dict, H, step_selected, ax, color='blue', linestyle='dashed', label=None)
    visualize_prediction_result(f_dict, H, ax, color='#ffd300', linestyle='solid', label='future')

    if intervals is not None:
        selected_steps = [1, 3, 5, 7, 9, 11]
        visualize_cp_result(intervals, p_dict, selected_steps, H, ax)

    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(video_dir, '{:03d}.pdf'.format(ts_key)), bbox_inches ='tight', pad_inches=0)
    plt.close()
    print('results visualized at {}'.format(video_dir))
    return


def draw_robot(
        ax,
        H,
        position_x,
        position_y,
        orientation_z,
        robot_img
):

    dx = 0.001 * np.cos(orientation_z)
    dy = 0.001 * np.sin(orientation_z)
    # position of the infinitesimal displacement vector's endpoint (w.r.t. the world frame)
    x_next, y_next = position_x + dx, position_y + dy

    # projective transformation of the vector
    pos_img_x, pos_img_y = to_image_frame(np.array([[position_x, position_y]]), H)
    pos_img_x, pos_img_y = pos_img_x.item(), pos_img_y.item()
    pos_img_x_next, pos_img_y_next = to_image_frame(np.array([[x_next, y_next]]), H)
    pos_img_x_next, pos_img_y_next = pos_img_x_next.item(), pos_img_y_next.item()

    # vector repr. w.r.t. the image frame
    dx_img = pos_img_x_next - pos_img_x
    dy_img = pos_img_y_next - pos_img_y

    orientation_z_img = np.arctan2(dy_img, dx_img)
    # orientation_z_img = orientation_z_img - np.pi / 2.

    # robot figure
    img_rotated = OffsetImage(Image.fromarray(ndimage.rotate(
        robot_img, np.rad2deg(-orientation_z_img))), zoom=0.05, zorder=80)

    ax.add_artist(AnnotationBbox(img_rotated, (pos_img_x, pos_img_y), frameon=False))