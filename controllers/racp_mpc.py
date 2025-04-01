import numpy as np
from itertools import product


class RobocentricACPMPC:
    def __init__(self,
                 n_steps=12,
                 dt=0.4,
                 min_linear_x=-0.8, max_linear_x=0.8,
                 min_angular_z=-0.7, max_angular_z=0.7,
                 n_skip=4,
                 robot_rad=0.4,
                 calibration_set_size=12,
                 miscoverage_level=0.1,
                 step_size=0.05
                 ):

        self._n_steps = n_steps
        self._dt = dt
        self._miscoverage_level = miscoverage_level

        self.max_linear_x = max_linear_x
        self.min_linear_x = min_linear_x

        self.max_angular_z = max_angular_z
        self.min_angular_z = min_angular_z

        n_decision_epochs = n_steps // n_skip

        n_points = 9

        n_paths = n_points ** n_decision_epochs

        self._alpha_t = miscoverage_level * np.ones((n_paths, n_steps))

        self.n_skip = n_skip

        self.robot_rad = robot_rad
        self.safe_rad = self.robot_rad + 1. / np.sqrt(2.)

        self.calibration_set_size = calibration_set_size

        self._gamma = step_size

        self._prediction_queue = []         # prediction results
        self._track_queue = []              # true configuration of dynamic obstacles

    def __call__(self, pos_x, pos_y, orientation_z, boxes, predictions, goal):

        paths, vels = self.generate_paths(pos_x, pos_y, orientation_z, n_skip=self.n_skip)
        # paths, vels = self.generate_paths_wheel_vel(pos_x, pos_y, orientation_z, linear_x, angular_z)
        safe_paths, vels = self.filter_unsafe_paths(paths, vels, boxes, predictions)
        if safe_paths is None:
            # print('MPC infeasible')
            return None, {'feasible': False}
        else:
            path, vel = self.score_paths(safe_paths, vels, goal)
            info = {
                'feasible': True,
                'candidate_paths': paths,
                'safe_paths': safe_paths,
                'final_path': path
            }
            return vel[1], info

    @staticmethod
    def score_paths(paths, vels, goal):
        intermediate_cost = np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2,-1))
        control_cost = .001 * np.sum(vels ** 2, axis=(-2, -1))
        terminal_cost = 10. * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        minimum_cost = np.argmin(intermediate_cost + control_cost + terminal_cost)
        return paths[minimum_cost], vels[minimum_cost]

    def filter_unsafe_paths(self,
                            paths,
                            vels,
                            boxes,
                            predictions
                            ):

        masks = []
        for box in boxes:
            center = box.pos
            sz = np.array([box.w, box.h])
            th = box.rad
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])  # rotate by -th w.r.t. the origin
            lb, ub = -.5 * sz - self.robot_rad, .5 * sz + self.robot_rad
            # robot's current coordinate frame -> rectangle's coordinate frame
            transformed_paths = (paths[:, 1:, :] - center) @ R  # first state: observed from the system
            # boolean array of shape (# paths, # steps)
            # True = collision
            mask = np.logical_and(np.all(transformed_paths <= ub, axis=-1), np.all(transformed_paths >= lb, axis=-1))
            masks.append(mask)
        masks = np.array(masks)

        mask_unsafe_static = np.sum(masks, axis=0, dtype=bool)
        mask_unsafe_static = np.sum(mask_unsafe_static, axis=-1)

        # shape: (# paths, prediction length)
        # min_distance[k, i] = dist(robot_pos_i[k], prediction[i])
        # min_distance = np.min([np.sum((paths[:, 1:, :] - prediction) ** 2, axis=-1) ** .5 for prediction in predictions.values()], axis=0)

        min_distance = self._calculate_minimum_distance(paths, predictions.values())

        confidence_intervals = self._calibrate_scores(paths)

        mask_unsafe_dynamic = np.any(min_distance - confidence_intervals < self.safe_rad, axis=-1)

        # True = no collision
        mask_safe = np.logical_and(np.logical_not(mask_unsafe_static), np.logical_not(mask_unsafe_dynamic))
        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe]
        else:
            # print('no safe paths found')
            return None, None

    def check_constraints(self, pos_batch, step, p_dict):
        """
        pos_batch: batch of shape (batch size, 2)
        step: prediction step to consider; range from 0 ~ prediction length - 1

        check if d(x, X(t+i|t)) >= r_safe + R_t^i(x) for each x in a given batch

        return: boolean array of shape (batch size,)
                i-th entry = True if pos_batch[i] satisfies the constraint
        """

        # (batch size, 2)
        # -> [collecting over nodes] -> (# nodes, batch size)
        # -> [minimum among nodes] -> (batch size,)

        min_distance = np.min(
            [np.sum((pos_batch - p[step, :]) ** 2, axis=-1) ** .5 for p in p_dict.values()],
            axis=0)

        differences = []
        # x_v(t-j), v in V
        for j, tracking in enumerate(reversed(self._track_queue)):
            # x_v(...|t-j-i), v in V

            if step + j > len(self._prediction_queue) or j > self.calibration_set_size:
                break
            else:
                prediction = self._prediction_queue[-step - j]

                # TODO: list of positions in the queue
                tracked_xy = [xy[-1] for xy in tracking.values()]

                # x_v(t-j|t-j-i)
                # list of arrays each with shape (sample size, 2)
                predicted_xy = [xy[step] for xy in prediction.values()]

                # shape: (batch size,)
                distance_tracked = np.min([np.sum((pos_batch - xy) ** 2, axis=-1) ** .5 for xy in tracked_xy],
                                          axis=0)

                # (batch size, 2) -> (batch size,) -> (# nodes, batch size)
                # -> (batch size,)

                distance_predicted = np.min(
                    [np.sum((pos_batch - xy) ** 2, axis=-1) ** .5 for xy in predicted_xy],
                    axis=0)

                difference = np.clip(distance_predicted - distance_tracked, a_min=0., a_max=None)
                differences.append(difference)

        differences = np.array(differences)
        # (batch size,)
        # This has to be corrected...
        intervals = np.quantile(differences, q=1.-self._miscoverage_level, axis=0)
        return min_distance >= self.safe_rad + intervals, min_distance >= self.safe_rad

    def _calibrate_scores(self, paths):
        n_paths = paths.shape[0]
        confidence_intervals = np.zeros((n_paths, self._n_steps))

        # TODO: optimize
        for i in range(1, self._n_steps+1):
            # prediction step i: 1 <= i <= N
            robot_xy = paths[:, i-1, :]
            differences = []
            # x_v(t-j), v in V
            for j, tracking in enumerate(reversed(self._track_queue)):
                # x_v(...|t-j-i), v in V
                if i + j > len(self._prediction_queue) or j > self.calibration_set_size:
                    break
                else:
                    prediction = self._prediction_queue[-i-j]
                    # TODO: list of positions in the queue
                    tracked_xy = [xy[-1] for xy in tracking.values()]
                    # x_v(t-j|t-j-i)
                    predicted_xy = [xy[i-1] for xy in prediction.values()]
                    # shape: (# paths)
                    distance_tracked = np.min([np.sum((robot_xy - xy) ** 2, axis=-1) ** .5 for xy in tracked_xy], axis=0)
                    distance_predicted = np.min([np.sum((robot_xy - xy) ** 2, axis=-1) ** .5 for xy in predicted_xy], axis=0)
                    difference = np.clip(distance_predicted - distance_tracked, a_min=0., a_max=None)
                    differences.append(difference)
            differences = np.array(differences)
            # shape (calibration set size, # paths)
            # TODO: This is terribly slow...
            q = 1. - self._alpha_t[:, i-1]      # (# paths)
            intervals = np.zeros(n_paths)
            intervals[q <= 0.] = 0.
            # TODO: refine these values?
            intervals[q >= 1.] = (i - 1) * 1.2 * self._dt
            for k in range(n_paths):
                if 0. < q[k] < 1.:
                    intervals[k] = np.quantile(differences[:, k], q=q[k])
            confidence_intervals[:, i-1] = intervals
        return confidence_intervals

    def _calculate_minimum_distance(self, paths, configurations):
        return np.min([np.sum((paths[:, 1:, :] - pos) ** 2, axis=-1) ** .5 for pos in configurations], axis=0)

    def update_observations(self, pos_x, pos_y, orientation_z, tracking_result):
        if self._n_steps <= len(self._track_queue):
            # Extend this!!!
            # tree of the robot's future states whose root is the current robot state
            paths, _ = self.generate_paths(pos_x, pos_y, orientation_z, n_skip=self.n_skip)
            # evaluate the states in the tree using the previous observations of the nodes
            # is of shape (# paths, prediction length)
            confidence_intervals = self._calibrate_scores(paths)
            scores = np.zeros_like(confidence_intervals)

            for i in range(1, self._n_steps+1):
                # prediction step i: 1 <= i <= N
                robot_xy = paths[:, i - 1, :]
                # x_v(t-j), v in V

                # x_v(...|t-j-i), v in V
                if i > len(self._prediction_queue):
                    # TODO: handle corner cases (when there are insufficient number of past observations)
                    break
                else:

                    prediction = self._prediction_queue[-i]
                    # TODO: list of positions in the queue
                    tracked_xy = [xy[-1] for xy in tracking_result.values()]
                    # x_v(t-j|t-j-i)
                    predicted_xy = [xy[i - 1] for xy in prediction.values()]
                    # shape: (# paths)
                    distance_tracked = np.min([np.sum((robot_xy - xy) ** 2, axis=-1) ** .5 for xy in tracked_xy],
                                              axis=0)
                    distance_predicted = np.min([np.sum((robot_xy - xy) ** 2, axis=-1) ** .5 for xy in predicted_xy],
                                                axis=0)
                    difference = np.clip(distance_predicted - distance_tracked, a_min=0., a_max=None)
                    scores[:, i-1] = difference

                covered = scores <= confidence_intervals
                self._alpha_t += self._gamma * (self._miscoverage_level - 1. + covered)
        self._track_queue.append(tracking_result)

    def update_predictions(self, prediction_result):
        self._prediction_queue.append(prediction_result)

    def generate_paths(
            self,
            pos_x,
            pos_y,
            orientation_z,
            n_skip=5
    ):
        """
        Generate multiple paths starting at (x, y, theta) = (0, 0, 0)
        """

        # TODO: Employing pruning techniques would reduce the number of the paths, but would be also challenging to optimize...
        # TODO: use numba?
        # physical parameters
        dt = self._dt
        # velocity & acceleration ranges

        linear_xs = np.array([self.min_linear_x, .0, self.max_linear_x])
        angular_zs = np.array([self.min_angular_z, .0, self.max_angular_z])

        n_points = linear_xs.size * angular_zs.size

        linear_xs, angular_zs = np.meshgrid(linear_xs, angular_zs)

        linear_xs = np.reshape(linear_xs, newshape=(-1,))
        angular_zs = np.reshape(angular_zs, newshape=(-1,))

        # (# grid points, 2)
        # velocity_profile = np.stack((linear_xs, angular_zs), axis=0)

        n_decision_epochs = self._n_steps // n_skip

        # profiles = [velocity_profile for _ in range(n_decision_epochs)]

        # n_paths = n_points ** n_decision_epochs

        state_shape = tuple(n_points for _ in range(n_decision_epochs)) + (self._n_steps+1,)
        x = np.zeros(state_shape)
        y = np.zeros(state_shape)
        th = np.zeros(state_shape)

        # state initialization
        x[..., 0] = pos_x
        y[..., 0] = pos_y
        th[..., 0] = orientation_z

        control_shape = tuple(n_points for _ in range(n_decision_epochs)) + (self._n_steps,)
        v = np.zeros(control_shape)
        w = np.zeros(control_shape)

        for e in range(n_decision_epochs):
            augmented_shape = [1] * n_decision_epochs
            augmented_shape[e] = -1
            v_epoch = linear_xs.reshape(augmented_shape)
            w_epoch = angular_zs.reshape(augmented_shape)
            for t in range(e * n_skip, (e + 1) * n_skip):
                v[..., t] = v_epoch
                w[..., t] = w_epoch

                x[..., t + 1] = x[..., t] + dt * v_epoch * np.cos(th[..., t])
                y[..., t + 1] = y[..., t] + dt * v_epoch * np.sin(th[..., t])
                th[..., t + 1] = th[..., t] + dt * w_epoch

        x = np.reshape(x, (-1, self._n_steps+1))
        y = np.reshape(y, (-1, self._n_steps+1))
        # th = np.reshape(th, (-1, self._n_steps))
        v = np.reshape(v, (-1, self._n_steps))
        w = np.reshape(w, (-1, self._n_steps))

        return np.stack((x, y), axis=-1), np.stack((v, w), axis=-1)