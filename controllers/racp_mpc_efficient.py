import numpy as np
import numpy.ma as ma
from controllers.utils import obs2numpy, pred2numpy, compute_pairwise_distances_along_axis, compute_pairwise_distances, compute_quantiles


class RobocentricACPMPCEfficient:
    def __init__(self,
                 n_steps=12,
                 dt=0.4,
                 min_linear_x=-0.8, max_linear_x=0.8,
                 min_angular_z=-0.7, max_angular_z=0.7,
                 n_skip=4,
                 robot_rad=0.4,
                 calibration_set_size=10,
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

        self.alpha_t = miscoverage_level * np.ones((n_paths, n_steps))

        self.n_skip = n_skip

        self.robot_rad = robot_rad
        self.safe_rad = self.robot_rad + 1. / np.sqrt(2.)

        self.path_history = []
        self.quantile_history = []

        self.calibration_set_size = calibration_set_size

        self._gamma = step_size

        self._prediction_queue = []         # prediction results
        self._track_queue = []              # true configuration of dynamic obstacles

    def __call__(self, pos_x, pos_y, orientation_z, boxes, predictions, goal):
        # Warning! The method can be invoked only when t >= N
        # Thus, the controller has to wait until at least N observations are collected.

        # update the observation queue & alpha^J_t's
        # The following line has been moved to the outer loop
        # self.update_observations(obs=tracking_res)

        # span a discrete search space (x^J_{...|t}: J in scr{J})
        paths, vels = self.generate_paths(pos_x, pos_y, orientation_z, n_skip=self.n_skip)

        self.path_history.append(paths)

        quantiles = self.evaluate_scores(paths=paths)       # compute R^J_{t+i|t} for all J & i
        self.quantile_history.append(quantiles)

        # Solve MPC:
        # Find x^J_{...|t}'s within the constraints
        safe_paths, vels = self.filter_unsafe_paths(paths, vels, boxes, predictions, quantiles)

        # update the prediction queue
        # moved to the outer loop
        # self.update_predictions(predictions)

        if safe_paths is None:
            # print('MPC infeasible')
            return None, {'feasible': False,
                'quantiles': quantiles}
        else:
            path, vel = self.score_paths(safe_paths, vels, goal)
            info = {
                'feasible': True,
                'candidate_paths': paths,
                'safe_paths': safe_paths,
                'final_path': path,
                'quantiles': quantiles
            }
            return vel[1], info


    def run_naive_mpc(self, pos_x, pos_y, orientation_z, boxes, predictions, goal):
        # Warning! The method can be invoked only when t >= N
        # Thus, the controller has to wait until at least N observations are collected.

        # update the observation queue & alpha^J_t's
        # The following line has been moved to the outer loop
        # self.update_observations(obs=tracking_res)

        # span a discrete search space (x^J_{...|t}: J in scr{J})
        paths, vels = self.generate_paths(pos_x, pos_y, orientation_z, n_skip=self.n_skip)

        # self.path_history.append(paths)

        # quantiles = self.evaluate_scores(paths=paths)       # compute R^J_{t+i|t} for all J & i
        # self.quantile_history.append(quantiles)

        # Solve MPC:
        # Find x^J_{...|t}'s within the constraints
        safe_paths, vels = self.filter_unsafe_paths(paths, vels, boxes, predictions, quantiles=0.)

        # update the prediction queue
        # moved to the outer loop
        # self.update_predictions(predictions)

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
                            predictions,
                            quantiles
                            ):
        # static constraints
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

        # shape = (search space size, prediction length)
        min_dist = self.compute_min_dist(paths=paths[:, 1:, :], obs=pred2numpy(predictions))
        mask_unsafe_dynamic = np.any(min_dist < self.safe_rad + quantiles, axis=-1)

        # True = no collision
        mask_safe = np.logical_and(np.logical_not(mask_unsafe_static), np.logical_not(mask_unsafe_dynamic))

        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe]
        else:
            # print('no safe paths found')
            return None, None

    def update_predictions(self, prediction_result):
        self._prediction_queue.append(pred2numpy(prediction_result))

    @staticmethod
    def compute_min_dist(paths, obs):
        """
        paths: shape (search space size, # steps, dim.)
        obs: shape (# nodes, # steps, dim.)
        return: shape (# search space size, # steps)
        """

        D = compute_pairwise_distances_along_axis(paths, obs, axis=(1, 1))      # (# steps, # search space size, # nodes)
        return np.min(D, axis=-1).T

    def load_recent_obs(self, batch_size):
        # list of length <= size whose elements are arrays of shape (|V|, 2) where |V|: # nodes
        # TODO: handle the case # observations < batch size
        # t' in [t-M+1, t]
        obs_batch = self._track_queue[-batch_size:]
        max_n_nodes = max(o.shape[0] for o in obs_batch)
        batch = np.zeros((batch_size, max_n_nodes, 2))        # batch with padded zeros (to store variable-size arrays)
        mask = np.ones((batch_size, max_n_nodes), dtype=bool)
        for t, o in enumerate(obs_batch):
            n_nodes = o.shape[0]
            batch[t, :n_nodes, :] = o
            mask[t, :n_nodes] = False
        return batch, mask      # (batch size, max # nodes)

    def load_recent_pred(self, batch_size, step):
        """
        If this method is called at time t, we assume that the queue only stores the prediction results up to t - 1:
        X_{...|t-1}, X{...|t-2}...
        """
        assert step >= 1
        # t' in [t-M+1-i, t-i]
        idx_begin = -batch_size + 1 - step
        idx_end = None if step == 1 else 1 - step

        # each element having of shape (|V(t')|, prediction length, 2)
        pred_batch = self._prediction_queue[idx_begin:idx_end]
        max_n_nodes = max(o.shape[0] for o in pred_batch)
        batch = np.zeros((batch_size, max_n_nodes, 2))  # batch with padded zeros (to store variable-size arrays)
        mask = np.ones((batch_size, max_n_nodes), dtype=bool)
        for t, p in enumerate(pred_batch):
            n_nodes = p.shape[0]
            batch[t, :n_nodes, :] = p[:, step-1, :]     # only the i-th step predictions
            mask[t, :n_nodes] = False
        return batch, mask      # (batch size, max # nodes)

    def evaluate_scores(self, paths):
        n_data = len(self._track_queue)  # X(0), ...., X(t) stored in the queue; thus represents t + 1
        # at least t >= N in order to compare X_t & X_{t|t-N}
        # This yields a condition for the batch size > 0:
        assert n_data >= self._n_steps + 1

        n_paths = paths.shape[0]

        # at most M recent data
        batch_size = min(n_data-self._n_steps, self.calibration_set_size)
        obs_batch, mask = self.load_recent_obs(batch_size=batch_size)
        # (search space size, prediction length, batch size, max # nodes)
        pdist_obs = compute_pairwise_distances(paths[:, 1:, :], obs_batch)
        pdist_obs[..., mask] = np.inf                   # fill the masked entries with a large value
        min_dist_obs = np.min(pdist_obs, axis=-1)       # (search space size, prediction length, batch size)

        # minimum distance from the robot to predicted positions
        min_dist_pred = np.zeros_like(min_dist_obs)
        for i in range(1, self._n_steps+1):     # 1 <= i <= N
            x_i = paths[:, i, :]        # (search space size, 2)
            # (search space size, batch size, max # nodes)
            # TODO: check the order of insertions & loading
            pred_batch, mask = self.load_recent_pred(batch_size=batch_size, step=i)
            pdist_obs = compute_pairwise_distances(x_i, pred_batch)
            pdist_obs[:, mask] = np.inf
            min_dist_pred_i = np.min(pdist_obs, axis=-1)   # (search space size, batch size)
            min_dist_pred[:, i-1, :] = min_dist_pred_i

        scores = np.clip(min_dist_pred - min_dist_obs, a_min=0., a_max=None)
        # final shape: (search space size, prediction length)
        quantiles = compute_quantiles(scores, axis=-1, levels=1.-self.alpha_t)
        # maximum for Q_{1 - alpha_t} when alpha_t <= 0
        # The value is estimated from the reported FDEs of Trajectron++ on UCY-ETH datasets
        max_scores = .2 * np.arange(1, self._n_steps+1)
        max_scores = np.tile(max_scores, (n_paths, 1))
        quantiles = np.where(np.isposinf(quantiles), max_scores, quantiles)
        return quantiles

    def update_observations(self, obs):
        obs_numpy = obs2numpy(obs)     # (# nodes, 2)
        quantiles = []
        min_dist_obs = []
        min_dist_pred = []

        n_data = len(self.quantile_history)     # R_{t_0}, ..., R_{t-1} stored; thus represents t - t_0 where t_0 >= N
        max_n_steps = min(self._n_steps + 1, n_data)
        for i in range(1, max_n_steps):

            q_i = self.quantile_history[-i][:, i-1]
            quantiles.append(q_i)
            # x^J_{t-1}, x^J_{t-2}, ...
            paths = self.path_history[-i]       # x^J_{...|t-i}
            x_i = paths[:, i, :]            # x^J_{t|t-i}; of shape (search space size, 2)
            min_dist_obs_i = np.min(compute_pairwise_distances(x_i, obs_numpy), axis=-1)        # (search space size,)
            min_dist_obs.append(min_dist_obs_i)

            pred = self._prediction_queue[-i]       # X_{...|t-i}; of shape (|V(t-i)|, prediction length, 2)
            pred_i = pred[:, i-1, :]                # (|V(t-i)|, 2)
            # TODO: this must have been calculated, thus caching the past result would save the computation
            min_dist_pred_i = np.min(compute_pairwise_distances(x_i, pred_i), axis=-1)            # (search space size,)
            min_dist_pred.append(min_dist_pred_i)

        n_paths = self.alpha_t.shape[0]

        if n_data > 1:
            min_dist_obs = np.stack(min_dist_obs, axis=-1)          # (search space size, prediction length)
            min_dist_pred = np.stack(min_dist_pred, axis=-1)

            quantiles = np.stack(quantiles, axis=-1)                #

            err = (quantiles < min_dist_pred - min_dist_obs)

            # if n_data < self._n_steps + 1:
            #     pad_width = self._n_steps + 1 - n_data
            #     err = np.hstack((err, np.zeros((n_paths, pad_width))))

            self.alpha_t[:, :max_n_steps - 1] += self._gamma * (self._miscoverage_level - err)
            if n_data < self._n_steps + 1:
                pad_width = self._n_steps + 1 - n_data
                err = np.hstack((err, np.zeros((n_paths, pad_width))))      # just for the consistent data size
        else:
            err = np.zeros((n_paths, self._n_steps))

        # online update of alpha^{J, i}_t's

        self.alpha_t = np.clip(self.alpha_t, 0., 1.)
        # update data
        self._track_queue.append(obs_numpy)
        return err

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

