import numpy as np
from typing import Dict, List
from itertools import product


class AdaptiveConformalPredictionModule:
    """
    Implementation of Adaptive Conformal Prediction (ACP) for pedestrian trajectory forecasting.
    For the details of ACP, refer to Gibbs & Candes, 2021, or Zaffran et al., 2022.
    Considering the lagged nature of prediction tasks, a lagged variant of ACP is implemented here; See Dixit et al., 2023.
    """

    def __init__(self,
                 target_miscoverage_level,
                 step_size,
                 n_scores,
                 max_interval_lengths,
                 sample_size,
                 offline_calibration_set: Dict[int, List[float]]
                 ):

        self._n_scores = n_scores
        self._alpha = target_miscoverage_level

        # effective miscoverage level alpha_t
        # initialization: alpha_0 := alpha
        self._alpha_t = target_miscoverage_level * np.ones(self._n_scores)

        self._history = {
            'score': [],
            'interval': [],
            'coverage': [],
            'effective': []
        }

        # TODO: real queue instead of the list implementation (in case we need memory usage reduction)
        # keeps the past K predictions, where K: prediction length
        # Each element of a queue is a dictionary representing a map track_id -> array of shape (K, 2)
        self._prediction_queue = []
        self._interval_queue = []

        # step size parameter
        # gamma = 0 corresponds to the standard split conformal prediction
        self._gamma = step_size
        self._max_interval_len = max_interval_lengths       # known upper bounds of the score functions

        # initialize each interval length to its maximum possible value
        # calibration set D^i_t containing the scores of i-th step prediction
        self._online_calibration_set = {
            i: [] for i in range(n_scores)
        }
        self._sample_size = sample_size
        # a set of pre-computed scores from offline data
        # corresponds to D^i_0
        self._offline_calibration_set = {i: [] for i in range(n_scores)}
        self._load_offline_calibration_set(offline_calibration_set)

        self._step = 0

    def _load_offline_calibration_set(self, offline_calibration_set: Dict[int, List[float]]):
        for i in range(self._n_scores):
            n_offline = len(offline_calibration_set[i])
            if n_offline < self._sample_size:
                # insufficient number of offline data (even smaller than calibration sample size)
                # fill the set with the known upper bound of the score
                self._offline_calibration_set[i] += [self._max_interval_len[i] for _ in range(self._sample_size - n_offline)]

        return

    def update(self, obs, pred):
        """
        observe x_t -> generate prediction(s) -> adaptive update of D^i_{cal, t} & alpha^i_t
                    -> build C_{alpha^i_t}(x_t|D^i_{cal,t})
        """
        scores = self._compute_scores(obs)
        self._update_calibration_sets(scores)
        intervals = self._compute_interval_lengths()
        coverages = self._evaluate_coverage(scores)
        effective_levels = self._update_miscoverage_levels(coverages)

        self._prediction_queue.append(pred)         # PREDICTION(t)
        self._interval_queue.append(intervals)      # INTERVAL(t)
        self._step += 1     # t -> t + 1

        # store intermediate results for future evaluation
        self._history['score'].append(scores)
        self._history['interval'].append(intervals)
        self._history['coverage'].append(coverages)
        self._history['effective'].append(effective_levels)

        return intervals, coverages

    def _compute_scores(self, obs):
        """
        Compare the past predictions x(t|t-1), ..., x(t|t-K) with a new observation x(t) and compute the scores.

        return: list of length min(t, K) whose entry represents s^i_t
                If t = 0, then this returns the empty list [].
        """
        scores = []
        for i, pred in enumerate(reversed(self._prediction_queue)):
            # trace the diagonal of the prediction queue
            if i >= self._n_scores:
                # Recall: # score functions = prediction length
                break
            else:
                errors = []
                if not pred:
                    # When there is no pedestrian in the scene, then report zero error
                    scores.append(0.)
                else:
                    for track_id, xy_seq in pred.items():
                        if track_id in obs.keys():
                            # only if a tracked object appears both in the past prediction result & real observation
                            err = np.sum((xy_seq[i] - obs[track_id][-1]) ** 2) ** .5
                            errors.append(err)
                    if not errors:
                        scores.append(0.)
                    else:
                        scores.append(np.max(errors))
        return scores

    def _update_calibration_sets(self, scores):
        # D^i_{t+1] := D^i_t + [s^i_t]
        for i, score in enumerate(scores):
            self._online_calibration_set[i].append(score)
        return

    def _compute_interval_lengths(self):
        # R^i_t := Q_{1 - alpha_t} (D), where D ~ D^i_{t+1]
        intervals = []
        for i in range(self._n_scores):
            n_online = len(self._online_calibration_set[i])
            if n_online < self._sample_size:
                # insufficient online data
                # use all online data & sample a subset of offline data
                n_offline_samples = self._sample_size - n_online
                samples = self._online_calibration_set[i] + list(np.random.choice(self._offline_calibration_set[i], size=n_offline_samples))
            else:
                samples = np.random.choice(self._online_calibration_set[i], size=self._sample_size, replace=True)
            alpha = self._alpha_t[i]
            if alpha <= 0.:
                interval = self._max_interval_len[i]
            elif alpha < 1.:
                interval = np.quantile(samples, q=1-self._alpha_t[i])
            else:
                # alpha >= 1.
                interval = 0.
            intervals.append(interval)
        return np.array(intervals)

    def _evaluate_coverage(self, scores):
        coverages = np.array(scores) <= np.diag(np.flip(self._interval_queue, axis=0))
        assert coverages.size == min(self._n_scores, self._step)  # min(t, K)
        # check s^i_t <= R^i_{t-i}, which is equivalent to x(t) in C^i_{t-i}
        return coverages

    def _update_miscoverage_levels(self, coverages):
        if coverages.size > 0:
            # alpha^i_t = alpha^{i+1}_t + gamma * (alpha - 1 + I(x(t) in C^i_{t-i}))
            self._alpha_t[:coverages.size] += self._gamma * (self._alpha - 1. + coverages)
            self._alpha_t = np.clip(self._alpha_t, 0., 1.)
        return np.copy(self._alpha_t)

    def get_summary(self):
        return self._history.copy()