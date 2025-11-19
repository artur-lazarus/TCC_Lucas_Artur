from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Optional

Point = Tuple[float, float]

def kalman_smoother(xs: List[np.ndarray], Ps: List[np.ndarray], 
                    F: np.ndarray, Q: np.ndarray):
    """
    RTS smoother that smooths only the position components (x, y)
    and preserves the filtered velocities (vx, vy).

    This avoids velocity bias caused by standard RTS on full state.
    """
    N = len(xs)
    if N == 0:
        return [], []

    # Initialize from last filtering step
    xs_smooth = [xs[-1].copy()]
    Ps_smooth = [Ps[-1].copy()]

    # Backward pass
    for k in range(N - 2, -1, -1):
        P_pred = F @ Ps[k] @ F.T + Q
        G = Ps[k] @ F.T @ np.linalg.inv(P_pred)

        # Standard RTS smoothing
        x_s = xs[k] + G @ (xs_smooth[0] - F @ xs[k])
        P_s = Ps[k] + G @ (Ps_smooth[0] - P_pred) @ G.T

        # -----------------------------------------
        # 🔑 DO NOT smooth velocity — preserve filtered vx, vy
        # -----------------------------------------
        x_s[2] = xs[k][2]   # vx (filtered)
        x_s[3] = xs[k][3]   # vy (filtered)

        xs_smooth.insert(0, x_s)
        Ps_smooth.insert(0, P_s)

    return xs_smooth, Ps_smooth

def _motion_matrices(dt: float):
    # State: [x, y, vx, vy]
    F = np.array([
        [1, 0, dt, 0 ],
        [0, 1, 0 , dt],
        [0, 0, 1 , 0 ],
        [0, 0, 0 , 1 ],
    ], dtype=float)
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)
    return F, H

def _q_cv(dt: float, sigma_a: float):
    # Continuous white acceleration model → discrete Q
    dt2 = dt*dt
    dt3 = dt2*dt
    dt4 = dt2*dt2
    q = sigma_a**2
    Q = q * np.array([
        [dt4/4,     0,   dt3/2,     0   ],
        [0,     dt4/4,     0,    dt3/2 ],
        [dt3/2,   0,    dt2,       0   ],
        [0,     dt3/2,     0,     dt2  ],
    ], dtype=float)
    return Q

_id_counter = 0
def _next_id():
    global _id_counter
    _id_counter += 1
    return _id_counter

@dataclass
class Track:
    id: int
    x: np.ndarray             # state (4,)
    P: np.ndarray             # covariance (4,4)
    F: np.ndarray             # (4,4)
    H: np.ndarray             # (2,4)
    Q: np.ndarray             # (4,4)
    R: np.ndarray             # (2,2)
    hits: int = 0
    time_since_update: int = 0
    age: int = 0
    history: list = field(default_factory=list)  # [[frame, (x, y), vel_x], ...]
    filtered_states: list = field(default_factory=list)
    filtered_covs: list = field(default_factory=list)
    last_detection: Point = None
    bboxes_with_framecount: list = field(default_factory=list)  # [(x, y, w, h), ...]
    frames: list = field(default_factory=list)  # [frame_count, ...]

    @staticmethod
    def from_detection(pt: Point, dt: float, sigma_a: float, sigma_z: float) -> "Track":
        F, H = _motion_matrices(dt)
        Q = _q_cv(dt, sigma_a)
        R = np.eye(2) * (sigma_z**2)

        x = np.zeros((4, 1))
        x[0, 0], x[1, 0] = pt
        # Start with a bit of uncertainty on position and larger on velocity
        P = np.diag([5.0, 5.0, 100.0, 100.0])

        return Track(
            id=_next_id(),
            x=x, P=P, F=F, H=H, Q=Q, R=R
        )

    def predict(self, frame_count: int):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        # Store frame_count, predicted position, and x velocity
        pos = self.position()
        vel_x = float(self.x[2, 0])
        self.history.append([frame_count, pos, vel_x])

    def update(self, z: Point, frame_count):
        self.last_detection = z
        #print(f"Speeds for track ID {self.id} before update: vx({float(self.x[2, 0])}) vy({float(self.x[3, 0])})")
        z = np.array([[z[0]], [z[1]]], dtype=float)
        y = z - (self.H @ self.x)                          # innovation
        #print(f"Innovation for track ID {self.id}: x({float(y[0,0])}) y({float(y[1,0])})")
        S = self.H @ self.P @ self.H.T + self.R            # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)           # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0
        #print(f"Appending speeds vx({float(self.x[2, 0])}) vy({float(self.x[3, 0])}) for track ID {self.id}")
        self.filtered_states.append(self.x.copy())
        self.filtered_covs.append(self.P.copy())
        self.history[-1] = [frame_count, self.position(), float(self.x[2, 0])]  # override last history entry

    def position(self) -> Point:
        return float(self.x[0, 0]), float(self.x[1, 0])

    def velocity(self) -> Point:
        return float(self.x[2, 0]), float(self.x[3, 0])
    
    @staticmethod
    def from_two_detections(pt0: Point, pt1: Point, dt: float, sigma_a: float, sigma_z: float) -> "Track":
        F, H = _motion_matrices(dt)
        Q = _q_cv(dt, sigma_a)
        R = np.eye(2) * (sigma_z**2)

        # State [x, y, vx, vy]
        vx = (pt1[0] - pt0[0]) / dt
        vy = (pt1[1] - pt0[1]) / dt

        x = np.array([[pt1[0]], [pt1[1]], [vx], [vy]], dtype=float)

        # Covariance: tighter on position (just observed), velocity from two-point FD
        var_pos = sigma_z**2  # we just measured it
        var_v = 0.1

        P = np.diag([var_pos, var_pos, var_v, var_v]).astype(float)

        t = Track(id=_next_id(), x=x, P=P, F=F, H=H, Q=Q, R=R)
        # We’ve effectively had two “hits” already:
        t.hits = 2  # or 2 if you prefer; this counts the birth evidence
        t.time_since_update = 0
        t.filtered_states.append(x.copy())
        return t

class Tracker:
    """
    Minimal multi-target tracker with per-target Kalman filters and greedy matching.

    Call update(cars) once per frame, where `cars` is a list of (x, y) points.
    Query active tracks via `tracks` property (list of Track objects).
    """
    def __init__(
        self,
        dt: float = 1/30,               # frame period (s)
        scale_lambda: float = 1.0,      # scale parameter
        sigma_a: float = 5.0,           # process noise (m/s^2-ish units)
        sigma_z: float = 3.0,           # measurement noise (pixels)
        distance_threshold: float = 100, # gating distance in pixels
        max_age: int = 10,              # drop tracks not seen for this many frames
        min_hits: int = 2               # require this many hits before reporting
    ):
        self.dt = float(dt)
        self.scale_lambda = float(scale_lambda)
        self.sigma_a = float(sigma_a)
        self.sigma_z = float(sigma_z)
        self.distance_threshold = float(distance_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._tracks: List[Track] = []
        self._newborns: List[Point] = []
        self._finished_tracks: List[Tuple[Track, float]] = []
        self.new_finished_tracks = 0
        

    @property
    def tracks(self) -> List[Track]:
        # Only return tracks that have matured (optional).
        return [t for t in self._tracks if t.hits >= self.min_hits or t.time_since_update == 0]

    def _predict_all(self, frame_count: int):
        for t in self._tracks:
            t.predict(frame_count)

    @staticmethod
    def _pairwise_dist(A: List[Point], B: List[Point]) -> np.ndarray:
        if len(A) == 0 or len(B) == 0:
            return np.zeros((len(A), len(B)))
        a = np.array(A, dtype=float)  # (na, 2)
        b = np.array(B, dtype=float)  # (nb, 2)
        # Euclidean distances
        d2 = (
            (a[:, None, 0] - b[None, :, 0])**2
          + (a[:, None, 1] - b[None, :, 1])**2
        )
        return np.sqrt(d2)

    def _greedy_match(self, cost: np.ndarray, threshold: float):
        """
        Greedy min-cost bipartite matching under a distance gate.
        Returns: list of (i_track, j_det), list of unmatched_track_idx, list of unmatched_det_idx
        """
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

        matches = []
        used_tracks = set()
        used_dets = set()

        # Flatten and sort by cost asc
        flat = [(i, j, cost[i, j]) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
        flat.sort(key=lambda x: x[2])

        for i, j, c in flat:
            if c > threshold:
                break
            if i in used_tracks or j in used_dets:
                continue
            matches.append((i, j))
            used_tracks.add(i)
            used_dets.add(j)

        unmatched_tracks = [i for i in range(cost.shape[0]) if i not in used_tracks]
        unmatched_dets = [j for j in range(cost.shape[1]) if j not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, bboxes: List[Tuple[Point, Point, Point, Point]], frame_count: int):
        # First: try to pair current detections with newborns (one-frame cache)
        detections = [(bbox[0], bbox[1]+bbox[3]) for bbox in bboxes]
        born = []
        if self._newborns:
            # greedy nearest pairing under the same gate
            D = self._pairwise_dist(self._newborns, detections)
            pairs, unused_newborns, unused_dets = self._greedy_match(D, self.distance_threshold)
            # Create seeded tracks for paired newborns
            for i_nb, j_det in pairs:
                pt0 = self._newborns[i_nb]
                pt1 = detections[j_det]
                self._tracks.append(
                    Track.from_two_detections(pt0, pt1, dt=self.dt, sigma_a=self.sigma_a, sigma_z=self.sigma_z)
                )
                born.append(j_det)
            # Keep only newborns that didn't find a mate (we'll drop them after this frame)
            self._newborns = [self._newborns[i] for i in unused_newborns]
            # Remove the detections that were just used to create tracks
            detections = [d for k, d in enumerate(detections) if k not in born]
            bboxes = [bb for k, bb in enumerate(bboxes) if k not in born]

        # 1) Predict everyone
        self._predict_all(frame_count)

        # 2) Associate remaining detections to predicted tracks
        pred_positions = [t.position() for t in self._tracks]
        cost = self._pairwise_dist(pred_positions, detections)
        matches, unmatched_tracks, unmatched_dets = self._greedy_match(cost, self.distance_threshold)

        # 3) Update matched
        for i, j in matches:
            #if self._tracks[i].last_detection is not None:
                #print(f"Detection based speed for track {self._tracks[i].id}= {np.hypot(detections[j][0]-self._tracks[i].last_detection[0], detections[j][1]-self._tracks[i].last_detection[1]) / self.dt}")
            self._tracks[i].update(detections[j], frame_count)
            self._tracks[i].bboxes_with_framecount.append((bboxes[j], frame_count))
            #print(f"Kalman based speed for track {self._tracks[i].id}= {np.hypot(self._tracks[i].x[2,0], self._tracks[i].x[3,0])}")

        # 4) Age/remove unmatched tracks
        survivors = []
        
        for idx, t in enumerate(self._tracks):
            survivor = True
            if (t.x[0]<30 or t.x[0]>1920) or (idx in unmatched_tracks and t.time_since_update > self.max_age):
                survivor = False
            if survivor: 
                survivors.append(t)
                continue
            if len(t.filtered_states)<2:
                continue
            full_track_distance_px = np.hypot(t.filtered_states[-1][0,0]-t.filtered_states[0][0,0],
                                            t.filtered_states[-1][1,0]-t.filtered_states[0][1,0])
            full_track_distance_m = full_track_distance_px * self.scale_lambda
            if full_track_distance_m > 10:
                avg_v = self.get_track_average_velocity(t)
                self._finished_tracks.append((t, avg_v))  # save before deleting
                self.new_finished_tracks += 1
        self._tracks = survivors

        # 5) For unmatched detections: do NOT create tracks immediately.
        # Store as newborns to wait for a second detection next frame.
        self._newborns = [detections[j] for j in unmatched_dets]
        try:
            self._newborns += [self._newborns[i] for i in unused_newborns]
        except Exception:
            pass  # no previous newborns

    # Convenience: get simple tuples for drawing or logs
    def get_active_states(self) -> List[Tuple[int, Point, Point]]:
        """
        Returns list of (id, position(x,y), velocity(vx,vy)) for tracks that are not stale.
        """
        out = []
        for t in self._tracks:
            if t.time_since_update <= self.max_age:
                out.append((t.id, t.position(), t.velocity()))
        return out
    
    def get_average_velocity_per_track(self) -> List[Tuple[int, float]]:
        """
        Run RTS smoothing for each (active or finished) track and return (track_id, avg_speed).
        """
        results = []
        all_tracks = self._tracks + [t for t, _ in self._finished_tracks]
        for t in all_tracks:
            avg_v = self.get_track_average_velocity(t)
            results.append((t.id,len(t.filtered_states), avg_v))
        return results
    
    def get_track_average_velocity(self, track) -> float:
        """
        Run RTS smoothing on positions-only and compute
        average speed using derivatives of the smoothed positions.
        """
        if len(track.filtered_states) < 3:
            return 0.0
        #print("Speeds before smoothing: ", [int(np.hypot(s[2,0], s[3,0])) for s in track.filtered_states])
        # RTS smoother (position-only)
        xs_s, _ = kalman_smoother(track.filtered_states,
                                track.filtered_covs,
                                track.F,
                                track.Q)

        # Extract smoothed positions
        xs_pos = [float(x[0]) for x in xs_s]
        ys_pos = [float(x[1]) for x in xs_s]

        # Time step from motion model (F matrix)
        dt = track.F[0, 2]   # assuming standard constant-velocity F

        speeds = []
        for i in range(1, len(xs_pos) - 1):
            # Central difference
            vx = (xs_pos[i+1] - xs_pos[i-1]) / (2 * dt)
            vy = (ys_pos[i+1] - ys_pos[i-1]) / (2 * dt)
            speeds.append(np.hypot(vx, vy))

        if not speeds:
            return 0.0
        #print("Speeds after smoothing: ", [int(s) for s in speeds])

        return float(np.mean(speeds))
    
    def retrieve_finished_tracks(self) -> List[Tuple[Track, float]]:
        """
        Returns list of (Track, avg_speed) for tracks that have just finished since last call.
        Resets the internal finished track list.
        """
        self.new_finished_tracks = 0
        return self._finished_tracks[-self.new_finished_tracks:]

