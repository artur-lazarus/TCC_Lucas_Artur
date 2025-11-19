#include "tracker.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

void motion_matrices(double dt, Mat4& F, Mat24& H) {
    F <<
        1, 0, dt, 0,
        0, 1, 0,  dt,
        0, 0, 1,  0,
        0, 0, 0,  1;

    H <<
        1, 0, 0, 0,
        0, 1, 0, 0;
}

Mat4 q_cv(double dt, double sigma_a) {
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt2 * dt2;
    double q   = sigma_a * sigma_a;

    Mat4 Q;
    Q <<
        dt4/4, 0,      dt3/2, 0,
        0,     dt4/4,  0,     dt3/2,
        dt3/2, 0,      dt2,   0,
        0,     dt3/2,  0,     dt2;
    return q * Q;
}

void kalman_smoother(
    const std::vector<Vec4>& xs,
    const std::vector<Mat4>& Ps,
    const Mat4& F,
    const Mat4& Q,
    std::vector<Vec4>& xs_smooth,
    std::vector<Mat4>& Ps_smooth
) {
    const int N = static_cast<int>(xs.size());
    if (N == 0) {
        xs_smooth.clear();
        Ps_smooth.clear();
        return;
    }

    xs_smooth.resize(N);
    Ps_smooth.resize(N);

    xs_smooth[N - 1] = xs[N - 1];
    Ps_smooth[N - 1] = Ps[N - 1];

    for (int k = N - 2; k >= 0; --k) {
        Mat4 P_pred = F * Ps[k] * F.transpose() + Q;
        Mat4 G      = Ps[k] * F.transpose() * P_pred.inverse();

        xs_smooth[k] =
            xs[k] + G * (xs_smooth[k + 1] - F * xs[k]);
        Ps_smooth[k] =
            Ps[k] + G * (Ps_smooth[k + 1] - P_pred) * G.transpose();

        xs_smooth[k](2) = xs[k](2);
        xs_smooth[k](3) = xs[k](3);
    }
}

int Track::next_id() {
    static std::atomic<int> counter{0};
    return ++counter;
}

Track Track::from_detection(
    const Point& pt,
    double dt,
    double sigma_a,
    double sigma_z
) {
    Track t;
    t.id = next_id();

    motion_matrices(dt, t.F, t.H);
    t.Q = q_cv(dt, sigma_a);
    t.R = Mat2::Identity() * (sigma_z * sigma_z);

    t.x.setZero();
    t.x(0) = pt.x();
    t.x(1) = pt.y();

    t.P = Mat4::Zero();
    t.P.diagonal() << 5.0, 5.0, 100.0, 100.0;

    return t;
}

Track Track::from_two_detections(
    const Point& pt0,
    const Point& pt1,
    double dt,
    double sigma_a,
    double sigma_z
) {
    Track t;
    t.id = next_id();

    motion_matrices(dt, t.F, t.H);
    t.Q = q_cv(dt, sigma_a);
    t.R = Mat2::Identity() * (sigma_z * sigma_z);

    double vx = (pt1.x() - pt0.x()) / dt;
    double vy = (pt1.y() - pt0.y()) / dt;

    t.x << pt1.x(), pt1.y(), vx, vy;

    double var_pos = sigma_z * sigma_z;
    double var_v   = 0.1;

    t.P = Mat4::Zero();
    t.P.diagonal() << var_pos, var_pos, var_v, var_v;

    t.hits             = 2;
    t.time_since_update = 0;
    t.filtered_states.push_back(t.x);
    return t;
}

void Track::predict(int frame_count) {
    x = F * x;
    P = F * P * F.transpose() + Q;

    ++age;
    ++time_since_update;

    Point  pos   = position();
    double vel_x = x(2);
    history.push_back({frame_count, pos, vel_x});
}

void Track::update(const Point& z_pt, int frame_count) {
    last_detection = z_pt;
    Eigen::Vector2d z = z_pt;
    Eigen::Vector2d y = z - H * x;
    Mat2  S = H * P * H.transpose() + R;
    Mat42 K = P * H.transpose() * S.inverse();

    x += K * y;
    Mat4 I = Mat4::Identity();
    P = (I - K * H) * P;

    ++hits;
    time_since_update = 0;

    filtered_states.push_back(x);
    filtered_covs.push_back(P);
    history.push_back({frame_count, position(), x(2)});
}

Point Track::position() const {
    return Point{x(0), x(1)};
}

Point Track::velocity() const {
    return Point{x(2), x(3)};
}

// ========================= Tracker implementation =========================

Tracker::Tracker(double dt,
                double scale_lambda,
                 double sigma_a,
                 double sigma_z,
                 double distance_threshold,
                 int max_age,
                 int min_hits)
    : dt_(dt),
      scale_lambda_(scale_lambda),
      sigma_a_(sigma_a),
      sigma_z_(sigma_z),
      distance_threshold_(distance_threshold),
      max_age_(max_age),
      min_hits_(min_hits) {}

std::vector<Track> Tracker::tracks() const {
    std::vector<Track> out;
    out.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.hits >= min_hits_ || t.time_since_update == 0) {
            out.push_back(t);
        }
    }
    return out;
}

void Tracker::predict_all(int frame_count) {
    for (auto& t : tracks_) {
        t.predict(frame_count);
    }
}

void Tracker::greedy_match_small(const std::vector<Point>& A,
                                 const std::vector<Point>& B,
                                 double threshold,
                                 std::vector<std::pair<int,int>>& matches,
                                 std::vector<int>& unmatched_A,
                                 std::vector<int>& unmatched_B)
{
    const int na = static_cast<int>(A.size());
    const int nb = static_cast<int>(B.size());

    matches.clear();
    unmatched_A.clear();
    unmatched_B.clear();

    if (na == 0 || nb == 0) {
        for (int i = 0; i < na; ++i) unmatched_A.push_back(i);
        for (int j = 0; j < nb; ++j) unmatched_B.push_back(j);
        return;
    }

    constexpr int MAXN = 30;  // safe for your ≤15 constraint
    
    // CRITICAL: Check bounds to prevent stack buffer overflow
    if (na > MAXN || nb > MAXN) {
        std::cerr << "WARNING: greedy_match_small exceeded MAXN=" << MAXN 
                  << " (na=" << na << ", nb=" << nb << "). Treating all as unmatched.\n";
        for (int i = 0; i < na; ++i) unmatched_A.push_back(i);
        for (int j = 0; j < nb; ++j) unmatched_B.push_back(j);
        return;
    }
    
    double cost[MAXN][MAXN];
    bool used_A[MAXN] = {false};
    bool used_B[MAXN] = {false};

    // Pairwise distances
    for (int i = 0; i < na; ++i) {
        for (int j = 0; j < nb; ++j) {
            double dx = A[i].x() - B[j].x();
            double dy = A[i].y() - B[j].y();
            cost[i][j] = std::sqrt(dx * dx + dy * dy);
        }
    }

    // Greedy global-min matching under threshold
    while (true) {
        double best = threshold;
        int best_i = -1;
        int best_j = -1;

        for (int i = 0; i < na; ++i) {
            if (used_A[i]) continue;
            for (int j = 0; j < nb; ++j) {
                if (used_B[j]) continue;
                double c = cost[i][j];
                if (c <= best) {
                    best = c;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i < 0 || best_j < 0) {
            break;
        }

        matches.emplace_back(best_i, best_j);
        used_A[best_i] = true;
        used_B[best_j] = true;
    }

    for (int i = 0; i < na; ++i) {
        if (!used_A[i]) unmatched_A.push_back(i);
    }
    for (int j = 0; j < nb; ++j) {
        if (!used_B[j]) unmatched_B.push_back(j);
    }
}

void Tracker::update(std::vector<Point> detections, int frame_count) {
    // --- Step 0: pair newborns with detections ---
    if (!newborns_.empty() && !detections.empty()) {
        std::vector<std::pair<int,int>> pairs;
        std::vector<int> unused_newborns;
        std::vector<int> unused_dets;

        greedy_match_small(newborns_, detections,
                           distance_threshold_,
                           pairs, unused_newborns, unused_dets);

        // Create seeded tracks for paired newborns
        std::vector<int> born_det_indices;
        born_det_indices.reserve(pairs.size());

        for (const auto& pr : pairs) {
            int i_nb  = pr.first;
            int j_det = pr.second;
            const Point& pt0 = newborns_[i_nb];
            const Point& pt1 = detections[j_det];
            tracks_.push_back(
                Track::from_two_detections(pt0, pt1, dt_, sigma_a_, sigma_z_)
            );
            born_det_indices.push_back(j_det);
        }

        // Keep only newborns that didn't find a mate
        std::vector<Point> kept_newborns;
        kept_newborns.reserve(unused_newborns.size());
        for (int idx : unused_newborns) {
            kept_newborns.push_back(newborns_[idx]);
        }
        newborns_.swap(kept_newborns);

        // Remove detections used to create tracks
        if (!born_det_indices.empty()) {
            std::sort(born_det_indices.begin(), born_det_indices.end());
            std::vector<Point> remaining;
            remaining.reserve(detections.size() - born_det_indices.size());
            int bi = 0;
            for (int k = 0; k < static_cast<int>(detections.size()); ++k) {
                if (bi < static_cast<int>(born_det_indices.size()) &&
                    k == born_det_indices[bi]) {
                    ++bi;
                } else {
                    remaining.push_back(detections[k]);
                }
            }
            detections.swap(remaining);
        }
    }

    // --- Step 1: predict all tracks ---
    predict_all(frame_count);

    // --- Step 2: associate remaining detections to predicted tracks ---
    std::vector<Point> pred_positions;
    pred_positions.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        pred_positions.push_back(t.position());
    }

    std::vector<std::pair<int,int>> matches;
    std::vector<int> unmatched_track_idx;
    std::vector<int> unmatched_det_idx;

    greedy_match_small(pred_positions, detections,
                       distance_threshold_,
                       matches, unmatched_track_idx, unmatched_det_idx);

    // --- Step 3: update matched tracks ---
    for (const auto& m : matches) {
        int ti = m.first;
        int dj = m.second;
        tracks_[ti].update(detections[dj], frame_count);
    }

    // --- Step 4: age/remove unmatched tracks ---
    std::vector<Track> survivors;
    survivors.reserve(tracks_.size());
    new_finished_tracks_ = 0;

    // Build mask for unmatched tracks (O(n) instead of find in loop)
    std::vector<char> is_unmatched(tracks_.size(), 0);
    for (int idx : unmatched_track_idx) {
        if (idx >= 0 && idx < (int)is_unmatched.size()) {
            is_unmatched[idx] = 1;
        }
    }

    for (int idx = 0; idx < static_cast<int>(tracks_.size()); ++idx) {
        Track& t = tracks_[idx];
        bool survivor = true;
        if ((t.x(0)<30 || t.x(0)>1920)||(is_unmatched[idx] && t.time_since_update > max_age_ )) {
            survivor = false;
        }
        if (survivor) {
            survivors.push_back(t);
            continue;
        }
        if (t.filtered_states.size() < 2) {
            continue;
        }
        double full_track_dist_px = std::hypot(
            t.filtered_states.back()(0) - t.filtered_states.front()(0),
            t.filtered_states.back()(1) - t.filtered_states.front()(1)
        );
        double full_track_dist_m = full_track_dist_px * scale_lambda_;
        if (full_track_dist_m > 10.0) {
            double avg_v = get_track_average_velocity(t);
                finished_tracks_.push_back(FinishedTrack{t, avg_v});
                ++new_finished_tracks_;
        }
    }
    tracks_.swap(survivors);

    // --- Step 5: unmatched detections become newborns ---
    std::vector<Point> new_newborns;
    new_newborns.reserve(unmatched_det_idx.size());
    for (int j : unmatched_det_idx) {
        new_newborns.push_back(detections[j]);
    }
    newborns_.insert(newborns_.end(), new_newborns.begin(), new_newborns.end());
}

std::vector<ActiveState> Tracker::get_active_states() const {
    std::vector<ActiveState> out;
    out.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.time_since_update <= max_age_) {
            out.push_back(ActiveState{t.id, t.position(), t.velocity()});
        }
    }
    return out;
}

std::vector<AverageVelocity> Tracker::get_average_velocity_per_track() const {
    std::vector<AverageVelocity> results;

    std::vector<const Track*> all_tracks;
    all_tracks.reserve(tracks_.size() + finished_tracks_.size());
    for (const auto& t : tracks_) {
        all_tracks.push_back(&t);
    }
    for (const auto& ft : finished_tracks_) {
        all_tracks.push_back(&ft.track);
    }

    for (const Track* t : all_tracks) {
        if (t->filtered_states.size() < 3) {
            continue;
        }

        double avg_v = get_track_average_velocity(*t);
        results.push_back(AverageVelocity{
            t->id,
            static_cast<int>(t->filtered_states.size()),
            avg_v
        });
    }

    return results;
}

double Tracker::get_track_average_velocity(const Track& track) const {
    if (track.filtered_states.size() < 3) {
        return 0.0;
    }

    std::vector<Vec4> xs_s;
    std::vector<Mat4> Ps_s;
    kalman_smoother(track.filtered_states, track.filtered_covs,
                    track.F, track.Q,
                    xs_s, Ps_s);
    std::vector<double> xs_pos;
    std::vector<double> ys_pos;
    for (const auto& x : xs_s) {
        xs_pos.push_back(x(0));
        ys_pos.push_back(x(1));
    }
    
    std::vector<double> velocities;
    for (int i = 1; i < static_cast<int>(xs_pos.size())-1; ++i) {
        double dx = (xs_pos[i+1] - xs_pos[i - 1])/2;
        double dy = (ys_pos[i+1] - ys_pos[i - 1])/2;
        double v  = std::hypot(dx, dy) / dt_;
        velocities.push_back(v);
    }

    if (velocities.empty()) {
        return 0.0;
    }
    return std::accumulate(velocities.begin(), velocities.end(), 0.0) / velocities.size();
}

std::vector<FinishedTrack> Tracker::retrieve_finished_tracks() {
    std::vector<FinishedTrack> out;
    if (new_finished_tracks_ <= 0) {
        new_finished_tracks_ = 0;
        return out;
    }

    int total = static_cast<int>(finished_tracks_.size());
    int start = std::max(0, total - new_finished_tracks_);

    out.reserve(new_finished_tracks_);
    for (int i = start; i < total; ++i) {
        out.push_back(finished_tracks_[i]);
    }

    new_finished_tracks_ = 0;
    return out;
}
