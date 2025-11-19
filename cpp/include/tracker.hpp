#pragma once

#include <Eigen/Dense>
#include <vector>
#include <atomic>
#include <utility>   

using Vec4   = Eigen::Matrix<double, 4, 1>;
using Mat4   = Eigen::Matrix<double, 4, 4>;
using Mat2   = Eigen::Matrix<double, 2, 2>;
using Mat24  = Eigen::Matrix<double, 2, 4>;
using Mat42  = Eigen::Matrix<double, 4, 2>;
using Point  = Eigen::Vector2d;

// Motion / process model
void motion_matrices(double dt, Mat4& F, Mat24& H);
Mat4 q_cv(double dt, double sigma_a);

// Rauch–Tung–Striebel smoother
void kalman_smoother(
    const std::vector<Vec4>& xs,
    const std::vector<Mat4>& Ps,
    const Mat4& F,
    const Mat4& Q,
    std::vector<Vec4>& xs_smooth,
    std::vector<Mat4>& Ps_smooth
);

struct Track {
    int   id;
    Vec4  x;         // state
    Mat4  P;         // covariance
    Mat4  F;
    Mat24 H;
    Mat4  Q;
    Mat2  R;
    int   hits            = 0;
    int   time_since_update = 0;
    int   age             = 0;
    Point last_detection;

    std::vector<Vec4> filtered_states;
    std::vector<Mat4> filtered_covs;

    struct HistoryEntry {
        int   frame;
        Point pos;
        double vel_x;
    };
    std::vector<HistoryEntry> history;

    static int next_id();

    static Track from_detection(
        const Point& pt,
        double dt,
        double sigma_a,
        double sigma_z
    );

    static Track from_two_detections(
        const Point& pt0,
        const Point& pt1,
        double dt,
        double sigma_a,
        double sigma_z
    );

    void predict(int frame_count);
    void update(const Point& z_pt, int frame_count);
    Point position() const;
    Point velocity() const;
};

struct FinishedTrack {
    Track  track;
    double avg_speed;
};

struct ActiveState {
    int   id;
    Point position;
    Point velocity;
};

struct AverageVelocity {
    int    id;
    int    num_states;
    double avg_speed;
};

class Tracker {
public:
    Tracker(double dt = 1.0 / 30.0,
            double scale_lambda = 1.0,
            double sigma_a = 5.0,
            double sigma_z = 3.0,
            double distance_threshold = 100.0,
            int max_age = 10,
            int min_hits = 2);

    // Equivalent to Python's `tracks` property
    std::vector<Track> tracks() const;

    const std::vector<Track>& tracks_ref() const { return tracks_; }

    // Main per-frame update
    void update(std::vector<Point> detections, int frame_count);

    // Convenience getters
    std::vector<ActiveState>      get_active_states() const;
    std::vector<AverageVelocity>  get_average_velocity_per_track() const;
    double                        get_track_average_velocity(const Track& track) const;
    std::vector<FinishedTrack>    retrieve_finished_tracks();

    int new_finished_tracks() const { return new_finished_tracks_; }

private:
    double dt_;
    double scale_lambda_;
    double sigma_a_;
    double sigma_z_;
    double distance_threshold_;
    int    max_age_;
    int    min_hits_;

    std::vector<Track>         tracks_;
    std::vector<Point>         newborns_;
    std::vector<FinishedTrack> finished_tracks_;
    int                        new_finished_tracks_ = 0;

    void predict_all(int frame_count);

    // Small-N greedy matcher, stack-based, no Eigen::MatrixXd / sort
    static void greedy_match_small(const std::vector<Point>& A,
                                   const std::vector<Point>& B,
                                   double threshold,
                                   std::vector<std::pair<int,int>>& matches,
                                   std::vector<int>& unmatched_A,
                                   std::vector<int>& unmatched_B);
};