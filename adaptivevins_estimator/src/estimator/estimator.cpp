/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator() : f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while (!accBuf.empty())
        accBuf.pop();
    while (!gyrBuf.empty())
        gyrBuf.pop();
    while (!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl
             << ric[i] << endl
             << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    // Initialize deep front-end
    featureTracker.readIntrinsicParameter(CAM_NAMES);
    featureTracker.initializeExtractorMatcher(0, extractor_weight_global_path, matcher_weight_global_path, MATCHER_THRESHOLD);

    // Initialize classical front-end
    // NOTE: each tracker has its own n_id counter starting at 0.
    // With a hardcoded switch this is safe (only one tracker runs per session).
    // For dynamic switching, assign featureTrackerClassical.n_id an offset (e.g. 1,000,000)
    // to prevent feature ID collisions across the two trackers.
    featureTrackerClassical.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if (!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if (USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if (USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if (restart)
    {
        clearState();
        setParameter();
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;

    // Signal the GPU logger (in main) that the rosbag has started delivering data
    if (inputImageCnt == 1)
    {
        BAG_STARTED.store(true);
        // Truncate aug_log.csv at start of each run
        mkdir("time_consumption", 0755);
        std::ofstream aug_init("time_consumption/aug_log.csv", std::ios::out | std::ios::trunc);
        aug_init << "timestamp,difficulty,n_feat,n_inlier,n_track,n_blur,n_bright,blur_var,brightness,n_classical,augmented,n_aug_features,mean_track_len\n";
    }

    // =========================================================================
    // HYBRID FRONT-END — classical always runs; deep augments sparse regions
    // =========================================================================

    // Difficulty signals — weights and bounds matched to original AdaptiveVINS
    // calibration (commit b7f7033), plus track-length as an additional signal.
    static constexpr int    FEAT_MIN    = 20;
    static constexpr int    FEAT_MAX    = 200;
    static constexpr double TRACK_MIN   = 2.0;
    static constexpr double TRACK_MAX   = 8.0;
    static constexpr double BRIGHT_MIN  = 50.0;   // below = dark / struggling
    static constexpr double BRIGHT_MAX  = 200.0;  // above = well-lit / fine
    static constexpr double BLUR_MIN    = 10.0;   // below = very blurry
    static constexpr double BLUR_MAX    = 200.0;  // above = sharp

    static constexpr double W_FEAT      = 0.30;
    static constexpr double W_INLIER    = 0.30;
    static constexpr double W_TRACK     = 0.15;
    static constexpr double W_BLUR      = 0.15;
    static constexpr double W_BRIGHT    = 0.10;

    int   prev_feat_count   = featureTrackerClassical.last_tracked_count;
    float prev_inlier_ratio = featureTrackerClassical.last_inlier_ratio;

    // Mean track length
    double mean_track_len = 1.0;
    if (!featureTrackerClassical.track_cnt.empty())
    {
        double sum = 0.0;
        for (int c : featureTrackerClassical.track_cnt) sum += c;
        mean_track_len = sum / featureTrackerClassical.track_cnt.size();
    }

    // Image blur: Laplacian variance — low = blurry (motion blur / defocus)
    cv::Mat lap;
    cv::Laplacian(_img, lap, CV_64F);
    cv::Scalar mean_lap, std_lap;
    cv::meanStdDev(lap, mean_lap, std_lap);
    double blur_var = std_lap[0] * std_lap[0];

    // Image brightness: darker = harder (matches original EuRoC calibration)
    cv::Scalar mean_bright = cv::mean(_img);

    auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };
    double n_feat   = clamp01(1.0 - (prev_feat_count  - FEAT_MIN)   / (double)(FEAT_MAX  - FEAT_MIN));
    double n_inlier = clamp01(1.0 - prev_inlier_ratio);
    double n_track  = clamp01(1.0 - (mean_track_len   - TRACK_MIN)  / (TRACK_MAX - TRACK_MIN));
    double n_blur   = clamp01(1.0 - (blur_var          - BLUR_MIN)   / (BLUR_MAX  - BLUR_MIN));
    double n_bright = clamp01(1.0 - (mean_bright[0]    - BRIGHT_MIN) / (BRIGHT_MAX - BRIGHT_MIN));
    double difficulty = W_FEAT * n_feat + W_INLIER * n_inlier + W_TRACK * n_track
                      + W_BLUR * n_blur + W_BRIGHT * n_bright;

    // Hysteresis: once augmentation fires, keep it running until difficulty
    // drops below the low threshold. Prevents rapid on/off switching that
    // breaks deep feature tracks. MIN_DWELL prevents switching back immediately
    // after turn-on (first frame after activation skews signals).
    static constexpr double AUG_THRESH_HIGH = 0.25;  // turn augmentation ON
    static constexpr double AUG_THRESH_LOW  = 0.12;  // turn augmentation OFF
    static constexpr int    MIN_DWELL       = 10;     // min frames per state
    static bool aug_active     = false;
    static int  frames_in_mode = 0;
    ++frames_in_mode;
    if (frames_in_mode > MIN_DWELL)
    {
        if (!aug_active && difficulty > AUG_THRESH_HIGH)
        {
            aug_active     = true;
            frames_in_mode = 0;
        }
        else if (aug_active && difficulty < AUG_THRESH_LOW)
        {
            aug_active     = false;
            frames_in_mode = 0;
        }
    }

    // Deep feature IDs are offset to prevent collision with classical IDs
    static constexpr int DEEP_ID_OFFSET = 1000000;

    // -------------------------------------------------------------------------
    // Step 1 — Always run classical front-end (FAST + optical flow)
    // -------------------------------------------------------------------------
    // Reset deep tracker timings each frame — they are only updated when
    // trackImage_dpl is actually called. Without this reset, stale values from
    // the last augmented frame would inflate timing on every non-augmented frame.
    featureTracker.last_extraction_ms = 0.0;
    featureTracker.last_matching_ms   = 0.0;

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    if (_img1.empty())
        featureFrame = featureTrackerClassical.trackImage(t, _img);
    else
        featureFrame = featureTrackerClassical.trackImage(t, _img, _img1);

    // -------------------------------------------------------------------------
    // Step 2 — Conditionally augment with full SuperPoint + LightGlue pipeline
    // -------------------------------------------------------------------------
    // featureTracker (FeatureTrackerDPL) maintains its own state across frames:
    // it extracts SuperPoint on the current image and matches with LightGlue
    // against the previous frame's keypoints — exactly like the full deep pipeline.
    // We merge its output into featureFrame using an ID offset so there are no
    // collisions with the classical feature IDs.
    int n_classical = (int)featureTrackerClassical.prev_pts.size();

    // Track the last frame index at which deep augmentation fired, so we can
    // detect gaps and reset the deep tracker's stale state.
    // If augmentation skips more than MAX_DEEP_GAP frames, prev_dplpts_descriptors
    // is too old for LightGlue to match reliably — clear prev_pts to force a
    // clean re-initialisation (extraction only, no matches on that one frame).
    static constexpr int MAX_DEEP_GAP = 3;
    static int last_deep_frame = -1;

    if (aug_active && solver_flag == NON_LINEAR)
    {
        if (last_deep_frame >= 0 && (inputImageCnt - last_deep_frame) > MAX_DEEP_GAP)

        {
            featureTracker.prev_pts.clear();
            featureTracker.prev_dplpts_descriptors.clear();
            ROS_WARN("[AdaptiveVINS] deep tracker gap %d frames — forcing re-init",
                     inputImageCnt - last_deep_frame);
        }
        last_deep_frame = inputImageCnt;

        auto deepFrame = (_img1.empty())
            ? featureTracker.trackImage_dpl(t, _img)
            : featureTracker.trackImage_dpl(t, _img, _img1);

        // Cap deep features to avoid overflowing the backend's NUM_OF_F budget.
        // Keep the first MAX_DEEP_FEATURES matches (LightGlue orders by confidence).
        static constexpr int MAX_DEEP_FEATURES = 100;
        int n_deep = 0;
        for (auto &[fid, obs] : deepFrame)
        {
            if (n_deep >= MAX_DEEP_FEATURES) break;
            featureFrame[DEEP_ID_OFFSET + fid] = obs;
            ++n_deep;
        }

        if (n_deep > 0)
        {
            ++aug_frame_count;
            printf("[AdaptiveVINS] #%4d | AUGMENTED +%d SP+LG features | "
                   "classical=%d difficulty=%.3f aug_frames=%d\n",
                   inputImageCnt, n_deep,
                   n_classical, difficulty, aug_frame_count);
        }
    }

    // Write per-frame augmentation log
    {
        int n_aug = (int)featureFrame.size() - (int)featureTrackerClassical.prev_pts.size()
                    + (/* injected this frame */ 0); // approximate: tracked below
        // Count aug features: featureFrame entries with ID >= DEEP_ID_OFFSET
        int n_aug_in_frame = 0;
        for (const auto &kv : featureFrame)
            if (kv.first >= DEEP_ID_OFFSET) n_aug_in_frame++;

        std::ofstream aug_log("time_consumption/aug_log.csv", std::ios::app);
        if (aug_log.is_open())
        {
            aug_log << std::fixed << std::setprecision(6)
                    << t << ","
                    << difficulty << ","
                    << n_feat << "," << n_inlier << "," << n_track << ","
                    << n_blur << "," << n_bright << ","
                    << blur_var << "," << mean_bright[0] << ","
                    << n_classical << ","
                    << (n_aug_in_frame > 0 ? 1 : 0) << ","
                    << n_aug_in_frame << ","
                    << mean_track_len << "\n";
        }
    }

    if (VERBOSE_LOGGING)
    {
        printf("[AdaptiveVINS] #%4d | feat=%3d inlier=%.2f difficulty=%.3f | "
               "augment=%s\n",
               inputImageCnt, prev_feat_count, prev_inlier_ratio, difficulty,
               aug_active ? "YES" : "no");
    }
    // =========================================================================

    if (SHOW_TRACK)
        pubTrackImage(featureTrackerClassical.getTrackImage(), t);

    // If multi-thread mode is on, processMeasurements() is already running in its own thread;
    // just push the new feature frame into featureBuf
    if (MULTIPLE_THREAD)
    {
        if (inputImageCnt % 2 == 0)
        // Only push every other frame to avoid overloading the estimator
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;

        // Single-thread mode: run processMeasurements() directly
        processMeasurements();
    }
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    // printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if (!MULTIPLE_THREAD)
        processMeasurements();
}

/// @brief Get IMU data between t0 and t1
/// @param t0 previous frame timestamp
/// @param t1 current frame timestamp
/// @param accVector
/// @param gyrVector
/// @return
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                               vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if (accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    // printf("get imu from %f %f\n", t0, t1);
    // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if (t1 <= accBuf.back().first) // current frame t1 is within available IMU data range
    {
        while (accBuf.front().first <= t0) // discard IMU data older than the previous frame
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        // buffer now contains only data after t0
        while (accBuf.front().first < t1) // push IMU data between t0 and t1 into the output vectors
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front()); // include one sample at or just past t1
        gyrVector.push_back(gyrBuf.front());
    }
    else // latest IMU data is still before t1; more data may be arriving
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

/// @brief Check whether IMU data is available after timestamp t
/// @param t
/// @return
bool Estimator::IMUAvailable(double t)
{
    if (!accBuf.empty() && t <= accBuf.back().first) // buffer is non-empty and latest IMU sample is past t
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (1)
    {
        // std_msgs::Header des_header;
        // des_header.frame_id = "world";
        // // publish per-frame SuperPoint descriptors
        // pubSuperPointDescriptors(*this, des_header);

        // printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature; // [timestamp, featureFrame]
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        if (!featureBuf.empty())
        {
            feature = featureBuf.front(); // take oldest feature frame from FIFO buffer
            curTime = feature.first + td; // current time = image timestamp + fixed time delay
            while (1)
            {
                if ((!USE_IMU || IMUAvailable(feature.first + td))) // check if sufficient IMU data is available
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (!MULTIPLE_THREAD) // in single-thread mode, return and wait for IMU data
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if (USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector); // collect IMU between previous and current frame

            featureBuf.pop(); // remove the feature we just consumed
            mBuf.unlock();

            if (USE_IMU)
            {
                if (!initFirstPoseFlag) // align initial pose with gravity direction
                    initFirstIMUPose(accVector);
                for (size_t i = 0; i < accVector.size(); i++)
                {
                    /* IMU interval structure:
                       prevTime ---dt---> imu[0] ---dt---> imu[1] ---...--- imu[i-1] |---dt---> curTime --- imu[i] */
                    double dt;
                    if (i == 0)
                        dt = accVector[i].first - prevTime;          // interval from prevTime to first IMU sample
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;       // interval from second-to-last sample to curTime
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second); // mechanization + pre-integration
                }
            }
            mProcess.lock(); // lock to prevent other threads from modifying intrinsics or state

            auto t_backend_start = std::chrono::high_resolution_clock::now();

            processImage(feature.second, feature.first); // main state estimation entry point

            double estimation_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_backend_start).count();
            // Classical always runs; deep augmentation timing is additive when it fires
            double extraction_ms = featureTrackerClassical.last_extraction_ms
                                 + featureTracker.last_extraction_ms;
            double matching_ms   = featureTrackerClassical.last_matching_ms
                                 + featureTracker.last_matching_ms;
            double frontend_ms   = extraction_ms + matching_ms;

            // Write unified per-frame timing to CSV.
            // First write each run: truncate (overwrite). Subsequent writes: append.
            static bool timing_initialized = false;
            static bool dir_created = false;
            if (!dir_created)
            {
                mkdir("time_consumption", 0755);
                dir_created = true;
            }
            {
                std::ios_base::openmode mode = timing_initialized
                                               ? std::ios::app
                                               : (std::ios::out | std::ios::trunc);
                std::ofstream timingFile("time_consumption/timing_log.csv", mode);
                if (timingFile.is_open())
                {
                    if (!timing_initialized)
                    {
                        timingFile << "timestamp,frontend_ms,extraction_ms,matching_ms,estimation_ms\n";
                        timing_initialized = true;
                    }
                    timingFile << std::fixed << std::setprecision(6)
                               << feature.first << ","
                               << frontend_ms   << ","
                               << extraction_ms << ","
                               << matching_ms   << ","
                               << estimation_ms << "\n";
                }
                else
                {
                    std::cerr << "Unable to open timing_log.csv" << std::endl;
                }
            }

            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);

            pubSuperPointDescriptors(*this, header);
            mProcess.unlock();
        }

        if (!MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/// @brief Initialize the first IMU pose by aligning with gravity
/// @param accVector
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    Eigen::Vector3d averAcc(0, 0, 0);
    // Compute average acceleration
    int n = (int)accVector.size();
    for (size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    // Compute rotation from average acceleration direction to [0,0,1]
    Matrix3d R0 = Utility::g2R(averAcc);
    // Extract yaw from R0
    double yaw = Utility::R2ypr(R0).x();
    // Remove yaw component to align initial pose with gravity only
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl
         << Rs[0] << endl;
    // Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

/// @brief IMU pre-integration and mechanization to predict state
/// @param t IMU data timestamp
/// @param dt time interval between this and the previous IMU sample
/// @param linear_acceleration accelerometer measurement at time t
/// @param angular_velocity gyroscope measurement at time t
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // Bootstrap: store first IMU sample as the previous measurement
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // Create pre-integration object for this frame if it doesn't exist yet
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // Only integrate if we are past the first frame
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
        // Store raw IMU data for this frame interval
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        // Mechanization: predict pose and velocity for frame_count
        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    // Shift current measurements to previous
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/// @brief Process image frame and run state estimation
/// @param image
/// @param header
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) // check parallax to determine marginalization; also adds features to feature_manager
    {
        marginalization_flag = MARGIN_OLD; // good parallax: marginalize oldest keyframe
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW; // poor parallax: marginalize second-newest frame
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; // reset temp pre-integration for the next frame

    if (ESTIMATE_EXTRINSIC == 2) // online extrinsic calibration: find feature correspondences and optimize with pre-integration
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                               << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // System not yet initialized; run initialization first
    if (solver_flag == INITIAL)
    {
        // Monocular + IMU initialization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE) // need a full window of frames before initializing
            {
                bool result = false;
                // Run SFM if extrinsic calibration is not needed and enough time has elapsed
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {

                    result = initialStructure();
                    initial_timestamp = header;
                }
                if (result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // Stereo + IMU initialization
        if (STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if (STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if (frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if (frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
    }
    else
    {
        TicToc t_solve;
        if (!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        optimization();
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (!MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }
}

/// @brief Initialize structure via SFM to get initial poses, positions, and landmark locations
/// @return
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // Check IMU observability (sufficient excitation); note this check is informational only and does not gate execution
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // average velocity change per frame
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // mean velocity change across all frames
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1)); // std dev of velocity changes
        // ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            ROS_INFO("IMU excitation not enough!");
            // return false;
        }
    }
    // Global SFM

    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;    // [feature_id, 3D point]
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature) // iterate over all features in the sliding window
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;             // false = not yet triangulated
        tmp_feature.id = it_per_id.feature_id;
        // Collect all observations (normalized image coordinates) for this feature
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            // observation: (frame_index, 2D normalized point)
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    // Find a reference frame in the window with sufficient parallax to the latest frame
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    // Run global SFM using the relative pose between frame l and the last frame
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD; // discard oldest frame and retry
        return false;
    }

    // Solve PnP for all frames not covered by SFM
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i]) // frame is in the window: pose already solved by SFM, mark as keyframe
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix(); // use window frame i as initial guess for PnP
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points) // find SFM-triangulated 3D points corresponding to this frame's features for PnP
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // If PnP fails for any frame, the entire initialization fails
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // After SFM initialization, run visual-inertial alignment
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    // Inertial initialization: solve for gyro bias, gravity direction, and scale to align visual and inertial frames.
    // Uses linear least-squares (Hx=b via Cholesky), not nonlinear optimization.
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // Update state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }
    double s = (x.tail<1>())(0);
    // Re-integrate with updated bias
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // Rescale translations with the recovered scale factor
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    // Update velocities
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

/// @brief Find a historical frame in the window with sufficient matches and parallax to the latest frame, and compute their relative pose
/// @param relative_R
/// @param relative_T
/// @param l
/// @return
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); // get feature correspondences between frame i and the last frame
        if (corres.size() > 20)                              // need at least 20 matches to check parallax
        {
            double sum_parallax = 0;
            double average_parallax;
            // Compute average feature parallax
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // If parallax is large enough, solve relative pose between frame i and WINDOW_SIZE
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if (USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if (USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                         para_Pose[0][3],
                                                         para_Pose[0][4],
                                                         para_Pose[0][5])
                                                 .toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        // TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5])
                                   .toRotationMatrix()
                                   .transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) +
                    origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if (USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                         .normalized()
                         .toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if (USE_IMU)
        td = para_Td[0][0];
}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        // return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        // ROS_INFO(" big translation");
        // return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        // ROS_INFO(" big z translation");
        // return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        // return true;
    }
    return false;
}

/// @brief Run sliding window nonlinear optimization
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    // loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    // loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    // ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if (USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if (!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            // ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            // ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if (USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if (STEREO && it_per_frame.is_stereo)
            {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    // printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    // printf("solver costs: %f \n", t_solver.toc());

    double2vector();
    // printf("frame_count: %d \n", frame_count);

    if (frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if (STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if (imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if (USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    // printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    // printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if (USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if (USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if (USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}
/// @brief Predict current frame feature positions in the next frame using constant-velocity motion
void Estimator::predictPtsInNextFrame()
{
    if (frame_count < 2)
        return;
    // Predict next pose assuming constant velocity
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);                   // pose at frame_count
    getPoseInWorldFrame(frame_count - 1, prevT); // pose at frame_count - 1
    nextT = curT * (prevT.inverse() * curT);     // extrapolate one step forward
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if (it_per_id.estimated_depth > 0) // only use features with a valid depth estimate
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            // Only predict for features observed at least twice and last seen in the current frame
            if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;                                                     // depth in the camera frame of the first observation
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];            // camera -> body frame (at first observation)
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];                                     // body -> world frame
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3)); // world -> predicted next body frame
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);                                 // body -> camera frame
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts); // pass predicted pixel locations to feature tracker
    // printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    // return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                     Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if (STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if (ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);
    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while (!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
