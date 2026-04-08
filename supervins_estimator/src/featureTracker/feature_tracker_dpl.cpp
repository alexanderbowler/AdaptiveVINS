/*
 * @Author: Hongkun Luo
 * @Date: 2024-11-05 21:16:57
 * @LastEditors: luohongk luohongkun@whu.edu.cn
 * @Description: 
 * 
 * Hongkun Luo
 */
/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "feature_tracker_dpl.h"
void FeatureTrackerDPL::initializeExtractorMatcher(int extractor_type_, string &extractor_weight_path, string &matcher_weight_path, float matcher_threshold = 0.5)
{
    extractor_type = extractor_type_;
    FeatureExtractorDPL = std::make_shared<Extractor_DPL>();
    FeatureExtractorDPL->initialize(extractor_weight_path, extractor_type_);

    FeatureMatcherDPL = std::make_shared<Matcher_DPL>();
    FeatureMatcherDPL->initialize(matcher_weight_path, extractor_type_, matcher_threshold);

    if (extractor_type_ == SUPERPOINT)
    {
        descriptor_size = SUPERPOINT_SIZE;
    }
    else if (extractor_type_ == DISK)
    {
        descriptor_size = DISK_SIZE;
    }
}

cv::Mat FeatureTrackerDPL::Extractor_PreProcess(const cv::Mat &Image, float &scale)
{
    float temp_scale = scale;
    cv::Mat tempImage = Image.clone();
    // std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;

    std::string fn = "max";
    std::string interp = "area";
    cv::Mat resize_img = ResizeImage(tempImage, IMAGE_SIZE_DPL, scale, fn, interp);
    cv::Mat resultImage = NormalizeImage(resize_img);
    // if (cfg.extractorType == "superpoint")
    //{
    // std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
    // resultImage = RGB2Grayscale(resultImage);
    //}
    // std::cout << "[INFO] Scale from " << temp_scale << " to " << scale << std::endl;

    return resultImage;
}

void FeatureTrackerDPL::goodFeaturesToTrack_dpl(cv::Mat img, vector<cv::Point2f> &pts, vector<pair<cv::Point2f, vector<float>>> &dplpts_descriptors, int max_num, double extractor_threshold, int radius, cv::Mat &mask)
{
    cv::Mat im = img.clone();
    cv::Mat im_preprocessed = Extractor_PreProcess(im, FeatureExtractorDPL->scale);
    std::pair<std::vector<cv::Point2f>, float *> result_dplpts_descriptors = FeatureExtractorDPL->extract_featurepoints(im_preprocessed);
    int n = result_dplpts_descriptors.first.size();
    for (int i = 0; i < n; i++)
    {
        cv::Point2f dplpt = result_dplpts_descriptors.first[i];
        cv::Point2f pt = cv::Point2f((dplpt.x + 0.5) / FeatureExtractorDPL->scale - 0.5, (dplpt.y + 0.5) / FeatureExtractorDPL->scale - 0.5);
        if (!inBorder(pt))
            continue;
        if (mask.at<uchar>(pt) == 255)
        {
            std::vector<float> descriptor(result_dplpts_descriptors.second + i * descriptor_size, result_dplpts_descriptors.second + (i + 1) * descriptor_size);
            pts.push_back(pt);
            dplpts_descriptors.push_back(make_pair(dplpt, descriptor));
        }
    }
}

void FeatureTrackerDPL::extract_features_dpl(cv::Mat img, vector<cv::Point2f> &pts, vector<pair<cv::Point2f, vector<float>>> &dplpts_descriptors)
{
    cv::Mat im = img.clone();
    cv::Mat im_preprocessed = Extractor_PreProcess(im, FeatureExtractorDPL->scale);
    std::pair<std::vector<cv::Point2f>, float *> result_dplpts_descriptors = FeatureExtractorDPL->extract_featurepoints(im_preprocessed);

    int n = result_dplpts_descriptors.first.size();
    for (int i = 0; i < n; i++)
    {

        cv::Point2f dplpt = result_dplpts_descriptors.first[i];
        cv::Point2f pt = cv::Point2f((dplpt.x + 0.5) / FeatureExtractorDPL->scale - 0.5, (dplpt.y + 0.5) / FeatureExtractorDPL->scale - 0.5);
        if (!inBorder(pt))
            continue;
        std::vector<float> descriptor(result_dplpts_descriptors.second + i * descriptor_size, result_dplpts_descriptors.second + (i + 1) * descriptor_size);
        pts.push_back(pt);
        dplpts_descriptors.push_back(make_pair(dplpt, descriptor));
    }
    // std::cout<<dplpts_descriptors[0].first<<endl;
    // std::cout<<dplpts_descriptors[0].second<<endl;
}

bool FeatureTrackerDPL::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

void FeatureTrackerDPL::reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTrackerDPL::reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTrackerDPL::FeatureTrackerDPL()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

void FeatureTrackerDPL::match_features_dpl(cv::Mat prev_img_, cv::Mat cur_img_, vector<pair<cv::Point2f, vector<float>>> &prev_dplpts_descriptors_, vector<pair<cv::Point2f, vector<float>>> &cur_dplpts_descriptors_, vector<pair<int, int>> &result_matches,double &ransacReprojThreshold)
{
    // Number of features in previous and current frames
    int n_pre = prev_dplpts_descriptors_.size();
    int n_cur = cur_dplpts_descriptors_.size();
    // debug
    // cout << "prev_dples size = " << n_pre << "cur_dpls size=" << n_cur << endl;
    vector<cv::Point2f> prev_dplpts, cur_dplpts;
    prev_dplpts.reserve(n_pre);
    cur_dplpts.reserve(n_cur);
    float prev_descriptors[n_pre * descriptor_size];
    float cur_descriptors[n_cur * descriptor_size];

    for (int i = 0; i < n_pre; i++)
    {
        prev_dplpts.push_back(prev_dplpts_descriptors_[i].first);
        vector<float> desc = prev_dplpts_descriptors_[i].second;
        int idx = i * descriptor_size;
        for (float desc_value : desc)
        {
            prev_descriptors[idx] = desc_value;
            idx++;
        }
    }

    for (int i = 0; i < n_cur; i++)
    {
        cur_dplpts.push_back(cur_dplpts_descriptors_[i].first);
        vector<float> desc = cur_dplpts_descriptors_[i].second;
        int idx = i * descriptor_size;
        for (float desc_value : desc)
        {
            cur_descriptors[idx] = desc_value;
            idx++;
        }
    }

    // Normalize keypoints for the matcher
    vector<cv::Point2f> prev_dplpts_normalized = FeatureMatcherDPL->pre_process(prev_dplpts, prev_img_.rows, prev_img_.cols);
    vector<cv::Point2f> cur_dplpts_normalized = FeatureMatcherDPL->pre_process(cur_dplpts, cur_img_.rows, cur_img_.cols);

    // Run deep learning feature matching
    vector<pair<int, int>> tem_matches;
    tem_matches = FeatureMatcherDPL->match_featurepoints(prev_dplpts_normalized, cur_dplpts_normalized, prev_descriptors, cur_descriptors);

    // std::cout<<"tem_matches size = "<<tem_matches.size()<<std::endl;

    // Extract matched point pairs for RANSAC
    vector<cv::Point2f> points1, points2;
    for (const auto &match : tem_matches)
    {
        points1.push_back(prev_dplpts_normalized[match.first]);
        points2.push_back(cur_dplpts_normalized[match.second]);
    }

    // RANSAC fundamental matrix estimation to filter outliers
    std::vector<uchar> inliersMask;
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, ransacReprojThreshold, 0.99, inliersMask);

    // Keep only inlier matches
    std::vector<pair<int, int>> inlierMatches;
    for (int i = 0; i < inliersMask.size(); ++i)
    {
        if (inliersMask[i])
        {
            result_matches.push_back(tem_matches[i]);
        }
    }
}

void FeatureTrackerDPL::match_with_predictions_dpl(cv::Mat prev_img_, cv::Mat cur_img_, vector<pair<cv::Point2f, vector<float>>> &prev_dplpts_descriptors_, vector<pair<cv::Point2f, vector<float>>> &cur_dplpts_descriptors_, vector<cv::Point2f> &predict_pts_, vector<cv::Point2f> &cur_pts_, vector<pair<int, int>> &result_matches,double &ransacReprojThreshold)
{
    int n_pre = prev_dplpts_descriptors_.size();
    int n_cur = cur_dplpts_descriptors_.size();
    vector<cv::Point2f> prev_dplpts, cur_dplpts;
    prev_dplpts.reserve(n_pre);
    cur_dplpts.reserve(n_cur);
    float prev_descriptors[n_pre * descriptor_size];
    float cur_descriptors[n_cur * descriptor_size];

    for (int i = 0; i < n_pre; i++)
    {
        prev_dplpts.push_back(prev_dplpts_descriptors_[i].first);
        vector<float> desc = prev_dplpts_descriptors_[i].second;
        int idx = i * descriptor_size;
        for (float desc_value : desc)
        {
            prev_descriptors[idx] = desc_value;
            idx++;
        }
    }

    for (int i = 0; i < n_cur; i++)
    {
        cur_dplpts.push_back(cur_dplpts_descriptors_[i].first);
        vector<float> desc = cur_dplpts_descriptors_[i].second;
        int idx = i * descriptor_size;
        for (float desc_value : desc)
        {
            cur_descriptors[idx] = desc_value;
            idx++;
        }
    }

    vector<cv::Point2f> prev_dplpts_normalized = FeatureMatcherDPL->pre_process(prev_dplpts, prev_img_.rows, prev_img_.cols);
    vector<cv::Point2f> cur_dplpts_normalized = FeatureMatcherDPL->pre_process(cur_dplpts, cur_img_.rows, cur_img_.cols);

    vector<pair<int, int>> matches = FeatureMatcherDPL->match_featurepoints(prev_dplpts_normalized, cur_dplpts_normalized, prev_descriptors, cur_descriptors);
    // RANSAC fundamental matrix estimation
    std::vector<uchar> inliersMask;
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(prev_dplpts_normalized, cur_dplpts_normalized, cv::FM_RANSAC, ransacReprojThreshold, 0.99, inliersMask);

    // Keep only inlier matches
    std::vector<cv::Point2f> inlierPrevPts, inlierCurPts;
    std::vector<pair<int, int>> inlierMatches;
    for (int i = 0; i < inliersMask.size(); ++i)
    {
        if (inliersMask[i])
        {
            inlierPrevPts.push_back(prev_dplpts_normalized[i]);
            inlierCurPts.push_back(cur_dplpts_normalized[i]);
            result_matches.push_back(matches[i]);
        }
    }
}

/// @brief Set feature extraction mask and sort features by track count (longest-tracked first)
void FeatureTrackerDPL::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    // Store current frame features as (track_count, (point, id)) for sorting
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    // Sort by track count descending
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    // Restore sorted features and apply minimum-distance mask
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);

            // Mark a MIN_DIST-radius exclusion zone around this point in the mask
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

cv::Mat FeatureTrackerDPL::setMask_dpl(vector<cv::Point2f> &matched_points, int radius = MIN_DIST)
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    for (cv::Point2f &pt : matched_points)
    {
        if (mask.at<uchar>(pt) == 255)
        {
            // Mark a radius-sized exclusion zone around this matched point
            cv::circle(mask, pt, radius, 0, -1);
        }
    }

    return mask.clone();
}

double FeatureTrackerDPL::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}



/// @brief Stereo inter-frame feature tracking
/// @param _cur_time
/// @param _img left image
/// @param _img1 right image
/// @return feature map for the current frame

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTrackerDPL::trackImage_dpl(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;

    // Clear current frame feature containers
    cur_pts.clear();
    cur_dplpts_descriptors.clear();

    // Extract feature points and descriptors using deep learning
    auto t_extraction_start = std::chrono::high_resolution_clock::now();
    extract_features_dpl(cur_img, cur_pts, cur_dplpts_descriptors);
    last_extraction_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_extraction_start).count();

    // Working containers for this frame
    set<int> cur_matched_indices;
    vector<cv::Point2f> temp_prev_pts;
    vector<pair<cv::Point2f, vector<float>>> temp_prev_dplpts_descriptors;
    vector<cv::Point2f> temp_cur_pts;
    vector<pair<cv::Point2f, vector<float>>> temp_cur_dplpts_descriptors;
    vector<int> temp_ids;
    vector<int> temp_track_cnt;

    cout << cur_pts.size() << " points extracted in current frame originally" << endl;

    // Cannot track without features from the previous frame
    ROS_INFO_STREAM("prev_pts.size() = " << prev_pts.size());

    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<pair<int, int>> matches;

        std::cout << "hasPrediction = " << hasPrediction << std::endl;
        if (hasPrediction)
        {
            // Match features using IMU-predicted point locations
            auto t_matching_start = std::chrono::high_resolution_clock::now();
            match_with_predictions_dpl(prev_img, cur_img, prev_dplpts_descriptors, cur_dplpts_descriptors, predict_pts, cur_pts, matches, ransacReprojThreshold);
            last_matching_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_matching_start).count();
        }
        else
        {
            // Match features without prediction
            auto t_matching_start = std::chrono::high_resolution_clock::now();
            match_features_dpl(prev_img, cur_img, prev_dplpts_descriptors, cur_dplpts_descriptors, matches, ransacReprojThreshold);
            last_matching_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_matching_start).count();

            ROS_INFO_STREAM("matches.size() = " << matches.size());
        }

        // the number of matched points
        int n_matches = matches.size();
        // cout << n_matches <<" points matched in current frame" <<endl;

        // first, save the successfully matched points in the temporary vector
        for (int i = 0; i < n_matches; i++)
        {
            pair<int, int> match = matches[i];
            temp_prev_pts.push_back(prev_pts[match.first]);
            temp_prev_dplpts_descriptors.push_back(prev_dplpts_descriptors[match.first]);
            temp_cur_pts.push_back(cur_pts[match.second]);
            temp_cur_dplpts_descriptors.push_back(cur_dplpts_descriptors[match.second]);
            temp_ids.push_back(ids[match.first]);
            temp_track_cnt.push_back(track_cnt[match.first] + 1);
            // record the indices of matched points
            cur_matched_indices.insert(match.second);
        }
    }

    // to avoid point gathering, create a mask depending on matched points in the current image
    cv::Mat mask_for_dpl = setMask_dpl(temp_cur_pts, MIN_DIST);
    
    // add new feature points but filter the ones which are close to matched points via the mask
    for (int i = 0; i < cur_pts.size(); i++)
    {
        if (!cur_matched_indices.count(i))
        {
            if (mask_for_dpl.at<uchar>(cur_pts[i]) == 255) // TO DO: annote this line can alleviate the problem of system crashing
            {
                temp_cur_pts.push_back(cur_pts[i]);
                temp_cur_dplpts_descriptors.push_back(cur_dplpts_descriptors[i]);
                temp_track_cnt.push_back(1);
                temp_ids.push_back(n_id++);
            }
        }
    }

    prev_pts = temp_prev_pts;
    prev_dplpts_descriptors = temp_prev_dplpts_descriptors;
    cur_pts.swap(temp_cur_pts);
    cur_dplpts_descriptors.swap(temp_cur_dplpts_descriptors);
    track_cnt = temp_track_cnt;
    ids = temp_ids;

    // cout << cur_pts.size()<<" points reserved in current frame finally" <<endl;

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
    if (!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        cur_dplpts_right_descriptors.clear();

        vector<cv::Point2f> temp_cur_right_pts;

        if (!cur_pts.empty())
        {
            extract_features_dpl(rightImg, cur_right_pts, cur_dplpts_right_descriptors);
            vector<pair<int, int>> right_matches;

            // double ransac_thresh = 0.5;
            match_features_dpl(cur_img, rightImg, cur_dplpts_descriptors, cur_dplpts_right_descriptors, right_matches,ransacReprojThreshold);
            int n_matches_right = right_matches.size();
            for (int i = 0; i < n_matches_right; i++)
            {
                pair<int, int> right_match = right_matches[i];
                temp_cur_right_pts.push_back(cur_right_pts[right_match.second]);
                ids_right.push_back(ids[right_match.first]);
            }
            cur_right_pts = temp_cur_right_pts;
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);                                              // undistort and back-project to normalized plane
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map); // compute right feature velocities on normalized plane
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    if (SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    // Tracking complete; shift current frame data to previous frame
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map; // map of (feature_id, normalized point) for previous frame
    prev_dplpts_descriptors = cur_dplpts_descriptors;
    prev_time = cur_time;
    hasPrediction = false; // reset prediction flag

    prevLeftPtsMap.clear(); // rebuild left feature map with current frame data
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    ROS_INFO_STREAM("start save ");
    

    // Build feature frame: key=feature_id, value=[(camera_id, [norm_x, norm_y, 1, px_u, px_v, vel_x, vel_y])]
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x = cur_un_pts[i].x; // normalized plane x
        double y = cur_un_pts[i].y; // normalized plane y
        double z = 1;
        double p_u = cur_pts[i].x;  // pixel x
        double p_v = cur_pts[i].y;  // pixel y
        int camera_id = 0;           // 0 = left (primary) camera
        double velocity_x = pts_velocity[i].x; // velocity on normalized plane
        double velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
    // Same for right camera (camera_id = 1)
    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x = cur_un_right_pts[i].x;
            double y = cur_un_right_pts[i].y;
            double z = 1;
            double p_u = cur_right_pts[i].x;
            double p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x = right_pts_velocity[i].x;
            double velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }

    // printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

void FeatureTrackerDPL::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTrackerDPL::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTrackerDPL::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

/// @brief Undistort pixel-plane feature points and back-project to normalized camera coordinates
/// @param pts
/// @param cam
/// @return
vector<cv::Point2f> FeatureTrackerDPL::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);                                    // returns undistorted normalized coordinates [x,y,1] for pinhole model
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z())); // b.z() == 1.0 for pinhole, so this gives the normalized coordinates
    }
    return un_pts;
}

/// @brief Compute feature point velocities on the normalized camera plane
/// @param ids
/// @param pts current frame feature points
/// @param cur_id_pts on entry holds the previous frame points; updated to current frame inside this function
/// @param prev_id_pts holds the previous frame points
/// @return
vector<cv::Point2f> FeatureTrackerDPL::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                                   map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear(); // clear and repopulate with current frame points
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // Compute velocities relative to previous frame
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;

        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end()) // matching feature found in previous frame
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0)); // no match in previous frame: velocity = 0
        }
    }
    else // no previous frame data: initialize all velocities to zero
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTrackerDPL::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                                  vector<int> &curLeftIds,
                                  vector<cv::Point2f> &curLeftPts,
                                  vector<cv::Point2f> &curRightPts,
                                  map<int, cv::Point2f> &prevLeftPtsMap)
{
    // int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_BGR2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            // cv::Point2f leftPt = curLeftPtsTrackRight[i];
            // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    // draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    // printf("predict pts size %d \n", (int)predict_pts_debug.size());

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

/// @brief Provide IMU-predicted 3D landmark locations to the feature tracker
/// @param predictPts
void FeatureTrackerDPL::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end()) // found a prediction for this feature
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv); // project predicted 3D point to pixel
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]); // no prediction: fall back to previous pixel location
    }
}

void FeatureTrackerDPL::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

cv::Mat FeatureTrackerDPL::getTrackImage()
{
    return imTrack;
}