// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <iostream>

#include "core.hpp"
#include "tracker.hpp"
#include "utils.hpp"
#include "kuhn_munkres.hpp"

#include<fstream>
#include<iostream>

namespace {
cv::Point Center(const cv::Rect& rect) {
    return cv::Point(static_cast<int>(rect.x + rect.width * 0.5),
                     static_cast<int>(rect.y + rect.height * 0.5));
}

std::vector<cv::Point> Centers(const TrackedObjects &detections) {
    std::vector<cv::Point> centers(detections.size());
    for (size_t i = 0; i < detections.size(); i++) {
        centers[i] = Center(detections[i].rect);
    }
    return centers;
}

DetectionLog ConvertTracksToDetectionLog(const ObjectTracks& tracks) {
    DetectionLog log;

    // Combine detected objects by respective frame indices.
    std::map<int, TrackedObjects> objects;
    for (const auto& track : tracks)
        for (const auto& object : track.second) {
            auto itr = objects.find(object.frame_idx);
            if (itr != objects.end())
                itr->second.emplace_back(object);
            else
                objects.emplace(object.frame_idx, TrackedObjects{object});
        }

    for (const auto& frame_res : objects) {
        DetectionLogEntry entry;
        entry.frame_idx = frame_res.first;
        entry.objects = std::move(frame_res.second);
        log.push_back(std::move(entry));
    }

    return log;
}

inline bool IsInRange(float val, float min, float max) {
    return min <= val && val <= max;
}

inline bool IsInRange(float val, cv::Vec2f range) {
    return IsInRange(val, range[0], range[1]);
}

std::vector<cv::Scalar> GenRandomColors(int colors_num) {
    std::vector<cv::Scalar> colors(colors_num);
    for (int i = 0; i < colors_num; i++) {
        colors[i] = cv::Scalar(static_cast<uchar>(255. * rand() / RAND_MAX),  // NOLINT
                               static_cast<uchar>(255. * rand() / RAND_MAX),  // NOLINT
                               static_cast<uchar>(255. * rand() / RAND_MAX));  // NOLINT
    }
    return colors;
}

}  // anonymous namespace

TrackerParams::TrackerParams()
    : min_track_duration(1000),
    forget_delay(150),
    aff_thr_fast(0.8f),
    aff_thr_strong(0.75f),
    shape_affinity_w(0.5f),
    motion_affinity_w(0.2f),
    time_affinity_w(0.0f),
    min_det_conf(0.65f),
    bbox_aspect_ratios_range(0.666f, 5.0f),
    bbox_heights_range(40, 1000),
    predict(25),
    strong_affinity_thr(0.2805f),
    reid_thr(0.61f),
    drop_forgotten_tracks(false),
    max_num_objects_in_track(300) {}

void ValidateParams(const TrackerParams &p) {
    PT_CHECK_GE(p.min_track_duration, static_cast<size_t>(500));
    PT_CHECK_LE(p.min_track_duration, static_cast<size_t>(10000));

    PT_CHECK_LE(p.forget_delay, static_cast<size_t>(10000));

    PT_CHECK_GE(p.aff_thr_fast, 0.0f);
    PT_CHECK_LE(p.aff_thr_fast, 1.0f);

    PT_CHECK_GE(p.aff_thr_strong, 0.0f);
    PT_CHECK_LE(p.aff_thr_strong, 1.0f);

    PT_CHECK_GE(p.shape_affinity_w, 0.0f);
    PT_CHECK_LE(p.shape_affinity_w, 100.0f);

    PT_CHECK_GE(p.motion_affinity_w, 0.0f);
    PT_CHECK_LE(p.motion_affinity_w, 100.0f);

    PT_CHECK_GE(p.time_affinity_w, 0.0f);
    PT_CHECK_LE(p.time_affinity_w, 100.0f);

    PT_CHECK_GE(p.min_det_conf, 0.0f);
    PT_CHECK_LE(p.min_det_conf, 1.0f);

    PT_CHECK_GE(p.bbox_aspect_ratios_range[0], 0.0f);
    PT_CHECK_LE(p.bbox_aspect_ratios_range[1], 10.0f);
    PT_CHECK_LT(p.bbox_aspect_ratios_range[0], p.bbox_aspect_ratios_range[1]);

    PT_CHECK_GE(p.bbox_heights_range[0], 10.0f);
    PT_CHECK_LE(p.bbox_heights_range[1], 1080.0f);
    PT_CHECK_LT(p.bbox_heights_range[0], p.bbox_heights_range[1]);

    PT_CHECK_GE(p.predict, 0);
    PT_CHECK_LE(p.predict, 10000);

    PT_CHECK_GE(p.strong_affinity_thr, 0.0f);
    PT_CHECK_LE(p.strong_affinity_thr, 1.0f);

    PT_CHECK_GE(p.reid_thr, 0.0f);
    PT_CHECK_LE(p.reid_thr, 1.0f);


    if (p.max_num_objects_in_track > 0) {
        int min_required_track_length = static_cast<int>(p.forget_delay);
        PT_CHECK_GE(p.max_num_objects_in_track, min_required_track_length);
        PT_CHECK_LE(p.max_num_objects_in_track, 10000);
    }
}

// Returns confusion matrix as:
//   |tp fn|
//   |fp tn|
cv::Mat PedestrianTracker::ConfusionMatrix(const std::vector<Match> &matches) {
    const bool kNegative = false;
    cv::Mat conf_mat(2, 2, CV_32F, cv::Scalar(0));
    for (const auto &m : matches) {
        conf_mat.at<float>(m.gt_label == kNegative, m.pr_label == kNegative)++;
    }

    return conf_mat;
}

PedestrianTracker::PedestrianTracker(const TrackerParams &params)
    : params_(params),
    descriptor_strong_(nullptr),
    distance_strong_(nullptr),
    collect_matches_(true),
    tracks_counter_(0),
    valid_tracks_counter_(0),
    frame_size_(0, 0),
    prev_timestamp_(std::numeric_limits<uint64_t>::max())
	{
        ValidateParams(params);
		iou_ = new iou();
    }

// Pipeline parameters getter.
const TrackerParams &PedestrianTracker::params() const { return params_; }

// Pipeline parameters setter.
void PedestrianTracker::set_params(const TrackerParams &params) {
    ValidateParams(params);
    params_ = params;
}

// Descriptor fast getter.
const PedestrianTracker::Descriptor &PedestrianTracker::descriptor_fast() const {
    return descriptor_fast_;
}

// Descriptor fast setter.
void PedestrianTracker::set_descriptor_fast(const Descriptor &val) {
    descriptor_fast_ = val;
}

// Descriptor strong getter.
const PedestrianTracker::Descriptor &PedestrianTracker::descriptor_strong() const {
    return descriptor_strong_;
}

// Descriptor strong setter.
void PedestrianTracker::set_descriptor_strong(const Descriptor &val) {
    descriptor_strong_ = val;
}

// Distance fast getter.
const PedestrianTracker::Distance &PedestrianTracker::distance_fast() const { return distance_fast_; }

// Distance fast setter.
void PedestrianTracker::set_distance_fast(const Distance &val) { distance_fast_ = val; }

// Distance strong getter.
const PedestrianTracker::Distance &PedestrianTracker::distance_strong() const { return distance_strong_; }

// Distance strong setter.
void PedestrianTracker::set_distance_strong(const Distance &val) { distance_strong_ = val; }

// Returns all tracks including forgotten (lost too many frames ago).
const std::unordered_map<size_t, Track> &
PedestrianTracker::tracks() const {
    return tracks_;
}

// Returns indexes of active tracks only.
const std::set<size_t> &PedestrianTracker::active_track_ids() const {
    return active_track_ids_;
}


// Returns detection log which is used for tracks saving.
DetectionLog PedestrianTracker::GetDetectionLog(const bool valid_only) const {
    return ConvertTracksToDetectionLog(all_tracks(valid_only));
}

// Returns decisions made by heuristic based on fast distance/descriptor and
// shape, motion and time affinity.
const std::vector<PedestrianTracker::Match> &
PedestrianTracker::base_classifier_matches() const {
    return base_classifier_matches_;
}

// Returns decisions made by heuristic based on strong distance/descriptor
// and
// shape, motion and time affinity.
const std::vector<PedestrianTracker::Match> &PedestrianTracker::reid_based_classifier_matches() const {
    return reid_based_classifier_matches_;
}

// Returns decisions made by strong distance/descriptor affinity.
const std::vector<PedestrianTracker::Match> &PedestrianTracker::reid_classifier_matches() const {
    return reid_classifier_matches_;
}

TrackedObjects PedestrianTracker::FilterDetections(
    const TrackedObjects &detections) const {
    TrackedObjects filtered_detections;
    for (const auto &det : detections) {
        float aspect_ratio = static_cast<float>(det.rect.height) / det.rect.width;
        if (det.confidence > params_.min_det_conf &&
            IsInRange(aspect_ratio, params_.bbox_aspect_ratios_range) &&
            IsInRange(static_cast<float>(det.rect.height), params_.bbox_heights_range)) {
            filtered_detections.emplace_back(det);
        }
    }
    return filtered_detections;
}

void PedestrianTracker::SolveAssignmentProblem(
    const std::set<size_t> &track_ids, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors, float thr,
    std::set<size_t> *unmatched_tracks, std::set<size_t> *unmatched_detections,
    std::set<std::tuple<size_t, size_t, float>> *matches) {
    PT_CHECK(unmatched_tracks);
    PT_CHECK(unmatched_detections);
    unmatched_tracks->clear();
    unmatched_detections->clear();

    PT_CHECK(!track_ids.empty());
    PT_CHECK(!detections.empty());
    PT_CHECK(descriptors.size() == detections.size());
    PT_CHECK(matches);
    matches->clear();

    cv::Mat dissimilarity;
    ComputeDissimilarityMatrix(track_ids, detections, descriptors,
                               &dissimilarity);

    auto res = KuhnMunkres().Solve(dissimilarity);

    for (size_t i = 0; i < detections.size(); i++) {
        unmatched_detections->insert(i);
    }

    size_t i = 0;
    for (auto id : track_ids) {
        if (res[i] < detections.size()) {
            matches->emplace(id, res[i], 1 - dissimilarity.at<float>(i, res[i]));
        } else {
            unmatched_tracks->insert(id);
        }
        i++;
    }
}

const ObjectTracks PedestrianTracker::all_tracks(bool valid_only) const {
    ObjectTracks all_objects;
    size_t counter = 0;

    std::set<size_t> sorted_ids;
    for (const auto &pair : tracks()) {
        sorted_ids.emplace(pair.first);
    }

    for (size_t id : sorted_ids) {
        if (!valid_only || IsTrackValid(id)) {
            TrackedObjects filtered_objects;
            for (const auto &object : tracks().at(id).objects) {
                filtered_objects.emplace_back(object);
                filtered_objects.back().object_id = counter;
            }
            all_objects.emplace(counter++, filtered_objects);
        }
    }
    return all_objects;
}

cv::Rect PedestrianTracker::PredictRect(size_t id, size_t k,
                                        size_t s) const {
    const auto &track = tracks_.at(id);
    PT_CHECK(!track.empty());

    if (track.size() == 1) {
        return track[0].rect;
    }

    size_t start_i = track.size() > k ? track.size() - k : 0;
    float width = 0, height = 0;

    for (size_t i = start_i; i < track.size(); i++) {
        width += track[i].rect.width;
        height += track[i].rect.height;
    }

    PT_CHECK(track.size() - start_i > 0);
    width /= (track.size() - start_i);
    height /= (track.size() - start_i);

    float delim = 0;
    cv::Point2f d(0, 0);

    for (size_t i = start_i + 1; i < track.size(); i++) {
        d += cv::Point2f(Center(track[i].rect) - Center(track[i - 1].rect));
        delim += (track[i].frame_idx - track[i - 1].frame_idx);
    }

    if (delim) {
        d /= delim;
    }

    s += 1;

    cv::Point c = Center(track.back().rect);
    return cv::Rect(static_cast<int>(c.x - width / 2 + d.x * s),
                    static_cast<int>(c.y - height / 2 + d.y * s),
                    static_cast<int>(width),
                    static_cast<int>(height));
}


bool PedestrianTracker::EraseTrackIfBBoxIsOutOfFrame(size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end()) return true;
    auto c = Center(tracks_.at(track_id).predicted_rect);
    if (!prev_frame_size_.empty() &&
        (c.x < 0 || c.y < 0 || c.x > prev_frame_size_.width ||
         c.y > prev_frame_size_.height)) {
        tracks_.at(track_id).lost = params_.forget_delay + 1;
        for (auto id : active_track_ids()) {
            size_t min_id = std::min(id, track_id);
            size_t max_id = std::max(id, track_id);
            tracks_dists_.erase(std::pair<size_t, size_t>(min_id, max_id));
        }
        active_track_ids_.erase(track_id);
        return true;
    }
    return false;
}

bool PedestrianTracker::EraseTrackIfItWasLostTooManyFramesAgo(
    size_t track_id) {
    if (tracks_.find(track_id) == tracks_.end()) return true;
    if (tracks_.at(track_id).lost > params_.forget_delay) {
        for (auto id : active_track_ids()) {
            size_t min_id = std::min(id, track_id);
            size_t max_id = std::max(id, track_id);
            tracks_dists_.erase(std::pair<size_t, size_t>(min_id, max_id));
        }
        active_track_ids_.erase(track_id);

        return true;
    }
    return false;
}

bool PedestrianTracker::UpdateLostTrackAndEraseIfItsNeeded(
    size_t track_id) {
    tracks_.at(track_id).lost++;
    tracks_.at(track_id).predicted_rect =
		iou_->get_correct_rect(frame_size, PredictRect(track_id, params().predict, tracks_.at(track_id).lost));

    /*bool erased = EraseTrackIfBBoxIsOutOfFrame(track_id);
    if (!erased) erased = EraseTrackIfItWasLostTooManyFramesAgo(track_id);*/
	bool erased = false;
	return erased;
}

void PedestrianTracker::UpdateLostTracks(
    const std::set<size_t> &track_ids) {
    for (auto track_id : track_ids) {
        UpdateLostTrackAndEraseIfItsNeeded(track_id);
    }
}

void PedestrianTracker::Process(const cv::Mat &frame,
                                const TrackedObjects &input_detections,
                                uint64_t timestamp) {

	frame_size = frame.size();

	std::ofstream out;
	if (write_match_result)
	{
		out.open("../../test.txt", ios::app);
	}

    if (prev_timestamp_ != std::numeric_limits<uint64_t>::max())
        PT_CHECK_LT(prev_timestamp_, timestamp);

    if (frame_size_ == cv::Size(0, 0)) {
        frame_size_ = frame.size();
    } else {
        PT_CHECK_EQ(frame_size_, frame.size());
    }

    //TrackedObjects detections = FilterDetections(input_detections);
	TrackedObjects detections = input_detections;
    for (auto &obj : detections) {
        obj.timestamp = timestamp;
    }

    std::vector<cv::Mat> descriptors;

	if (USE_NETWORK_FOR_REID)
	{
		ComputeStrongDesciptors(frame, detections, &descriptors);
	}
	else
	{
		ComputeFastDesciptors(frame, detections, &descriptors);
	}
	int num = descriptors.size();
	std::cout << "descriptor num:" << num <<std::endl;
	if (num > 0) {
		for (int i = 0; i < num; i++) {
			std::cout << "decriptor " << i << "'s (w,h) :(" << (descriptors.at(i)).clone().cols<<","<< (descriptors.at(i)).clone().rows << ")" <<std::endl;
		}
	}
    auto active_tracks = active_track_ids_;

	if (detections.size() < target_person_num)
	{
		std::cout << "Activate dist+descriptor+iou match : " << tracks_.size() << std::endl;
		UpdateLostTracks(active_tracks);

		std::vector<cv::Rect> detections_rects;
		for (int ii = 0; ii < detections.size(); ++ii)
		{
			detections_rects.emplace_back(detections[ii].rect);
		}

		std::vector<cv::Rect> track_rects;
		std::vector<cv::Mat> track_descriptors;
		for (int ii = 0; ii < tracks_.size(); ++ii)
		{
			track_rects.emplace_back(tracks_.at(ii).objects.back().rect);
			track_descriptors.emplace_back(tracks_.at(ii).descriptor_fast);
		}

		iou_->get_iou_pairs(track_rects);

		iou_->get_close_rect_id_pairs();

		iou_->get_rect_match_by_dist2(detections_rects, track_rects);
		iou_->get_rect_match_by_descriptor(descriptors, track_descriptors);

		std::vector<int> detections_matches_by_dist2 = iou_->detections_matches_by_dist2;
		std::vector<int> detections_matches_by_descriptor = iou_->detections_matches_by_descriptor;

		std::vector<double> detections_match_dists_by_descriptor = iou_->detections_match_dists_by_descriptor;

		for (int ii = 0; ii < detections_rects.size(); ++ii)
		{
			std::cout << "------pair : (" << ii << "," << detections_matches_by_dist2[ii] << "&&" << detections_matches_by_descriptor[ii] << ")" << std::endl;
			std::cout << "------dist : " << iou_->detections_match_dists_by_dist2[ii] << " , " << iou_->detections_match_dists_by_descriptor[ii] << std::endl;
			for (int jj = 0; jj < track_rects.size(); ++jj)
			{
				if (detections_matches_by_dist2[ii] > -1 && detections_matches_by_dist2[ii] != jj && iou_->iou_pairs[detections_matches_by_dist2[ii]][jj] > 0)
				{
					std::cout << "------iou  : (" << detections_matches_by_dist2[ii] << "," << jj << ") : " << iou_->iou_pairs[detections_matches_by_dist2[ii]][jj] << std::endl;
				}
				if(detections_matches_by_descriptor[ii] > -1 && detections_matches_by_descriptor[ii] != jj && iou_->iou_pairs[detections_matches_by_descriptor[ii]][jj] > 0)
				{
					std::cout << "------iou  : (" << detections_matches_by_descriptor[ii] << "," << jj << ") : " << iou_->iou_pairs[detections_matches_by_descriptor[ii]][jj] << std::endl;
				}
			}

			double max_current_iou = 0;
			int current_max_episodes_to_show_predict = 0;
			int matched_track_id = -1;

			for (int jj = 0; jj < track_rects.size(); ++jj)
			{
				if (jj != detections_matches_by_dist2[ii] && iou_->iou_pairs[jj][detections_matches_by_dist2[ii]] > max_current_iou)
				{
					max_current_iou = iou_->iou_pairs[jj][detections_matches_by_dist2[ii]];
				}
			}

			bool update_descriptor = true;

			if (max_current_iou > 0.8)
			{
				update_descriptor = false;
			}

			if (detections_matches_by_dist2[ii] != -1 && detections_matches_by_dist2[ii] == detections_matches_by_descriptor[ii])
			{
				AppendToTrack(frame, detections_matches_by_dist2[ii], detections[ii], descriptors[ii], descriptors[ii], update_descriptor);
				current_max_episodes_to_show_predict = tracks_.at(detections_matches_by_dist2[ii]).max_episodes_to_show_predict;
				tracks_.at(detections_matches_by_dist2[ii]).max_episodes_to_show_predict = 0;

				matched_track_id = detections_matches_by_dist2[ii];
			}
			else if (detections_matches_by_dist2[ii] != -1 && max_current_iou < 0.8 && detections_match_dists_by_descriptor[ii] > 0.2)
			{
				AppendToTrack(frame, detections_matches_by_dist2[ii], detections[ii], descriptors[ii], descriptors[ii], update_descriptor);
				current_max_episodes_to_show_predict = tracks_.at(detections_matches_by_dist2[ii]).max_episodes_to_show_predict;
				tracks_.at(detections_matches_by_dist2[ii]).max_episodes_to_show_predict = 0;

				matched_track_id = detections_matches_by_dist2[ii];
			}

			if (detections_matches_by_descriptor[ii] != -1 && detections_matches_by_dist2[ii] != detections_matches_by_descriptor[ii] && detections_match_dists_by_descriptor[ii] < 0.2)
			{
				if (matched_track_id != -1)
				{
					std::cout << "reset current match : (" << ii << " , " << detections_matches_by_dist2[ii] << ") to (" << ii << " , " << detections_matches_by_descriptor[ii] << ")" << std::endl;
					tracks_.at(detections_matches_by_dist2[ii]).objects.pop_back();
					--tracks_.at(detections_matches_by_dist2[ii]).length;
					tracks_.at(detections_matches_by_dist2[ii]).max_episodes_to_show_predict = current_max_episodes_to_show_predict;
				}
				AppendToTrack(frame, detections_matches_by_descriptor[ii], detections[ii], descriptors[ii], descriptors[ii], update_descriptor);
				tracks_.at(detections_matches_by_descriptor[ii]).max_episodes_to_show_predict = 0;

				matched_track_id = detections_matches_by_descriptor[ii];
			}

			if (matched_track_id == -1)
			{
				std::cout << "!!!!!!!!!!!!!!!!!!!!!!!detection match failed : " << ii << std::endl;
			}
			else if (write_match_result)
			{
				out << "Frame:" << current_frame_idx << std::endl << "match:" << ii << "," << matched_track_id << std::endl;
			}
		}

		for (int ii = 0; ii < tracks_.size(); ++ii)
		{
			if (tracks_.at(ii).lost > 0)
			{
				double max_iou_to_other = 0;

				for (int jj = 0; jj < tracks_.size(); ++jj)
				{
					if (jj != ii && iou_->iou_pairs[ii][jj] > max_iou_to_other)
					{
						max_iou_to_other = iou_->iou_pairs[ii][jj];
					}
				}

				bool update_descriptor = true;

				if (max_iou_to_other > 0.8)
				{
					update_descriptor = false;
				}

				if (max_iou_to_other < 0.2 && tracks_.at(ii).max_episodes_to_show_predict < 1)
				{
					std::cout << "track max_episodes_to_show_predict : " << ii << " -> " << tracks_.at(ii).max_episodes_to_show_predict << std::endl;
					TrackedObject predict_detection = tracks_.at(ii).objects.back();
					cv::Rect predict_detection_rect = iou_->get_correct_rect(frame.size(), PredictRect(ii, params().predict, tracks_.at(ii).max_episodes_to_show_predict));

					predict_detection.rect = predict_detection_rect;

					AppendToTrack(frame, ii, predict_detection, tracks_.at(ii).descriptor_fast, tracks_.at(ii).descriptor_strong, update_descriptor);

					++tracks_.at(ii).max_episodes_to_show_predict;
				}
			}
		}



		//cv::imshow("dbg", frame);
		
		//cv::waitKey();

		out.close();

		return;
	}

	/*for (int ii = detections.size() - 1; ii > 1; --ii)
	{
		detections.erase(detections.begin() + ii);
		descriptors.erase(descriptors.begin() + ii);
	}*/

    if (!active_tracks.empty() && !detections.empty())
	{
        std::set<size_t> unmatched_tracks, unmatched_detections;
        std::set<std::tuple<size_t, size_t, float>> matches;

        SolveAssignmentProblem(active_tracks, detections, descriptors,
                               params_.aff_thr_fast, &unmatched_tracks,
                               &unmatched_detections, &matches);

        std::map<size_t, std::pair<bool, cv::Mat>> is_matching_to_track;

        if (distance_strong_)
		{
            std::vector<std::pair<size_t, size_t>> reid_track_and_det_ids =
                GetTrackToDetectionIds(matches);
            is_matching_to_track = StrongMatching(
                frame, detections, reid_track_and_det_ids);
        }

        for (const auto &match : matches)
		{
            size_t track_id = std::get<0>(match);
            size_t det_id = std::get<1>(match);
            float conf = std::get<2>(match);

            auto last_det = tracks_.at(track_id).objects.back();
            last_det.rect = tracks_.at(track_id).predicted_rect;

            if (collect_matches_ && last_det.object_id >= 0 &&
                detections[det_id].object_id >= 0)
			{
                base_classifier_matches_.emplace_back(
                    tracks_.at(track_id).objects.back(), last_det.rect,
                    detections[det_id], conf > params_.aff_thr_fast);
            }

            if (conf > params_.aff_thr_fast)
			{
                AppendToTrack(frame, track_id, detections[det_id],
                              descriptors[det_id], cv::Mat());
                unmatched_detections.erase(det_id);

				if (write_match_result)
				{
					out << "Frame:" << current_frame_idx << std::endl << "match:" << det_id << "," << track_id << std::endl;
				}
            }
			else
			{
                if (conf > params_.strong_affinity_thr)
				{
                    if (distance_strong_ && is_matching_to_track[track_id].first)
					{
                        AppendToTrack(frame, track_id, detections[det_id],
                                      descriptors[det_id],
                                      is_matching_to_track[track_id].second.clone());

						if (write_match_result)
						{
							out << "Frame:" << current_frame_idx << std::endl << "match:" << det_id << "," << track_id << std::endl;
						}
                    }
					else
					{
                        if (UpdateLostTrackAndEraseIfItsNeeded(track_id))
						{
                            AddNewTrack(frame, detections[det_id], descriptors[det_id],
                                        distance_strong_
                                        ? is_matching_to_track[track_id].second.clone()
                                        : cv::Mat());

							if (write_match_result)
							{
								out << "Frame:" << current_frame_idx << std::endl << "match:" << det_id << "," << track_id << std::endl;
							}
                        }
                    }

                    unmatched_detections.erase(det_id);
                }
				else
				{
                    unmatched_tracks.insert(track_id);
                }
            }
        }

        AddNewTracks(frame, detections, descriptors, unmatched_detections);
        UpdateLostTracks(unmatched_tracks);

        /*for (size_t id : active_tracks)
		{
            EraseTrackIfBBoxIsOutOfFrame(id);
        }*/
    }
	else
	{
        AddNewTracks(frame, detections, descriptors);
        UpdateLostTracks(active_tracks);
    }

    prev_frame_size_ = frame.size();
    if (params_.drop_forgotten_tracks) DropForgottenTracks();

    tracks_dists_.clear();
    prev_timestamp_ = timestamp;
	
	++current_frame_idx;

	out.close();
}

void PedestrianTracker::DropForgottenTracks() {
    std::unordered_map<size_t, Track> new_tracks;
    std::set<size_t> new_active_tracks;

    size_t max_id = 0;
    if (!active_track_ids_.empty())
        max_id =
            *std::max_element(active_track_ids_.begin(), active_track_ids_.end());

    const size_t kMaxTrackID = 10000;
    bool reassign_id = max_id > kMaxTrackID;

    size_t counter = 0;
    for (const auto &pair : tracks_) {
        if (!IsTrackForgotten(pair.first)) {
            new_tracks.emplace(reassign_id ? counter : pair.first, pair.second);
            new_active_tracks.emplace(reassign_id ? counter : pair.first);
            counter++;

        } else {
            if (IsTrackValid(pair.first)) {
                valid_tracks_counter_++;
            }
        }
    }
    tracks_.swap(new_tracks);
    active_track_ids_.swap(new_active_tracks);

    tracks_counter_ = reassign_id ? counter : tracks_counter_;
}

void PedestrianTracker::DropForgottenTrack(size_t track_id) {
    PT_CHECK(IsTrackForgotten(track_id));
    PT_CHECK(active_track_ids_.count(track_id) == 0);
    tracks_.erase(track_id);
}

float PedestrianTracker::ShapeAffinity(float weight, const cv::Rect &trk,
                                       const cv::Rect &det) {
    float w_dist = static_cast<float>(std::abs(trk.width - det.width) / (trk.width + det.width));
    float h_dist = static_cast<float>(std::abs(trk.height - det.height) / (trk.height + det.height));
    return static_cast<float>(exp(static_cast<double>(-weight * (w_dist + h_dist))));
}

float PedestrianTracker::MotionAffinity(float weight, const cv::Rect &trk,
                                        const cv::Rect &det) {
    float x_dist = static_cast<float>(trk.x - det.x) * (trk.x - det.x) /
        (det.width * det.width);
    float y_dist = static_cast<float>(trk.y - det.y) * (trk.y - det.y) /
        (det.height * det.height);
    return static_cast<float>(exp(static_cast<double>(-weight * (x_dist + y_dist))));
}

float PedestrianTracker::TimeAffinity(float weight, const float &trk_time,
                                      const float &det_time) {
    return static_cast<float>(exp(static_cast<double>(-weight * std::fabs(trk_time - det_time))));
}

void PedestrianTracker::ComputeFastDesciptors(
    const cv::Mat &frame, const TrackedObjects &detections,
    std::vector<cv::Mat> *desriptors) {
    *desriptors = std::vector<cv::Mat>(detections.size(), cv::Mat());
    for (size_t i = 0; i < detections.size(); i++) {
		if (use_det_bbox)
		{
			descriptor_fast_->Compute(frame(detections[i].det_rect).clone(),
				&((*desriptors)[i]));
		}
		else
		{
			descriptor_fast_->Compute(frame(detections[i].rect).clone(),
				&((*desriptors)[i]));
		}
    }
}

void PedestrianTracker::ComputeStrongDesciptors(
	const cv::Mat &frame, const TrackedObjects &detections,
	std::vector<cv::Mat> *desriptors) {
	*desriptors = std::vector<cv::Mat>(detections.size(), cv::Mat());
	for (size_t i = 0; i < detections.size(); i++) {
		if (use_det_bbox)
		{
			descriptor_strong_->Compute(frame(detections[i].det_rect).clone(),
										&((*desriptors)[i]));
		}
		else
		{
			descriptor_strong_->Compute(frame(detections[i].rect).clone(),
										&((*desriptors)[i]));
		}
	}
}

void PedestrianTracker::ComputeDissimilarityMatrix(
    const std::set<size_t> &active_tracks, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors,
    cv::Mat *dissimilarity_matrix) {
    cv::Mat am(active_tracks.size(), detections.size(), CV_32F, cv::Scalar(0));
    size_t i = 0;
    for (auto id : active_tracks) {
        auto ptr = am.ptr<float>(i);
        for (size_t j = 0; j < descriptors.size(); j++) {
            auto last_det = tracks_.at(id).objects.back();
            last_det.rect = tracks_.at(id).predicted_rect;
			if (USE_NETWORK_FOR_REID)
			{
				ptr[j] = AffinityStrong(tracks_.at(id).descriptor_fast, last_det,
										descriptors[j], detections[j]);
			}
			else
			{
				ptr[j] = AffinityFast(tracks_.at(id).descriptor_fast, last_det,
									  descriptors[j], detections[j]);
			}
        }
        i++;
    }
    *dissimilarity_matrix = 1.0 - am;
}

std::vector<float> PedestrianTracker::ComputeDistances(
    const cv::Mat &frame,
    const TrackedObjects& detections,
    const std::vector<std::pair<size_t, size_t>> &track_and_det_ids,
    std::map<size_t, cv::Mat> *det_id_to_descriptor) {
    std::map<size_t, size_t> det_to_batch_ids;
    std::map<size_t, size_t> track_to_batch_ids;

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> descriptors;
    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        if (tracks_.at(track_id).descriptor_strong.empty()) {
            images.push_back(tracks_.at(track_id).last_image);
            descriptors.push_back(cv::Mat());
            track_to_batch_ids[track_id] = descriptors.size() - 1;
        }

        images.push_back(frame(detections[det_id].rect));
        descriptors.push_back(cv::Mat());
        det_to_batch_ids[det_id] = descriptors.size() - 1;
    }

    descriptor_strong_->Compute(images, &descriptors);

    std::vector<cv::Mat> descriptors1;
    std::vector<cv::Mat> descriptors2;
    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        if (tracks_.at(track_id).descriptor_strong.empty()) {
            tracks_.at(track_id).descriptor_strong =
                descriptors[track_to_batch_ids[track_id]].clone();
        }
        (*det_id_to_descriptor)[det_id] = descriptors[det_to_batch_ids[det_id]];

        descriptors1.push_back(descriptors[det_to_batch_ids[det_id]]);
        descriptors2.push_back(tracks_.at(track_id).descriptor_strong);
    }

    std::vector<float> distances =
        distance_strong_->Compute(descriptors1, descriptors2);

    return distances;
}

std::vector<std::pair<size_t, size_t>>
PedestrianTracker::GetTrackToDetectionIds(
    const std::set<std::tuple<size_t, size_t, float>> &matches) {
    std::vector<std::pair<size_t, size_t>> track_and_det_ids;

    for (const auto &match : matches) {
        size_t track_id = std::get<0>(match);
        size_t det_id = std::get<1>(match);
        float conf = std::get<2>(match);
        if (conf < params_.aff_thr_fast && conf > params_.strong_affinity_thr) {
            track_and_det_ids.emplace_back(track_id, det_id);
        }
		/*if (conf < params_.aff_thr_fast) {
			track_and_det_ids.emplace_back(track_id, det_id);
		}*/
    }
    return track_and_det_ids;
}

std::map<size_t, std::pair<bool, cv::Mat>>
PedestrianTracker::StrongMatching(
    const cv::Mat &frame,
    const TrackedObjects& detections,
    const std::vector<std::pair<size_t, size_t>> &track_and_det_ids) {
    std::map<size_t, std::pair<bool, cv::Mat>> is_matching;

    if (track_and_det_ids.size() == 0) {
        return is_matching;
    }

    std::map<size_t, cv::Mat> det_ids_to_descriptors;
    std::vector<float> distances =
        ComputeDistances(frame, detections,
                         track_and_det_ids, &det_ids_to_descriptors);

    for (size_t i = 0; i < track_and_det_ids.size(); i++) {
        auto reid_affinity = 1.0 - distances[i];

        size_t track_id = track_and_det_ids[i].first;
        size_t det_id = track_and_det_ids[i].second;

        const auto& track = tracks_.at(track_id);
        const auto& detection = detections[det_id];

        auto last_det = track.objects.back();
        last_det.rect = track.predicted_rect;

        float affinity = static_cast<float>(reid_affinity) * Affinity(last_det, detection);

        if (collect_matches_ && last_det.object_id >= 0 &&
            detection.object_id >= 0) {
            reid_classifier_matches_.emplace_back(track.objects.back(), last_det.rect,
                                                  detection,
                                                  reid_affinity > params_.reid_thr);

            reid_based_classifier_matches_.emplace_back(
                track.objects.back(), last_det.rect, detection,
                affinity > params_.aff_thr_strong);
        }

        bool is_detection_matching =
            reid_affinity > params_.reid_thr && affinity > params_.aff_thr_strong;

        is_matching[track_id] = std::pair<bool, cv::Mat>(
            is_detection_matching, det_ids_to_descriptors[det_id]);
    }
    return is_matching;
}

void PedestrianTracker::AddNewTracks(
    const cv::Mat &frame, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors) {
    PT_CHECK(detections.size() == descriptors.size());
    for (size_t i = 0; i < detections.size(); i++) {
        AddNewTrack(frame, detections[i], descriptors[i]);
    }
}

void PedestrianTracker::AddNewTracks(
    const cv::Mat &frame, const TrackedObjects &detections,
    const std::vector<cv::Mat> &descriptors, const std::set<size_t> &ids) {
    PT_CHECK(detections.size() == descriptors.size());
    for (size_t i : ids) {
        PT_CHECK(i < detections.size());
        AddNewTrack(frame, detections[i], descriptors[i]);
    }
}

void PedestrianTracker::AddNewTrack(const cv::Mat &frame,
                                    const TrackedObject &detection,
                                    const cv::Mat &descriptor_fast,
                                    const cv::Mat &descriptor_strong) {
    auto detection_with_id = detection;
    detection_with_id.object_id = tracks_counter_;
    tracks_.emplace(std::pair<size_t, Track>(
            tracks_counter_,
            Track({detection_with_id}, frame(detection.rect).clone(),
                  descriptor_fast.clone(), descriptor_strong.clone())));

    for (size_t id : active_track_ids_) {
        tracks_dists_.emplace(std::pair<size_t, size_t>(id, tracks_counter_),
                              std::numeric_limits<float>::max());
    }

    active_track_ids_.insert(tracks_counter_);
    tracks_counter_++;
}

void PedestrianTracker::AppendToTrack(const cv::Mat &frame,
                                      size_t track_id,
                                      const TrackedObject &detection,
                                      const cv::Mat &descriptor_fast,
                                      const cv::Mat &descriptor_strong,
									  bool update_descriptor) {
    //PT_CHECK(!IsTrackForgotten(track_id));

    auto detection_with_id = detection;
    detection_with_id.object_id = track_id;

    auto &cur_track = tracks_.at(track_id);
    cur_track.objects.emplace_back(detection_with_id);
    cur_track.predicted_rect = detection.rect;
    cur_track.lost = 0;
    cur_track.last_image = frame(detection.rect).clone();
	if (cur_track.descriptor_fast.empty()) {
		cur_track.descriptor_fast = descriptor_fast.clone();
	}
	else 
	{
		if (update_descriptor)
		{
			cur_track.descriptor_fast = 0.05 * descriptor_fast + 0.95 * cur_track.descriptor_fast;
		}
	}
    
    cur_track.length++;

    if (cur_track.descriptor_strong.empty()) {
        cur_track.descriptor_strong = descriptor_strong.clone();
    } else if (!descriptor_strong.empty()) 
	{
		if (update_descriptor)
		{
			cur_track.descriptor_strong = 0.1 * descriptor_strong + 0.9 * cur_track.descriptor_strong;
		}
    }


    if (params_.max_num_objects_in_track > 0) {
        while (cur_track.size() >
               static_cast<size_t>(params_.max_num_objects_in_track)) {
            cur_track.objects.erase(cur_track.objects.begin());
        }
    }
}

float PedestrianTracker::AffinityFast(const cv::Mat &descriptor1,
                                      const TrackedObject &obj1,
                                      const cv::Mat &descriptor2,
                                      const TrackedObject &obj2) {
    const float eps = 1e-6f;
    float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
    //if (shp_aff < eps) return 0.0f;

    float mot_aff =
        MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
    //if (mot_aff < eps) return 0.0f;
    float time_aff =
        TimeAffinity(params_.time_affinity_w, static_cast<float>(obj1.frame_idx), static_cast<float>(obj2.frame_idx));

    //if (time_aff < eps) return 0.0f;

    float app_aff = 1.0f - distance_fast_->Compute(descriptor1, descriptor2);

    //return shp_aff * mot_aff * app_aff * time_aff;
	return app_aff;
}

float PedestrianTracker::AffinityStrong(const cv::Mat &descriptor1,
	const TrackedObject &obj1,
	const cv::Mat &descriptor2,
	const TrackedObject &obj2) {
	const float eps = 1e-6f;
	float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
	//if (shp_aff < eps) return 0.0f;

	float mot_aff =
		MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
	//if (mot_aff < eps) return 0.0f;
	float time_aff =
		TimeAffinity(params_.time_affinity_w, static_cast<float>(obj1.frame_idx), static_cast<float>(obj2.frame_idx));

	//if (time_aff < eps) return 0.0f;

	float app_aff = 1.0f - distance_strong_->Compute(descriptor1, descriptor2);

	return 0.1 * mot_aff + 0.9 * app_aff;
	//return app_aff;
}

float PedestrianTracker::Affinity(const TrackedObject &obj1,
                                  const TrackedObject &obj2) {
    float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
    float mot_aff =
        MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
    float time_aff =
        TimeAffinity(params_.time_affinity_w, static_cast<float>(obj1.frame_idx), static_cast<float>(obj2.frame_idx));
    //return shp_aff * mot_aff * time_aff;
	return mot_aff;
}

bool PedestrianTracker::IsTrackValid(size_t id) const {
    const auto& track = tracks_.at(id);
    const auto &objects = track.objects;
    if (objects.empty()) {
        return false;
    }
    int64_t duration_ms = objects.back().timestamp - track.first_object.timestamp;
    if (duration_ms < static_cast<int64_t>(params_.min_track_duration))
        return false;
    return true;
}

bool PedestrianTracker::IsTrackForgotten(size_t id) const {
    return IsTrackForgotten(tracks_.at(id));
}

bool PedestrianTracker::IsTrackForgotten(const Track &track) const {
    return (track.lost > params_.forget_delay);
}

size_t PedestrianTracker::Count() const {
    size_t count = valid_tracks_counter_;
    for (const auto &pair : tracks_) {
        count += (IsTrackValid(pair.first) ? 1 : 0);
    }
    return count;
}

std::unordered_map<size_t, std::vector<cv::Point>>
PedestrianTracker::GetActiveTracks() const {
    std::unordered_map<size_t, std::vector<cv::Point>> active_tracks;
    for (size_t idx : active_track_ids()) {
        auto track = tracks().at(idx);
        if (IsTrackValid(idx) && !IsTrackForgotten(idx)) {
            active_tracks.emplace(idx, Centers(track.objects));
        }
		//active_tracks.emplace(idx, Centers(track.objects));
    }
    return active_tracks;
}

TrackedObjects PedestrianTracker::TrackedDetections() const {
    TrackedObjects detections;
    for (size_t idx : active_track_ids()) {
        auto track = tracks().at(idx);
        if (IsTrackValid(idx) && !track.lost) {
            detections.emplace_back(track.objects.back());
        }
    }
    return detections;
}

cv::Mat PedestrianTracker::DrawActiveTracks(const cv::Mat &frame) {
    cv::Mat out_frame = frame.clone();

    if (colors_.empty()) {
        int num_colors = 100;
        colors_ = GenRandomColors(num_colors);
    }

    auto active_tracks = GetActiveTracks();
    for (auto active_track : active_tracks) {
        size_t idx = active_track.first;
        auto centers = active_track.second;
		auto track = tracks().at(idx);

		if (SHOW_TRACK_LINES)
		{
			DrawPolyline(centers, colors_[idx % colors_.size()], &out_frame);
			std::stringstream ss;
			ss << idx;
			cv::putText(out_frame, ss.str(), centers.back(), cv::FONT_HERSHEY_SCRIPT_COMPLEX, 2.0,
				colors_[idx % colors_.size()], 3);

			if (track.lost) {
				cv::line(out_frame, active_track.second.back(),
					Center(track.predicted_rect), cv::Scalar(0, 0, 0), 4);
			}
		}

		if (!track.descriptor_fast.empty()) {
			//cv::Mat diagram(100, 100, CV_8UC3);
			//DrawDesciptorDiagram(diagram, track.descriptor_fast);
			cv::Point topLeft = (track.predicted_rect).tl();
			std::cout << "predict rect topLeft : (" << topLeft.x << "," << topLeft.y << ")" << std::endl;
			//diagram.copyTo(out_frame( cv::Rect(topLeft.x,topLeft.y,100,100)));
			//diagram.copyTo(out_frame(cv::Rect(110 * idx, 0, 100, 100)));
		}
		
    }

    return out_frame;
}

void PedestrianTracker::DrawDesciptorDiagram(cv::Mat &outImage , cv::Mat & descriptor)
{
	
	int row = descriptor.rows, col = descriptor.cols;
	int num = row * col;
	std::vector<double> values(num);
	double min_value = 10000, max_value = -10000;
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			values[i*col + j] = descriptor.at<double>(i, j);
			if (i == 0 && j == 0) {
				min_value = values[i*col + j];
			}
			else {
				min_value = MIN(min_value, values[i*col + j]);
				max_value = MAX(max_value, values[i*col + j]);
			}
		}
	}
	double scale = num / max_value;
	cv::Mat diagram= cv::Mat::zeros(num, num, CV_8UC3);
	for (int i = 0; i < num; ++i) {
		cv::line(diagram, cv::Point(i, 0), cv::Point(i, int(values[i] * scale)), cv::Scalar(255, 255, 255));
	}
	cv::resize(diagram, outImage, cv::Size(100, 100));

}

const cv::Size kMinFrameSize = cv::Size(320, 240);
const cv::Size kMaxFrameSize = cv::Size(1920, 1080);

void PedestrianTracker::PrintConfusionMatrices() const {
    std::cout << "Base classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(base_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }

    std::cout << "Reid-based classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(reid_based_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }

    std::cout << "Reid only classifier quality: " << std::endl;
    {
        auto cm = ConfusionMatrix(reid_classifier_matches());
        std::cout << cm << std::endl;
        std::cout << "or" << std::endl;
        cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
        cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
        std::cout << cm << std::endl << std::endl;
    }
}

void PedestrianTracker::PrintReidPerformanceCounts(std::string fullDeviceName) const {
    if (descriptor_strong_) {
        descriptor_strong_->PrintPerformanceCounts(fullDeviceName);
    }
}
