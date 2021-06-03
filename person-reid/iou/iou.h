#pragma once
#include <iostream>
#include <vector>
#include <json/json.h>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <opencv2/core.hpp>

using namespace std;

class iou
{
public:
	iou();
	~iou();

	double get_iou(cv::Rect rect_1, cv::Rect rect_2);

	void get_iou_pairs(vector<cv::Rect> rects);

	void get_close_rect_id_pairs();

	double get_close_rect_dist2(cv::Rect rect_1, cv::Rect rect_2);
	double get_close_rect_dist(cv::Rect rect_1, cv::Rect rect_2);

	void get_rect_match_by_dist2(vector<cv::Rect> detections_rects, vector<cv::Rect> track_rects);
	void get_rect_match_by_dist(vector<cv::Rect> detections_rects, vector<cv::Rect> track_rects);

	double get_descriptor_dist(cv::Mat descriptor_1, cv::Mat descriptor_2);

	void get_rect_match_by_descriptor(vector<cv::Mat> detections_descriptors, vector<cv::Mat> track_descriptors);

	cv::Rect get_correct_rect(cv::Size image_size, cv::Rect rect);

public:
	vector<vector<double>> iou_pairs;

	vector<vector<int>> close_rect_pairs;

	vector<int> detections_matches_by_dist2;
	vector<double> detections_match_dists_by_dist2;
	vector<int> detections_matches_by_dist;
	vector<double> detections_match_dists_by_dist;
	vector<int> detections_matches_by_descriptor;
	vector<double> detections_match_dists_by_descriptor;


};