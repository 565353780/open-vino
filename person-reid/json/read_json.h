#pragma once
#include <iostream>
#include <vector>
#include <json/json.h>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

using namespace std;

class read_json
{
public:
	read_json();
	~read_json();

	double string2double(const string& str);
	int intlen(int num);
	vector<vector<vector<double>>> load_json(string json_path);
	vector<vector<double>> bbox_xyxy_to_xywh(vector<vector<double>> bbox);
	vector<vector<double>> bbox_xyxy_to_tlwh(vector<vector<double>> bbox);
	vector<vector<double>> bbox_xywh_to_xyxy(vector<vector<double>> bbox);
	vector<vector<double>> bbox_xywh_to_tlwh(vector<vector<double>> bbox);
	vector<vector<double>> bbox_tlwh_to_xyxy(vector<vector<double>> bbox);
	vector<vector<double>> bbox_tlwh_to_xywh(vector<vector<double>> bbox);
	vector<vector<vector<double>>> get_bbox_xyxy(vector<vector<vector<double>>> json_dict, double conf_thresh, double expand_coefficient, double det_expand_coefficient, vector<int> keypoints_idx_used, vector<int> keypoints_idx_det);
	vector<vector<vector<double>>> get_bbox_xywh(vector<vector<vector<double>>> json_dict, double conf_thresh, double expand_coefficient, double det_expand_coefficient, vector<int> keypoints_idx_used, vector<int> keypoints_idx_det);
	vector<vector<vector<double>>> get_bbox_tlwh(vector<vector<vector<double>>> json_dict, double conf_thresh, double expand_coefficient, double det_expand_coefficient, vector<int> keypoints_idx_used, vector<int> keypoints_idx_det);
	

};