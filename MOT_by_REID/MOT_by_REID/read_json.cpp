#include "read_json.h"

read_json::read_json()
{

}

read_json::~read_json()
{

}

double read_json::string2double(const string& str)
{
	istringstream iss(str);
	double num;
	iss >> num;
	return num;
}

int read_json::intlen(int num)
{
	if (num < 0)
	{
		return 0;
	}

	int i = 1;

	while (i > 0)
	{
		if (num < pow(10, i))
		{
			return i;
		}

		++i;
	}
}

vector<vector<vector<double>>> read_json::load_json(string json_path)
{
	vector<vector<vector<double>>> people_keypoints;

	ifstream is(json_path, ios::binary);

	if (!is.is_open())
	{
		cout << "open json file failed." << endl;
		return people_keypoints;
	}

	Json::Reader reader;
	Json::Value root;

	// ½âÎöjsonÄÚÈÝ
	if (reader.parse(is, root))
	{
		int people_size = root["people"].size();
		for (int i = 0; i < people_size; ++i)
		{
			vector<vector<double>> keypoints;

			int keypoints_size = root["people"][i]["pose_keypoints_2d"].size();
			for (int j = 0; j < keypoints_size / 3; ++j)
			{
				vector<double> keypoint;
				keypoint.emplace_back(root["people"][i]["pose_keypoints_2d"][3 * j].asDouble());
				keypoint.emplace_back(root["people"][i]["pose_keypoints_2d"][3 * j + 1].asDouble());
				keypoint.emplace_back(root["people"][i]["pose_keypoints_2d"][3 * j + 2].asDouble());

				keypoints.emplace_back(keypoint);
			}

			people_keypoints.emplace_back(keypoints);
		}
	}

	is.close();

	return people_keypoints;
}

vector<vector<double>> read_json::bbox_xyxy_to_xywh(vector<vector<double>> bbox)
{
	for (int i = 0; i < bbox.size(); ++i)
	{
		double x_min = bbox[i][0];
		double y_min = bbox[i][1];
		double x_max = bbox[i][2];
		double y_max = bbox[i][3];

		bbox[i][0] = (x_min + x_max) / 2;
		bbox[i][1] = (y_min + y_max) / 2;
		bbox[i][2] = x_max - x_min;
		bbox[i][3] = y_max - y_min;
	}

	return bbox;
}

vector<vector<double>> read_json::bbox_xyxy_to_tlwh(vector<vector<double>> bbox)
{
	for (int i = 0; i < bbox.size(); ++i)
	{
		bbox[i][2] -= bbox[i][0];
		bbox[i][3] -= bbox[i][1];
	}

	return bbox;
}

vector<vector<double>> read_json::bbox_xywh_to_xyxy(vector<vector<double>> bbox)
{
	for (int i = 0; i < bbox.size(); ++i)
	{
		double x_mid = bbox[i][0];
		double y_mid = bbox[i][1];
		double x_delta = bbox[i][2];
		double y_delta = bbox[i][3];

		bbox[i][0] = x_mid - x_delta / 2;
		bbox[i][1] = y_mid - y_delta / 2;
		bbox[i][2] = x_mid + x_delta / 2;
		bbox[i][3] = y_mid + y_delta / 2;
	}

	return bbox;
}

vector<vector<double>> read_json::bbox_xywh_to_tlwh(vector<vector<double>> bbox)
{
	for (int i = 0; i < bbox.size(); ++i)
	{
		bbox[i][0] -= bbox[i][2] / 2;
		bbox[i][1] -= bbox[i][3] / 2;
	}

	return bbox;
}

vector<vector<double>> read_json::bbox_tlwh_to_xyxy(vector<vector<double>> bbox)
{
	for (int i = 0; i < bbox.size(); ++i)
	{
		bbox[i][2] += bbox[i][0];
		bbox[i][3] += bbox[i][1];
	}

	return bbox;
}

vector<vector<double>> read_json::bbox_tlwh_to_xywh(vector<vector<double>> bbox)
{
	for (int i = 0; i < bbox.size(); ++i)
	{
		bbox[i][0] += bbox[i][2] / 2;
		bbox[i][1] += bbox[i][3] / 2;
	}

	return bbox;
}

vector<vector<double>> read_json::get_bbox_xyxy(vector<vector<vector<double>>> people_keypoints, double conf_thresh, double expand_coefficient, vector<int> keypoints_idx_used)
{
	vector<vector<double>> bbox_dict;

	int people_size = people_keypoints.size();

	for (int i = 0; i < people_size; ++i)
	{
		vector<double> people_bbox;

		int keypoints_size = keypoints_idx_used.size();

		int xy_min_idx = -1;

		for (int j = 0; j < keypoints_size; ++j)
		{
			int keypoint_idx = keypoints_idx_used[j];

			if (people_keypoints[i][keypoint_idx][2] > 0)
			{
				xy_min_idx = j;
				break;
			}
		}

		if (xy_min_idx >= 0)
		{
			int keypoint_idx = keypoints_idx_used[xy_min_idx];

			int min_x = people_keypoints[i][keypoint_idx][0];
			int min_y = people_keypoints[i][keypoint_idx][1];
			int max_x = min_x;
			int max_y = min_y;
			double conf = 0;
			int conf_size = 0;
			double weak_coefficient = 0.95;
			int zero_conf_num = xy_min_idx;

			for (int j = xy_min_idx + 1; j < keypoints_size; ++j)
			{
				keypoint_idx = keypoints_idx_used[j];

				if (people_keypoints[i][keypoint_idx][2] > 0)
				{
					int temp_x = people_keypoints[i][keypoint_idx][0];
					int temp_y = people_keypoints[i][keypoint_idx][1];

					if (temp_x < min_x)
					{
						min_x = temp_x;
					}
					else if (temp_x > max_x)
					{
						max_x = temp_x;
					}
					if (temp_y < min_y)
					{
						min_y = temp_y;
					}
					else if (temp_y > max_y)
					{
						max_y = temp_y;
					}
					conf += people_keypoints[i][keypoint_idx][2];
					++conf_size;
				}
				else
				{
					++zero_conf_num;
				}
			}
			conf = conf * pow(weak_coefficient, zero_conf_num) / conf_size;

			if (min_x < max_x && min_y < max_y && conf > conf_thresh)
			{
				people_bbox.emplace_back(min_x);
				people_bbox.emplace_back(((max_y + min_y) - (max_y - min_y) * expand_coefficient) / 2);
				people_bbox.emplace_back(max_x);
				people_bbox.emplace_back(((max_y + min_y) + (max_y - min_y) * expand_coefficient) / 2);
				people_bbox.emplace_back(conf);

				bbox_dict.emplace_back(people_bbox);
			}
		}
	}

	return bbox_dict;
}

vector<vector<double>> read_json::get_bbox_xywh(vector<vector<vector<double>>> people_keypoints, double conf_thresh, double expand_coefficient, vector<int> keypoints_idx_used)
{
	vector<vector<double>> bbox_dict;

	int people_size = people_keypoints.size();

	for (int i = 0; i < people_size; ++i)
	{
		vector<double> people_bbox;

		int keypoints_size = keypoints_idx_used.size();

		int xy_min_idx = -1;

		for (int j = 0; j < keypoints_size; ++j)
		{
			int keypoint_idx = keypoints_idx_used[j];

			if (people_keypoints[i][keypoint_idx][2] > 0)
			{
				xy_min_idx = j;
				break;
			}
		}

		if (xy_min_idx >= 0)
		{
			int keypoint_idx = keypoints_idx_used[xy_min_idx];

			int min_x = people_keypoints[i][keypoint_idx][0];
			int min_y = people_keypoints[i][keypoint_idx][1];
			int max_x = min_x;
			int max_y = min_y;
			double conf = 0;
			int conf_size = 0;
			double weak_coefficient = 0.95;
			int zero_conf_num = xy_min_idx;

			for (int j = xy_min_idx + 1; j < keypoints_size; ++j)
			{
				keypoint_idx = keypoints_idx_used[j];

				if (people_keypoints[i][keypoint_idx][2] > 0)
				{
					int temp_x = people_keypoints[i][keypoint_idx][0];
					int temp_y = people_keypoints[i][keypoint_idx][1];

					if (temp_x < min_x)
					{
						min_x = temp_x;
					}
					else if (temp_x > max_x)
					{
						max_x = temp_x;
					}
					if (temp_y < min_y)
					{
						min_y = temp_y;
					}
					else if (temp_y > max_y)
					{
						max_y = temp_y;
					}
					conf += people_keypoints[i][keypoint_idx][2];
					++conf_size;
				}
				else
				{
					++zero_conf_num;
				}
			}
			conf = conf * pow(weak_coefficient, zero_conf_num) / conf_size;

			if (min_x < max_x && min_y < max_y && conf > conf_thresh)
			{
				people_bbox.emplace_back((min_x + max_x) / 2);
				people_bbox.emplace_back((min_y + max_y) / 2);
				people_bbox.emplace_back(max_x - min_x);
				people_bbox.emplace_back((max_y - min_y) * expand_coefficient);
				people_bbox.emplace_back(conf);

				bbox_dict.emplace_back(people_bbox);
			}
		}
	}

	return bbox_dict;
}

vector<vector<double>> read_json::get_bbox_tlwh(vector<vector<vector<double>>> people_keypoints, double conf_thresh, double expand_coefficient, vector<int> keypoints_idx_used)
{
	vector<vector<double>> bbox_dict;

	int people_size = people_keypoints.size();

	for (int i = 0; i < people_size; ++i)
	{
		vector<double> people_bbox;

		int keypoints_size = keypoints_idx_used.size();

		int xy_min_idx = -1;

		for (int j = 0; j < keypoints_size; ++j)
		{
			int keypoint_idx = keypoints_idx_used[j];

			if (people_keypoints[i][keypoint_idx][2] > 0)
			{
				xy_min_idx = j;
				break;
			}
		}

		if (xy_min_idx >= 0)
		{
			int keypoint_idx = keypoints_idx_used[xy_min_idx];

			int min_x = people_keypoints[i][keypoint_idx][0];
			int min_y = people_keypoints[i][keypoint_idx][1];
			int max_x = min_x;
			int max_y = min_y;
			double conf = 0;
			int conf_size = 0;
			double weak_coefficient = 0.95;
			int zero_conf_num = xy_min_idx;

			for (int j = xy_min_idx + 1; j < keypoints_size; ++j)
			{
				keypoint_idx = keypoints_idx_used[j];

				if (people_keypoints[i][keypoint_idx][2] > 0)
				{
					int temp_x = people_keypoints[i][keypoint_idx][0];
					int temp_y = people_keypoints[i][keypoint_idx][1];

					if (temp_x < min_x)
					{
						min_x = temp_x;
					}
					else if (temp_x > max_x)
					{
						max_x = temp_x;
					}
					if (temp_y < min_y)
					{
						min_y = temp_y;
					}
					else if (temp_y > max_y)
					{
						max_y = temp_y;
					}
					conf += people_keypoints[i][keypoint_idx][2];
					++conf_size;
				}
				else
				{
					++zero_conf_num;
				}
			}
			conf = conf * pow(weak_coefficient, zero_conf_num) / conf_size;

			if (min_x < max_x && min_y < max_y && conf > conf_thresh)
			{
				people_bbox.emplace_back(min_x);
				people_bbox.emplace_back(((min_y + max_y) - (max_y - min_y) * expand_coefficient) / 2);
				people_bbox.emplace_back(max_x - min_x);
				people_bbox.emplace_back((max_y - min_y) * expand_coefficient);
				people_bbox.emplace_back(conf);

				bbox_dict.emplace_back(people_bbox);
			}
		}
	}

	return bbox_dict;
}

