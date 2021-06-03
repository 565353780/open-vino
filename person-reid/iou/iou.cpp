#include "iou.h"

iou::iou()
{

}

iou::~iou()
{

}

double iou::get_iou(cv::Rect rectA, cv::Rect rectB)
{
	if (rectA.x > rectB.x + rectB.width)
	{
		return 0.;
	}
	if (rectA.y > rectB.y + rectB.height)
	{
		return 0.;
	}
	if ((rectA.x + rectA.width) < rectB.x)
	{
		return 0.;
	}
	if ((rectA.y + rectA.height) < rectB.y)
	{
		return 0.;
	}

	double colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
	double rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);

	double intersection = colInt * rowInt;

	double areaA = rectA.width * rectA.height;
	double areaB = rectB.width * rectB.height;

	return intersection / (areaA + areaB - intersection);
}

void iou::get_iou_pairs(vector<cv::Rect> rects)
{
	iou_pairs.clear();

	iou_pairs.resize(rects.size());
	for (int i = 0; i < rects.size(); ++i)
	{
		iou_pairs[i].resize(rects.size());
	}

	for (int i = 0; i < rects.size(); ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			if (i == j)
			{
				iou_pairs[i][i] = 1;
			}
			else
			{
				iou_pairs[i][j] = get_iou(rects[i], rects[j]);

				iou_pairs[j][i] = iou_pairs[i][j];
			}
		}
	}
}

void iou::get_close_rect_id_pairs()
{
	close_rect_pairs.clear();

	for (int i = 0; i < iou_pairs.size(); ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (iou_pairs[i][j] > 0)
			{
				vector<int> close_rect_pair;

				close_rect_pair.emplace_back(j);
				close_rect_pair.emplace_back(i);

				close_rect_pairs.emplace_back(close_rect_pair);
			}
		}
	}
}

double iou::get_close_rect_dist2(cv::Rect rect_1, cv::Rect rect_2)
{
	double rect_dist = 0;

	rect_dist += (rect_1.x - rect_2.x) * (rect_1.x - rect_2.x);
	rect_dist += (rect_1.y - rect_2.y) * (rect_1.y - rect_2.y);
	rect_dist += (rect_1.x + rect_1.width - rect_2.x - rect_2.width) * (rect_1.x + rect_1.width - rect_2.x - rect_2.width);
	rect_dist += (rect_1.y + rect_1.height - rect_2.y - rect_2.height) * (rect_1.y + rect_1.height - rect_2.y - rect_2.height);

	return rect_dist;
}

double iou::get_close_rect_dist(cv::Rect rect_1, cv::Rect rect_2)
{
	return sqrt(get_close_rect_dist2(rect_1, rect_2));
}

void iou::get_rect_match_by_dist2(vector<cv::Rect> detections_rects, vector<cv::Rect> track_rects)
{
	detections_matches_by_dist2.clear();
	detections_match_dists_by_dist2.clear();

	detections_matches_by_dist2.resize(detections_rects.size());
	detections_match_dists_by_dist2.resize(detections_rects.size());

	for (int i = 0; i < detections_rects.size(); ++i)
	{
		detections_matches_by_dist2[i] = -1;
		detections_match_dists_by_dist2[i] = -1;
	}

	for (int i = 0; i < detections_rects.size(); ++i)
	{
		for (int j = 0; j < track_rects.size(); ++j)
		{
			double current_dist2 = get_close_rect_dist2(detections_rects[i], track_rects[j]);

			if (detections_match_dists_by_dist2[i] == -1 || detections_match_dists_by_dist2[i] > current_dist2)
			{
				detections_match_dists_by_dist2[i] = current_dist2;
				detections_matches_by_dist2[i] = j;
			}
		}

		/*int current_det_match = detections_matches_by_dist2[i];
		if (current_det_match != -1)
		{
			double min_track_rect_dist2_to_current_det_match = -1;

			for (int j = 0; j < track_rects.size(); ++j)
			{
				if (j != current_det_match)
				{
					double current_track_rect_dist2 = get_close_rect_dist2(track_rects[j], track_rects[current_det_match]);

					if (min_track_rect_dist2_to_current_det_match == -1 || min_track_rect_dist2_to_current_det_match > current_track_rect_dist2)
					{
						cout << "!!!! current dist id : " << j << " , " << current_det_match << " , " << current_track_rect_dist2 << endl;
						min_track_rect_dist2_to_current_det_match = current_track_rect_dist2;
					}
				}
			}

			if (min_track_rect_dist2_to_current_det_match > 0 && min_track_rect_dist2_to_current_det_match < detections_match_dists_by_dist2[i])
			{
				cout << "!!!!reset current match : " << i << " , match_dist : " << detections_match_dists_by_dist2[i] << " , min_track_dist : " << min_track_rect_dist2_to_current_det_match << endl;
				detections_match_dists_by_dist2[i] = -1;
				detections_matches_by_dist2[i] = -1;
			}
		}*/
	}
}

void iou::get_rect_match_by_dist(vector<cv::Rect> detections_rects, vector<cv::Rect> track_rects)
{
	detections_matches_by_dist.clear();
	detections_match_dists_by_dist.clear();

	detections_matches_by_dist.resize(detections_rects.size());
	detections_match_dists_by_dist.resize(detections_rects.size());

	for (int i = 0; i < detections_rects.size(); ++i)
	{
		detections_matches_by_dist[i] = -1;
		detections_match_dists_by_dist[i] = -1;
	}

	for (int i = 0; i < detections_rects.size(); ++i)
	{
		for (int j = 0; j < track_rects.size(); ++j)
		{
			double current_dist = get_close_rect_dist(detections_rects[i], track_rects[j]);

			if (detections_match_dists_by_dist[i] == -1 || detections_match_dists_by_dist[i] > current_dist)
			{
				detections_match_dists_by_dist[i] = current_dist;
				detections_matches_by_dist[i] = j;
			}
		}
	}
}

double iou::get_descriptor_dist(cv::Mat descriptor_1, cv::Mat descriptor_2)
{
	double xy = descriptor_1.dot(descriptor_2);
	double xx = descriptor_1.dot(descriptor_1);
	double yy = descriptor_2.dot(descriptor_2);
	double norm = sqrt(xx * yy) + 1e-6;
	return 1.0 - xy / norm;
}

void iou::get_rect_match_by_descriptor(vector<cv::Mat> detections_descriptors, vector<cv::Mat> track_descriptors)
{
	detections_matches_by_descriptor.clear();
	detections_match_dists_by_descriptor.clear();

	detections_matches_by_descriptor.resize(detections_descriptors.size());
	detections_match_dists_by_descriptor.resize(detections_descriptors.size());

	for (int i = 0; i < detections_descriptors.size(); ++i)
	{
		detections_matches_by_descriptor[i] = -1;
		detections_match_dists_by_descriptor[i] = -1;
	}

	for (int i = 0; i < detections_descriptors.size(); ++i)
	{
		for (int j = 0; j < track_descriptors.size(); ++j)
		{
			double current_dist = get_descriptor_dist(detections_descriptors[i], track_descriptors[j]);

			if (detections_match_dists_by_descriptor[i] == -1 || detections_match_dists_by_descriptor[i] > current_dist)
			{
				detections_match_dists_by_descriptor[i] = current_dist;
				detections_matches_by_descriptor[i] = j;
			}
		}
	}
}

cv::Rect iou::get_correct_rect(cv::Size image_size, cv::Rect rect)
{
	if (rect.x + rect.width < 0)
	{
		rect.x = 0;
		rect.width = 0;
	}
	else if (rect.x >= image_size.width)
	{
		rect.x = image_size.width - 1;
		rect.width = 0;
	}

	if (rect.y + rect.height < 0)
	{
		rect.y = 0;
		rect.height = 0;
	}
	else if (rect.y >= image_size.height)
	{
		rect.y = image_size.height - 1;
		rect.height = 0;
	}

	if (rect.x < 0)
	{
		rect.x = 0;
	}
	else if (rect.x + rect.width >= image_size.width)
	{
		rect.width = image_size.width - 1 - rect.x;
	}

	if (rect.y < 0)
	{
		rect.y = 0;
	}
	else if (rect.y + rect.height >= image_size.height)
	{
		rect.height = image_size.height - 1 - rect.y;
	}

	return rect;
}