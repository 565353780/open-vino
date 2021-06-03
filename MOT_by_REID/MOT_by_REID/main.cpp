#include <iostream>
#include <vector>
#include <string>
#include <experimental/filesystem>
#include <Windows.h>
#include <ctime>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <cstdlib>
#include "read_json.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main()
{
	VideoCapture cap("E:/chLi/deep_sort_pytorch/2019_2person_cut.mp4");
	long frameToStart = 0;
	double conf_thresh = 0.5;
	double expand_coefficient = 1.2;
	int max_wait_time = 60;

	if (!cap.isOpened())
	{
		return 0;
	}

	long totalFrameNumber = cap.get(CAP_PROP_FRAME_COUNT);
	double rate = cap.get(CAP_PROP_FPS);

	cap.set(CAP_PROP_POS_FRAMES, frameToStart);

	read_json* json_reader = new read_json();

	int frame_idx = int(frameToStart);

	while (frame_idx >= 0)
	{
		Mat frame;

		cap >> frame;

		int frame_len = json_reader->intlen(frame_idx);

		stringstream ss;

		ss << "E:/chLi/openpose-master/json/2019_2person_cut/2019_2person_cut_";
		for (int i = 0; i < 12 - frame_len; ++i)
		{
			ss << "0";
		}
		ss << frame_idx;
		ss << "_keypoints.json";

		string json_file = ss.str();

		clock_t start = clock();
		while (!experimental::filesystem::exists(json_file))
		{
			if (int(clock() - start) < max_wait_time * 1000)
			{
				Sleep(500);
			}
			else
			{
				return 1;
			}
		}

		vector<vector<vector<double>>> people_keypoints;

		people_keypoints = json_reader->load_json(json_file);

		vector<int> keypoints_idx_used({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
		//vector<int> keypoints_idx_used({5, 2, 9, 12, 13, 14, 19, 20, 21, 22, 23, 24});

		vector<vector<double>> people_bboxs;

		people_bboxs = json_reader->get_bbox_xyxy(people_keypoints, conf_thresh, expand_coefficient, keypoints_idx_used);
		
		for (int i = 0; i < people_bboxs.size(); ++i)
		{
			rectangle(frame, Point(int(people_bboxs[i][0]), int(people_bboxs[i][1])), Point(int(people_bboxs[i][2]), int(people_bboxs[i][3])), Scalar(0, 255, 255), 1, 1, 0);
		}

		imshow("test", frame);
		waitKey(1);

		//Net net = readNetFromTensorflow("E:/chLi/tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb");
		//Net net = readNetFromONNX("E:/chLi/OpenVINO/osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.onnx");
		Net net = readNetFromTorch("E:\chLi\py-curling-reconstruct\LapNet\trained_model\LapNet_chkpt_better_epoch9969_GPU0.pth");
		if (net.empty())
		{
			cout << "error:no model" << endl;
		}
		net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
		net.setPreferableTarget(DNN_TARGET_CPU);

		printf("ssd network model loaded...\n");
		Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(), true, false, 5);
		net.setInput(blob);

		float threshold = 0.5;
		Mat detection = net.forward();

		//cout << prob << endl;

		++frame_idx;

		break;
	}

	return 1;
}