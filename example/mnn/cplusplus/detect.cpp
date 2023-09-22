//
// Created by zj on 23-9-21.
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/AutoTime.hpp>
#include <opencv2/opencv.hpp>

using namespace MNN;
using namespace MNN::CV;

float Sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float Tanh(float x) {
	return 2.0f / (1.0f + exp(-2 * x)) - 1;
}

class TargetBox {
 private:
  float GetWidth() { return (x2 - x1); };
  float GetHeight() { return (y2 - y1); };

 public:
  int x1;
  int y1;
  int x2;
  int y2;

  int category;
  float score;

  float area() { return GetWidth() * GetHeight(); };
};

float IntersectionArea(const TargetBox &a, const TargetBox &b) {
	if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
		// no intersection
		return 0.f;
	}

	float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
	float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

	return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b) {
	return (a.score > b.score);
}

//NMS处理
int nmsHandle(std::vector<TargetBox> &src_boxes, std::vector<TargetBox> &dst_boxes) {
	std::vector<int> picked;

	sort(src_boxes.begin(), src_boxes.end(), scoreSort);

	for (int i = 0; i < src_boxes.size(); i++) {
		int keep = 1;
		for (int j = 0; j < picked.size(); j++) {
			//交集
			float inter_area = IntersectionArea(src_boxes[i], src_boxes[picked[j]]);
			//并集
			float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
			float IoU = inter_area / union_area;

			if (IoU > 0.45 && src_boxes[i].category == src_boxes[picked[j]].category) {
				keep = 0;
				break;
			}
		}

		if (keep) {
			picked.push_back(i);
		}
	}

	for (int i = 0; i < picked.size(); i++) {
		dst_boxes.push_back(src_boxes[picked[i]]);
	}

	return 0;
}

/*
 * 1. 创建模型解释器/会话
 * 2. 读取图片, 进行图像预处理
 * 3. 模型推理, 提取输出结果
 * 4. 推理结果后处理, 得到最终预测结果
 * 5. 绘制预测结果
 */
int main(int argc, char *argv[]) {
	if (argc != 3) {
		MNN_PRINT("Usage: ./pictureRecognition.out model.mnn input.jpg\n");
		return 0;
	}

	// 1. 创建模型解释器/会话
	std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]), Interpreter::destroy);
	net->setCacheFile(".cachefile");
	net->setSessionMode(Interpreter::Session_Backend_Auto);
	net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
	ScheduleConfig config;
	config.type = MNN_FORWARD_AUTO;
	// BackendConfig bnconfig;
	// bnconfig.precision = BackendConfig::Precision_Low;
	// config.backendConfig = &bnconfig;
	auto session = net->createSession(config);

	auto input = net->getSessionInput(session, NULL);
	input->printShape();
	auto shape = input->shape();
	// Set Batch Size
	shape[0] = argc - 2;
	net->resizeTensor(input, shape);
	net->resizeSession(session);
	float memoryUsage = 0.0f;
	net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryUsage);
	float flops = 0.0f;
	net->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
	int backendType[2];
	net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);
	MNN_PRINT("Session Info: memory use %f MB, flops is %f M, backendType is %d, batch size = %d\n",
			  memoryUsage,
			  flops,
			  backendType[0],
			  argc - 2);
	auto output = net->getSessionOutput(session, NULL);
	if (nullptr == output || output->elementSize() == 0) {
		MNN_ERROR("Resize error, the model can't run batch: %d\n", shape[0]);
		return 0;
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	// Set Data
	std::shared_ptr<Tensor> inputUser(new Tensor(input, Tensor::TENSORFLOW));
	inputUser->printShape();
	auto bpp = inputUser->channel();
	auto size_h = inputUser->height();
	auto size_w = inputUser->width();
	MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

	auto inputPatch = argv[2];
	cv::Mat src_img = cv::imread(inputPatch, cv::IMREAD_COLOR);
	if (src_img.empty()) {
		MNN_ERROR("Can't open %s\n", inputPatch);
		return 0;
	}
	int src_width = src_img.cols;
	int src_height = src_img.rows;

	cv::Mat img;
	cv::resize(src_img, img, cv::Size(size_w, size_h));
//	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

//	float mean[3] = {103.94f, 116.78f, 123.68f};
//	float normals[3] = {0.017f, 0.017f, 0.017f};
	img.convertTo(img, CV_32F);
//	cv::Mat blob = img - cv::Scalar(103.94f, 116.78f, 123.68f);
	cv::Mat blob = img * (1 / 255.);

	int width = blob.cols;
	int height = blob.rows;
	MNN_PRINT("origin size: %d, %d\n", width, height);

	int input_size = inputUser->elementSize();
	assert(1 * 3 * 352 * 352 == input_size);
	::memcpy(inputUser->host<uchar>(), blob.data, input_size * 4);

	printf("inputUser->elementSize() = %d\n", inputUser->elementSize());
	input->copyFromHostTensor(inputUser.get());
	for (int i = 0; i < 50; i++) {
		std::cout << input->host<float>()[i] << " ";
	}
	std::cout << std::endl;
	for (int i = input_size - 50; i < input_size; i++) {
		std::cout << input->host<float>()[i] << " ";
	}
	std::cout << std::endl;

	// Infer
	net->runSession(session);
	auto dimType = output->getDimensionType();
	if (output->getType().code != halide_type_float) {
		std::cout << "dimType = Tensor::TENSORFLOW" << std::endl;
		dimType = Tensor::TENSORFLOW;
	}
	std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));
	outputUser->printShape();
	output->copyToHostTensor(outputUser.get());

	// handle output tensor
	std::vector<TargetBox> target_boxes;
	// NMS处理
	std::vector<TargetBox> nms_boxes;

	int category_num = 1;
	float thresh = 0.65;
	MNN_PRINT("For Image: %s\n", argv[2]);
	auto size = outputUser->stride(0);
	int B = outputUser->batch();
	int C = outputUser->channel();
	int H = outputUser->height();
	int W = outputUser->width();
	printf("dim=%d size=%d C=%d H=%d W=%d\n", outputUser->dimensions(), B, C, H, W);

	for (int hi = 0; hi < H; hi++) {
		for (int wi = 0; wi < W; wi++) {
			float data[C];
			for (int ci = 0; ci < C; ci++) {
				int idx = ci * W * H + hi * W + wi;
				data[ci] = outputUser->host<float>()[idx];
			}

			int category = 0;
			float max_score = 0.0f;
			for (int i = 0; i < category_num; i++) {
				if (i == 0) {
					max_score = data[5 + i];
				} else {
					float cls_score = data[5 + i];
					if (max_score < cls_score) {
						category = i;
						max_score = cls_score;
					}
				}
			}
			float obj_score = data[0];
			float score = std::pow(max_score, 0.4) * std::pow(obj_score, 0.6);

			if (score > thresh) {
				// 解析坐标
				float x_offset = Tanh(data[1]);
				float y_offset = Tanh(data[2]);
				float box_width = Sigmoid(data[3]);
				float box_height = Sigmoid(data[4]);

				float cx = 1.0 * (wi + x_offset) / W;
				float cy = 1.0 * (hi + y_offset) / H;

				// xc/yc/box_w/box_h -> x1/y1/x2/y2
				int x1 = (int)((cx - box_width * 0.5) * src_width);
				int y1 = (int)((cy - box_height * 0.5) * src_height);
				int x2 = (int)((cx + box_width * 0.5) * src_width);
				int y2 = (int)((cy + box_height * 0.5) * src_height);

				target_boxes.push_back(TargetBox{x1, y1, x2, y2, category, score});
			}
		}
	}

	nmsHandle(target_boxes, nms_boxes);

	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
	std::cout << "Process() took " << fp_ms.count() << " ms." << std::endl;

	net->updateCacheFile(session);

	// draw result
	for (auto box : nms_boxes) {
		printf("x1:%d y1:%d x2:%d y2:%d  %s:%.2f%%\n", box.x1, box.y1, box.x2, box.y2, "box", box.score * 100);

		cv::rectangle(src_img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 0, 255), 2);
		cv::putText(src_img,
					"box",
					cv::Point(box.x1, box.y1),
					cv::FONT_HERSHEY_SIMPLEX,
					0.75,
					cv::Scalar(0, 255, 0),
					2);
	}
	cv::imwrite("result.jpg", src_img);

	return 0;
}