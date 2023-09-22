//
//  pictureRecognition.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
 * Usage: Compile
 * 		$ cd /path/to/FastestDet/example/mnn/classify/
 * 		$ mkdir build && cd build
 * 		$ cmake .. && make -j12
 *
 * Usage: Classify
 * 		$ ./classify ../../mobilenet_demo/mobilenet_v1.mnn ../../mobilenet_demo/ILSVRC2012_val_00049999.JPEG
 *
 * INPUT FORMAT: NHWC
 * OUTPUT FORMAT:
 */

#include <cstdio>
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

int main(int argc, const char *argv[]) {
	if (argc != 3) {
		MNN_PRINT("Usage: ./pictureRecognition.out model.mnn input0.jpg\n");
		return 0;
	}
	// Create Interpreter and Session
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

	// Set Data
	std::shared_ptr<Tensor> inputUser(new Tensor(input, Tensor::TENSORFLOW));
	inputUser->printShape();
	auto bpp = inputUser->channel();
	auto size_h = inputUser->height();
	auto size_w = inputUser->width();
	MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

	auto inputPatch = argv[2];
	cv::Mat img = cv::imread(inputPatch, cv::IMREAD_COLOR);
	if (img.empty()) {
		MNN_ERROR("Can't open %s\n", inputPatch);
		return 0;
	}

	cv::resize(img, img, cv::Size(size_w, size_h));
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

//	float mean[3] = {103.94f, 116.78f, 123.68f};
//	float normals[3] = {0.017f, 0.017f, 0.017f};
	img.convertTo(img, CV_32F);
	cv::Mat blob = img - cv::Scalar(103.94f, 116.78f, 123.68f);
	blob *= 0.017f;

	int width = blob.cols;
	int height = blob.rows;
	MNN_PRINT("origin size: %d, %d\n", width, height);

	int input_size = inputUser->elementSize();
	assert(1 * 3 * 224 * 224 == input_size);
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
	auto type = outputUser->getType();
	for (int batch = 0; batch < shape[0]; ++batch) {
		MNN_PRINT("For Image: %s\n", argv[batch + 2]);
		auto size = outputUser->stride(0);
		std::vector<std::pair<int, float>> tempValues(size);
		if (type.code == halide_type_float) {
			printf("type.code == halide_type_float\n");
			auto values = outputUser->host<float>() + batch * outputUser->stride(0);
			for (int i = 0; i < size; ++i) {
				tempValues[i] = std::make_pair(i, values[i]);
			}
		}
		if (type.code == halide_type_uint && type.bytes() == 1) {
			auto values = outputUser->host<uint8_t>() + batch * outputUser->stride(0);
			for (int i = 0; i < size; ++i) {
				tempValues[i] = std::make_pair(i, values[i]);
			}
		}
		if (type.code == halide_type_int && type.bytes() == 1) {
			auto values = outputUser->host<int8_t>() + batch * outputUser->stride(0);
			for (int i = 0; i < size; ++i) {
				tempValues[i] = std::make_pair(i, values[i]);
			}
		}
		// Find Max
		std::sort(tempValues.begin(), tempValues.end(),
				  [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });

		int length = size > 10 ? 10 : size;
		for (int i = 0; i < length; ++i) {
			MNN_PRINT("%d, %f\n", tempValues[i].first, tempValues[i].second);
		}
	}
	net->updateCacheFile(session);
	return 0;
}
