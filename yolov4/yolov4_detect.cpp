#include "cuda_runtime.h"
#include <fstream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace chrono;
using namespace cv;
using namespace dnn;

const string PATH = "./src/";
const string CLASSES_PATH = PATH + "classes.txt";
const string CONFIG_PATH = PATH + "yolov4-tiny.cfg";
const string MODEL_PATH = PATH + "yolov4-tiny.weights";
const int INPUT_WIDTH = 416;
const int INPUT_HEIGHT = 416;
const float CONFIDENCE_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.4;

const vector<Scalar> colors = {Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0)};

vector<string> load_class_list() {
    vector<string> class_list;
    ifstream ifs(CLASSES_PATH);
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    } 
    return class_list;
}

void load_net(Net &net, bool is_cuda) {

    auto result = readNetFromDarknet(CONFIG_PATH, MODEL_PATH);
    
    if (is_cuda) {
        cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(DNN_BACKEND_CUDA);
        result.setPreferableTarget(DNN_TARGET_CUDA);
    } else {
        cout << "Running on CPU\n";
        result.setPreferableBackend(DNN_BACKEND_OPENCV);
        result.setPreferableTarget(DNN_TARGET_CPU);
    }

    net = result;
}

int main(int argc, char ** argv)
{

    vector<string> class_list = load_class_list();

    Mat frame;
    VideoCapture capture(0);
    if(!capture.isOpened()){
	    cerr << "Error opening video file\n";
	    return -1;
	}

    bool is_cuda = cudaGetDevice;
    
    Net net;
    load_net(net, is_cuda);

    auto model = DetectionModel(net);
    model.setInputParams(1./255, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true);

    auto start = high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    while(true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            cout << "End of stream\n";
            break;
        }

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        model.detect(frame, classIds, confidences, boxes, CONFIDENCE_THRESHOLD, NMS_THRESHOLD);
        
        frame_count++;
        total_frames++;

        int detections = classIds.size();

        for (int i = 0; i < detections; ++i) {

            auto box = boxes[i];
            auto classId = classIds[i];
            const auto color = colors[classId % colors.size()];
            rectangle(frame, box, color, 3);

            rectangle(frame, Point(box.x, box.y - 20), Point(box.x + box.width, box.y), color, FILLED);
            putText(frame, class_list[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }

        auto end = high_resolution_clock::now();
        fps = frame_count * 1000.0 / duration_cast<milliseconds>(end - start).count();

        if (fps > 0) {

            ostringstream fps_label;
            fps_label << fixed << setprecision(2);
            fps_label << "FPS: " << fps;
            string fps_label_str = fps_label.str();

            putText(frame, fps_label_str.c_str(), Point(10, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            
        }

        imshow("DETECTED WINDOW", frame);

        if(waitKey(1) != -1) {
            capture.release();
            cout << "finished by user\n";
            break;
        }
    }

    cout << "Total frames: " << total_frames << "\n";

    return 0;
}

