#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <fstream>


using namespace std;
using namespace chrono;
using namespace cv;
using namespace dnn;

const string PATH = "./src/";
const string CLASSES_PATH = PATH + "classes.txt";
const string MODEL_PATH = PATH + "yolov5n.onnx";
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

const vector<Scalar> colors = {Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0)};

// import classes
vector<string> load_class_list()
{
    vector<string> class_list;
    ifstream ifs(CLASSES_PATH);
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

// loading the model
void load_net(Net &net, bool is_cuda)
{
    auto result = readNetFromONNX(MODEL_PATH);

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


struct Detection
{
    int class_id;
    float confidence;
    Rect box;
};

Mat format_yolov5(const Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

void detect(Mat &image, Net &net, vector<Detection> &output, const vector<string> &className) {
    Mat blob;

    auto input_image = format_yolov5(image);
    
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data + 5;
            Mat scores(1, className.size(), CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(Rect(left, top, width, height));
            }

        }

        data += 85;

    }

    vector<int> nms_result;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

int main(int argc, char **argv)
{

    vector<string> class_list = load_class_list();

    Mat frame;
    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        cerr << "Error opening video file\n";
        return -1;
    }

    bool is_cuda = cudaGetDevice;

    Net net;
    load_net(net, is_cuda);

    auto start = high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            cout << "End of stream\n";
            break;
        }

        vector<Detection> output;
        detect(frame, net, output, class_list);

        frame_count++;
        total_frames++;

        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            rectangle(frame, box, color, 3);

            rectangle(frame, Point(box.x, box.y - 20), Point(box.x + box.width, box.y), color, FILLED);
            putText(frame, class_list[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }

        auto end = high_resolution_clock::now();
        fps = frame_count * 1000.0 / duration_cast<milliseconds>(end - start).count();

        if (fps > 0)
        {

            ostringstream fps_label;
            fps_label << fixed << setprecision(2);
            fps_label << "FPS: " << fps;
            string fps_label_str = fps_label.str();

            putText(frame, fps_label_str.c_str(), Point(10, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        imshow("output", frame);

        if (waitKey(1) != -1)
        {
            capture.release();
            cout << "finished by user\n";
            break;
        }
    }

    cout << "Total frames: " << total_frames << "\n";

    return 0;
}