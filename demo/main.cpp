/*********************************************************
 * @brief         libfacedetection人脸检测demo
 * @Inparam
 * @Outparam
 * @return
 * @author        hscoder
 * @date          2020-02-01
********************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <unistd.h>
#include "facedetectcnn.h"
#include <math.h>
//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;

struct Bbox
{
    int x;
    int y;
    int w;
    int h;
    float score;
};

int facedetection(cv::Mat &src , std::vector<Bbox> &bb);

inline void drawRect(cv::Mat &src , std::vector<Bbox> &boxes , cv::Scalar color = cv::Scalar(0 , 255 , 0));

int main(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    cv::Mat frame;
    while (cap.isOpened())
    {
        cap >> frame;
        if(frame.empty())
        {
            break;
        }

        cv::resize(frame , frame , cv::Size(640 , 480));

        cv::flip(frame , frame , 1);

        std::vector<Bbox> vec_boxes;
        auto start_time = std::chrono::steady_clock::now();
        facedetection(frame , vec_boxes);
        auto end_time = std::chrono::steady_clock::now();
        auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout<<"cost time: " << cost_time << std::endl;
        int fps = 1000/cost_time;
        drawRect(frame , vec_boxes);

        std::string text = "fps:" + std::to_string(fps);
        cv::putText(frame , text , cv::Point(20 , 50) , cv::FONT_HERSHEY_SIMPLEX , 1 , cv::Scalar(255 , 0 , 0));
        cv::imshow("frame" , frame);
        int key = cv::waitKey(int(cost_time));
        if (key == 27)
        {
           break;
        }

    }
    return 0;
}

int facedetection(cv::Mat &src , std::vector<Bbox> &bb)
{
    int *pResults = NULL;
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        std::cout << "can not alloc buffer" << std::endl;
        return -1;
    }
    pResults = facedetect_cnn(pBuffer, (unsigned char*)(src.ptr(0)), src.cols, src.rows, (int)src.step);
    std::cout << (pResults ? *pResults : 0) << "faces detected." << std::endl;
    Mat result_cnn = src.clone();

    for(int i = 0; i < (pResults ? *pResults : 0); i++)
    {
        short * p = ((short*)(pResults+1))+142*i;
        int x = p[0];
        int y = p[1];
        int w = p[2];
        int h = p[3];
        float score = std::sqrt(float(p[4])/100);

        std::cout << "score: " << score << std::endl;

        if (score >= 0.8)
        {
            Bbox face_boxes;
            face_boxes.x = x;
            face_boxes.y = y;
            face_boxes.w = w;
            face_boxes.h = h;
            face_boxes.score = score;

            bb.push_back(face_boxes);
        }
    }
    free(pBuffer);
    return 1;
}

void drawRect(cv::Mat &src , std::vector<Bbox> &boxes , cv::Scalar color)
{
    for(auto bb : boxes)
    {
        int x = bb.x;
        int y = bb.y;
        int w = bb.w;
        int h = bb.h;
        cv::rectangle(src , cv::Rect(x , y , w , h) , color , 2);
        std::string text = std::to_string(bb.score);
        cv::putText(src , text , cv::Point(x , y) , cv::FONT_HERSHEY_SIMPLEX , 0.6 , cv::Scalar(0 , 0 , 255));
    }
}
