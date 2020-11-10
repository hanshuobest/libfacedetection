
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
#include <file_system.hpp>
#include <portability_fixes.hpp>
#include <wildcard.hpp>

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

static bool preprocess(cv::Mat &frame, const cv::Size re_size = cv::Size(640, 480), bool flip_flag = true);

static int facedetection(cv::Mat &src, std::vector<Bbox> &bb);

inline void drawRect(cv::Mat &src, std::vector<Bbox> &boxes, cv::Scalar color = cv::Scalar(0, 255, 0));

bool get_file_lst(std::vector<std::string> &files_lst, const std::string &suffix, const std::string &dir_path)
{
    if (suffix.empty() || dir_path.empty())
    {
        std::cerr << "please check input suffix or dir_path\n";
        return false;
    }

    files_lst = stlplus::folder_wildcard(dir_path, suffix, false, true);
    return true;
}

//字符串分割函数
std::vector<std::string> str_split(std::string str, std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern; //扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "exe image.jpg" << std::endl;
        return 0;
    }

    std::string dir_path = argv[1];
    std::vector<std::string> img_lst;
    get_file_lst(img_lst, "*.jpg", dir_path);
    std::sort(img_lst.begin(), img_lst.end(), [](std::string a, std::string b) { return atoi(str_split(a, ".")[0].c_str()) < atoi(str_split(b, ".")[0].c_str()); });

    int count = 0;
    for (auto name : img_lst)
    {
        std::string full_name = dir_path + "/" + name;
        cv::Mat frame = cv::imread(full_name);
        if (!frame.data)
        {
            std::cerr << name << ": read image is null" << std::endl ;
            continue;
        }

        preprocess(frame, cv::Size(640, 480), false);
        std::vector<Bbox> vec_boxes;

        auto start_time = std::chrono::steady_clock::now();
        facedetection(frame, vec_boxes);
        auto end_time = std::chrono::steady_clock::now();
        auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "cost time: " << cost_time << std::endl;
        int fps = 1000 / cost_time;
        drawRect(frame, vec_boxes);

        std::string save_name = std::to_string(count) + ".jpg";
        cv::imwrite(save_name, frame);
        count++;
    }

    // std::string filename = argv[1];
    // cv::Mat frame = cv::imread(filename);

    // preprocess(frame, cv::Size(640, 480), true);
    // std::vector<Bbox> vec_boxes;

    // auto start_time = std::chrono::steady_clock::now();
    // facedetection(frame, vec_boxes);
    // auto end_time = std::chrono::steady_clock::now();
    // auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    // std::cout << "cost time: " << cost_time << std::endl;
    // int fps = 1000 / cost_time;
    // drawRect(frame, vec_boxes);

    // std::string text = "fps:" + std::to_string(fps);
    // cv::putText(frame, text, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
    // cv::imshow("frame", frame);
    // cv::waitKey(0);

    return 0;
}

bool preprocess(cv::Mat &frame, const cv::Size re_size, bool flip_flag)
{
    if (!frame.data)
    {
        std::cerr << "read image is null\n";
        return false;
    }
    cv::resize(frame, frame, re_size);
    if (flip_flag)
    {
        cv::flip(frame, frame, 1);
    }

    return true;
}

int facedetection(cv::Mat &src, std::vector<Bbox> &bb)
{
    int *pResults = NULL;
    unsigned char *pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if (!pBuffer)
    {
        std::cout << "can not alloc buffer" << std::endl;
        return -1;
    }
    pResults = facedetect_cnn(pBuffer, (unsigned char *)(src.ptr(0)), src.cols, src.rows, (int)src.step);
    std::cout << (pResults ? *pResults : 0) << "faces detected." << std::endl;
    Mat result_cnn = src.clone();

    for (int i = 0; i < (pResults ? *pResults : 0); i++)
    {
        short *p = ((short *)(pResults + 1)) + 142 * i;
        int x = p[0];
        int y = p[1];
        int w = p[2];
        int h = p[3];
        float score = std::sqrt(float(p[4]) / 100);

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

void drawRect(cv::Mat &src, std::vector<Bbox> &boxes, cv::Scalar color)
{
    for (auto bb : boxes)
    {
        int x = bb.x;
        int y = bb.y;
        int w = bb.w;
        int h = bb.h;
        cv::rectangle(src, cv::Rect(x, y, w, h), color, 2);
        std::string text = std::to_string(bb.score);
        cv::putText(src, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));
    }
}
