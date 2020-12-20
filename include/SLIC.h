//
// Created by wang on 2020/12/17.
//

#ifndef ORB_SLAM2_SLIC_H
#define ORB_SLAM2_SLIC_H

#define ITERATION_NUM 3
#define THREAD_NUM 4
#define SP_SIZE 9
#define HUBER_RANGE 0.4

#include <iostream>
#include <vector>
#include "superpixel.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include <thread>
#include <opencv2/highgui/highgui.hpp>

class SLIC{
public:
    SLIC(int h,int w):image_height(h),image_width(w){
        sp_width = image_width / SP_SIZE;
        sp_height = image_height / SP_SIZE;
        superpixel_seeds.resize(sp_width * sp_height);
        superpixel_index.resize(image_width * image_height);
    };

    void generate_super_pixels(cv::Mat &_image, cv::Mat &_depth);

    std::vector<Superpixel_seed> superpixel_seeds;

    void debug_show();


protected:
    bool calculate_cost(
                float &nodepth_cost, float &depth_cost,
                const float &pixel_R, const float &pixel_G, const float &pixel_B, const float &pixel_inverse_depth,
                const int &x, const int &y,
                const int &sp_x, const int &sp_y);

    void update_pixels_kernel(int thread_i, int thread_num);
    void update_pixels();
    void update_seeds_kernel(
                int thread_i, int thread_num);
    void update_seeds();
    void initialize_seeds_kernel(
                int thread_i, int thread_num);
    void initialize_seeds();

    std::vector<int> superpixel_index;
    int image_height, image_width;
    int sp_width, sp_height;

    cv::Mat image;
    cv::Mat depth;

};


#endif //ORB_SLAM2_SLIC_H
