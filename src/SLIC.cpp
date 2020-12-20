//
// Created by wang on 2020/12/17.
//
#include "SLIC.h"

void SLIC::generate_super_pixels(cv::Mat &_image, cv::Mat &_depth)
{
    image = _image.clone();
    depth = _depth.clone();
    superpixel_seeds.clear();
    superpixel_seeds.resize(sp_width * sp_height);
    superpixel_index.clear();
    superpixel_index.resize(image_width * image_height);
    initialize_seeds();
    for (int it_i = 0; it_i < ITERATION_NUM; it_i++)
    {
        update_pixels();
        update_seeds();
    }
    for (int j = 0; j < image_height; j++)
        for (int i = 0; i < image_width; i++) {
            int sp_index = superpixel_index[j * image_width + i];
            superpixel_seeds[sp_index].size++;
        }
}

void SLIC::initialize_seeds()
{
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < THREAD_NUM; i++)
    {
        std::thread this_thread(&SLIC::initialize_seeds_kernel, this, i, THREAD_NUM);
        thread_pool.push_back(std::move(this_thread));
    }
    for (int i = 0; i < thread_pool.size(); i++)
        if (thread_pool[i].joinable())
            thread_pool[i].join();
}

void SLIC::initialize_seeds_kernel(
        int thread_i, int thread_num)
{
    int step = superpixel_seeds.size() / thread_num;
    int begin_index = step * thread_i;
    int end_index = begin_index + step;
    if (thread_i == thread_num - 1)
        end_index = superpixel_seeds.size();
    for (int seed_i = begin_index; seed_i < end_index; seed_i++)
    {
        int sp_x = seed_i % sp_width;
        int sp_y = seed_i / sp_width;
        int image_x = sp_x * SP_SIZE + SP_SIZE / 2;
        int image_y = sp_y * SP_SIZE + SP_SIZE / 2;
        image_x = image_x < (image_width - 1) ? image_x : (image_width - 1);
        image_y = image_y < (image_height - 1) ? image_y : (image_height - 1);
        Superpixel_seed this_sp;
        this_sp.x = image_x;
        this_sp.y = image_y;
        this_sp.size = 0;
        // RGB
        cv::Vec3b pix = image.at<cv::Vec3b>(image_y, image_x);
        this_sp.R = pix[2];
        this_sp.G = pix[1];
        this_sp.B = pix[0];

        this_sp.stable = false;
        this_sp.mean_depth = depth.at<float>(image_y, image_x);
        if(this_sp.mean_depth < 0.01)
        {
            int check_x_begin = sp_x * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
            int check_y_begin = sp_y * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
            int check_x_end = check_x_begin + SP_SIZE * 2;
            int check_y_end = check_y_begin + SP_SIZE * 2;
            check_x_begin = check_x_begin > 0 ? check_x_begin : 0;
            check_y_begin = check_y_begin > 0 ? check_y_begin : 0;
            check_x_end = check_x_end < image_width - 1 ? check_x_end : image_width - 1;
            check_y_end = check_y_end < image_height - 1 ? check_y_end : image_height - 1;
            bool find_depth = false;
            for (int check_j = check_y_begin; check_j < check_y_end; check_j++)
            {
                for (int check_i = check_x_begin; check_i < check_x_end; check_i ++)
                {
                    float this_depth = depth.at<float>(check_j, check_i);
                    if(this_depth > 0.01)
                    {
                        this_sp.mean_depth = this_depth;
                        find_depth = true;
                        break;
                    }
                }
                if(find_depth)
                    break;
            }
        }
        superpixel_seeds[seed_i] = this_sp;
    }
}

void SLIC::update_pixels()
{
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < THREAD_NUM; i++)
    {
        std::thread this_thread(&SLIC::update_pixels_kernel, this, i, THREAD_NUM);
        thread_pool.push_back(std::move(this_thread));
    }
    for (int i = 0; i < thread_pool.size(); i++)
        if (thread_pool[i].joinable())
            thread_pool[i].join();
}

void SLIC::update_pixels_kernel(
        int thread_i, int thread_num)
{
    int step_row = image_height / thread_num;
    int start_row = step_row * thread_i;
    int end_row = start_row + step_row;
    if(thread_i == thread_num - 1)
        end_row = image_height;
    for(int row_i = start_row; row_i < end_row; row_i++)
        for(int col_i = 0; col_i < image_width; col_i++)
        {
            if(superpixel_seeds[superpixel_index[row_i * image_width + col_i]].stable)
                continue;
            // RGB
            cv::Vec3b pix = image.at<cv::Vec3b>(row_i, col_i);
            float my_R = pix[2];
            float my_G = pix[1];
            float my_B = pix[0];

            float my_inv_depth = 0.0;
            if (depth.at<float>(row_i, col_i) > 0.01)
                my_inv_depth = 1.0 / depth.at<float>(row_i, col_i);
            int base_sp_x = col_i / SP_SIZE;
            int base_sp_y = row_i / SP_SIZE;
            float min_dist_depth = 1e6;
            int min_sp_index_depth = -1;
            float min_dist_nodepth = 1e6;
            int min_sp_index_nodepth = -1;
            bool all_has_depth = true;
            for(int check_i = -1; check_i <= 1; check_i ++)
                for(int check_j = -1; check_j <= 1; check_j ++)
                {
                    int check_sp_x = base_sp_x + check_i;
                    int check_sp_y = base_sp_y + check_j;
                    int dist_sp_x = fabs(check_sp_x * SP_SIZE + SP_SIZE/2 - col_i);
                    int dist_sp_y = fabs(check_sp_y * SP_SIZE + SP_SIZE/2 - row_i);
                    if (dist_sp_x < SP_SIZE && dist_sp_y < SP_SIZE &&
                        check_sp_x >= 0 && check_sp_x < sp_width &&
                        check_sp_y >= 0 && check_sp_y < sp_height)
                    {
                        float dist_depth, dist_nodepth;
                        all_has_depth &= calculate_cost(
                                dist_nodepth,
                                dist_depth,
                                my_R,my_G,my_B, my_inv_depth,
                                col_i, row_i, check_sp_x, check_sp_y);
                        if (dist_depth < min_dist_depth)
                        {
                            min_dist_depth = dist_depth;
                            min_sp_index_depth = (base_sp_y + check_j) * sp_width + base_sp_x + check_i;
                        }
                        if (dist_nodepth < min_dist_nodepth)
                        {
                            min_dist_nodepth = dist_nodepth;
                            min_sp_index_nodepth = (base_sp_y + check_j) * sp_width + base_sp_x + check_i;
                        }
                    }
                }
            if(all_has_depth)
            {
                superpixel_index[row_i * image_width + col_i] = min_sp_index_depth;
                superpixel_seeds[min_sp_index_depth].stable = false;
            }
            else
            {
                superpixel_index[row_i * image_width + col_i] = min_sp_index_nodepth;
                superpixel_seeds[min_sp_index_nodepth].stable = false;
            }
        }
}

void SLIC::update_seeds()
{
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < THREAD_NUM; i++)
    {
        std::thread this_thread(&SLIC::update_seeds_kernel, this, i, THREAD_NUM);
        thread_pool.push_back(std::move(this_thread));
    }
    for (int i = 0; i < thread_pool.size(); i++)
        if (thread_pool[i].joinable())
            thread_pool[i].join();
}

void SLIC::update_seeds_kernel(
        int thread_i, int thread_num)
{
    int step = superpixel_seeds.size() / thread_num;
    int begin_index = step * thread_i;
    int end_index = begin_index + step;
    if(thread_i == thread_num - 1)
        end_index = superpixel_seeds.size();
    for (int seed_i = begin_index; seed_i < end_index; seed_i++)
    {
        if(superpixel_seeds[seed_i].stable)
            continue;
        int sp_x = seed_i % sp_width;
        int sp_y = seed_i / sp_width;
        int check_x_begin = sp_x * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        int check_y_begin = sp_y * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        int check_x_end = check_x_begin + SP_SIZE * 2;
        int check_y_end = check_y_begin + SP_SIZE * 2;
        check_x_begin = check_x_begin > 0 ? check_x_begin : 0;
        check_y_begin = check_y_begin > 0 ? check_y_begin : 0;
        check_x_end = check_x_end < image_width - 1 ? check_x_end : image_width - 1;
        check_y_end = check_y_end < image_height - 1 ? check_y_end : image_height - 1;
        float sum_x = 0;
        float sum_y = 0;
        float sum_R = 0.0, sum_G = 0.0, sum_B = 0.0;
        float sum_intensity_num = 0.0;
        float sum_depth = 0.0;
        float sum_depth_num = 0.0;
        std::vector<float> depth_vector;
        for (int check_j = check_y_begin; check_j < check_y_end; check_j++)
            for (int check_i = check_x_begin; check_i < check_x_end; check_i ++)
            {
                int pixel_index = check_j * image_width + check_i;
                if (superpixel_index[pixel_index] == seed_i)
                {
                    sum_x += check_i;
                    sum_y += check_j;
                    sum_intensity_num += 1.0;
                    // RGB
                    cv::Vec3b pix = image.at<cv::Vec3b>(check_j, check_i);
                    float my_R = pix[2];
                    float my_G = pix[1];
                    float my_B = pix[0];
                    sum_R += my_R;
                    sum_G += my_G;
                    sum_B += my_B;

                    //sum_intensity += image.at<uchar>(check_j, check_i);
                    float check_depth = depth.at<float>(check_j, check_i);
                    if (check_depth > 0.1)
                    {
                        depth_vector.push_back(check_depth);
                        sum_depth += check_depth;
                        sum_depth_num += 1.0;
                    }
                }
            }
        if (sum_intensity_num == 0)
            return;
        sum_R /= sum_intensity_num;
        sum_G /= sum_intensity_num;
        sum_B /= sum_intensity_num;

        sum_x /= sum_intensity_num;
        sum_y /= sum_intensity_num;
        float pre_R = superpixel_seeds[seed_i].R;
        float pre_G = superpixel_seeds[seed_i].G;
        float pre_B = superpixel_seeds[seed_i].B;

        float pre_x = superpixel_seeds[seed_i].x;
        float pre_y = superpixel_seeds[seed_i].y;

        superpixel_seeds[seed_i].R = sum_R;
        superpixel_seeds[seed_i].G = sum_G;
        superpixel_seeds[seed_i].B = sum_B;

        superpixel_seeds[seed_i].x = sum_x;
        superpixel_seeds[seed_i].y = sum_y;
        float update_diff = fabs(pre_R - sum_R) + fabs(pre_G - sum_G)+fabs(pre_B - sum_B) + fabs(pre_x - sum_x) + fabs(pre_y - sum_y);
        if (update_diff < 0.4)
            superpixel_seeds[seed_i].stable = true;
        if (sum_depth_num > 0)
        {
            float mean_depth = sum_depth / sum_depth_num;
            float sum_a, sum_b;
            for (int newton_i = 0; newton_i < 5; newton_i++)
            {
                sum_a = sum_b = 0;
                for (int p_i = 0; p_i < depth_vector.size(); p_i++)
                {
                    float residual = mean_depth - depth_vector[p_i];
                    if (residual < HUBER_RANGE && residual > -HUBER_RANGE)
                    {
                        sum_a += 2 * residual;
                        sum_b += 2;
                    }
                    else
                    {
                        sum_a += residual > 0 ? HUBER_RANGE : -1 * HUBER_RANGE;
                    }
                }
                float delta_depth = -sum_a / (sum_b + 10.0);
                mean_depth = mean_depth + delta_depth;
                if (delta_depth < 0.01 && delta_depth > -0.01)
                    break;
            }
            superpixel_seeds[seed_i].mean_depth = mean_depth;
        }
        else
        {
            superpixel_seeds[seed_i].mean_depth = 0.0;
        }
    }
}

bool SLIC::calculate_cost(
        float &nodepth_cost, float &depth_cost,
        const float &pixel_R, const float &pixel_G, const float &pixel_B, const float &pixel_inverse_depth,
        const int &x, const int &y,
        const int &sp_x, const int &sp_y)
{
    int sp_index = sp_y * sp_width + sp_x;
    nodepth_cost = 0;
    float dist =
            (superpixel_seeds[sp_index].x - x) * (superpixel_seeds[sp_index].x - x) + (superpixel_seeds[sp_index].y - y) * (superpixel_seeds[sp_index].y - y);
    nodepth_cost += dist / ((SP_SIZE / 2) * (SP_SIZE / 2));
    float R_diff = (superpixel_seeds[sp_index].R - pixel_R);
    float G_diff = (superpixel_seeds[sp_index].G - pixel_G);
    float B_diff = (superpixel_seeds[sp_index].B - pixel_B);

    nodepth_cost += (R_diff * R_diff + G_diff * G_diff + B_diff * B_diff) / 300.0;
    depth_cost = nodepth_cost;
    if (superpixel_seeds[sp_index].mean_depth > 0 && pixel_inverse_depth > 0)
    {
        float inverse_depth_diff = 1.0 / superpixel_seeds[sp_index].mean_depth - pixel_inverse_depth;
        depth_cost += inverse_depth_diff * inverse_depth_diff * 400.0;
        // float inverse_depth_diff = superpixel_seeds[sp_index].mean_depth - 1.0/pixel_inverse_depth;
        // depth_cost += inverse_depth_diff * inverse_depth_diff * 400.0;
        return true;
    }
    return false;
}

void SLIC::debug_show()
{
    cv::Mat result = cv::Mat(image_height, image_width, CV_8UC3);
    for (int j = 0; j < image_height; j++)
        for (int i = 0; i < image_width; i++)
        {
            // pixel sp index
            int sp_index = superpixel_index[j * image_width + i];
            cv::Vec3b this_norm;
            this_norm[0] = fabs(superpixel_seeds[sp_index].B);
            this_norm[1] = fabs(superpixel_seeds[sp_index].G);
            this_norm[2] = fabs(superpixel_seeds[sp_index].R);
            result.at<cv::Vec3b>(j, i) = this_norm;
        }
    for (int i = 0; i < superpixel_index.size(); i++)
    {
        int p_x = i % image_width;
        int p_y = i / image_width;
        int my_index = superpixel_index[i];
        if (p_x + 1 < image_width && superpixel_index[i + 1] != my_index)
            result.at<cv::Vec3b>(p_y, p_x) = cv::Vec3b(0, 0, 0);
        if (p_y + 1 < image_height && superpixel_index[i + image_width] != my_index)
            result.at<cv::Vec3b>(p_y, p_x) = cv::Vec3b(0, 0, 0);
    }
    cv::Mat d_res = cv::Mat(image_height, image_width, CV_32FC1);
    float max=0,min=100000;
    for (int j = 0; j < image_height; j++){
        for (int i = 0; i < image_width; i++) {
            // pixel sp index
            int sp_index = superpixel_index[j * image_width + i];
            //std::cout<<this_norm<<std::endl;
            d_res.at<float>(j, i) = fabs(superpixel_seeds[sp_index].mean_depth);
            if(d_res.at<float>(j, i)>max) max = d_res.at<float>(j, i);
            if(d_res.at<float>(j, i)<min) min = d_res.at<float>(j, i);
        }
    }
    if (min!=max)
        d_res.convertTo(d_res,CV_8U,255.0/(max-min),-255.0*min/(max-min));

    for (int i = 0; i < superpixel_index.size(); i++)
    {
        int p_x = i % image_width;
        int p_y = i / image_width;
        int my_index = superpixel_index[i];
        if (p_x + 1 < image_width && superpixel_index[i + 1] != my_index)
            d_res.at<char>(p_y, p_x) = 0;
        if (p_y + 1 < image_height && superpixel_index[i + image_width] != my_index)
            d_res.at<char>(p_y, p_x) = 0;
    }
    cv::imwrite("raw.png",image);
    cv::imwrite("RGB.png",result);
    cv::imwrite("depth.png",d_res);

    //cv::waitKey(0);
}