#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>


double SSD(const cv::Mat& img1, cv::Mat& img2) {

    const auto width = img1.cols;
    const auto height = img1.rows;
    double err = 0;


    if (img1.rows != img2.rows || img1.cols != img2.cols) {
        std::cout << "Dimensios mismatch!" << std::endl;
        return 0;

    }

    const cv::Mat output = (img1 - img2);
    cv::Mat dest(cv::Size(width, height), CV_64FC1);
    cv::pow(output, 2, dest);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            err += dest.at<uchar>(r, c);
        }
    }
    return err;

}

double mse(const cv::Mat& img1, cv::Mat& img2) {

    const auto width = img1.cols;
    const auto height = img1.rows;
    double err = 0;

    if (img1.rows != img2.rows || img1.cols != img2.cols) {
        std::cout << "Dimensios mismatch!" << std::endl;
        return 0;

    }

    const cv::Mat output = (img1 - img2);
    cv::Mat dest(cv::Size(width, height), CV_64FC1);
    cv::pow(output, 2, dest);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            err += dest.at<uchar>(r, c);
        }
    }

    err /= width * height;
    return err;

}

double rmse(const cv::Mat& img1, cv::Mat& img2) {

    const auto width = img1.cols;
    const auto height = img1.rows;
    double err = 0;

    if (img1.rows != img2.rows || img1.cols != img2.cols) {
        std::cout << "Dimensios mismatch!" << std::endl;
        return 0;

    }

    const cv::Mat output = (img1 - img2);
    cv::Mat dest(cv::Size(width, height), CV_64FC1);
    cv::pow(output, 2, dest);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            err += dest.at<uchar>(r, c);
        }
    }

    err /= width * height;
    double res = pow(err, 0.5);
    return res;

}

double psnr(const cv::Mat& img1, cv::Mat& img2) {

    double mse_val = mse(img1, img2);

    if (mse_val <= 1e-10) // for small values return zero
        return 0;

    else
    {
        double psnr = 10.0 * log10((255.0 * 255.0) / mse_val);
        std::cout << "PSNR is " << psnr;
        return psnr;
    }

}

void compute_quality_metric(const cv::Mat& img1, cv::Mat& img2, std::string title) {

    std::cout << std::endl;
    std::cout << "Similiarity Metrics of (" << title << "): " << std::endl;
    std::cout << "SSD: " << SSD(img1, img2) << std::endl;
    std::cout << "MSE: " << mse(img1, img2) << std::endl;;
    std::cout << "RMSE: " << rmse(img1, img2) << std::endl;;
    std::cout << "PNSR: " << psnr(img1, img2) << std::endl;;
    std::cout << std::endl;
}

cv::Mat create_gaussian_filter(int sigma) {

    int rmax = 2.5 * sigma;
    int window_size = 2 * rmax + 1;

    cv::Mat Gaussian_filter = cv::Mat::zeros(window_size, window_size, CV_32F);

    for (int i = 0; i < window_size; ++i) {
        for (int j = 0; j < window_size; ++j) {
            float value = std::exp(-(((i - rmax) * (i - rmax) + (j - rmax) * (j - rmax)) / (2.0 * sigma * sigma)));
            Gaussian_filter.at<float>(i, j) = value;
        }
    }
   
    return Gaussian_filter;
}

cv::Mat bilateral_filter(cv::Mat image, int sigma_s, int sigma_r) {

    cv::Mat Gaussian_filter = create_gaussian_filter(sigma_s);

    int window_size = Gaussian_filter.size().height;
    int rmax = window_size / 2;

    int height = image.size().height;
    int width = image.size().width;

    cv::Mat blurred = cv::Mat::zeros(height, width, CV_8UC1);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            float sum = 0;
            float normalize = 0;

            for (int k = 0; k < window_size; ++k) {
                for (int l = 0; l < window_size; ++l) {

                    //avoid out of boundary exception
                    if (i + k - rmax < 0 || i + k - rmax >= height || j + l - rmax < 0 || j + l - rmax >= width) continue;

                    //calculate range difference 
                    int intensity_difference = (image.at<uchar>(i, j) - image.at<uchar>(i + k - rmax, j + l - rmax)) * (image.at<uchar>(i, j) - image.at<uchar>(i + k - rmax, j + l - rmax));

                    //final weight, combination of the range kernel and the Gaussian kernel
                    float weight = std::exp(-((intensity_difference) / (2.0 * sigma_r * sigma_r))) * Gaussian_filter.at<float>(k, l);

                    //accumulate the weights for normalization
                    normalize += weight;

                    //taking the weighted sum of the input values
                    sum += weight * image.at<uchar>(i + k - rmax, j + l - rmax);
                }
            }

            blurred.at<uchar>(i, j) = std::round(sum / normalize);
        }
    }

    return blurred;

}

cv::Mat joint_bilateral_filter(cv::Mat f, cv::Mat g, int sigma_s, int sigma_r) {

    cv::Mat Gaussian_filter = create_gaussian_filter(sigma_s);
    int window_size = Gaussian_filter.size().height;
    int rmax = window_size / 2;

    int height = f.size().height;
    int width = f.size().width;

    cv::Mat blurred = cv::Mat::zeros(height, width, CV_8UC1);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            float sum = 0;
            
            float normalize = 0;

            for (int k = 0; k < window_size; ++k) {
                for (int l = 0; l < window_size; ++l) {
                    if (i + k - rmax < 0 || i + k - rmax >= height || j + l - rmax < 0 || j + l - rmax >= width) continue;

                    int intensity_difference = (f.at<uchar>(i, j) - f.at<uchar>(i + k - rmax, j + l - rmax)) * (f.at<uchar>(i, j) - f.at<uchar>(i + k - rmax, j + l - rmax));

                    float weight = std::exp(-((intensity_difference) / (2.0 * sigma_r * sigma_r))) * Gaussian_filter.at<float>(k, l);

                    normalize += weight;

                    sum += weight * g.at<uchar>(i + k - rmax, j + l - rmax);
                }
            }

            blurred.at<uchar>(i, j) = std::round(sum / normalize);
        }
    }

    return blurred;
}

cv::Mat upsample(int sigma_s, int sigma_r, cv::Mat disp, cv::Mat image) {

    int upsample_factor = std::floor(std::log2(image.size().height / (float)disp.size().height)); //  round down to the smaller int number

    cv::Mat upsampled_disp = disp;

    for (int i = 1; i < upsample_factor - 1; ++i) {
        std::cout << "Upsampling disparity image... " << std::floor(((float)i / upsample_factor) * 100) << "%\r" << std::flush;
        cv::resize(upsampled_disp, upsampled_disp, cv::Size(), 2.0, 2.0);
        cv::Mat lowres_image;
        cv::resize(image, lowres_image, upsampled_disp.size());
        upsampled_disp = joint_bilateral_filter(lowres_image, upsampled_disp, sigma_s, sigma_r);
    }
    std::cout << "Upsampling disparity image... " << std::floor(((float)(upsample_factor - 1) / upsample_factor) * 100) << "%\r" << std::flush;
    cv::resize(upsampled_disp, upsampled_disp, image.size());
    upsampled_disp = joint_bilateral_filter(image, upsampled_disp, sigma_s, sigma_r);
    std::cout << "Upsampling disparity image... Done." << std::endl;
    return upsampled_disp;
}

int main(int argc, char** argv) {

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << "image Depth_image Sigma_S Sigma_R" << std::endl;
        return 1;
    }

    int sigma_s = std::stoi(argv[3]);
    int sigma_r = std::stoi(argv[4]);

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    cv::imshow("image", image);

    int sigmas_s[4] = { 1, 3, 5, 10 };
    int sigmas_r[4] = { 10, 30, 100, 300 };

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << "Calculating filtered images... [" << i * 4 + j + 1 << "/16]\r" << std::flush;
            cv::Mat filtered = bilateral_filter(image, sigmas_s[i], sigmas_r[j]);
            std::stringstream ss;
            ss << "filtered_" << sigmas_s[i] << "_" << sigmas_r[j] << ".png";
            cv::imwrite(ss.str(), filtered);


            cv::Mat cv_bilateral_output;

            cv::bilateralFilter(image, cv_bilateral_output, sigmas_r[j], sigmas_s[i] , 4);

            compute_quality_metric(image, cv_bilateral_output, "OpenCV bilateral filter vs ours");


        }
    }
    std::cout << "Calculating filtered images... Done.     " << std::endl;

    cv::Mat disp = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    cv::imshow("lowres_disp", disp);

    cv::Mat upsampled_disp = upsample(sigma_s, sigma_r, disp, image);
    cv::imshow("upsampled_disp", upsampled_disp);
    cv::imwrite("upsampled_disp.png", upsampled_disp);

  



    cv::waitKey(0);


    return 0;
}