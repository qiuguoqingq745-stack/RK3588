/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <vector>
#include <opencv2/opencv.hpp>

struct Mbuffer {
    cv::Mat img;
    std::mutex mtx;

    // Incremented whenever decoder publishes a new BGR frame.
    uint64_t frame_id = 0;
    long long pts = 0;

    // Local-file playback throttling.
    int frame_interval_ms = 0;
    bool throttle = false;

    std::vector<uint8_t> yuv_work;
    cv::Mat bgr_work;
};
