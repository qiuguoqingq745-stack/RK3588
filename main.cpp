/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "detection_fusion_manager.h"
#include "im2d.h"
#include "rknnPool.hpp"
#include "stream_loader.h"
#include "streaming_manager.h"

char *model_person = const_cast<char *>("../../model/person_relu.rknn");
char *model_helmet = const_cast<char *>("../../model/helmet_relu.rknn");
char *model_callplay = const_cast<char *>("../../model/callplay_relu.rknn");

StreamLoaderManager &manager = StreamLoaderManager::getInstance();
std::vector<std::thread> rk_threads;
std::vector<cv::Mat> images(6);
std::vector<std::mutex> mutexes(6);

StreamingManager streaming_manager;
DetectionFusionManager fusion_manager;

const size_t RKNN_WORKER_THREADS = 3;
dpool::ThreadPool pool(RKNN_WORKER_THREADS);

void combineImage(StreamLoaderManager &manager)
{
    cv::Mat combinedImage(1080, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat lastCombinedImage;
    bool hasLastFrame = false;

    const int target_fps = 24;
    const int frame_interval_ms = 1000 / target_fps;
    auto last_frame_time = std::chrono::steady_clock::now();

    while (true) {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_frame_time).count();

        if (elapsed < frame_interval_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_interval_ms - elapsed));
            current_time = std::chrono::steady_clock::now();
        }
        last_frame_time = current_time;

        bool hasNewFrame = false;
        const int tile_w = 640;
        const int tile_h = 360;

        for (int i = 0; i < manager.num_stream; ++i) {
            cv::Mat local_img;
            {
                std::lock_guard<std::mutex> lock(mutexes[i]);
                if (images[i].empty()) {
                    continue;
                }
                local_img = std::move(images[i]);
                images[i] = cv::Mat();
            }

            cv::Mat resizedImage(tile_h, tile_w, CV_8UC3);
            int src_w = local_img.cols;
            int src_h = local_img.rows;

            rga_buffer_t src_buf = wrapbuffer_virtualaddr(local_img.data, src_w, src_h, RK_FORMAT_BGR_888);
            rga_buffer_t dst_buf = wrapbuffer_virtualaddr(resizedImage.data, tile_w, tile_h, RK_FORMAT_BGR_888);

            IM_STATUS status = imresize(src_buf, dst_buf);
            if (status != IM_STATUS_SUCCESS) {
                cv::resize(local_img, resizedImage, cv::Size(tile_w, tile_h));
            }

            int row = i / 2;
            int col = i % 2;
            int x = col * tile_w;
            int y = row * tile_h;
            resizedImage.copyTo(combinedImage(cv::Rect(x, y, tile_w, tile_h)));
            hasNewFrame = true;
        }

        cv::Mat frameToSend;
        if (hasNewFrame) {
            lastCombinedImage = combinedImage.clone();
            frameToSend = lastCombinedImage;
            hasLastFrame = true;
        } else if (hasLastFrame) {
            frameToSend = lastCombinedImage;
        } else {
            frameToSend = combinedImage.clone();
        }

        StreamingData stream_data;
        stream_data.stream_id = 0;
        stream_data.frame = frameToSend;
        stream_data.timestamp = std::chrono::system_clock::now();
        memset(&stream_data.person_results, 0, sizeof(detect_result_group_t));
        memset(&stream_data.helmet_results, 0, sizeof(detect_result_group_t));
        memset(&stream_data.tired_results, 0, sizeof(detect_result_group_t));
        memset(&stream_data.callplay_results, 0, sizeof(detect_result_group_t));

        streaming_manager.addStreamingData(stream_data);
    }
}

void rknn_infer(std::shared_ptr<rknn_lite> p1,
                std::shared_ptr<rknn_lite> p2,
                std::shared_ptr<rknn_lite> p3,
                std::shared_ptr<rknn_lite> p4,
                int i)
{
    detect_result_group_t g1, g2, g3, g4;
    memset(&g1, 0, sizeof(detect_result_group_t));
    memset(&g2, 0, sizeof(detect_result_group_t));
    memset(&g3, 0, sizeof(detect_result_group_t));
    memset(&g4, 0, sizeof(detect_result_group_t));

    const int INFER_INTERVAL = 2;
    const int TARGET_PROCESS_FPS = 12;
    const auto PROCESS_INTERVAL = std::chrono::milliseconds(1000 / TARGET_PROCESS_FPS);

    int frame_count = 0;
    uint64_t last_frame_id = 0;
    auto last_process_time = std::chrono::steady_clock::now() - PROCESS_INTERVAL;
    std::vector<FusedDetection> last_fused;

    while (!manager.stream_loaders[i]->stopFlag) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_process_time < PROCESS_INTERVAL) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        long long frame_pts = 0;
        uint64_t frame_id = 0;
        {
            std::unique_lock<std::mutex> lock(manager.stream_loaders[i]->buffer.mtx);
            Mbuffer &buffer = manager.stream_loaders[i]->buffer;
            if (buffer.img.empty() || buffer.frame_id == last_frame_id) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            p1->ori_img = buffer.img.clone();
            frame_id = buffer.frame_id;
            frame_pts = buffer.pts;
        }

        last_frame_id = frame_id;
        last_process_time = now;

        p2->ori_img = p1->ori_img;
        if (p3) {
            p3->ori_img = p1->ori_img;
        }
        p4->ori_img = p1->ori_img;

        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (frame_pts > 0 && now_ms - frame_pts > 500) {
            continue;
        }

        frame_count++;
        bool do_infer = (frame_count % INFER_INTERVAL == 1) || last_fused.empty();
        if (do_infer) {
            auto f1 = pool.submit([&]() { p1->interf(g1); });
            auto f2 = pool.submit([&]() { p2->interf(g2); });
            std::future<void> f3;
            if (p3) {
                f3 = pool.submit([&]() { p3->interf(g3); });
            }
            auto f4 = pool.submit([&]() { p4->interf(g4); });

            f1.get();
            f2.get();
            if (p3) {
                f3.get();
            }
            f4.get();

            last_fused = fusion_manager.fuseDetections(g1, g2, g3, g4);
        }

        fusion_manager.drawFusedDetections(p1->ori_img, last_fused);
        {
            std::lock_guard<std::mutex> lockimage(mutexes[i]);
            images[i] = std::move(p1->ori_img);
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_stream>" << std::endl;
        return -1;
    }

    manager.num_stream = std::min(std::stoi(argv[1]), static_cast<int>(images.size()));

    StreamingConfig stream_config;
    stream_config.rtmp_url = "rtmp://172.20.10.3:1935/live/livestream";
    stream_config.width = 1280;
    stream_config.height = 720;
    stream_config.fps = 24;
    stream_config.bitrate = 2000000;
    stream_config.enable_rtmp = true;
    stream_config.draw_detections = true;

    for (int i = 0; i < manager.num_stream; ++i) {
        manager.load_stream(i);

        auto ptr1 = std::make_shared<rknn_lite>(model_person, 0, 1, 0);
        auto ptr2 = std::make_shared<rknn_lite>(model_helmet, 1, 2, 1);
        auto ptr3 = nullptr;
        auto ptr4 = std::make_shared<rknn_lite>(model_callplay, 2, 2, 3);

        rk_threads.emplace_back(rknn_infer, ptr1, ptr2, ptr3, ptr4, i);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (!streaming_manager.initialize(stream_config)) {
        std::cerr << "Failed to initialize streaming manager" << std::endl;
        return -1;
    }

    streaming_manager.startStreaming();

    std::thread readerThread(combineImage, std::ref(manager));
    readerThread.join();

    streaming_manager.stopStreaming();

    for (int i = 0; i < manager.num_stream; ++i) {
        manager.unload_stream(i);
    }
    for (auto &t : rk_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    return 0;
}
