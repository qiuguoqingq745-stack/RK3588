/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/imgutils.h>
#include <libavutil/rational.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
}

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>

#include "m_buffer.hpp"
#include "mpp_decoder.h"

using std::queue;
using std::vector;

using MppDecoderFrameCallback = std::function<void(void *userdata,
                                                   int width_stride,
                                                   int height_stride,
                                                   int width,
                                                   int height,
                                                   int format,
                                                   int fd,
                                                   void *data,
                                                   int id)>;

class StreamLoader {
public:
    MppDecoder decoder;

    int videoStreamIndex = -1;
    AVDictionary *options = nullptr;
    AVFormatContext *fmtCtx = nullptr;
    AVCodecParameters *codecPar = nullptr;

    AVBSFContext *bsf_ctx = nullptr;
    const AVBitStreamFilter *bsf = nullptr;

    bool got_key_frame = false;
    AVPacket *temp_pkt = nullptr;
    int current_pkt_id = 0;
    int stream_loader_id = 0;
    char *stream_url = nullptr;
    int width = 0;
    int height = 0;
    int status = 0;
    bool isnotAnnexB = false;
    MppDecoderFrameCallback callback;

    Mbuffer buffer;
    std::atomic<bool> stopFlag;

    double source_fps_ = 25.0;
    bool is_local_file_ = false;

    StreamLoader(char *url, int id);
    ~StreamLoader();

    int open();
    void close();
    bool read_frame();
    void operator()();
    void update_queue();
};

class StreamLoaderManager {
public:
    vector<char *> urls = {
        const_cast<char *>("rtsp://172.20.10.3:8554/live"),
        const_cast<char *>("rtsp://172.20.10.3:8554/live"),
        const_cast<char *>("rtsp://172.20.10.3:8554/live"),
        const_cast<char *>("rtsp://172.20.10.3:8554/live"),
        const_cast<char *>("rtsp://172.20.10.3:8554/live"),
        const_cast<char *>("rtsp://172.20.10.3:8554/live")
    };
    int num_stream = 4;

    StreamLoaderManager(const StreamLoaderManager &) = delete;
    StreamLoaderManager &operator=(const StreamLoaderManager &) = delete;

    static StreamLoaderManager &getInstance()
    {
        static StreamLoaderManager instance;
        return instance;
    }

    void load_stream(int id);
    void unload_stream(int id);

    vector<StreamLoader *> stream_loaders;
    vector<std::thread> threads;

private:
    StreamLoaderManager()
    {
        std::cout << "StreamLoaderManager created" << std::endl;
    }

    ~StreamLoaderManager()
    {
        std::cout << "StreamLoaderManager destroyed" << std::endl;
    }
};
