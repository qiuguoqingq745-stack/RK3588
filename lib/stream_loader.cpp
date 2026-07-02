/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#include "stream_loader.h"
#include "im2d.h"
#include <chrono>
#include <string>
#include <thread>

// 判断是否为 Annex B 格式
// 该函数并没有使用
int is_annexb(const uint8_t *buf, size_t buf_size)
{
    // Annex B 格式以 0x000001 或 0x00000001 开头
    if (buf_size >= 4)
    {
        if ((buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x01) ||
            (buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x00 && buf[3] == 0x01))
        {
            return 1; // 是 Annex B 格式
        }
    }
    return 0; // 不是 Annex B 格式
}

//YUV 裸数据->转成 BGR
void mpp_decoder_frame_callback(void *buffer, int width_stride, int height_stride, int width, int height, int format, int fd, void *data, int id)
{
        Mbuffer *mbuffer = (Mbuffer *)buffer;

        mbuffer->bgr_work.create(height, width, CV_8UC3);//BGR 彩图
        if (mbuffer->bgr_work.empty()) {
            return;
        }

        // 真正的零拷贝：直接使用 RGA 硬件加速 NV12->BGR，传入物理 fd 和 stride
        rga_buffer_t src_buf = wrapbuffer_virtualaddr_t(data, width, height, width_stride, height_stride, RK_FORMAT_YCbCr_420_SP);
        rga_buffer_t dst_buf = wrapbuffer_virtualaddr_t(mbuffer->bgr_work.data, width, height, width, height, RK_FORMAT_BGR_888);

        IM_STATUS status = imcvtcolor(src_buf, //输入
                                    dst_buf,//输出
                                    RK_FORMAT_YCbCr_420_SP, //源格式
                                    RK_FORMAT_BGR_888,//目标格式
                                    IM_COLOR_SPACE_DEFAULT);
        //瑞芯微 RGA 硬件转换失败时的备用方案
        if (status != IM_STATUS_SUCCESS) {
            static int fallback_count = 0;
            if (fallback_count++ < 3) {
                fprintf(stderr, "RGA NV12->BGR failed (%d), hardware zero-copy failed\n", (int)status);
            }
        }

        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::unique_lock<std::mutex> mlock(mbuffer->mtx);
        mbuffer->img = std::move(mbuffer->bgr_work);
        mbuffer->pts = now_ms;
        ++mbuffer->frame_id;
        mlock.unlock();

        // 每输出一帧限速，解决一包多帧导致的倍速
        if (mbuffer->throttle && mbuffer->frame_interval_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(mbuffer->frame_interval_ms));
}


void StreamLoader::close()
{
    decoder.Reset();
    if (temp_pkt)
    {
        av_packet_free(&temp_pkt); // 释放 temp_pkt 并将指针置为 nullptr
    }

    if (fmtCtx)
    {
        avformat_close_input(&fmtCtx); // 关闭输入流
        fmtCtx = nullptr;              // 确保指针在关闭后被设置为 nullptr
    }

    if (codecPar)
    {
        avcodec_parameters_free(&codecPar); // 释放 codecPar 结构
    }
}

bool StreamLoader::read_frame()
{
    using namespace std::chrono_literals;
    int eof_retry = 0;           // 连续 av_read_frame 失败次数（EOF 时递增）
    int no_frame_count = 0;      // 已读视频包但解码未出帧的次数，防止异常时死循环
    const int MAX_EOF_RETRY = 10;       // EOF 时最多重试次数，超过则触发 reconnect
    const int MAX_PACKETS_NO_FRAME = 100; // 连续读包未出帧的上限，避免异常流导致死循环

    while (true)
    {
        int x = av_read_frame(fmtCtx, temp_pkt);
        if (x < 0)
        {
            status = x;
            eof_retry++;
            if (eof_retry >= MAX_EOF_RETRY) {
                return false;  // 确认 EOF，触发 reconnect 循环播放
            }
            std::this_thread::sleep_for(2ms);
            av_packet_unref(temp_pkt);
            continue;
        }

        eof_retry = 0;  // 成功读到包，重置 EOF 计数

        if (temp_pkt->stream_index != videoStreamIndex)
        {
            av_packet_unref(temp_pkt);
            continue;
        }

        // 视频包
        if (isnotAnnexB)
        {
            int ret = av_bsf_send_packet(bsf_ctx, temp_pkt);//将temp_pkt放入bsf_ctx转换
            if (ret < 0)
            {
                fprintf(stderr, "Error sending packet to filter\n");
                av_packet_unref(temp_pkt);
                return false;
            }
            ret = av_bsf_receive_packet(bsf_ctx, temp_pkt);//将转换后的数据放入temp_pkt
            if (ret < 0)
            {
                fprintf(stderr, "Error receiving packet from filter\n");
                av_packet_unref(temp_pkt);
                return false;
            }
        }

        bool decode_success = decoder.Decode(temp_pkt->data, temp_pkt->size, 0);//解码
        av_packet_unref(temp_pkt);

        if (decode_success)
        {
            status = 0;
            return true;
        }

        no_frame_count++;
        if (no_frame_count >= MAX_PACKETS_NO_FRAME)
        {
            // 异常：连续多包无输出，避免死循环
            return false;
        }
        // 删除 sleep_for(2ms)，因为网络流需要尽可能快地吃包，不能人为限速
    }
}

StreamLoader::StreamLoader(char *url, int id)
{
    stream_loader_id = id;
    std::cout << "StreamLoader: " << std::to_string(id) << std::endl;
    callback = mpp_decoder_frame_callback;
    // mat_ptr = new cv::Mat();

    stream_url = url; 
    status = 0;
    stopFlag = false;
}

StreamLoader::~StreamLoader()
{
    std::cout << "destory stream loader: " << stream_loader_id << std::endl;
    close();
    // delete mat_ptr;
}

int StreamLoader::open()
{
    temp_pkt = av_packet_alloc();//输出：创建空 AVPacket（存压缩数据）->存从 RTSP 读取的压缩帧
    // av_init_packet is deprecated in FFmpeg 7.x, av_packet_alloc() already initializes the packet
    codecPar = avcodec_parameters_alloc();// 输出：创建空编码参数容器->保存视频编码信息（宽高、格式等）

    //设置 RTSP 拉流参数（输入配置）
    av_dict_set(&options, "rtbufsize", "8192000", 0);// 缓冲区大小
    av_dict_set(&options, "start_time_realtime", "0", 0);
    av_dict_set(&options, "rtsp_transport", "tcp", 0);// 强制 TCP 拉流（更稳定)
    av_dict_set(&options, "stimeout", "2000000", 0); // 超时 2s
    av_dict_set(&options, "max_delay", "500000", 0);// 最大延迟

    // 打开RTSP流  fmtCtx:格式上下文，包含整个流的所有信息
    if (avformat_open_input(&fmtCtx, stream_url, NULL, &options) != 0)
    {
        std::cout << "open rtsp stream failed" << std::endl;
        return -1;
    }
    // 查找RTSP流信息
    if (avformat_find_stream_info(fmtCtx, NULL) < 0)
    {
        return -1;
    }

    // 打印视频相关信息
    av_dump_format(fmtCtx, 0, stream_url, 0);
    // 获取视频的信息
    videoStreamIndex = -1;
    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++)
    {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            width = fmtCtx->streams[i]->codecpar->width;
            height = fmtCtx->streams[i]->codecpar->height;
            videoStreamIndex = i;
            break;
        }
    }
    std::cout << "videoindex: " << videoStreamIndex << std::endl;
    if (videoStreamIndex < 0)
    {
        return -2;
    }
    //获取视频编码格式
    AVCodecID rtsp_format = fmtCtx->streams[videoStreamIndex]->codecpar->codec_id;
    if (status == 0)
    {
        int ret = 0;
        void *src_buffer = &(this->buffer);
        switch (rtsp_format)
        {
        case AV_CODEC_ID_H264:
            ret = decoder.Init(264, 25, src_buffer, stream_loader_id);
            // ----------------------------------------------------------
            // 查找H.264比特流过滤器
            bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf)
            {
                fprintf(stderr, "Could not find h264_mp4toannexb filter\n");
                avformat_close_input(&fmtCtx);
                return -3;
            }

            // 初始化比特流过滤器上下文创建实例bsf_ctx
            if (av_bsf_alloc(bsf, &bsf_ctx) < 0)
            {
                fprintf(stderr, "Could not allocate bsf context\n");
                avformat_close_input(&fmtCtx);
                return -3;
            }
            // 设置过滤器参数
            //streams[0] → 一般是 视频流  streams[videoStreamIndex]更保险
            //• streams[1] → 一般是 音频流
            avcodec_parameters_copy(bsf_ctx->par_in, fmtCtx->streams[0]->codecpar);
            bsf_ctx->time_base_in = fmtCtx->streams[0]->time_base;

            if (av_bsf_init(bsf_ctx) < 0)
            {
                fprintf(stderr, "Could not initialize bsf context\n");
                av_bsf_free(&bsf_ctx);
                avformat_close_input(&fmtCtx);
                return -3;
            }
            isnotAnnexB = true;
            // ----------------------------------------------------------
            std::cout << "H264 " << ret << std::endl;
            break;
        case AV_CODEC_ID_HEVC:
            ret = decoder.Init(265, 25, src_buffer, stream_loader_id);
            std::cout << "HEVC " << ret << std::endl;
            break;
        }
    }
//设置解码转换的回调函数
    decoder.SetCallback(this->callback);
    avcodec_parameters_copy(codecPar, fmtCtx->streams[videoStreamIndex]->codecpar);

    // 获取源视频帧率，用于本地文件限速
    AVStream *st = fmtCtx->streams[videoStreamIndex];
    double fps = av_q2d(st->avg_frame_rate);//获取平均帧率 av_q2d 是把 FFmpeg 的分数转成小数
    if (fps <= 0) fps = av_q2d(st->r_frame_rate);//如果 avg_frame_rate 拿不到，用 r_frame_rate
    if (fps <= 0) fps = 25.0;
    source_fps_ = fps;

    is_local_file_ = false;
    if (stream_url) {
        std::string u(stream_url);
        if (u.rfind("rtsp://", 0) != 0 && u.rfind("rtmp://", 0) != 0)
            is_local_file_ = true;
    }

    if (is_local_file_ && source_fps_ > 0) {
        buffer.throttle = true;
        buffer.frame_interval_ms = (int)(1000.0 / source_fps_ * 1.2);
    }

    return 0;
}


void StreamLoader::operator()()
{
    while (stopFlag == false)
    {
        try
        {
            read_frame();
        }
        catch (std::exception &e)
        {
            std::cout << "exception ............" << std::endl;
            std::cout << e.what() << std::endl;
        }
        if (status)
        {
            // status < 0 通常为 AVERROR_EOF（文件播完），触发 reconnect 实现循环播放
            std::cout << "Stream " << stream_loader_id << " EOF, reconnecting..." << std::endl;

            // 先关闭当前流，清理解码器和 AVFormatContext
            close();

            // 根据 URL 类型区分本地文件与网络流：
            // - 本地文件：立即重新 open，相当于从头开始播放，实现循环播放
            // - 网络流（rtsp/rtmp 等）：按原来的逻辑，失败时 10 秒后重试
            bool is_network_stream = false;
            if (stream_url)
            {
                std::string url_str(stream_url);
                if (url_str.rfind("rtsp://", 0) == 0 ||
                    url_str.rfind("rtmp://", 0) == 0)
                {
                    is_network_stream = true;
                }
            }

            if (is_network_stream)
            {
                // 原有 RTSP 重连逻辑：失败则 10s 后重试
                while (open() != 0)
                {
                    std::cout << "Reconnect (network) failed, retry after 10s, id = "
                              << stream_loader_id << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
                }
            }
            else
            {
                // 本地文件：立即重新 open，相当于从头开始播放
                // 如果打开失败，短暂等待后快速重试
                while (open() != 0)
                {
                    std::cout << "Reopen local file failed, retry shortly, id = "
                              << stream_loader_id << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                std::cout << "Local file reopened, loop playback, id = "
                          << stream_loader_id << std::endl;
            }

            // 重置状态，继续正常读取帧
            status = 0;
        }
    }
}

// ==================================================================================================

void StreamLoaderManager::load_stream(int id)
{
    std::cout << "Loading stream id: " << id << std::endl;
    StreamLoader *loader = new StreamLoader(urls[id], id);
    loader->open();
    stream_loaders.push_back(loader);
    //调用operator()启动线程
    threads.emplace_back(std::thread(std::ref(*loader)));
}

// 卸载流
void StreamLoaderManager::unload_stream(int id)
{
    std::cout << "Unloading stream id: " << id << std::endl;
    stream_loaders[id]->stopFlag = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if(threads[id].joinable())
        threads[id].join();
    delete stream_loaders[id];
}
