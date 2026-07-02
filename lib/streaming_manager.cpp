/*
 * Copyright (c) 2025-04-01 HeXiaotian
 *
 * This source code is licensed for learning and research purposes only.
 * Commercial use, redistribution, resale, and creation of derivative works
 * are strictly prohibited without prior written permission from the author.
 */

#include "streaming_manager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
}

StreamingManager::StreamingManager() 
    : streaming_active_(false)
    , should_stop_(false)
    , avformat_context_(nullptr)
    , mpp_encoder_(nullptr)
    , rtmp_frame_index_(0)
    , frame_count_(0)
{
}

StreamingManager::~StreamingManager() {
    stopStreaming();
}

bool StreamingManager::initialize(const StreamingConfig& config) {
    config_ = config;
    
    if (config_.enable_rtmp) {
        if (!initializeRTMP()) {
            std::cerr << "Failed to initialize RTMP streaming" << std::endl;
            return false;
        }
    }
    
    if (config_.enable_rtsp) {
        if (!initializeRTSP()) {
            std::cerr << "Failed to initialize RTSP streaming" << std::endl;
            return false;
        }
    }
    
    return true;
}

void StreamingManager::addStreamingData(const StreamingData& data) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // 限制队列大小，避免内存溢出
    if (streaming_queue_.size() > 10) {
        streaming_queue_.pop(); // 丢弃最旧的数据
        stats_.frames_dropped++;
    }
    
    streaming_queue_.push(data);
    queue_cv_.notify_one();
    
    // 调试信息：每100帧打印一次
    static int frame_count = 0;
    if (++frame_count % 100 == 0) {
        std::cout << "Added frame to streaming queue, queue size: " << streaming_queue_.size() << std::endl;
    }
}

void StreamingManager::startStreaming() {
    if (streaming_active_.load()) {
        return;
    }
    
    streaming_active_ = true;
    should_stop_ = false;
    streaming_thread_ = std::thread(&StreamingManager::streamingWorker, this);
    
    std::cout << "Streaming started" << std::endl;
}

void StreamingManager::stopStreaming() {
    if (!streaming_active_.load()) {
        return;
    }
    
    should_stop_ = true;
    queue_cv_.notify_all();  // 防止线程在 wait 中阻塞
    
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }
    
    streaming_active_ = false;
    
    // 清理FFmpeg / MPP 资源  先写 trailer（确保流结束标记），再关闭网络句柄，最后释放上下文。
    if (avformat_context_) {
        AVFormatContext* fmt_ctx = (AVFormatContext*)avformat_context_;
        av_write_trailer(fmt_ctx); // 写文件尾（RTMP/RTSP 结束帧）
        if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE) && fmt_ctx->pb) {
            avio_closep(&fmt_ctx->pb);   // 关闭网络 IO
        }
        avformat_free_context(fmt_ctx);
        avformat_context_ = nullptr;
    }

    if (mpp_encoder_) {
        mpp_encoder_->Release();
        delete mpp_encoder_;
        mpp_encoder_ = nullptr;
    }
    
    std::cout << "Streaming stopped" << std::endl;
}

 void StreamingManager::streamingWorker() {
     cv::Mat last_frame;               // 保存最后一帧，用于超时情况下的“保活帧”
     bool has_last_frame = false;      // 标记是否已经收到过有效帧
 
    while (!should_stop_.load()) {    // 主循环：只要外部没有请求停止就一直跑
         StreamingData data;           // 本轮要处理的帧（包括图像 + 检测结果）
        bool got_data = false;        // 标记本轮是否成功取到数据

       // --------------------------------------------------------------
      // ① 等待生产者（推理线程）放入数据，使用条件变量 + 超时
     // --------------------------------------------------------------
        {
            std::unique_lock<std::mutex> lock(queue_mutex_); // 加锁保护队列
            // 计算本帧的期望间隔（毫秒），与用户配置的 fps 对齐
           auto timeout = std::chrono::milliseconds(1000 / config_.fps);

            // wait_for 会在两种情况返回 true：
             //   1) 队列非空（有新帧）   2) should_stop_ 被置位（外部要求退出）
            if (queue_cv_.wait_for(lock, timeout, [this] {
                 return !streaming_queue_.empty() || should_stop_.load();
             })) 
             {
                // ------------------- 有数据或需要停止 -------------------
                if (should_stop_.load()) {   // 立即响应退出信号
                   break;                  // 跳出 while 循环，结束线程
               }

                if (!streaming_queue_.empty()) { // 正常取帧路径
                   data = streaming_queue_.front(); // 取最旧的帧（FIFO）
                    streaming_queue_.pop();          // 移除队列头部
                   got_data = true;                 // 标记本轮成功获取
                 }
           } else {
               // ------------------- 超时路径 -------------------
                // 超时说明在本帧间隔内没有新数据到来。为防止 RTMP/RTSP
                // 连接因“长时间无数据”而被服务器端踢掉，需要发送
              // “保活帧”。如果已经缓存了上一帧，则复用它；否则
                // 直接跳过本轮，继续等待。
               if (has_last_frame && !last_frame.empty()) {
                   // 使用上一帧的图像保持流活跃
                    data.frame = last_frame.clone(); // 深拷贝，防止后续修改
                    data.stream_id = 0;               // 保活帧不关心 stream_id
                    data.timestamp = std::chrono::system_clock::now();
                   // 检测结果全部清零，表示“没有检测”
                    memset(&data.person_results,   0, sizeof(detect_result_group_t));
                    memset(&data.helmet_results,   0, sizeof(detect_result_group_t));
                    memset(&data.tired_results,    0, sizeof(detect_result_group_t));
                    memset(&data.callplay_results, 0, sizeof(detect_result_group_t));
                    got_data = true;                 // 仍然视为本轮有数据
                 } else {
                     // 没有任何帧可用（首次启动或前一帧被丢弃），
                    // 直接 continue 进入下一次循环等待
                     continue;
                }
             }
         } // <-- 这里锁会在离开作用域时自动释放

        if (!got_data) {               // 防御性检查：理论上不可能走到这里
             continue;
        }

         // --------------------------------------------------------------
        // ② 对取到的帧做本地处理（绘制、分辨率统一、拷贝）
        // --------------------------------------------------------------
        cv::Mat frame = data.frame.clone(); // 深拷贝，防止外部数据被修改

         // 绘制检测框（如果用户打开了绘制开关）
         if (config_.draw_detections) {
            drawDetections(frame, data);
       }

        // 若输入分辨率与推流配置不一致，则统一缩放到目标宽高
        if (frame.cols != config_.width || frame.rows != config_.height) {
            cv::resize(frame, frame, cv::Size(config_.width, config_.height));
        }
         // --------------------------------------------------------------
         // ④ 实际推流：依据配置分别走 RTMP / RTSP 两条路径
        // --------------------------------------------------------------
         bool success = false;         // 记录本轮是否至少成功发送一次
        if (config_.enable_rtmp) {
             success |= sendRTMPFrame(frame);   // 发送 RTMP
         }
        if (config_.enable_rtsp) {
            success |= sendRTSPFrame(frame);   // 发送 RTSP（目前是占位实现）
        }

        // ==========================================
        // ✨ 推流完成，frame 不再使用 → 用 move！
        // --------------------------------------------------------------
        // 保存当前帧为“保活帧”，供后续超时使用
        // ==========================================
        last_frame = std::move(frame);
        has_last_frame = true;
         // --------------------------------------------------------------
         // ⑤ 更新统计信息（线程安全）——帧计数、FPS、发送成功率
        // --------------------------------------------------------------
         {
             std::lock_guard<std::mutex> lock(stats_mutex_); // 保护 stats_
             if (success) {
                stats_.frames_sent++;          // 成功发送计数
             } else {
                 stats_.frames_dropped++;       // 发送失败计数
             }

             // 计算实时 FPS：每满 config_.fps 帧更新一次
            auto now = std::chrono::system_clock::now();
            frame_count_++;                     // 计数器累计所有处理的帧（包括保活帧）
            if (frame_count_ % config_.fps == 0) { // 每秒（或每 config_.fps 帧）更新一次
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                     now - stats_.last_frame_time).count();
                 if (duration > 0) {
                    // fps = 目标帧数 * 1000 / 实际耗时（ms）
                    stats_.fps = (config_.fps * 1000.0) / duration;
                 }
                 stats_.last_frame_time = now; // 记录本次统计的时间基准
             }
        }
 
         // --------------------------------------------------------------
         // ⑥ 控制推流速率：让线程睡到下一个目标帧的时间点
         // --------------------------------------------------------------
         std::this_thread::sleep_for(std::chrono::milliseconds(1000 / config_.fps));
     } // end while
 } // end streamingWorker

void StreamingManager::drawDetections(cv::Mat& frame, const StreamingData& data) {
    // 绘制人员检测结果 (绿色)
    for (int i = 0; i < data.person_results.count; i++) {
        const auto& result = data.person_results.results[i];
        cv::rectangle(frame, 
            cv::Point(result.box.left, result.box.top),
            cv::Point(result.box.right, result.box.bottom),
            cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Person", 
            cv::Point(result.box.left, result.box.top - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    
    // 绘制安全帽检测结果 (红色)
    for (int i = 0; i < data.helmet_results.count; i++) {
        const auto& result = data.helmet_results.results[i];
        cv::rectangle(frame, 
            cv::Point(result.box.left, result.box.top),
            cv::Point(result.box.right, result.box.bottom),
            cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "Helmet", 
            cv::Point(result.box.left, result.box.top - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    
    // 绘制疲劳检测结果 (黄色)
    for (int i = 0; i < data.tired_results.count; i++) {
        const auto& result = data.tired_results.results[i];
        cv::rectangle(frame, 
            cv::Point(result.box.left, result.box.top),
            cv::Point(result.box.right, result.box.bottom),
            cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, "Tired", 
            cv::Point(result.box.left, result.box.top - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    }
    
    // 绘制通话/游戏检测结果 (蓝色)
    for (int i = 0; i < data.callplay_results.count; i++) {
        const auto& result = data.callplay_results.results[i];
        cv::rectangle(frame, 
            cv::Point(result.box.left, result.box.top),
            cv::Point(result.box.right, result.box.bottom),
            cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "Call/Play", 
            cv::Point(result.box.left, result.box.top - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }
    
    // 添加时间戳和统计信息
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    
    cv::putText(frame, ss.str(), cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // 添加检测统计
    std::string stats = "P:" + std::to_string(data.person_results.count) +
                       " H:" + std::to_string(data.helmet_results.count) +
                       " T:" + std::to_string(data.tired_results.count) +
                       " C:" + std::to_string(data.callplay_results.count);
    
    cv::putText(frame, stats, cv::Point(10, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
}

std::string StreamingManager::createDetectionJSON(const StreamingData& data) {
    std::stringstream json;
    json << "{";
    json << "\"stream_id\":" << data.stream_id << ",";
    json << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(
        data.timestamp.time_since_epoch()).count() << ",";
    
    json << "\"detections\":{";
    json << "\"person\":" << data.person_results.count << ",";
    json << "\"helmet\":" << data.helmet_results.count << ",";
    json << "\"tired\":" << data.tired_results.count << ",";
    json << "\"callplay\":" << data.callplay_results.count;
    json << "}";
    
    json << "}";
    return json.str();
}

bool StreamingManager::initializeRTMP() {
    // 初始化 FFmpeg 网络
    avformat_network_init();

    AVFormatContext* fmt_ctx = nullptr;

     // ② 创建输出上下文（FLV 容器，RTMP 常用封装）
    //    参数解释：<fmt_ctx> 输出指针，nullptr 表示让 FFmpeg 自动选择合适的封装器，
    //    "flv" 为容器格式，config_.rtmp_url 为目标地址（rtmp://...）
    if (avformat_alloc_output_context2(&fmt_ctx, nullptr, "flv", config_.rtmp_url.c_str()) < 0 || !fmt_ctx) {
        std::cerr << "Could not create output context" << std::endl;
        return false;
    }

    // 创建视频流（不再让 FFmpeg 编码，只做封装）
    AVStream* stream = avformat_new_stream(fmt_ctx, nullptr);
    if (!stream) {
        std::cerr << "Could not create stream" << std::endl;
        avformat_free_context(fmt_ctx);
        return false;
    }
    // ④ 为该流填充 **CodecParameters**（告诉接收端我们要发送的是什么编码）
    AVCodecParameters* codecpar = stream->codecpar;
    codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    codecpar->codec_id   = AV_CODEC_ID_H264;           // 我们自己用 MPP 编 H.264
    codecpar->width      = config_.width;
    codecpar->height     = config_.height;
    codecpar->format     = AV_PIX_FMT_YUV420P;         // 编码输出格式

    // ⑤ 设置 **时间基**（time_base）——帧率的时间尺度
    //    这里使用 1/fps，表示每帧的时间戳间隔为 1/fps 秒
    stream->time_base = AVRational{1, config_.fps};

    // 初始化 MPP 硬编码器（只负责把 BGR → H.264）
    mpp_encoder_ = new MppEncoder();
    if (mpp_encoder_->Init(config_.width, config_.height, config_.fps, config_.bitrate, 264) != 0) {
        std::cerr << "Failed to init MPP encoder" << std::endl;
        delete mpp_encoder_;
        mpp_encoder_ = nullptr;
        avformat_free_context(fmt_ctx);
        return false;
    }

   // ⑦ 从 MPP 编码器获取 **SPS / PPS**（H.264 关键的 extra info）
   //    这些信息在流的头部（extradata）中必须提供，接收端才能正确解码
    uint8_t header_buf[1024];
    int header_size = sizeof(header_buf);
    if (mpp_encoder_->GetHeader(header_buf, &header_size) == 0 && header_size > 0) {
        // 为 extradata 分配 FFmpeg 所需的缓冲区（加上填充字节，防止越界读取）
        codecpar->extradata = (uint8_t*)av_malloc(header_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (codecpar->extradata) {
            memcpy(codecpar->extradata, header_buf, header_size);
            memset(codecpar->extradata + header_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
            codecpar->extradata_size = header_size;
            std::cout << "H.264 extradata from MPP, size: " << header_size << " bytes" << std::endl;
        }
    } else {
        std::cerr << "Warning: failed to get H.264 extra info from MPP encoder" << std::endl;
    }

    // 打开输出（RTMP）
    if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_ctx->pb, config_.rtmp_url.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Could not open output URL: " << config_.rtmp_url << std::endl;
            avformat_free_context(fmt_ctx);
            return false;
        }
    }
// ⑨ 写 **文件头**（FLV/RTMP 头部），此时会把 SPS/PPS、流信息发送给服务器
    if (avformat_write_header(fmt_ctx, nullptr) < 0) {
        std::cerr << "Error occurred when writing header to RTMP" << std::endl;
        if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE) && fmt_ctx->pb) {
            avio_closep(&fmt_ctx->pb);
        }
        avformat_free_context(fmt_ctx);
        return false;
    }

    avformat_context_ = fmt_ctx;
    rtmp_frame_index_ = 0;
    return true;
}

bool StreamingManager::sendRTMPFrame(const cv::Mat& frame) {
    // 入口函数：把一帧 BGR 图像推送到已经打开的 RTMP 连接
    if (!avformat_context_ || !mpp_encoder_) {
        return false;   // 若上下文或硬件编码器未初始化，直接返回，防止空指针崩溃
    }
// 取得 FFmpeg 的输出上下文指针，后面需要通过它写入 AVPacket
    AVFormatContext* fmt_ctx = (AVFormatContext*)avformat_context_;
    if (fmt_ctx->nb_streams == 0) {
        return false;
    }
    AVStream* stream = fmt_ctx->streams[0];

    // 使用 MPP 编码当前帧（BGR -> H.264）
    // 预估一个足够大的缓冲区（经验值：分辨率 * 2 一般足够）
    int max_packet_size = config_.width * config_.height * 2;
    std::vector<uint8_t> enc_buf(max_packet_size);
    int packet_size = max_packet_size;
    if (!frame.isContinuous()) {
        std::cerr << "Frame is not continuous, skip\n";
        return false;
    }

    int ret = mpp_encoder_->EncodeFrame(
        frame.data,
        frame.cols,
        frame.rows,
        enc_buf.data(),
        &packet_size,
        (int)frame.step
    );

    if (ret != 0 || packet_size <= 0) {
        // 本帧没有有效编码输出，直接跳过
        return false;
    }

    // 构造 AVPacket 并发送
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = enc_buf.data();
    pkt.size = packet_size;
    pkt.stream_index = stream->index;

    // 简单的基于帧序号的 PTS
    pkt.pts = rtmp_frame_index_;
    pkt.dts = rtmp_frame_index_;
    rtmp_frame_index_++;

    AVRational src_tb{1, config_.fps};
    av_packet_rescale_ts(&pkt, src_tb, stream->time_base);

    ret = av_interleaved_write_frame(fmt_ctx, &pkt);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "Error writing MPP-encoded frame to RTMP: " << errbuf << std::endl;
        return false;
    }

    static int send_count = 0;
    if (++send_count % 100 == 0) {
        std::cout << "Sent frame " << send_count
                  << " (MPP encoded, size=" << packet_size << " bytes)" << std::endl;
    }

    return true;
}

bool StreamingManager::initializeRTSP() {
    // RTSP推流实现（类似RTMP，但使用不同的输出格式）
    // 这里简化实现，实际项目中需要更复杂的RTSP服务器设置
    return true;
}

bool StreamingManager::sendRTSPFrame(const cv::Mat& frame) {
    // RTSP推流实现
    return true;
}

StreamingManager::StreamingStats StreamingManager::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}
