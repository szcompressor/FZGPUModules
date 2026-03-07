#pragma once

#include <cstdarg>
#include <cstdio>

namespace fz {

/**
 * Log severity levels
 */
enum class LogLevel {
    TRACE = 0,  // Very verbose, per-buffer/per-stage
    DEBUG = 1,  // Pipeline construction, allocation events
    INFO  = 2,  // High-level status (finalize, compress complete)
    WARN  = 3   // Unexpected but recoverable situations
};

/**
 * Lightweight callback-based logger
 *
 * Default: silent (no output). Set a callback to receive log messages.
 *
 * Usage:
 *   // Enable logging to stderr:
 *   fz::Logger::setCallback([](fz::LogLevel level, const char* msg) {
 *       fprintf(stderr, "[fzgmod] %s\n", msg);
 *   });
 *
 *   // Or use the built-in stderr helper:
 *   fz::Logger::enableStderr(fz::LogLevel::DEBUG);
 *
 *   // Disable logging:
 *   fz::Logger::setCallback(nullptr);
 */
class Logger {
public:
    using Callback = void(*)(LogLevel level, const char* msg);

    static void setCallback(Callback cb) { callback() = cb; }

    /**
     * Convenience: log everything >= min_level to stderr
     */
    static void enableStderr(LogLevel min_level = LogLevel::DEBUG) {
        minLevel() = min_level;
        setCallback([](LogLevel level, const char* msg) {
            if (level >= Logger::minLevel()) {
                const char* tag = "???";
                switch (level) {
                    case LogLevel::TRACE: tag = "TRACE"; break;
                    case LogLevel::DEBUG: tag = "DEBUG"; break;
                    case LogLevel::INFO:  tag = "INFO";  break;
                    case LogLevel::WARN:  tag = "WARN";  break;
                }
                fprintf(stderr, "[fzgmod:%s] %s\n", tag, msg);
            }
        });
    }

    static void log(LogLevel level, const char* fmt, ...) {
        if (!callback()) return;
        char buf[512];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        callback()(level, buf);
    }

private:
    static Callback& callback() {
        static Callback cb = nullptr;
        return cb;
    }
    static LogLevel& minLevel() {
        static LogLevel lvl = LogLevel::DEBUG;
        return lvl;
    }
};

} // namespace fz

// ===== Convenience Macros =====
// Define FZ_LOG_DISABLE to compile out all logging

#ifndef FZ_LOG_DISABLE
#define FZ_LOG(level, ...) ::fz::Logger::log(::fz::LogLevel::level, __VA_ARGS__)
#else
#define FZ_LOG(level, ...) ((void)0)
#endif
