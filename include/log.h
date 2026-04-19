#pragma once

/**
 * include/log.h — FZGPUModules logging infrastructure
 *
 * Design goals:
 *   1. Zero overhead when disabled — FZ_LOG calls below the compile-time
 *      threshold expand to ((void)0); arguments are never evaluated.
 *   2. Zero overhead in release builds by default — FZ_LOG_MIN_LEVEL defaults
 *      to INFO (2), so TRACE and DEBUG calls compile out completely.
 *   3. Flexible output — a single callback receives all log messages; the
 *      caller decides where they go (stderr, file, test buffer, etc.).
 *   4. Explicit diagnostic helpers (printDAG, printBufferLifetimes) always
 *      produce output via FZ_PRINT, which uses the callback if set or stdout.
 *
 * Compile-time control (set via CMake `-DFZ_LOG_MIN_LEVEL=N` or `\#define` before
 * including this header):
 *
 *   FZ_LOG_MIN_LEVEL=0  (TRACE)  — all log calls compiled in
 *   FZ_LOG_MIN_LEVEL=1  (DEBUG)  — TRACE compiled out
 *   FZ_LOG_MIN_LEVEL=2  (INFO)   — TRACE+DEBUG compiled out  ← default
 *   FZ_LOG_MIN_LEVEL=3  (WARN)   — only WARN compiled in
 *   FZ_LOG_MIN_LEVEL=255 (SILENT) — all log calls compiled out
 *
 * Runtime control:
 *   fz::Logger::setCallback(cb)       — set output sink (nullptr = silent)
 *   fz::Logger::setMinLevel(level)    — runtime filter (≥ compile-time floor)
 *   fz::Logger::enableStderr(level)   — convenience: log ≥ level to stderr
 *
 * Usage:
 *   FZ_LOG(INFO,  "finalize: %zu stages", n);
 *   FZ_LOG(DEBUG, "buffer %s allocated %.1f KB", tag, kb);
 *   FZ_LOG(TRACE, "execute: input=%zu output=%zu", in, out);
 *   FZ_LOG(WARN,  "outlier overflow: %u > capacity", count);
 *
 *   FZ_PRINT("  node [%d] %s", id, name);   // always outputs (diagnostic)
 */

#include <cstdarg>
#include <cstdio>

namespace fz {

// ── Log levels ───────────────────────────────────────────────────────────────

enum class LogLevel : int {
    TRACE  = 0,  ///< Per-stage execute(), per-chunk details — very verbose
    DEBUG  = 1,  ///< Pipeline construction, buffer allocation, data stats
    INFO   = 2,  ///< High-level milestones: finalize, compress, decompress
    WARN   = 3,  ///< Unexpected but recoverable: outlier overflow, fallbacks
    SILENT = 255 ///< Compile-time sentinel — do not pass to log()
};

// ── Logger ───────────────────────────────────────────────────────────────────

class Logger {
public:
    using Callback = void(*)(LogLevel level, const char* msg);

    // ── Sink configuration ────────────────────────────────────────────────

    /**
     * Set the output callback.  All log messages at or above the runtime
     * minimum level are forwarded to this function.  Pass nullptr to silence
     * all runtime output (compile-time disabled calls are already gone).
     */
    static void setCallback(Callback cb) { callback() = cb; }

    /**
     * Set the runtime minimum log level.  Messages below this level are
     * dropped even if the compile-time floor allows them.  Useful for
     * toggling verbosity at runtime without recompiling.
     *
     * Note: the effective floor is max(FZ_LOG_MIN_LEVEL, runtime minLevel).
     * Setting a runtime level lower than the compile-time floor has no effect
     * because those call sites are already compiled out.
     */
    static void setMinLevel(LogLevel level) { minLevel() = level; }
    static LogLevel getMinLevel()           { return minLevel(); }

    /**
     * Convenience: enable logging at or above min_level to stderr.
     * Format: "[fzgmod:LEVEL] message"
     */
    static void enableStderr(LogLevel min_level = LogLevel::INFO) {
        minLevel() = min_level;
        setCallback([](LogLevel level, const char* msg) {
            if (level >= Logger::minLevel()) {
                fprintf(stderr, "[fzgmod:%s] %s\n", levelTag(level), msg);
            }
        });
    }

    // ── Logging ───────────────────────────────────────────────────────────

    /**
     * Emit a log message.  Called by the FZ_LOG macro only for levels that
     * passed compile-time gating; do not call directly.
     */
    static void log(LogLevel level, const char* fmt, ...) {
        if (!callback()) return;
        if (level < minLevel()) return;
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        callback()(level, buf);
    }

    /**
     * Diagnostic output — always emits, not filtered by log level.
     * Used by printDAG(), printBufferLifetimes(), and similar explicit
     * diagnostic helpers that the caller has explicitly invoked.
     *
     * Routes through the callback (as INFO) if one is set; otherwise writes
     * to stdout so explicit diagnostic calls are never silently swallowed.
     */
    static void print(const char* fmt, ...) {
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        if (callback()) {
            callback()(LogLevel::INFO, buf);
        } else {
            puts(buf);
        }
    }

    static const char* levelTag(LogLevel l) {
        switch (l) {
            case LogLevel::TRACE:  return "TRACE";
            case LogLevel::DEBUG:  return "DEBUG";
            case LogLevel::INFO:   return "INFO";
            case LogLevel::WARN:   return "WARN";
            default:               return "???";
        }
    }

private:
    static Callback& callback() {
        static Callback cb = nullptr;
        return cb;
    }
    static LogLevel& minLevel() {
        static LogLevel lvl = LogLevel::INFO;
        return lvl;
    }
};

} // namespace fz

// ── Compile-time level gating ─────────────────────────────────────────────────
//
// FZ_LOG_MIN_LEVEL is set by CMake (default 2 = INFO).  Log calls below this
// level expand to ((void)0) — the compiler eliminates them entirely and their
// arguments are never evaluated, giving true zero overhead.
//
// Override at the CMake level:
//   cmake -DFZ_LOG_MIN_LEVEL=0   # full TRACE logging (debug builds)
//   cmake -DFZ_LOG_MIN_LEVEL=255 # all logging compiled out (embedded)

#ifndef FZ_LOG_MIN_LEVEL
#  define FZ_LOG_MIN_LEVEL 2  // INFO
#endif

// FZ_LOG_DISABLE is kept for backward compatibility; equivalent to SILENT.
#ifdef FZ_LOG_DISABLE
#  undef  FZ_LOG_MIN_LEVEL
#  define FZ_LOG_MIN_LEVEL 255
#endif

/**
 * FZ_LOG(LEVEL, fmt, ...) — primary logging macro.
 *
 * If the level's integer value is below FZ_LOG_MIN_LEVEL the entire call —
 * including all arguments — is compiled out via if constexpr.  No function
 * call, no string evaluation, no branch at runtime.
 *
 * Example:
 *   FZ_LOG(DEBUG, "allocated %s: %.1f KB", tag.c_str(), sz / 1024.0);
 *   FZ_LOG(WARN,  "outlier overflow: %u > %u capacity", actual, max);
 */
#define FZ_LOG(level, ...)                                                      \
    do {                                                                        \
        if constexpr (static_cast<int>(::fz::LogLevel::level) >=               \
                      static_cast<int>(FZ_LOG_MIN_LEVEL)) {                     \
            ::fz::Logger::log(::fz::LogLevel::level, __VA_ARGS__);              \
        }                                                                       \
    } while (0)

/**
 * FZ_PRINT(fmt, ...) — always-output diagnostic macro.
 *
 * Not filtered by FZ_LOG_MIN_LEVEL.  Use for explicitly-called diagnostic
 * functions (printDAG, printBufferLifetimes) where the caller has opted in
 * to output and silence would be confusing.
 *
 * Routes through the Logger callback if one is set; falls back to stdout.
 */
#define FZ_PRINT(...) ::fz::Logger::print(__VA_ARGS__)
