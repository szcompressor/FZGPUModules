/**
 * Dedicated Huffman throughput/latency benchmark.
 *
 * Times HuffmanStage<uint16_t> directly so pipeline scheduling and archive
 * concatenation are not charged to the coder. Input construction, device
 * allocation, uploads, stage scratch allocation, and validation copies stay
 * outside timed regions. Transient allocations performed by execute() remain
 * included because they are part of the implementation being evaluated.
 *
 * Usage:
 *   fzgmod-profile-huffman-throughput [options]
 *     --sizes LIST          comma-separated input bytes (default 4K,64K,1M,64M)
 *     --warmups N           untimed warmups per steady-state row (default 10)
 *     --repetitions N       measured iterations per clock (default 31)
 *     --distribution NAME   uniform|skewed|low_entropy|scientific|all
 *     --mode NAME           host|device|all
 *     --book NAME           per_block|adaptive_cold|adaptive_warm|fixed|all
 *     --operation NAME      compress|decompress|all
 *     --csv PATH            write machine-readable results
 *
 * The GPU event interval ends after the last encode/decode work enqueued by
 * execute(). Host latency additionally includes stream completion and
 * postStreamSync(), including DeviceResident's required header/book-status
 * readback. All rows are warmed repeated measurements. "Adaptive cold fit"
 * describes codebook state, not the first timed CUDA invocation: every warmup
 * and measured repetition uses a newly initialized, preallocated stage and must
 * fit a book. Adaptive warm reuse fits once before warmups and retains that book.
 * PerBlock also warms normally but rebuilds its book on every call by definition.
 */

#include "backend/api.h"
#include "coders/huffman/huffman_stage.h"
#include "mem/mempool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Symbol = uint16_t;
using Clock = std::chrono::steady_clock;
constexpr uint32_t kBookLength = 1024;

void check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) { check(cudaMalloc(&ptr, bytes), "cudaMalloc"); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Event {
    cudaEvent_t event = nullptr;
    Event() { check(cudaEventCreate(&event), "cudaEventCreate"); }
    ~Event() { if (event) cudaEventDestroy(event); }
};

struct Options {
    std::vector<size_t> sizes = {4ULL << 10, 64ULL << 10, 1ULL << 20, 64ULL << 20};
    int warmups = 10;
    int repetitions = 31;
    std::string distribution = "all";
    std::string mode = "all";
    std::string book = "all";
    std::string operation = "all";
    std::string csv;
};

size_t parse_size(std::string value)
{
    if (value.empty()) throw std::runtime_error("empty size");
    uint64_t multiplier = 1;
    const char suffix = value.back();
    if (suffix == 'K' || suffix == 'k') { multiplier = 1ULL << 10; value.pop_back(); }
    else if (suffix == 'M' || suffix == 'm') { multiplier = 1ULL << 20; value.pop_back(); }
    else if (suffix == 'G' || suffix == 'g') { multiplier = 1ULL << 30; value.pop_back(); }
    const unsigned long long base = std::stoull(value);
    if (base > std::numeric_limits<size_t>::max() / multiplier)
        throw std::runtime_error("size overflow");
    return static_cast<size_t>(base * multiplier);
}

std::vector<size_t> parse_sizes(const std::string& text)
{
    std::vector<size_t> result;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) result.push_back(parse_size(item));
    if (result.empty()) throw std::runtime_error("--sizes must not be empty");
    for (size_t bytes : result) {
        if (bytes < sizeof(Symbol) || bytes % sizeof(Symbol) != 0)
            throw std::runtime_error("each input size must be a positive multiple of 2 bytes");
    }
    return result;
}

Options parse_args(int argc, char** argv)
{
    Options options;
    auto value = [&](int& i, const char* option) -> std::string {
        if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + option);
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sizes") options.sizes = parse_sizes(value(i, "--sizes"));
        else if (arg == "--warmups") options.warmups = std::stoi(value(i, "--warmups"));
        else if (arg == "--repetitions") options.repetitions = std::stoi(value(i, "--repetitions"));
        else if (arg == "--distribution") options.distribution = value(i, "--distribution");
        else if (arg == "--mode") options.mode = value(i, "--mode");
        else if (arg == "--book") options.book = value(i, "--book");
        else if (arg == "--operation") options.operation = value(i, "--operation");
        else if (arg == "--csv") options.csv = value(i, "--csv");
        else if (arg == "--help" || arg == "-h") {
            std::cout << "See the usage block at the top of profiling/huffman_throughput.cpp\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + arg);
    }
    if (options.warmups < 0 || options.repetitions < 3)
        throw std::runtime_error("warmups must be >= 0 and repetitions must be >= 3");
    return options;
}

bool selected(const std::string& filter, const std::string& value)
{
    return filter == "all" || filter == value;
}

std::vector<float> load_scientific_field()
{
    std::ifstream input(FZ_DATA_DIR "/CLDHGH.f32", std::ios::binary | std::ios::ate);
    if (!input) return {};
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) return {};
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(data.data()), bytes);
    if (!input) return {};
    return data;
}

std::vector<Symbol> make_input(size_t bytes, const std::string& distribution,
                               const std::vector<float>& scientific)
{
    const size_t n = bytes / sizeof(Symbol);
    std::vector<Symbol> data(n);
    std::mt19937_64 rng(0x48554646ULL + bytes);
    if (distribution == "uniform") {
        std::uniform_int_distribution<uint32_t> pick(0, kBookLength - 1);
        for (auto& x : data) x = static_cast<Symbol>(pick(rng));
    } else if (distribution == "skewed") {
        std::uniform_int_distribution<uint32_t> percent(0, 99);
        std::uniform_int_distribution<uint32_t> tail(0, kBookLength - 1);
        for (auto& x : data) {
            const uint32_t p = percent(rng);
            x = static_cast<Symbol>(p < 82 ? 512 : (p < 94 ? 511 : tail(rng)));
        }
    } else if (distribution == "low_entropy") {
        static constexpr Symbol alphabet[] = {508, 509, 510, 511, 512, 513, 514, 515};
        for (size_t i = 0; i < n; ++i) data[i] = alphabet[(i * 5 + i / 31) % 8];
    } else if (distribution == "scientific") {
        if (scientific.size() < 2)
            throw std::runtime_error("scientific distribution requested but CLDHGH.f32 is unavailable");
        const auto [lo, hi] = std::minmax_element(scientific.begin(), scientific.end());
        const double step = std::max(2.0 * (static_cast<double>(*hi) - *lo) * 1e-4, 1e-20);
        // Span the field even for a 4 KiB probe; the leading CLDHGH slab can be
        // constant and would exercise Adaptive's degeneracy guard rather than a
        // representative warm book. Adjacent values at each sampled location
        // retain a scientific predictive-code shape.
        const size_t stride = std::max<size_t>(1, scientific.size() / n);
        for (size_t i = 0; i < n; ++i) {
            const size_t j = (i * stride) % scientific.size();
            const size_t prev = (j == 0) ? 0 : j - 1;
            long long q = std::llround((static_cast<double>(scientific[j]) - scientific[prev]) / step);
            q = std::max(-512LL, std::min(511LL, q));
            data[i] = static_cast<Symbol>(q + 512);
        }
    } else throw std::runtime_error("unknown distribution: " + distribution);
    return data;
}

enum class BookCase { PerBlock, AdaptiveCold, AdaptiveWarm, Fixed };

const char* mode_name(fz::HuffmanExecutionMode mode)
{
    return mode == fz::HuffmanExecutionMode::HostCoordinated
        ? "HostCoordinated" : "DeviceResident";
}

const char* book_name(BookCase book)
{
    switch (book) {
        case BookCase::PerBlock: return "PerBlock";
        case BookCase::AdaptiveCold: return "AdaptiveColdFit";
        case BookCase::AdaptiveWarm: return "AdaptiveWarmReuse";
        case BookCase::Fixed: return "Fixed";
    }
    return "unknown";
}

std::string book_filter_name(BookCase book)
{
    switch (book) {
        case BookCase::PerBlock: return "per_block";
        case BookCase::AdaptiveCold: return "adaptive_cold";
        case BookCase::AdaptiveWarm: return "adaptive_warm";
        case BookCase::Fixed: return "fixed";
    }
    return "unknown";
}

void configure(fz::HuffmanStage<Symbol>& stage, fz::HuffmanExecutionMode mode,
               BookCase book)
{
    stage.setBklen(kBookLength);
    stage.setExecutionMode(mode);
    stage.setTerminalOutput(true);
    stage.setValidateSymbolRange(true);
    if (book == BookCase::Fixed) {
        stage.setFixedBookFromModel(
            {fz::HuffmanBookModel::Laplace, 512.0, 64.0, 2.0});
    } else if (book == BookCase::AdaptiveCold || book == BookCase::AdaptiveWarm) {
        stage.setBookSource(fz::HuffmanBookSource::Adaptive);
        // This benchmark distinguishes explicit cold fit from indefinite reuse.
        stage.setRefitThreshold(0.0f);
        stage.setRefitInterval(0);
    } else {
        stage.setBookSource(fz::HuffmanBookSource::PerBlock);
    }
}

struct Context {
    std::unique_ptr<fz::MemoryPool> pool;
    std::unique_ptr<fz::HuffmanStage<Symbol>> stage;
    std::vector<void*> inputs;
    std::vector<void*> outputs;
    std::vector<size_t> sizes;

    Context(size_t raw_bytes, fz::HuffmanExecutionMode mode, BookCase book,
            void* input, void* output)
        : pool(std::make_unique<fz::MemoryPool>(fz::MemoryPoolConfig(raw_bytes, 12.0f))),
          stage(std::make_unique<fz::HuffmanStage<Symbol>>()),
          inputs{input}, outputs{output}, sizes{raw_bytes}
    {
        configure(*stage, mode, book);
        stage->onFinalize(raw_bytes, pool.get());
    }
};

void finish(Context& context, cudaStream_t stream)
{
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    context.stage->postStreamSync(stream);
}

void execute(Context& context, cudaStream_t stream)
{
    context.stage->execute(stream, context.pool.get(), context.inputs,
                           context.outputs, context.sizes);
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

template <class Prepare>
std::vector<double> measure_host(int warmups, int repetitions, Prepare prepare,
                                 cudaStream_t stream)
{
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (int i = -warmups; i < repetitions; ++i) {
        auto context = prepare();
        const auto begin = Clock::now();
        execute(*context, stream);
        finish(*context, stream);
        const auto end = Clock::now();
        if (i >= 0)
            samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return samples;
}

template <class Prepare>
std::vector<double> measure_gpu(int warmups, int repetitions, Prepare prepare,
                                cudaStream_t stream)
{
    Event begin, end;
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (int i = -warmups; i < repetitions; ++i) {
        auto context = prepare();
        check(cudaEventRecord(begin.event, stream), "cudaEventRecord(begin)");
        execute(*context, stream);
        check(cudaEventRecord(end.event, stream), "cudaEventRecord(end)");
        check(cudaEventSynchronize(end.event), "cudaEventSynchronize");
        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, begin.event, end.event), "cudaEventElapsedTime");
        finish(*context, stream);
        if (i >= 0) samples.push_back(elapsed);
    }
    return samples;
}

std::shared_ptr<Context> non_owning(Context& context)
{
    return std::shared_ptr<Context>(&context, [](Context*) {});
}

struct Result {
    std::string distribution;
    size_t input_bytes = 0;
    std::string mode;
    std::string book;
    std::string operation;
    double host_ms = 0;
    double gpu_ms = 0;
    double host_gbs = 0;
    double gpu_gbs = 0;
    size_t compressed_bytes = 0;
    double ratio = 0;
    bool roundtrip = false;
    bool byte_match = false;
    bool graph_compatible = false;
    std::string status = "supported";
};

std::vector<uint8_t> download_bytes(void* device, size_t bytes, cudaStream_t stream)
{
    std::vector<uint8_t> host(bytes);
    if (bytes) check(cudaMemcpyAsync(host.data(), device, bytes, cudaMemcpyDeviceToHost, stream),
                     "download compressed bytes");
    check(cudaStreamSynchronize(stream), "download sync");
    return host;
}

std::vector<Symbol> download_symbols(void* device, size_t n, cudaStream_t stream)
{
    std::vector<Symbol> host(n);
    check(cudaMemcpyAsync(host.data(), device, n * sizeof(Symbol), cudaMemcpyDeviceToHost, stream),
          "download symbols");
    check(cudaStreamSynchronize(stream), "download sync");
    return host;
}

std::string key(const std::string& distribution, size_t bytes, BookCase book)
{
    return distribution + ":" + std::to_string(bytes) + ":" + book_name(book);
}

} // namespace

int main(int argc, char** argv)
try {
    const Options options = parse_args(argc, argv);
    const std::vector<std::string> distributions =
        options.distribution == "all"
        ? std::vector<std::string>{"uniform", "skewed", "low_entropy", "scientific"}
        : std::vector<std::string>{options.distribution};
    const std::vector<fz::HuffmanExecutionMode> modes = {
        fz::HuffmanExecutionMode::HostCoordinated,
        fz::HuffmanExecutionMode::DeviceResident};
    const std::vector<BookCase> books = {
        BookCase::PerBlock, BookCase::AdaptiveCold,
        BookCase::AdaptiveWarm, BookCase::Fixed};

    int device_id = 0;
    cudaDeviceProp prop{};
    check(cudaGetDevice(&device_id), "cudaGetDevice");
    check(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
    int runtime_version = 0, driver_version = 0;
    check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    std::cout << "device=" << prop.name << " sm=" << prop.major << prop.minor
              << " driver_api=" << driver_version << " runtime=" << runtime_version
              << " warmups=" << options.warmups
              << " repetitions=" << options.repetitions << "\n";
    std::cout << "GPU elapsed ends after execute-enqueued work; host latency also includes "
                 "completion and required metadata readback.\n";

    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    const auto scientific = load_scientific_field();
    std::vector<Result> results;
    std::vector<std::pair<std::string, std::vector<uint8_t>>> host_archives;

    for (const auto& distribution : distributions) {
        for (size_t raw_bytes : options.sizes) {
            const auto input = make_input(raw_bytes, distribution, scientific);
            DeviceBuffer d_input(raw_bytes);
            check(cudaMemcpyAsync(d_input.ptr, input.data(), raw_bytes,
                                  cudaMemcpyHostToDevice, stream), "upload input");
            check(cudaStreamSynchronize(stream), "upload sync");

            fz::HuffmanStage<Symbol> estimator;
            estimator.setBklen(kBookLength);
            const size_t output_capacity = estimator.estimateOutputSizes({raw_bytes})[0];
            DeviceBuffer d_compressed(output_capacity);
            DeviceBuffer d_decoded(raw_bytes);

            for (auto mode : modes) {
                const std::string mode_filter =
                    mode == fz::HuffmanExecutionMode::HostCoordinated ? "host" : "device";
                if (!selected(options.mode, mode_filter)) continue;
                for (BookCase book : books) {
                    if (!selected(options.book, book_filter_name(book))) continue;

                    // A retained context is used by all steady-state cases. Cold
                    // Adaptive prepares a fresh, preallocated context per iteration.
                    Context retained(raw_bytes, mode, book, d_input.ptr, d_compressed.ptr);
                    auto retained_prepare = [&]() { return non_owning(retained); };
                    auto cold_prepare = [&]() {
                        return std::make_shared<Context>(raw_bytes, mode, book,
                                                        d_input.ptr, d_compressed.ptr);
                    };
                    const bool cold = book == BookCase::AdaptiveCold;
                    if (book == BookCase::AdaptiveWarm) {
                        execute(retained, stream);
                        finish(retained, stream); // untimed fit
                        if (retained.stage->getFitBitsPerSymbol() == 0.0)
                            throw std::runtime_error(
                                "Adaptive warm-reuse input produced a degenerate cold sample: "
                                + distribution + " " + std::to_string(raw_bytes));
                    }

                    auto prepare = cold ? std::function<std::shared_ptr<Context>()>(cold_prepare)
                                        : std::function<std::shared_ptr<Context>()>(retained_prepare);
                    std::vector<double> host_compress, gpu_compress;
                    if (selected(options.operation, "compress")) {
                        host_compress = measure_host(options.warmups, options.repetitions,
                                                     prepare, stream);
                        gpu_compress = measure_gpu(options.warmups, options.repetitions,
                                                   prepare, stream);
                    }

                    // Produce one canonical archive after timing. For cold Adaptive
                    // this intentionally uses a new stage, preserving cold semantics.
                    auto archive_context = cold ? cold_prepare() : retained_prepare();
                    execute(*archive_context, stream);
                    finish(*archive_context, stream);
                    const size_t compressed_bytes = archive_context->stage->getActualOutputSize(0);
                    const auto archive = download_bytes(d_compressed.ptr, compressed_bytes, stream);

                    const std::string archive_key = key(distribution, raw_bytes, book);
                    bool byte_match = true;
                    if (mode == fz::HuffmanExecutionMode::HostCoordinated) {
                        host_archives.emplace_back(archive_key, archive);
                    } else {
                        auto it = std::find_if(host_archives.begin(), host_archives.end(),
                            [&](const auto& item) { return item.first == archive_key; });
                        // A device-only filtered run still performs the required
                        // deterministic-format comparison; the host reference is
                        // validation work and remains outside every timed region.
                        if (it == host_archives.end()) {
                            Context reference(raw_bytes,
                                fz::HuffmanExecutionMode::HostCoordinated, book,
                                d_input.ptr, d_compressed.ptr);
                            if (book == BookCase::AdaptiveWarm) {
                                execute(reference, stream);
                                finish(reference, stream);
                            }
                            execute(reference, stream);
                            finish(reference, stream);
                            host_archives.emplace_back(
                                archive_key,
                                download_bytes(d_compressed.ptr,
                                    reference.stage->getActualOutputSize(0), stream));
                            it = std::prev(host_archives.end());
                        }
                        byte_match = it->second == archive;
                        // Restore the device archive consumed by the decoder below.
                        check(cudaMemcpyAsync(d_compressed.ptr, archive.data(), archive.size(),
                                              cudaMemcpyHostToDevice, stream),
                              "restore device archive");
                        check(cudaStreamSynchronize(stream), "restore archive sync");
                    }

                    uint8_t stage_header[128]{};
                    const size_t header_size =
                        archive_context->stage->serializeHeader(0, stage_header, sizeof(stage_header));
                    if (header_size == 0) throw std::runtime_error("failed to serialize stage header");

                    Context decoder(raw_bytes, mode, book, d_compressed.ptr, d_decoded.ptr);
                    decoder.stage->deserializeHeader(stage_header, header_size);
                    decoder.stage->setExecutionMode(mode);
                    decoder.stage->setInverse(true);
                    decoder.inputs[0] = d_compressed.ptr;
                    decoder.outputs[0] = d_decoded.ptr;
                    decoder.sizes[0] = compressed_bytes;
                    auto decoder_prepare = [&]() { return non_owning(decoder); };

                    // Correctness is verified for every mode/book/size/distribution.
                    execute(decoder, stream);
                    finish(decoder, stream);
                    const bool roundtrip = download_symbols(d_decoded.ptr, input.size(), stream) == input;
                    if (!roundtrip)
                        throw std::runtime_error("round-trip mismatch for " + distribution + " "
                            + std::to_string(raw_bytes) + " " + mode_name(mode) + " " + book_name(book));
                    if (mode == fz::HuffmanExecutionMode::DeviceResident && !byte_match)
                        throw std::runtime_error("HostCoordinated/DeviceResident byte mismatch for "
                            + distribution + " " + std::to_string(raw_bytes) + " " + book_name(book));

                    std::vector<double> host_decompress, gpu_decompress;
                    if (selected(options.operation, "decompress")) {
                        host_decompress = measure_host(options.warmups, options.repetitions,
                                                       decoder_prepare, stream);
                        gpu_decompress = measure_gpu(options.warmups, options.repetitions,
                                                     decoder_prepare, stream);
                    }

                    const bool graph_forward = retained.stage->isGraphCompatible();
                    auto append = [&](const char* operation,
                                      const std::vector<double>& host_samples,
                                      const std::vector<double>& gpu_samples,
                                      bool graph_compatible) {
                        if (host_samples.empty()) return;
                        Result result;
                        result.distribution = distribution;
                        result.input_bytes = raw_bytes;
                        result.mode = mode_name(mode);
                        result.book = book_name(book);
                        result.operation = operation;
                        result.host_ms = median(host_samples);
                        result.gpu_ms = median(gpu_samples);
                        result.host_gbs = static_cast<double>(raw_bytes) / result.host_ms / 1e6;
                        result.gpu_gbs = static_cast<double>(raw_bytes) / result.gpu_ms / 1e6;
                        result.compressed_bytes = compressed_bytes;
                        result.ratio = static_cast<double>(raw_bytes) / compressed_bytes;
                        result.roundtrip = roundtrip;
                        result.byte_match = mode == fz::HuffmanExecutionMode::HostCoordinated
                            ? true : byte_match;
                        result.graph_compatible = graph_compatible;
                        results.push_back(result);
                        std::cout << std::left << std::setw(12) << distribution
                                  << std::right << std::setw(10) << raw_bytes
                                  << "  " << std::setw(15) << mode_name(mode)
                                  << "  " << std::setw(17) << book_name(book)
                                  << "  " << std::setw(10) << operation
                                  << "  host " << std::fixed << std::setprecision(3)
                                  << std::setw(9) << result.host_ms << " ms "
                                  << std::setw(8) << std::setprecision(2) << result.host_gbs << " GB/s"
                                  << "  gpu " << std::setprecision(3) << std::setw(9)
                                  << result.gpu_ms << " ms " << std::setprecision(2)
                                  << std::setw(8) << result.gpu_gbs << " GB/s"
                                  << "  ratio " << result.ratio << "x\n";
                    };
                    append("compress", host_compress, gpu_compress, graph_forward);
                    append("decompress", host_decompress, gpu_decompress, false);
                }
            }
        }
    }

    if (!options.csv.empty()) {
        std::ofstream csv(options.csv);
        if (!csv) throw std::runtime_error("cannot open CSV: " + options.csv);
        csv << "distribution,input_bytes,mode,book,operation,host_median_ms,gpu_median_ms,"
               "host_gbs,gpu_gbs,compressed_bytes,compression_ratio,roundtrip,byte_match,"
               "graph_compatible,status\n";
        csv << std::setprecision(9);
        for (const auto& r : results) {
            csv << r.distribution << ',' << r.input_bytes << ',' << r.mode << ',' << r.book
                << ',' << r.operation << ',' << r.host_ms << ',' << r.gpu_ms << ','
                << r.host_gbs << ',' << r.gpu_gbs << ',' << r.compressed_bytes << ','
                << r.ratio << ',' << r.roundtrip << ',' << r.byte_match << ','
                << r.graph_compatible << ',' << r.status << '\n';
        }
    }

    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "huffman_throughput: " << error.what() << '\n';
    return 1;
}
