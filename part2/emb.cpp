#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;

static int perf_ctl_fd = -1;
static int perf_ack_fd = -1;

static void perf_ctl_init() {
    const char* ctl = getenv("EMB_CTL_FIFO");
    const char* ack = getenv("EMB_ACK_FIFO");
    if (!ctl || !ack) return;
    perf_ctl_fd = open(ctl, O_RDWR);
    perf_ack_fd = open(ack, O_RDWR);
}

static void perf_cmd(const char* cmd) {
    if (perf_ctl_fd < 0 || perf_ack_fd < 0) return;
    if (write(perf_ctl_fd, cmd, strlen(cmd)) < 0) return;
    char buf[16];
    if (read(perf_ack_fd, buf, sizeof(buf)) < 0) return;
}

static inline void perf_begin() { perf_cmd("enable\n"); }
static inline void perf_end()   { perf_cmd("disable\n"); }

static int embedding_table_size = 1000000;
static int embedding_dim        = 128;
static int input_size           = 720;
static int num_bags             = 20;
static int PREFETCH_DISTANCE    = 4;
static int PREFETCH_HINT        = 0;
static int SIMD_BITS            = 256;

static int random_int(int range) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(0, range - 1);
    return dis(gen);
}

static inline void do_prefetch(const void *addr, int hint) {
    switch (hint) {
        case 0: _mm_prefetch((const char*)addr, _MM_HINT_T0);  break;
        case 1: _mm_prefetch((const char*)addr, _MM_HINT_T1);  break;
        case 2: _mm_prefetch((const char*)addr, _MM_HINT_T2);  break;
        case 3: _mm_prefetch((const char*)addr, _MM_HINT_NTA); break;
    }
}

static void flush_table(const vector<float>& t) {
    for (size_t i = 0; i < t.size(); i += 16) _mm_clflush(&t[i]);
    _mm_mfence();
}

long long naive_emb(const vector<float>& embedding_table,
                    const vector<int>& input, const vector<int>& offsets) {
    auto start = high_resolution_clock::now();
    vector<vector<float>> output;
    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = ((i + 1) < offsets.size()) ? offsets[i + 1] : (int)input.size();
        vector<float> bag(embedding_dim, 0.0f);
        for (int j = start_idx; j < end_idx; ++j) {
            const float* data = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) bag[d] += data[d];
        }
        output.push_back(bag);
    }
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count();
}

long long run_with_prefetching(const vector<float>& embedding_table,
                               const vector<int>& input, const vector<int>& offsets) {
    auto start = high_resolution_clock::now();
    const int line_floats = 16;
    const int lines_per_row = (embedding_dim + line_floats - 1) / line_floats;

    vector<vector<float>> output;
    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = ((i + 1) < offsets.size()) ? offsets[i + 1] : (int)input.size();
        vector<float> bag(embedding_dim, 0.0f);
        for (int j = start_idx; j < end_idx; ++j) {
            int pj = j + PREFETCH_DISTANCE;
            if (pj < end_idx) {
                int pi = input[pj];
                const float* pref_base = &embedding_table[(size_t)pi * embedding_dim];
                for (int l = 0; l < lines_per_row; ++l) {
                    do_prefetch(pref_base + l * line_floats, PREFETCH_HINT);
                }
            }
            const float* data = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) bag[d] += data[d];
        }
        output.push_back(bag);
    }
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count();
}

long long run_with_simd(const vector<float>& embedding_table,
                        const vector<int>& input, const vector<int>& offsets) {
    auto start = high_resolution_clock::now();
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = ((i + 1) < offsets.size()) ? offsets[i + 1] : (int)input.size();
        vector<float> bag(embedding_dim, 0.0f);

        if (SIMD_BITS == 512) {
            const int W = 16;
            int num = embedding_dim / W;
            vector<__m512> acc(num);
            for (int k = 0; k < num; ++k) acc[k] = _mm512_setzero_ps();
            for (int j = start_idx; j < end_idx; ++j) {
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int k = 0; k < num; ++k)
                    acc[k] = _mm512_add_ps(acc[k], _mm512_loadu_ps(d + k * W));
            }
            for (int k = 0; k < num; ++k) _mm512_storeu_ps(&bag[k * W], acc[k]);
            for (int j = start_idx; j < end_idx; ++j) {
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int r = num * W; r < embedding_dim; ++r) bag[r] += d[r];
            }
        } else if (SIMD_BITS == 256) {
            const int W = 8;
            int num = embedding_dim / W;
            vector<__m256> acc(num);
            for (int k = 0; k < num; ++k) acc[k] = _mm256_setzero_ps();
            for (int j = start_idx; j < end_idx; ++j) {
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int k = 0; k < num; ++k)
                    acc[k] = _mm256_add_ps(acc[k], _mm256_loadu_ps(d + k * W));
            }
            for (int k = 0; k < num; ++k) _mm256_storeu_ps(&bag[k * W], acc[k]);
            for (int j = start_idx; j < end_idx; ++j) {
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int r = num * W; r < embedding_dim; ++r) bag[r] += d[r];
            }
        } else {
            const int W = 4;
            int num = embedding_dim / W;
            vector<__m128> acc(num);
            for (int k = 0; k < num; ++k) acc[k] = _mm_setzero_ps();
            for (int j = start_idx; j < end_idx; ++j) {
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int k = 0; k < num; ++k)
                    acc[k] = _mm_add_ps(acc[k], _mm_loadu_ps(d + k * W));
            }
            for (int k = 0; k < num; ++k) _mm_storeu_ps(&bag[k * W], acc[k]);
            for (int j = start_idx; j < end_idx; ++j) {
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int r = num * W; r < embedding_dim; ++r) bag[r] += d[r];
            }
        }
        output.push_back(bag);
    }
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count();
}

long long run_with_prefetching_simd(const vector<float>& embedding_table,
                                    const vector<int>& input, const vector<int>& offsets) {
    auto start = high_resolution_clock::now();
    const int line_floats = 16;
    const int lines_per_row = (embedding_dim + line_floats - 1) / line_floats;
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = ((i + 1) < offsets.size()) ? offsets[i + 1] : (int)input.size();
        vector<float> bag(embedding_dim, 0.0f);

        if (SIMD_BITS == 512) {
            const int W = 16;
            int num = embedding_dim / W;
            vector<__m512> acc(num);
            for (int k = 0; k < num; ++k) acc[k] = _mm512_setzero_ps();
            for (int j = start_idx; j < end_idx; ++j) {
                int pj = j + PREFETCH_DISTANCE;
                if (pj < end_idx) {
                    const float* p = &embedding_table[(size_t)input[pj] * embedding_dim];
                    for (int l = 0; l < lines_per_row; ++l)
                        do_prefetch(p + l * line_floats, PREFETCH_HINT);
                }
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int k = 0; k < num; ++k)
                    acc[k] = _mm512_add_ps(acc[k], _mm512_loadu_ps(d + k * W));
            }
            for (int k = 0; k < num; ++k) _mm512_storeu_ps(&bag[k * W], acc[k]);
        } else if (SIMD_BITS == 256) {
            const int W = 8;
            int num = embedding_dim / W;
            vector<__m256> acc(num);
            for (int k = 0; k < num; ++k) acc[k] = _mm256_setzero_ps();
            for (int j = start_idx; j < end_idx; ++j) {
                int pj = j + PREFETCH_DISTANCE;
                if (pj < end_idx) {
                    const float* p = &embedding_table[(size_t)input[pj] * embedding_dim];
                    for (int l = 0; l < lines_per_row; ++l)
                        do_prefetch(p + l * line_floats, PREFETCH_HINT);
                }
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int k = 0; k < num; ++k)
                    acc[k] = _mm256_add_ps(acc[k], _mm256_loadu_ps(d + k * W));
            }
            for (int k = 0; k < num; ++k) _mm256_storeu_ps(&bag[k * W], acc[k]);
        } else {
            const int W = 4;
            int num = embedding_dim / W;
            vector<__m128> acc(num);
            for (int k = 0; k < num; ++k) acc[k] = _mm_setzero_ps();
            for (int j = start_idx; j < end_idx; ++j) {
                int pj = j + PREFETCH_DISTANCE;
                if (pj < end_idx) {
                    const float* p = &embedding_table[(size_t)input[pj] * embedding_dim];
                    for (int l = 0; l < lines_per_row; ++l)
                        do_prefetch(p + l * line_floats, PREFETCH_HINT);
                }
                const float* d = &embedding_table[input[j] * embedding_dim];
                for (int k = 0; k < num; ++k)
                    acc[k] = _mm_add_ps(acc[k], _mm_loadu_ps(d + k * W));
            }
            for (int k = 0; k < num; ++k) _mm_storeu_ps(&bag[k * W], acc[k]);
        }
        output.push_back(bag);
    }
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count();
}

static int env_int(const char* k, int def) {
    const char *v = getenv(k);
    return v ? atoi(v) : def;
}

int main() {
    embedding_table_size = env_int("EMB_TABLE_SIZE",   embedding_table_size);
    embedding_dim        = env_int("EMB_DIM",          embedding_dim);
    input_size           = env_int("EMB_INPUT_SIZE",   input_size);
    num_bags             = env_int("EMB_NUM_BAGS",     num_bags);
    PREFETCH_DISTANCE    = env_int("EMB_PREFETCH_DIST",PREFETCH_DISTANCE);
    PREFETCH_HINT        = env_int("EMB_HINT",         PREFETCH_HINT);
    SIMD_BITS            = env_int("EMB_SIMD_WIDTH",   SIMD_BITS);
    int only             = env_int("EMB_ONLY", 0);
    int reps             = env_int("EMB_REPS", 1);

    perf_ctl_init();

    vector<float> embedding_table((size_t)embedding_table_size * embedding_dim);
    for (auto& v : embedding_table) v = static_cast<float>(random_int(1000));
    vector<int> input(input_size);
    for (auto& x : input) x = random_int(embedding_table_size);
    vector<int> offsets;
    for (int i = 0; i < num_bags; ++i) offsets.push_back((input_size * i) / num_bags);

    cout << "run,us\n";

    for (int r = 0; r < reps; ++r) {
        if (only == 0 || only == 1) {
            flush_table(embedding_table);
            perf_begin();
            long long t = naive_emb(embedding_table, input, offsets);
            perf_end();
            cout << "naive," << t << "\n";
        }
        if (only == 0 || only == 2) {
            flush_table(embedding_table);
            perf_begin();
            long long t = run_with_prefetching(embedding_table, input, offsets);
            perf_end();
            cout << "prefetch," << t << "\n";
        }
        if (only == 0 || only == 3) {
            flush_table(embedding_table);
            perf_begin();
            long long t = run_with_simd(embedding_table, input, offsets);
            perf_end();
            cout << "simd," << t << "\n";
        }
        if (only == 0 || only == 4) {
            flush_table(embedding_table);
            perf_begin();
            long long t = run_with_prefetching_simd(embedding_table, input, offsets);
            perf_end();
            cout << "prefetch_simd," << t << "\n";
        }
    }
    return 0;
}
