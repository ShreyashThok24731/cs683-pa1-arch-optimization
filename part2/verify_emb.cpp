#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <immintrin.h>
#include <cstdlib>
#include <string>
using namespace std;

#define main emb_original_main
#include "emb.cpp"
#undef main

static void run_naive_out(const vector<float>& t, const vector<int>& in, const vector<int>& off, vector<vector<float>>& out) {
    out.clear();
    for (size_t i = 0; i < off.size(); ++i) {
        int s = off[i], e = (i + 1) < off.size() ? off[i+1] : (int)in.size();
        vector<float> b(embedding_dim, 0.0f);
        for (int j = s; j < e; ++j) {
            const float* d = &t[in[j] * embedding_dim];
            for (int r = 0; r < embedding_dim; ++r) b[r] += d[r];
        }
        out.push_back(b);
    }
}

static float diff(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    float m = 0;
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[i].size(); j++)
            m = max(m, fabsf(A[i][j] - B[i][j]));
    return m;
}

int main() {
    embedding_table_size = 100000;
    embedding_dim        = 128;
    input_size           = 500;
    num_bags             = 10;
    PREFETCH_DISTANCE    = 4;

    vector<float> table((size_t)embedding_table_size * embedding_dim);
    for (auto& v : table) v = static_cast<float>(random_int(1000));
    vector<int> in(input_size);
    for (auto& x : in) x = random_int(embedding_table_size);
    vector<int> off;
    for (int i = 0; i < num_bags; ++i) off.push_back((input_size * i) / num_bags);

    vector<vector<float>> ref;
    run_naive_out(table, in, off, ref);

    int fails = 0;
    for (int W : {128, 256, 512}) {
        SIMD_BITS = W;
        {
            vector<vector<float>> got;
            (void)got;
        }
        vector<vector<float>> got;
        for (size_t i = 0; i < off.size(); ++i) {
            int s = off[i], e = (i + 1) < off.size() ? off[i+1] : (int)in.size();
            vector<float> b(embedding_dim, 0.0f);
            int Wf = (W == 512 ? 16 : (W == 256 ? 8 : 4));
            int num = embedding_dim / Wf;
            if (W == 512) {
                vector<__m512> acc(num);
                for (int k=0;k<num;k++) acc[k]=_mm512_setzero_ps();
                for (int j=s;j<e;j++) {
                    const float* d = &table[in[j]*embedding_dim];
                    for (int k=0;k<num;k++) acc[k]=_mm512_add_ps(acc[k], _mm512_loadu_ps(d+k*Wf));
                }
                for (int k=0;k<num;k++) _mm512_storeu_ps(&b[k*Wf], acc[k]);
            } else if (W == 256) {
                vector<__m256> acc(num);
                for (int k=0;k<num;k++) acc[k]=_mm256_setzero_ps();
                for (int j=s;j<e;j++) {
                    const float* d = &table[in[j]*embedding_dim];
                    for (int k=0;k<num;k++) acc[k]=_mm256_add_ps(acc[k], _mm256_loadu_ps(d+k*Wf));
                }
                for (int k=0;k<num;k++) _mm256_storeu_ps(&b[k*Wf], acc[k]);
            } else {
                vector<__m128> acc(num);
                for (int k=0;k<num;k++) acc[k]=_mm_setzero_ps();
                for (int j=s;j<e;j++) {
                    const float* d = &table[in[j]*embedding_dim];
                    for (int k=0;k<num;k++) acc[k]=_mm_add_ps(acc[k], _mm_loadu_ps(d+k*Wf));
                }
                for (int k=0;k<num;k++) _mm_storeu_ps(&b[k*Wf], acc[k]);
            }
            got.push_back(b);
        }
        float d = diff(ref, got);
        cout << "SIMD W=" << W << " max_err=" << d << " " << (d < 1e-2f ? "OK" : "FAIL") << "\n";
        if (d >= 1e-2f) fails++;
    }
    return fails ? 1 : 0;
}
