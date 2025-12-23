#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <memory>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <x86intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr size_t DEFAULT_BATCH_SIZE = 32;
static constexpr float LEARNING_RATE = 0.001f;
static constexpr float GRADIENT_TOLERANCE = 1e-5f;

class EngineConfig {
public:
    static constexpr size_t MIN_WORK_SIZE_FOR_PARALLEL = 4096;
    static constexpr size_t PARAM_ARENA_SIZE = 32 * 1024 * 1024;
    static constexpr size_t ACTIVATION_ARENA_SIZE = 16 * 1024 * 1024;
    static constexpr size_t GRADIENT_ARENA_SIZE = 8 * 1024 * 1024;
    static constexpr size_t MAX_BATCH_SIZE = 64;
    
    static EngineConfig& instance() {
        static EngineConfig config;
        return config;
    }
    
    bool enable_simd = true;
    bool enable_parallel = true;
    
private:
    EngineConfig() = default;
};

class CPUFeatures {
    bool avx_supported = false;
    bool avx2_supported = false;
    int num_threads = 1;

    void detect_features() {
#ifdef _OPENMP
        num_threads = std::max(1, omp_get_max_threads());
#else
        num_threads = 1;
#endif

#if defined(__x86_64__) || defined(_M_X64)
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        
        __cpuid(1, eax, ebx, ecx, edx);
        avx_supported = (ecx & (1 << 28)) != 0;
        
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        avx2_supported = (ebx & (1 << 5)) != 0;
#endif
    }

    CPUFeatures() { detect_features(); }

public:
    static CPUFeatures& get() {
        static CPUFeatures instance;
        return instance;
    }

    bool has_avx() const { return avx_supported && EngineConfig::instance().enable_simd; }
    bool has_avx2() const { return avx2_supported && EngineConfig::instance().enable_simd; }
    int get_thread_count() const { return num_threads; }
    bool should_use_parallel(size_t work_size) const {
        return EngineConfig::instance().enable_parallel && 
               num_threads > 1 && 
               work_size >= EngineConfig::MIN_WORK_SIZE_FOR_PARALLEL;
    }
};

class ArenaAllocator {
    std::unique_ptr<uint8_t[]> memory;
    size_t capacity_;
    uint8_t* current;
    uint8_t* end;

public:
    ArenaAllocator(size_t size) : capacity_(size) {
        try {
            memory = std::make_unique<uint8_t[]>(size);
            current = memory.get();
            end = current + size;
        } catch (const std::bad_alloc&) {
            memory = std::make_unique<uint8_t[]>(size / 2);
            current = memory.get();
            end = current + size / 2;
        }
    }

    void* allocate(size_t size, size_t alignment = 64) {
        uintptr_t ptr = reinterpret_cast<uintptr_t>(current);
        size_t padding = (alignment - (ptr % alignment)) % alignment;

        if (current + padding + size > end) {
            return nullptr;
        }

        current += padding;
        void* result = current;
        current += size;
        return result;
    }

    void reset() { current = memory.get(); }
    size_t used() const { return current - memory.get(); }
    size_t capacity() const { return capacity_; }
};

class MemoryManager {
    std::unique_ptr<ArenaAllocator> param_arena;
    std::unique_ptr<ArenaAllocator> activation_arena;
    std::unique_ptr<ArenaAllocator> gradient_arena;

public:
    MemoryManager() {
        param_arena = std::make_unique<ArenaAllocator>(EngineConfig::PARAM_ARENA_SIZE);
        activation_arena = std::make_unique<ArenaAllocator>(EngineConfig::ACTIVATION_ARENA_SIZE);
        gradient_arena = std::make_unique<ArenaAllocator>(EngineConfig::GRADIENT_ARENA_SIZE);
    }

    ArenaAllocator& get_param_arena() { return *param_arena; }
    ArenaAllocator& get_activation_arena() { return *activation_arena; }
    ArenaAllocator& get_gradient_arena() { return *gradient_arena; }

    void reset_activation() { activation_arena->reset(); }
    void reset_gradient() { gradient_arena->reset(); }
};

static MemoryManager memory_manager;

class Tensor {
    float* data_ptr = nullptr;
    std::vector<size_t> shape_;
    size_t total_size_ = 0;
    bool is_param_ = false;
    bool is_gradient_ = false;

public:
    Tensor() = default;

    Tensor(const std::vector<size_t>& dims, bool param = false, bool gradient = false) 
        : shape_(dims), is_param_(param), is_gradient_(gradient) {
        total_size_ = 1;
        for(auto dim : dims) total_size_ *= dim;

        ArenaAllocator* arena = nullptr;
        if(param) {
            arena = &memory_manager.get_param_arena();
        } else if(gradient) {
            arena = &memory_manager.get_gradient_arena();
        } else {
            arena = &memory_manager.get_activation_arena();
        }

        data_ptr = static_cast<float*>(arena->allocate(total_size_ * sizeof(float)));
        if(data_ptr) {
            std::fill(data_ptr, data_ptr + total_size_, 0.0f);
        }
    }

    Tensor(float* ptr, const std::vector<size_t>& dims) 
        : data_ptr(ptr), shape_(dims) {
        total_size_ = 1;
        for(auto dim : dims) total_size_ *= dim;
    }

    Tensor(Tensor&& other) noexcept 
        : data_ptr(other.data_ptr), 
          shape_(std::move(other.shape_)), 
          total_size_(other.total_size_),
          is_param_(other.is_param_),
          is_gradient_(other.is_gradient_) {
        other.data_ptr = nullptr;
        other.total_size_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if(this != &other) {
            data_ptr = other.data_ptr;
            shape_ = std::move(other.shape_);
            total_size_ = other.total_size_;
            is_param_ = other.is_param_;
            is_gradient_ = other.is_gradient_;
            other.data_ptr = nullptr;
            other.total_size_ = 0;
        }
        return *this;
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    float* ptr() { return data_ptr; }
    const float* ptr() const { return data_ptr; }
    size_t size() const { return total_size_; }
    const std::vector<size_t>& get_shape() const { return shape_; }

    float& operator[](size_t idx) { return data_ptr[idx]; }
    const float& operator[](size_t idx) const { return data_ptr[idx]; }

    void fill(float val) { 
        if(data_ptr) std::fill(data_ptr, data_ptr + total_size_, val); 
    }
};

class MathUtils {
private:
    static inline float exp_approx(float x) {
        x = std::min(std::max(x, -88.0f), 88.0f);
        union { float f; int32_t i; } u;
        u.i = (int32_t)(12102203 * x + 1065353216);
        return u.f;
    }
    
    static inline float log_approx(float x) {
        x = std::max(x, 1e-12f);
        union { float f; int32_t i; } u;
        u.f = x;
        return (u.i - 1065353216) * 8.262958405e-8f;
    }

#if defined(__x86_64__) || defined(_M_X64)
    static inline float exp_avx(float x) {
        __m128 x_vec = _mm_set_ss(x);
        __m128 result = exp_ps(x_vec);
        return _mm_cvtss_f32(result);
    }
    
    static inline float log_avx(float x) {
        __m128 x_vec = _mm_set_ss(x);
        __m128 result = log_ps(x_vec);
        return _mm_cvtss_f32(result);
    }
    
    static __m128 exp_ps(__m128 x) {
        const __m128 a = _mm_set1_ps(12102203.0f);
        const __m128 b = _mm_set1_ps(1065353216.0f);
        x = _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(-88.0f)), _mm_set1_ps(88.0f));
        __m128 result = _mm_add_ps(_mm_mul_ps(a, x), b);
        union { __m128 vec; float f[4]; } u = { result };
        u.f[0] = reinterpret_cast<float&>(static_cast<int32_t>(u.f[0]));
        u.f[1] = reinterpret_cast<float&>(static_cast<int32_t>(u.f[1]));
        u.f[2] = reinterpret_cast<float&>(static_cast<int32_t>(u.f[2]));
        u.f[3] = reinterpret_cast<float&>(static_cast<int32_t>(u.f[3]));
        return u.vec;
    }
    
    static __m128 log_ps(__m128 x) {
        x = _mm_max_ps(x, _mm_set1_ps(1e-12f));
        const __m128 inv = _mm_set1_ps(8.262958405e-8f);
        const __m128 b = _mm_set1_ps(1065353216.0f);
        union { __m128 vec; float f[4]; } u = { x };
        u.f[0] = static_cast<float>(reinterpret_cast<int32_t&>(u.f[0]));
        u.f[1] = static_cast<float>(reinterpret_cast<int32_t&>(u.f[1]));
        u.f[2] = static_cast<float>(reinterpret_cast<int32_t&>(u.f[2]));
        u.f[3] = static_cast<float>(reinterpret_cast<int32_t&>(u.f[3]));
        __m128 result = _mm_sub_ps(u.vec, b);
        return _mm_mul_ps(result, inv);
    }
#endif

public:
    static float fast_exp(float x) {
#if defined(__x86_64__) || defined(_M_X64)
        if(CPUFeatures::get().has_avx()) {
            return exp_avx(x);
        }
#endif
        return exp_approx(x);
    }
    
    static float fast_log(float x) {
#if defined(__x86_64__) || defined(_M_X64)
        if(CPUFeatures::get().has_avx()) {
            return log_avx(x);
        }
#endif
        return log_approx(x);
    }
};

class Adam {
    struct ParamState {
        Tensor m;
        Tensor v;
        float* param;
        float* grad;
    };

    std::vector<ParamState> params;
    std::unordered_map<float*, size_t> param_index;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float beta1_pow;
    float beta2_pow;

public:
    Adam(float lr = LEARNING_RATE, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f) 
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), beta1_pow(1.0f), beta2_pow(1.0f) {}

    void add_param(float* param, float* grad, size_t size, const std::vector<size_t>& shape) {
        if(param_index.find(param) != param_index.end()) return;

        param_index[param] = params.size();
        params.push_back({
            Tensor(shape, false, true),
            Tensor(shape, false, true),
            param,
            grad
        });
    }

    void step() {
        beta1_pow *= beta1;
        beta2_pow *= beta2;

        float inv_beta1_pow = 1.0f / (1.0f - beta1_pow);
        float inv_beta2_pow = 1.0f / (1.0f - beta2_pow);

#if defined(__x86_64__) || defined(_M_X64)
        bool use_simd = CPUFeatures::get().has_avx2();
#endif

        for(auto& p : params) {
            float* param_ptr = p.param;
            float* grad_ptr = p.grad;
            float* m_ptr = p.m.ptr();
            float* v_ptr = p.v.ptr();
            size_t n = p.m.size();

#if defined(__x86_64__) || defined(_M_X64)
            if(use_simd) {
                __m256 beta1_vec = _mm256_set1_ps(beta1);
                __m256 beta1_inv_vec = _mm256_set1_ps(1.0f - beta1);
                __m256 beta2_vec = _mm256_set1_ps(beta2);
                __m256 beta2_inv_vec = _mm256_set1_ps(1.0f - beta2);
                __m256 lr_vec = _mm256_set1_ps(learning_rate);
                __m256 eps_vec = _mm256_set1_ps(epsilon);
                __m256 beta1_pow_vec = _mm256_set1_ps(inv_beta1_pow);
                __m256 beta2_pow_vec = _mm256_set1_ps(inv_beta2_pow);
                
                size_t i = 0;
                for(; i + 8 <= n; i += 8) {
                    __m256 g_vec = _mm256_loadu_ps(grad_ptr + i);
                    __m256 m_vec = _mm256_loadu_ps(m_ptr + i);
                    __m256 v_vec = _mm256_loadu_ps(v_ptr + i);
                    __m256 p_vec = _mm256_loadu_ps(param_ptr + i);
                    
                    __m256 m_new = _mm256_fmadd_ps(beta1_inv_vec, g_vec, 
                                                 _mm256_mul_ps(beta1_vec, m_vec));
                    __m256 g_square = _mm256_mul_ps(g_vec, g_vec);
                    __m256 v_new = _mm256_fmadd_ps(beta2_inv_vec, g_square,
                                                 _mm256_mul_ps(beta2_vec, v_vec));
                    
                    _mm256_storeu_ps(m_ptr + i, m_new);
                    _mm256_storeu_ps(v_ptr + i, v_new);
                    
                    __m256 m_hat = _mm256_mul_ps(m_new, beta1_pow_vec);
                    __m256 v_hat = _mm256_mul_ps(v_new, beta2_pow_vec);
                    __m256 v_sqrt = _mm256_sqrt_ps(v_hat);
                    __m256 denom = _mm256_add_ps(v_sqrt, eps_vec);
                    __m256 update = _mm256_div_ps(m_hat, denom);
                    update = _mm256_mul_ps(lr_vec, update);
                    
                    __m256 p_new = _mm256_sub_ps(p_vec, update);
                    _mm256_storeu_ps(param_ptr + i, p_new);
                }
                
                for(; i < n; ++i) {
                    float gi = grad_ptr[i];
                    float mi = beta1 * m_ptr[i] + (1.0f - beta1) * gi;
                    float vi = beta2 * v_ptr[i] + (1.0f - beta2) * gi * gi;
                    
                    m_ptr[i] = mi;
                    v_ptr[i] = vi;
                    
                    float m_hat = mi * inv_beta1_pow;
                    float v_hat = vi * inv_beta2_pow;
                    
                    param_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            } else 
#endif
            {
                for(size_t i = 0; i < n; ++i) {
                    float gi = grad_ptr[i];
                    float mi = beta1 * m_ptr[i] + (1.0f - beta1) * gi;
                    float vi = beta2 * v_ptr[i] + (1.0f - beta2) * gi * gi;
                    
                    m_ptr[i] = mi;
                    v_ptr[i] = vi;
                    
                    float m_hat = mi * inv_beta1_pow;
                    float v_hat = vi * inv_beta2_pow;
                    
                    param_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
        }
    }

    void zero_grad() {
        for(auto& p : params) {
            std::fill(p.grad, p.grad + p.m.size(), 0.0f);
        }
    }
};

class OpNode {
public:
    virtual ~OpNode() = default;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void register_params(Adam& optimizer) = 0;
    virtual void zero_grad() = 0;
    virtual void allocate_tensors() = 0;
    
    std::string name;
    std::vector<OpNode*> inputs;
    std::vector<OpNode*> outputs;
    std::vector<Tensor*> input_tensors;
    std::vector<Tensor*> output_tensors;
    std::vector<Tensor*> param_tensors;
    std::vector<Tensor*> grad_tensors;
    
protected:
    OpNode(const std::string& n) : name(n) {}
};

class LinearOp : public OpNode {
    Tensor weights;
    Tensor bias;
    Tensor weight_grad;
    Tensor bias_grad;
    Tensor* input_cache;
    size_t in_features;
    size_t out_features;

    void forward_scalar(const float* x_ptr, float* y_ptr, 
                       const float* w_ptr, const float* b_ptr,
                       size_t batch_size) {
        for(size_t i = 0; i < batch_size; ++i) {
            const float* x_row = x_ptr + i * in_features;
            float* y_row = y_ptr + i * out_features;
            
            for(size_t j = 0; j < out_features; ++j) {
                float sum = b_ptr[j];
                const float* w_row = w_ptr + j * in_features;
                for(size_t k = 0; k < in_features; ++k) {
                    sum += x_row[k] * w_row[k];
                }
                y_row[j] = sum;
            }
        }
    }

#if defined(__x86_64__) || defined(_M_X64)
    void forward_avx(const float* x_ptr, float* y_ptr,
                    const float* w_ptr, const float* b_ptr,
                    size_t batch_size) {
        for(size_t i = 0; i < batch_size; ++i) {
            const float* x_row = x_ptr + i * in_features;
            float* y_row = y_ptr + i * out_features;
            
            for(size_t j = 0; j < out_features; ++j) {
                float sum = b_ptr[j];
                const float* w_row = w_ptr + j * in_features;
                
                __m256 sum_vec = _mm256_setzero_ps();
                size_t k = 0;
                for(; k + 8 <= in_features; k += 8) {
                    __m256 x_vec = _mm256_loadu_ps(x_row + k);
                    __m256 w_vec = _mm256_loadu_ps(w_row + k);
                    sum_vec = _mm256_fmadd_ps(x_vec, w_vec, sum_vec);
                }
                
                alignas(32) float temp[8];
                _mm256_store_ps(temp, sum_vec);
                for(size_t t = 0; t < 8; ++t) sum += temp[t];
                
                for(; k < in_features; ++k) {
                    sum += x_row[k] * w_row[k];
                }
                
                y_row[j] = sum;
            }
        }
    }
#endif

    void compute_weight_grad_scalar(const float* x_ptr, const float* g_ptr, 
                                   float* wg_ptr, size_t batch_size) {
        for(size_t i = 0; i < batch_size; ++i) {
            const float* x_row = x_ptr + i * in_features;
            const float* g_row = g_ptr + i * out_features;
            
            for(size_t j = 0; j < out_features; ++j) {
                float gj = g_row[j];
                float* wg_row = wg_ptr + j * in_features;
                for(size_t k = 0; k < in_features; ++k) {
                    wg_row[k] += gj * x_row[k];
                }
            }
        }
    }

#if defined(__x86_64__) || defined(_M_X64)
    void compute_weight_grad_avx(const float* x_ptr, const float* g_ptr,
                                float* wg_ptr, size_t batch_size) {
        for(size_t i = 0; i < batch_size; ++i) {
            const float* x_row = x_ptr + i * in_features;
            const float* g_row = g_ptr + i * out_features;
            
            for(size_t j = 0; j < out_features; ++j) {
                float gj = g_row[j];
                __m256 gj_vec = _mm256_set1_ps(gj);
                float* wg_row = wg_ptr + j * in_features;
                
                size_t k = 0;
                for(; k + 8 <= in_features; k += 8) {
                    __m256 x_vec = _mm256_loadu_ps(x_row + k);
                    __m256 wg_vec = _mm256_loadu_ps(wg_row + k);
                    wg_vec = _mm256_fmadd_ps(gj_vec, x_vec, wg_vec);
                    _mm256_storeu_ps(wg_row + k, wg_vec);
                }
                
                for(; k < in_features; ++k) {
                    wg_row[k] += gj * x_row[k];
                }
            }
        }
    }
#endif

    void compute_input_grad_scalar(const float* g_ptr, const float* w_ptr,
                                  float* ig_ptr, size_t batch_size) {
        for(size_t i = 0; i < batch_size; ++i) {
            const float* g_row = g_ptr + i * out_features;
            float* ig_row = ig_ptr + i * in_features;
            
            for(size_t k = 0; k < in_features; ++k) {
                float sum = 0.0f;
                for(size_t j = 0; j < out_features; ++j) {
                    sum += g_row[j] * w_ptr[j * in_features + k];
                }
                ig_row[k] = sum;
            }
        }
    }

#if defined(__x86_64__) || defined(_M_X64)
    void compute_input_grad_avx(const float* g_ptr, const float* w_ptr,
                               float* ig_ptr, size_t batch_size) {
        for(size_t i = 0; i < batch_size; ++i) {
            const float* g_row = g_ptr + i * out_features;
            float* ig_row = ig_ptr + i * in_features;
            
            for(size_t k = 0; k < in_features; ++k) {
                __m256 sum_vec = _mm256_setzero_ps();
                size_t j = 0;
                for(; j + 8 <= out_features; j += 8) {
                    __m256 g_vec = _mm256_loadu_ps(g_row + j);
                    __m256 w_vec = _mm256_setr_ps(
                        w_ptr[j * in_features + k],
                        w_ptr[(j+1) * in_features + k],
                        w_ptr[(j+2) * in_features + k],
                        w_ptr[(j+3) * in_features + k],
                        w_ptr[(j+4) * in_features + k],
                        w_ptr[(j+5) * in_features + k],
                        w_ptr[(j+6) * in_features + k],
                        w_ptr[(j+7) * in_features + k]
                    );
                    sum_vec = _mm256_fmadd_ps(g_vec, w_vec, sum_vec);
                }
                
                alignas(32) float temp[8];
                _mm256_store_ps(temp, sum_vec);
                float sum = 0.0f;
                for(size_t t = 0; t < 8; ++t) sum += temp[t];
                
                for(; j < out_features; ++j) {
                    sum += g_row[j] * w_ptr[j * in_features + k];
                }
                
                ig_row[k] = sum;
            }
        }
    }
#endif

public:
    LinearOp(const std::string& n, size_t in, size_t out) 
        : OpNode(n),
          weights({out, in}, true),
          bias({out}, true),
          weight_grad({out, in}, false, true),
          bias_grad({out}, false, true),
          input_cache(nullptr),
          in_features(in),
          out_features(out) {
        
        param_tensors.push_back(&weights);
        param_tensors.push_back(&bias);
        
        float stddev = std::sqrt(2.0f / in);
        weights.random_normal(0.0f, stddev);
        bias.fill(0.0f);
    }

    void forward() override {
        if(input_tensors.empty() || output_tensors.empty()) return;

        Tensor* x = input_tensors[0];
        Tensor* y = output_tensors[0];
        input_cache = x;

        size_t batch_size = x->size() / in_features;
        const float* x_ptr = x->ptr();
        float* y_ptr = y->ptr();
        const float* w_ptr = weights.ptr();
        const float* b_ptr = bias.ptr();

#if defined(__x86_64__) || defined(_M_X64)
        if(CPUFeatures::get().has_avx2() && 
           batch_size * out_features * in_features >= 1024) {
            if(CPUFeatures::get().should_use_parallel(batch_size * out_features)) {
#ifdef _OPENMP
                #pragma omp parallel for
#endif
                for(size_t i = 0; i < batch_size; ++i) {
                    const float* x_row = x_ptr + i * in_features;
                    float* y_row = y_ptr + i * out_features;
                    
                    for(size_t j = 0; j < out_features; ++j) {
                        float sum = b_ptr[j];
                        const float* w_row = w_ptr + j * in_features;
                        
                        __m256 sum_vec = _mm256_setzero_ps();
                        size_t k = 0;
                        for(; k + 8 <= in_features; k += 8) {
                            __m256 x_vec = _mm256_loadu_ps(x_row + k);
                            __m256 w_vec = _mm256_loadu_ps(w_row + k);
                            sum_vec = _mm256_fmadd_ps(x_vec, w_vec, sum_vec);
                        }
                        
                        alignas(32) float temp[8];
                        _mm256_store_ps(temp, sum_vec);
                        for(size_t t = 0; t < 8; ++t) sum += temp[t];
                        
                        for(; k < in_features; ++k) {
                            sum += x_row[k] * w_row[k];
                        }
                        
                        y_row[j] = sum;
                    }
                }
            } else {
                forward_avx(x_ptr, y_ptr, w_ptr, b_ptr, batch_size);
            }
        } else 
#endif
        {
            if(CPUFeatures::get().should_use_parallel(batch_size * out_features)) {
#ifdef _OPENMP
                #pragma omp parallel for
#endif
                for(size_t i = 0; i < batch_size; ++i) {
                    const float* x_row = x_ptr + i * in_features;
                    float* y_row = y_ptr + i * out_features;
                    
                    for(size_t j = 0; j < out_features; ++j) {
                        float sum = b_ptr[j];
                        const float* w_row = w_ptr + j * in_features;
                        for(size_t k = 0; k < in_features; ++k) {
                            sum += x_row[k] * w_row[k];
                        }
                        y_row[j] = sum;
                    }
                }
            } else {
                forward_scalar(x_ptr, y_ptr, w_ptr, b_ptr, batch_size);
            }
        }
    }

    void backward() override {
        if(input_tensors.size() < 2 || output_tensors.size() < 2 || !input_cache) return;

        Tensor* grad_output = input_tensors[1];
        Tensor* grad_input = output_tensors[1];

        size_t batch_size = grad_output->size() / out_features;
        const float* g_ptr = grad_output->ptr();
        const float* x_ptr = input_cache->ptr();
        float* gi_ptr = grad_input->ptr();
        float* wg_ptr = weight_grad.ptr();
        float* bg_ptr = bias_grad.ptr();

        std::fill(wg_ptr, wg_ptr + weight_grad.size(), 0.0f);
        std::fill(bg_ptr, bg_ptr + bias_grad.size(), 0.0f);

        bool use_parallel = CPUFeatures::get().should_use_parallel(batch_size * out_features * in_features);
        
#if defined(__x86_64__) || defined(_M_X64)
        bool use_simd = CPUFeatures::get().has_avx2();
#endif

        if(use_parallel && batch_size > 1) {
            size_t num_tiles = (batch_size + 15) / 16;
            
#ifdef _OPENMP
            #pragma omp parallel
#endif
            {
                std::vector<float> local_wg(weight_grad.size(), 0.0f);
                std::vector<float> local_bg(bias_grad.size(), 0.0f);
                
#ifdef _OPENMP
                #pragma omp for
#endif
                for(size_t tile = 0; tile < num_tiles; ++tile) {
                    size_t start = tile * 16;
                    size_t end = std::min(start + 16, batch_size);
                    
                    for(size_t i = start; i < end; ++i) {
                        const float* x_row = x_ptr + i * in_features;
                        const float* g_row = g_ptr + i * out_features;
                        float* gi_row = gi_ptr + i * in_features;
                        
                        for(size_t j = 0; j < out_features; ++j) {
                            float gj = g_row[j];
                            local_bg[j] += gj;
                            
                            float* wg_row = local_wg.data() + j * in_features;
#if defined(__x86_64__) || defined(_M_X64)
                            if(use_simd) {
                                __m256 gj_vec = _mm256_set1_ps(gj);
                                size_t k = 0;
                                for(; k + 8 <= in_features; k += 8) {
                                    __m256 x_vec = _mm256_loadu_ps(x_row + k);
                                    __m256 wg_vec = _mm256_loadu_ps(wg_row + k);
                                    wg_vec = _mm256_fmadd_ps(gj_vec, x_vec, wg_vec);
                                    _mm256_storeu_ps(wg_row + k, wg_vec);
                                }
                                for(; k < in_features; ++k) {
                                    wg_row[k] += gj * x_row[k];
                                }
                            } else 
#endif
                            {
                                for(size_t k = 0; k < in_features; ++k) {
                                    wg_row[k] += gj * x_row[k];
                                }
                            }
                        }
                        
                        const float* w_ptr = weights.ptr();
                        for(size_t k = 0; k < in_features; ++k) {
                            float sum = 0.0f;
#if defined(__x86_64__) || defined(_M_X64)
                            if(use_simd) {
                                __m256 sum_vec = _mm256_setzero_ps();
                                size_t j = 0;
                                for(; j + 8 <= out_features; j += 8) {
                                    __m256 g_vec = _mm256_loadu_ps(g_row + j);
                                    __m256 w_vec = _mm256_setr_ps(
                                        w_ptr[j * in_features + k],
                                        w_ptr[(j+1) * in_features + k],
                                        w_ptr[(j+2) * in_features + k],
                                        w_ptr[(j+3) * in_features + k],
                                        w_ptr[(j+4) * in_features + k],
                                        w_ptr[(j+5) * in_features + k],
                                        w_ptr[(j+6) * in_features + k],
                                        w_ptr[(j+7) * in_features + k]
                                    );
                                    sum_vec = _mm256_fmadd_ps(g_vec, w_vec, sum_vec);
                                }
                                
                                alignas(32) float temp[8];
                                _mm256_store_ps(temp, sum_vec);
                                for(size_t t = 0; t < 8; ++t) sum += temp[t];
                                
                                for(; j < out_features; ++j) {
                                    sum += g_row[j] * w_ptr[j * in_features + k];
                                }
                            } else 
#endif
                            {
                                for(size_t j = 0; j < out_features; ++j) {
                                    sum += g_row[j] * w_ptr[j * in_features + k];
                                }
                            }
                            gi_row[k] = sum;
                        }
                    }
                }
                
#ifdef _OPENMP
                #pragma omp critical
#endif
                {
                    for(size_t i = 0; i < weight_grad.size(); ++i) {
                        wg_ptr[i] += local_wg[i];
                    }
                    for(size_t i = 0; i < bias_grad.size(); ++i) {
                        bg_ptr[i] += local_bg[i];
                    }
                }
            }
        } else {
            compute_weight_grad_scalar(x_ptr, g_ptr, wg_ptr, batch_size);
#if defined(__x86_64__) || defined(_M_X64)
            if(use_simd) {
                compute_input_grad_avx(g_ptr, weights.ptr(), gi_ptr, batch_size);
            } else 
#endif
            {
                compute_input_grad_scalar(g_ptr, weights.ptr(), gi_ptr, batch_size);
            }
            
            for(size_t i = 0; i < batch_size; ++i) {
                const float* g_row = g_ptr + i * out_features;
                for(size_t j = 0; j < out_features; ++j) {
                    bg_ptr[j] += g_row[j];
                }
            }
        }
        
        float inv_batch = 1.0f / batch_size;
        for(size_t i = 0; i < weight_grad.size(); ++i) {
            wg_ptr[i] *= inv_batch;
        }
        for(size_t i = 0; i < bias_grad.size(); ++i) {
            bg_ptr[i] *= inv_batch;
        }
    }

    void register_params(Adam& optimizer) override {
        optimizer.add_param(weights.ptr(), weight_grad.ptr(), weights.size(), weights.get_shape());
        optimizer.add_param(bias.ptr(), bias_grad.ptr(), bias.size(), bias.get_shape());
    }

    void zero_grad() override {
        weight_grad.fill(0.0f);
        bias_grad.fill(0.0f);
    }

    void allocate_tensors() override {
        grad_tensors.push_back(&weight_grad);
        grad_tensors.push_back(&bias_grad);
    }
};

class ReLUOp : public OpNode {
    std::unique_ptr<float[]> mask_data;
    size_t mask_size = 0;

public:
    ReLUOp(const std::string& n) : OpNode(n) {}

    void forward() override {
        if(input_tensors.empty() || output_tensors.empty()) return;

        Tensor* x = input_tensors[0];
        Tensor* y = output_tensors[0];
        size_t n = x->size();
        
        if(!mask_data || mask_size < n) {
            mask_data = std::make_unique<float[]>(n);
            mask_size = n;
        }
        
        const float* x_ptr = x->ptr();
        float* y_ptr = y->ptr();
        float* m_ptr = mask_data.get();

        bool use_parallel = CPUFeatures::get().should_use_parallel(n);
        
        if(use_parallel) {
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for(size_t i = 0; i < n; ++i) {
                float val = x_ptr[i];
                m_ptr[i] = val > 0 ? 1.0f : 0.0f;
                y_ptr[i] = val * m_ptr[i];
            }
        } else {
            for(size_t i = 0; i < n; ++i) {
                float val = x_ptr[i];
                m_ptr[i] = val > 0 ? 1.0f : 0.0f;
                y_ptr[i] = val * m_ptr[i];
            }
        }
    }

    void backward() override {
        if(input_tensors.size() < 2 || output_tensors.size() < 2) return;

        Tensor* grad_output = input_tensors[1];
        Tensor* grad_input = output_tensors[1];
        size_t n = grad_output->size();
        const float* grad_ptr = grad_output->ptr();
        float* gi_ptr = grad_input->ptr();
        const float* m_ptr = mask_data.get();

        bool use_parallel = CPUFeatures::get().should_use_parallel(n);
        
        if(use_parallel) {
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for(size_t i = 0; i < n; ++i) {
                gi_ptr[i] = grad_ptr[i] * m_ptr[i];
            }
        } else {
            for(size_t i = 0; i < n; ++i) {
                gi_ptr[i] = grad_ptr[i] * m_ptr[i];
            }
        }
    }

    void register_params(Adam&) override {}
    void zero_grad() override {}
    void allocate_tensors() override {}
};

class SoftmaxCrossEntropyOp : public OpNode {
    std::unique_ptr<float[]> probs_data;
    size_t probs_size = 0;

public:
    SoftmaxCrossEntropyOp(const std::string& n) : OpNode(n) {}

    void forward() override {
        if(input_tensors.size() < 2 || output_tensors.empty()) return;

        Tensor* logits = input_tensors[0];
        Tensor* target = input_tensors[1];
        Tensor* loss = output_tensors[0];

        size_t batch_size = logits->size() / logits->get_shape().back();
        size_t features = logits->get_shape().back();
        size_t total_size = batch_size * features;
        
        if(!probs_data || probs_size < total_size) {
            probs_data = std::make_unique<float[]>(total_size);
            probs_size = total_size;
        }

        const float* logits_ptr = logits->ptr();
        float* probs_ptr = probs_data.get();
        const float* target_ptr = target->ptr();

        float total_loss = 0.0f;

        bool use_parallel = CPUFeatures::get().should_use_parallel(batch_size);
        
        if(use_parallel) {
#ifdef _OPENMP
            #pragma omp parallel for reduction(+:total_loss)
#endif
            for(size_t i = 0; i < batch_size; ++i) {
                const float* logits_row = logits_ptr + i * features;
                float* probs_row = probs_ptr + i * features;
                const float* target_row = target_ptr + i * features;

                float max_val = -std::numeric_limits<float>::max();
                for(size_t j = 0; j < features; ++j) {
                    max_val = std::max(max_val, logits_row[j]);
                }

                float sum = 0.0f;
                for(size_t j = 0; j < features; ++j) {
                    float val = MathUtils::fast_exp(logits_row[j] - max_val);
                    probs_row[j] = val;
                    sum += val;
                }

                float inv_sum = 1.0f / sum;
                for(size_t j = 0; j < features; ++j) {
                    probs_row[j] *= inv_sum;
                }

                for(size_t j = 0; j < features; ++j) {
                    if(target_row[j] > 0.5f) {
                        float prob = std::max(probs_row[j], 1e-12f);
                        total_loss -= MathUtils::fast_log(prob);
                        break;
                    }
                }
            }
        } else {
            for(size_t i = 0; i < batch_size; ++i) {
                const float* logits_row = logits_ptr + i * features;
                float* probs_row = probs_ptr + i * features;
                const float* target_row = target_ptr + i * features;

                float max_val = -std::numeric_limits<float>::max();
                for(size_t j = 0; j < features; ++j) {
                    max_val = std::max(max_val, logits_row[j]);
                }

                float sum = 0.0f;
                for(size_t j = 0; j < features; ++j) {
                    float val = MathUtils::fast_exp(logits_row[j] - max_val);
                    probs_row[j] = val;
                    sum += val;
                }

                float inv_sum = 1.0f / sum;
                for(size_t j = 0; j < features; ++j) {
                    probs_row[j] *= inv_sum;
                }

                for(size_t j = 0; j < features; ++j) {
                    if(target_row[j] > 0.5f) {
                        float prob = std::max(probs_row[j], 1e-12f);
                        total_loss -= MathUtils::fast_log(prob);
                        break;
                    }
                }
            }
        }

        (*loss)[0] = total_loss / batch_size;
    }

    void backward() override {
        if(input_tensors.size() < 2 || output_tensors.size() < 1) return;

        Tensor* target = input_tensors[1];
        Tensor* grad_output = output_tensors[0];

        size_t batch_size = probs_size / target->get_shape().back();
        size_t features = target->get_shape().back();

        const float* p = probs_data.get();
        const float* t = target->ptr();
        float* g = grad_output->ptr();

        float inv_batch = 1.0f / batch_size;
        
        bool use_parallel = CPUFeatures::get().should_use_parallel(batch_size * features);
        
        if(use_parallel) {
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for(size_t i = 0; i < batch_size * features; ++i) {
                g[i] = (p[i] - t[i]) * inv_batch;
            }
        } else {
            for(size_t i = 0; i < batch_size * features; ++i) {
                g[i] = (p[i] - t[i]) * inv_batch;
            }
        }
    }

    void register_params(Adam&) override {}
    void zero_grad() override {}
    void allocate_tensors() override {}
};

class StaticGraph {
    std::unordered_map<std::string, std::unique_ptr<OpNode>> ops;
    std::vector<OpNode*> forward_order;
    std::vector<OpNode*> backward_order;
    Adam optimizer;
    
public:
    StaticGraph(float lr = LEARNING_RATE) : optimizer(lr, 0.9f, 0.999f) {}
    
    void add_op(std::unique_ptr<OpNode> op) {
        std::string name = op->name;
        ops[name] = std::move(op);
    }
    
    void connect(const std::string& from, const std::string& to) {
        auto from_it = ops.find(from);
        auto to_it = ops.find(to);
        if(from_it != ops.end() && to_it != ops.end()) {
            from_it->second->outputs.push_back(to_it->second.get());
            to_it->second->inputs.push_back(from_it->second.get());
        }
    }
    
    void build() {
        for(auto& kv : ops) {
            kv.second->allocate_tensors();
        }
        
        for(auto& kv : ops) {
            kv.second->register_params(optimizer);
        }
        
        std::unordered_map<OpNode*, size_t> in_degree;
        std::queue<OpNode*> ready;
        
        for(auto& kv : ops) {
            in_degree[kv.second.get()] = kv.second->inputs.size();
            if(kv.second->inputs.empty()) {
                ready.push(kv.second.get());
            }
        }
        
        while(!ready.empty()) {
            OpNode* current = ready.front();
            ready.pop();
            forward_order.push_back(current);
            
            for(OpNode* next : current->outputs) {
                if(--in_degree[next] == 0) {
                    ready.push(next);
                }
            }
        }
        
        backward_order = forward_order;
        std::reverse(backward_order.begin(), backward_order.end());
    }
    
    void forward() {
        memory_manager.reset_activation();
        for(auto op : forward_order) {
            op->forward();
        }
    }
    
    void backward() {
        for(auto op : backward_order) {
            op->backward();
        }
    }
    
    void step() {
        optimizer.step();
        memory_manager.reset_gradient();
    }
    
    void zero_grad() {
        optimizer.zero_grad();
    }
    
    OpNode* get_op(const std::string& name) {
        auto it = ops.find(name);
        return it != ops.end() ? it->second.get() : nullptr;
    }
};

bool validate_gradients() {
    std::cout << "Validating gradients...\n";

    StaticGraph test_graph(0.001f);
    test_graph.add_op(std::make_unique<LinearOp>("linear", 3, 2));
    test_graph.add_op(std::make_unique<SoftmaxCrossEntropyOp>("loss"));

    test_graph.connect("linear", "loss");

    Tensor test_input({1, 3}, false);
    Tensor test_target({1, 2}, false);
    Tensor test_loss({1}, false);
    Tensor test_grad({1, 2}, false);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for(size_t i = 0; i < test_input.size(); ++i) {
        test_input[i] = dist(rng);
    }
    test_target[0] = 1.0f;
    test_target[1] = 0.0f;

    auto linear_op = dynamic_cast<LinearOp*>(test_graph.get_op("linear"));
    auto loss_op = dynamic_cast<SoftmaxCrossEntropyOp*>(test_graph.get_op("loss"));
    
    if(!linear_op || !loss_op) return false;

    linear_op->input_tensors.push_back(&test_input);
    linear_op->output_tensors.push_back(&test_grad);
    loss_op->input_tensors.push_back(&test_grad);
    loss_op->input_tensors.push_back(&test_target);
    loss_op->output_tensors.push_back(&test_loss);

    test_graph.build();

    float eps = 1e-4f;
    float* w_ptr = linear_op->weights.ptr();
    float* b_ptr = linear_op->bias.ptr();
    float* w_grad = linear_op->weight_grad.ptr();
    float* b_grad = linear_op->bias_grad.ptr();

    std::vector<float> original_w(w_ptr, w_ptr + linear_op->weights.size());
    std::vector<float> original_b(b_ptr, b_ptr + linear_op->bias.size());

    test_graph.zero_grad();
    test_graph.forward();
    test_graph.backward();

    std::vector<float> analytic_w_grad(w_grad, w_grad + linear_op->weights.size());
    std::vector<float> analytic_b_grad(b_grad, b_grad + linear_op->bias.size());

    std::vector<float> numeric_w_grad(linear_op->weights.size(), 0.0f);
    std::vector<float> numeric_b_grad(linear_op->bias.size(), 0.0f);

    for(size_t i = 0; i < linear_op->weights.size(); ++i) {
        float original = w_ptr[i];
        
        w_ptr[i] = original + eps;
        test_graph.zero_grad();
        test_graph.forward();
        float loss_plus = test_loss[0];
        
        w_ptr[i] = original - eps;
        test_graph.zero_grad();
        test_graph.forward();
        float loss_minus = test_loss[0];
        
        w_ptr[i] = original;
        numeric_w_grad[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    for(size_t i = 0; i < linear_op->bias.size(); ++i) {
        float original = b_ptr[i];
        
        b_ptr[i] = original + eps;
        test_graph.zero_grad();
        test_graph.forward();
        float loss_plus = test_loss[0];
        
        b_ptr[i] = original - eps;
        test_graph.zero_grad();
        test_graph.forward();
        float loss_minus = test_loss[0];
        
        b_ptr[i] = original;
        numeric_b_grad[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    std::copy(original_w.begin(), original_w.end(), w_ptr);
    std::copy(original_b.begin(), original_b.end(), b_ptr);

    bool passed = true;
    for(size_t i = 0; i < analytic_w_grad.size(); ++i) {
        float diff = std::abs(analytic_w_grad[i] - numeric_w_grad[i]);
        float scale = std::max(std::abs(analytic_w_grad[i]), std::abs(numeric_w_grad[i])) + 1e-8f;
        if(diff / scale > GRADIENT_TOLERANCE) {
            std::cout << "Weight gradient mismatch at " << i << ": " 
                      << analytic_w_grad[i] << " vs " << numeric_w_grad[i] << "\n";
            passed = false;
        }
    }

    for(size_t i = 0; i < analytic_b_grad.size(); ++i) {
        float diff = std::abs(analytic_b_grad[i] - numeric_b_grad[i]);
        float scale = std::max(std::abs(analytic_b_grad[i]), std::abs(numeric_b_grad[i])) + 1e-8f;
        if(diff / scale > GRADIENT_TOLERANCE) {
            std::cout << "Bias gradient mismatch at " << i << ": " 
                      << analytic_b_grad[i] << " vs " << numeric_b_grad[i] << "\n";
            passed = false;
        }
    }

    std::cout << (passed ? "Gradient validation PASSED\n" : "Gradient validation FAILED\n");
    return passed;
}

int main() {
    std::cout << "CPU capabilities:\n";
    std::cout << "Threads: " << CPUFeatures::get().get_thread_count() << "\n";
    std::cout << "AVX: " << (CPUFeatures::get().has_avx() ? "yes" : "no") << "\n";
    std::cout << "AVX2: " << (CPUFeatures::get().has_avx2() ? "yes" : "no") << "\n";

    if(!validate_gradients()) {
        std::cerr << "Gradient validation failed. Exiting.\n";
        return 1;
    }

    const size_t batch_size = std::min(DEFAULT_BATCH_SIZE, EngineConfig::MAX_BATCH_SIZE);
    
    StaticGraph graph(LEARNING_RATE);
    
    graph.add_op(std::make_unique<LinearOp>("linear1", 784, 256));
    graph.add_op(std::make_unique<ReLUOp>("relu1"));
    graph.add_op(std::make_unique<LinearOp>("linear2", 256, 128));
    graph.add_op(std::make_unique<ReLUOp>("relu2"));
    graph.add_op(std::make_unique<LinearOp>("linear3", 128, 64));
    graph.add_op(std::make_unique<ReLUOp>("relu3"));
    graph.add_op(std::make_unique<LinearOp>("linear4", 64, 10));
    graph.add_op(std::make_unique<SoftmaxCrossEntropyOp>("loss"));
    
    graph.connect("linear1", "relu1");
    graph.connect("relu1", "linear2");
    graph.connect("linear2", "relu2");
    graph.connect("relu2", "linear3");
    graph.connect("linear3", "relu3");
    graph.connect("relu3", "linear4");
    graph.connect("linear4", "loss");
    
    Tensor batch_data({batch_size, 784}, false);
    Tensor batch_labels({batch_size, 10}, false);
    Tensor loss_tensor({1}, false);
    Tensor intermediate1({batch_size, 256}, false);
    Tensor intermediate2({batch_size, 128}, false);
    Tensor intermediate3({batch_size, 64}, false);
    Tensor intermediate4({batch_size, 10}, false);
    Tensor grad1({batch_size, 256}, false);
    Tensor grad2({batch_size, 128}, false);
    Tensor grad3({batch_size, 64}, false);
    Tensor grad4({batch_size, 10}, false);
    
    auto linear1 = dynamic_cast<LinearOp*>(graph.get_op("linear1"));
    auto relu1 = dynamic_cast<ReLUOp*>(graph.get_op("relu1"));
    auto linear2 = dynamic_cast<LinearOp*>(graph.get_op("linear2"));
    auto relu2 = dynamic_cast<ReLUOp*>(graph.get_op("relu2"));
    auto linear3 = dynamic_cast<LinearOp*>(graph.get_op("linear3"));
    auto relu3 = dynamic_cast<ReLUOp*>(graph.get_op("relu3"));
    auto linear4 = dynamic_cast<LinearOp*>(graph.get_op("linear4"));
    auto loss_op = dynamic_cast<SoftmaxCrossEntropyOp*>(graph.get_op("loss"));
    
    if(!linear1 || !linear2 || !linear3 || !linear4 || !loss_op) return 1;
    
    linear1->input_tensors.push_back(&batch_data);
    linear1->output_tensors.push_back(&intermediate1);
    linear1->input_tensors.push_back(&grad1);
    
    relu1->input_tensors.push_back(&intermediate1);
    relu1->output_tensors.push_back(&intermediate1);
    relu1->input_tensors.push_back(&grad1);
    relu1->output_tensors.push_back(&grad1);
    
    linear2->input_tensors.push_back(&intermediate1);
    linear2->output_tensors.push_back(&intermediate2);
    linear2->input_tensors.push_back(&grad2);
    
    relu2->input_tensors.push_back(&intermediate2);
    relu2->output_tensors.push_back(&intermediate2);
    relu2->input_tensors.push_back(&grad2);
    relu2->output_tensors.push_back(&grad2);
    
    linear3->input_tensors.push_back(&intermediate2);
    linear3->output_tensors.push_back(&intermediate3);
    linear3->input_tensors.push_back(&grad3);
    
    relu3->input_tensors.push_back(&intermediate3);
    relu3->output_tensors.push_back(&intermediate3);
    relu3->input_tensors.push_back(&grad3);
    relu3->output_tensors.push_back(&grad3);
    
    linear4->input_tensors.push_back(&intermediate3);
    linear4->output_tensors.push_back(&intermediate4);
    linear4->input_tensors.push_back(&grad4);
    
    loss_op->input_tensors.push_back(&intermediate4);
    loss_op->input_tensors.push_back(&batch_labels);
    loss_op->output_tensors.push_back(&loss_tensor);
    loss_op->input_tensors.push_back(&grad4);
    
    graph.build();
    
    Tensor x_train({60000, 784}, false);
    Tensor y_train({60000, 10}, false);
    
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for(size_t i = 0; i < x_train.size(); ++i) {
        x_train[i] = dist(rng);
    }
    
    std::uniform_int_distribution<> label_dist(0, 9);
    for(size_t i = 0; i < 60000; ++i) {
        y_train[i * 10 + label_dist(rng)] = 1.0f;
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    const size_t epochs = 100;
    
    for(size_t epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        std::vector<size_t> indices(60000);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        float total_loss = 0.0f;
        size_t num_batches = 0;
        
        for(size_t batch = 0; batch < 60000; batch += batch_size) {
            size_t end = std::min(batch + batch_size, 60000UL);
            size_t current_batch = end - batch;
            
            for(size_t i = 0; i < current_batch; ++i) {
                size_t src_idx = indices[batch + i];
                std::memcpy(batch_data.ptr() + i * 784,
                           x_train.ptr() + src_idx * 784,
                           784 * sizeof(float));
                std::memcpy(batch_labels.ptr() + i * 10,
                           y_train.ptr() + src_idx * 10,
                           10 * sizeof(float));
            }
            
            graph.zero_grad();
            graph.forward();
            graph.backward();
            graph.step();
            
            total_loss += loss_tensor[0];
            num_batches++;
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        if(epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / num_batches 
                      << ", Time: " << epoch_duration.count() << "ms\n";
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "Total: " << total_duration.count() << "ms\n";
    
    return 0;
}
