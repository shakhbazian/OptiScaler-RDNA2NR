#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

inline void hip_check(hipError_t e,const char* call) {
    if (e!=hipSuccess) throw std::runtime_error(std::string(call)+": "+hipGetErrorString(e));
}
#define HIP_CHECK(call) hip_check((call),#call)
inline bool cleanup_failed=false;
// The graph binds one instance-owned arena and one owning stream while it
// records work. The binding is thread-local only to keep the recovered kernel
// call graph readable; concurrent calls on one graph instance are forbidden.
inline thread_local hipStream_t active_graph_stream=nullptr;
// Standalone probes keep canaries by default.  The connected graph may disable
// their per-allocation memset/check cost after the same layouts have passed the
// guarded corpus; each Buffer remembers the policy used at construction.
inline thread_local bool active_buffer_guards=true;
struct DeviceBufferArena {
    struct Slot { void* data=nullptr; size_t bytes=0; bool used=false; };
    std::vector<Slot> slots;
    size_t allocation_count=0,reuse_count=0,reserved_bytes=0,in_use_bytes=0,peak_in_use_bytes=0;
    void* acquire(size_t bytes) {
        Slot* best=nullptr;
        for(auto&slot:slots)if(!slot.used&&slot.bytes>=bytes&&(!best||slot.bytes<best->bytes))best=&slot;
        if(!best){void*data=nullptr;HIP_CHECK(hipMalloc(&data,bytes));slots.push_back({data,bytes,true});best=&slots.back();++allocation_count;reserved_bytes+=bytes;}
        else{best->used=true;++reuse_count;}
        in_use_bytes+=best->bytes;peak_in_use_bytes=std::max(peak_in_use_bytes,in_use_bytes);
        return best->data;
    }
    void release(void*data) {
        for(auto&slot:slots)if(slot.data==data){if(!slot.used)throw std::runtime_error("double workspace release");slot.used=false;in_use_bytes-=slot.bytes;return;}
        throw std::runtime_error("foreign workspace pointer");
    }
    bool release_all() noexcept {
        for(auto&slot:slots){
            if(slot.used){cleanup_failed=true;return false;}
            if(slot.data){if(hipFree(slot.data)!=hipSuccess){cleanup_failed=true;return false;}slot.data=nullptr;}
        }
        slots.clear();reserved_bytes=0;in_use_bytes=0;return true;
    }
    ~DeviceBufferArena(){static_cast<void>(release_all());}
};
inline thread_local DeviceBufferArena* active_buffer_arena=nullptr;
inline hipStream_t current_stream(){return active_graph_stream;}
template<class T> struct Buffer {
    static constexpr size_t Guard=256/sizeof(T);
    static_assert(256%sizeof(T)==0,"guard alignment");
    T* data=nullptr; size_t count; DeviceBufferArena* arena=nullptr;bool guards=true;
    explicit Buffer(size_t n):count(n),arena(active_buffer_arena),guards(!arena||active_buffer_guards) {
        if(arena)data=static_cast<T*>(arena->acquire((n+2*Guard)*sizeof(T)));
        else HIP_CHECK(hipMalloc(&data,(n+2*Guard)*sizeof(T)));
        if(!guards)return;
        const auto firstResult=active_graph_stream?
            hipMemsetAsync(data,0xff,Guard*sizeof(T),active_graph_stream):hipMemset(data,0xff,Guard*sizeof(T));
        const auto lastResult=firstResult==hipSuccess?(active_graph_stream?
            hipMemsetAsync(data+n+Guard,0xff,Guard*sizeof(T),active_graph_stream):
            hipMemset(data+n+Guard,0xff,Guard*sizeof(T))):firstResult;
        if(lastResult!=hipSuccess){
            if(arena){try{arena->release(data);}catch(...){cleanup_failed=true;}}
            else if(hipFree(data)!=hipSuccess)cleanup_failed=true;
            data=nullptr;HIP_CHECK(lastResult);
        }
    }
    ~Buffer() { if(data){if(arena){try{arena->release(data);}catch(...){cleanup_failed=true;}}else if(hipFree(data)!=hipSuccess)cleanup_failed=true;} }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    T* ptr() { return data+Guard; }
    void upload(const std::vector<T>& v) {
        if (v.size()!=count) throw std::runtime_error("upload size mismatch");
        if(active_graph_stream){
            HIP_CHECK(hipMemcpyAsync(ptr(),v.data(),count*sizeof(T),hipMemcpyHostToDevice,
                                     active_graph_stream));
            HIP_CHECK(hipStreamSynchronize(active_graph_stream));
        }else HIP_CHECK(hipMemcpy(ptr(),v.data(),count*sizeof(T),hipMemcpyHostToDevice));
    }
    std::vector<T> read() {
        std::vector<T> v(guards?count+2*Guard:count);
        if(active_graph_stream){
            HIP_CHECK(hipMemcpyAsync(v.data(),guards?data:ptr(),v.size()*sizeof(T),hipMemcpyDeviceToHost,
                                     active_graph_stream));
            HIP_CHECK(hipStreamSynchronize(active_graph_stream));
        }else HIP_CHECK(hipMemcpy(v.data(),guards?data:ptr(),v.size()*sizeof(T),hipMemcpyDeviceToHost));
        if(!guards)return v;
        const auto* first=reinterpret_cast<const unsigned char*>(&v.front());
        const auto* last=reinterpret_cast<const unsigned char*>(v.data()+Guard+count);
        for (size_t i=0;i<Guard*sizeof(T);++i) if (first[i]!=255 || last[i]!=255)
            throw std::runtime_error("GPU buffer guard overwritten");
        return std::vector<T>(v.begin()+Guard,v.end()-Guard);
    }
};
struct Reader {
    std::ifstream file;
    explicit Reader(const char* path,std::streamoff max_bytes=32*1024*1024):file(path,std::ios::binary|std::ios::ate) {
        if (!file || file.tellg()>max_bytes) throw std::runtime_error("invalid probe fixture");
        file.seekg(0);
    }
    template<class T> std::vector<T> read(size_t n) {
        if (n>8*1024*1024 || n*sizeof(T)>32*1024*1024) throw std::runtime_error("fixture read too large");
        std::vector<T> v(n);
        if (!file.read(reinterpret_cast<char*>(v.data()),n*sizeof(T))) throw std::runtime_error("truncated fixture");
        return v;
    }
    void end() { if (file.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing fixture bytes"); }
};
inline double half_value(std::uint16_t v) {
    unsigned e=(v>>10)&31,f=v&1023;
    double x=e==31 ? std::numeric_limits<double>::quiet_NaN() :
        e ? std::ldexp(1+f/1024.0,int(e)-15) : std::ldexp(double(f),-24);
    return v&0x8000 ? -x : x;
}
inline void sync_gpu() {
    HIP_CHECK(hipGetLastError());
    if(!active_graph_stream) HIP_CHECK(hipDeviceSynchronize());
}
