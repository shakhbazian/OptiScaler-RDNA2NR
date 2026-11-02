// Diagnostic client of the production neural kernels; no D3D/host scheduling.
#include <hip/hip_runtime.h>
#include "../../rdna2_nr/include/nr_spatial_graph.h"
#include "../../rdna2_nr/include/nr_gpu_model.h"
#include "../../rdna2_nr/include/nr_extent.h"
#include "../../rdna2_nr/include/nr_model_spec.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "model_b_observer.h"

extern "C" hipError_t launch_v2_prepare_features(
    const void*, std::size_t, const void*, std::size_t, const void*, std::size_t,
    const float*, void*, unsigned, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, bool, std::uint32_t, bool, unsigned, unsigned,
    unsigned, unsigned, float, float, float, float, float, float, float, float,
    float, hipStream_t);
extern "C" hipError_t launch_v2_compose(
    const void*, std::size_t, const void*, std::size_t, const void*, const void*,
    const void*, void*, std::size_t, float*, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, bool, bool, float, hipStream_t);
extern "C" hipError_t launch_v2_validate_neural(
    const void*, const void*, const float*, unsigned, unsigned, std::uint32_t*, hipStream_t);

static void Check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
static void Require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T> static std::vector<T> Read(std::ifstream& f,size_t n){
    std::vector<T> v(n);Require(bool(f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T))),"truncated corpus");return v;
}
struct Device {
    void* p=nullptr;size_t bytes;
    explicit Device(size_t n):bytes(n){Check(hipMalloc(&p,n));}
    ~Device(){if(p)static_cast<void>(hipFree(p));}
    Device(const Device&)=delete;
    void Put(const void* data){Check(hipMemcpy(p,data,bytes,hipMemcpyHostToDevice));}
    void Save(const std::string& path)const{
        std::vector<unsigned char> data(bytes);Check(hipMemcpy(data.data(),p,bytes,hipMemcpyDeviceToHost));
        // Image-quality archives can deduplicate identical historical tensors
        // with NTFS hard links. Unlink this pathname before writing so rerunning
        // one report gives it a private file instead of mutating every link.
        static_cast<void>(std::remove(path.c_str()));
        std::ofstream out(path,std::ios::binary);Require(bool(out.write(reinterpret_cast<const char*>(data.data()),bytes)),"write output");
    }
};
struct Stream {
    hipStream_t s=nullptr;Stream(){Check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));}
    ~Stream(){if(s){static_cast<void>(hipStreamSynchronize(s));static_cast<void>(hipStreamDestroy(s));}}
};
struct Events {
    std::array<hipEvent_t,76> e{};
    Events(){for(auto& v:e)Check(hipEventCreate(&v));}
    ~Events(){for(auto v:e)if(v)static_cast<void>(hipEventDestroy(v));}
    float Ms(size_t a,size_t b){float ms=0;Check(hipEventElapsedTime(&ms,e[a],e[b]));return ms;}
};
// Research-only replay at fixed shape and stable device addresses. Controls,
// input preparation, composition and history swaps stay outside the capture.
// This deliberately does not implement production graph cache/lifecycle logic.
struct CapturedGraph {
    hipGraph_t graph=nullptr;hipGraphExec_t executable=nullptr;
    ~CapturedGraph(){if(executable)static_cast<void>(hipGraphExecDestroy(executable));if(graph)static_cast<void>(hipGraphDestroy(graph));}
};
struct StageEvents {
    struct Range { const char* name; size_t begin,end,depth; };
    std::vector<hipEvent_t> events;
    std::vector<Range> ranges;
    std::vector<size_t> stack;
    std::string filter;
    size_t used=0;
    explicit StageEvents(bool enabled,const std::string& selected):events(enabled?2048:0),filter(selected){
        for(auto& event:events)Check(hipEventCreate(&event));
    }
    ~StageEvents(){for(auto event:events)if(event)static_cast<void>(hipEventDestroy(event));}
    static void Mark(void* context,const char* name,bool begin,hipStream_t stream){
        auto& self=*static_cast<StageEvents*>(context);
        if(!self.filter.empty()&&self.filter!=name)return;
        Require(self.used<self.events.size(),"stage event capacity");
        const size_t event=self.used++;
        Check(hipEventRecord(self.events[event],stream));
        if(begin){self.ranges.push_back({name,event,0,self.stack.size()});self.stack.push_back(self.ranges.size()-1);}
        else{
            Require(!self.stack.empty(),"stage end without begin");
            auto& range=self.ranges[self.stack.back()];
            Require(!std::strcmp(range.name,name),"stage nesting mismatch");
            range.end=event;self.stack.pop_back();
        }
    }
    void Save(std::ofstream& out,unsigned frame){
        Require(stack.empty(),"unfinished stage range");
        for(const auto& range:ranges){float ms=0;Check(hipEventElapsedTime(&ms,events[range.begin],events[range.end]));
            out<<frame<<','<<range.name<<','<<range.depth<<','<<ms<<'\n';}
        out.flush();ranges.clear();used=0;
    }
};
int main(int argc,char** argv){try{
    Require(argc>=6&&argc<=10,"usage: model_ab package package_sha source_sha corpus output_directory [--first-frame-only] [--images-only] [--stage-profile] [--hip-graph] [--calibrate]");
    bool firstFrameOnly=false,stageProfile=false,hipGraph=false,imagesOnly=false,calibrate=false;
    std::string stageFilter;
    for(int i=6;i<argc;++i){
        if(!std::strcmp(argv[i],"--calibrate"))calibrate=true;
        else if(!std::strcmp(argv[i],"--first-frame-only"))firstFrameOnly=true;
        else if(!std::strcmp(argv[i],"--images-only"))imagesOnly=true;
        else if(!std::strcmp(argv[i],"--hip-graph"))hipGraph=true;
        else if(!std::strcmp(argv[i],"--stage-profile"))stageProfile=true;
        else if(!std::strncmp(argv[i],"--stage-profile=",16)){stageProfile=true;stageFilter=argv[i]+16;}
        else throw std::runtime_error("unknown probe argument");
    }
    int count=0,selected=-1;Check(hipGetDeviceCount(&count));
    for(int i=0;i<count;++i){hipDeviceProp_t p{};Check(hipGetDeviceProperties(&p,i));
        if(!std::strcmp(p.gcnArchName,"gfx1030")){selected=i;std::printf("DEVICE %s %s\n",p.name,p.gcnArchName);break;}}
    Require(selected>=0,"gfx1030 required");Check(hipSetDevice(selected));
    std::ifstream in(argv[4],std::ios::binary);auto magic=Read<char>(in,8);
    Require(!std::memcmp(magic.data(),"NRQUAL01",8),"corpus magic");
    auto dims=Read<unsigned>(in,5);const unsigned frames=dims[0],h=dims[1],w=dims[2],nh=dims[3],nw=dims[4];
    Require(frames>0&&frames<=128&&h>0&&w>0&&h<=nh&&w<=nw&&
        NrV2::SupportedLogicalExtent(w,h)&&NrV2::SupportedGraphExtent(nw,nh),"corpus dimensions");
    auto package=NrV2::WeightPackage::LoadFile(argv[1],NrV2::ParseSha256(argv[2]),NrV2::ParseSha256(argv[3]));
    const auto spec=NrModel::Get(package.GraphVersion());
    const auto expectedExtent=NrModel::WorkingExtent(package.GraphVersion(),w,h);
    Require(nw==expectedExtent.width&&nh==expectedExtent.height,"model working extent mismatch");
    NrV2::GpuModel model(package);Stream stream;NrV2::SpatialGraphRuntime graph(model,stream.s);
    Device color(size_t(h)*w*8),motion(size_t(h)*w*4),mask(size_t(h)*w*8);
    Device features(size_t(nh)*nw*32),head(size_t(nh)*nw*8),output(size_t(h)*w*8);
    Device historyA(size_t(h)*w*12),historyB(size_t(h)*w*12);
    float* previous=static_cast<float*>(historyA.p);float* candidate=static_cast<float*>(historyB.p);
    Check(hipMemsetAsync(features.p,0,features.bytes,stream.s));
    Check(hipMemsetAsync(previous,0,historyA.bytes,stream.s));
    graph.Prepare(features.p,nh,nw,head.p);const auto prepared=graph.Stats();Events events;
    CapturedGraph captured;size_t graphNodes=0;
    Require(!hipGraph||(!stageProfile&&!calibrate),"HIP replay cannot be combined with stage profiling");
    if(hipGraph){
        Check(hipStreamBeginCapture(stream.s,hipStreamCaptureModeThreadLocal));
        graph.Enqueue(features.p,nh,nw,head.p);
        Check(hipStreamEndCapture(stream.s,&captured.graph));
        Check(hipGraphGetNodes(captured.graph,nullptr,&graphNodes));
        Check(hipGraphInstantiate(&captured.executable,captured.graph,nullptr,nullptr,0));
        std::printf("HIP_GRAPH nodes=%zu fixed_extent=%ux%u\n",graphNodes,nw,nh);
    }
    const std::string folder=argv[5];std::ofstream timings(folder+"/timings.csv");
    ModelBObserver observer;
    StageEvents stages(stageProfile,stageFilter);std::ofstream stageTimings;
    if(stageProfile){stageTimings.open(folder+"/stages.csv");stageTimings<<"frame,stage,depth,ms\n";}
    timings<<"frame,profile,prepare_ms,graph_ms,compose_ms,stream_ms,enqueue_ms,wall_ms";
    for(unsigned i=0;i<spec.blocks;++i)timings<<",block"<<i;
    for(unsigned i=0;i<spec.blocks;++i)timings<<",isolated_host_block"<<i;
    timings<<"\n";
    for(unsigned f=0;f<frames;++f){
        auto flags=Read<unsigned>(in,4); // temporal, noise, mask, per-block profile
        auto p=Read<float>(in,8); // style,tone,structure,skin,auto,intensity,jitterX,jitterY
        Require(flags[0]<=1&&flags[2]<=1&&flags[3]<=3&&(!flags[0]||f>0),"corpus flags");
        Require(!hipGraph||flags[3]==0,"HIP replay cannot be combined with block profiling");
        for(float v:p)Require(std::isfinite(v),"nonfinite parameter");
        auto c=Read<std::uint16_t>(in,size_t(h)*w*4),m=Read<std::uint16_t>(in,size_t(h)*w*2),k=Read<std::uint16_t>(in,size_t(h)*w*4);
        color.Put(c.data());motion.Put(m.data());mask.Put(k.data());
        const auto start=std::chrono::steady_clock::now();
        Check(hipEventRecord(events.e[0],stream.s));
        if(flags[3]>=2)Check(hipEventSynchronize(events.e[0]));
        const float local=!flags[2]&&(p[3]>=0||p[4]>=0)?1.f:p[2];
        const float skin=p[3]>=0?p[3]:(p[4]>=0?p[2]:-1.f);
        const float automatic=p[4]>=0?p[4]:(p[3]>=0?p[2]:-1.f);
        Check(launch_v2_prepare_features(color.p,w*8,motion.p,w*4,flags[2]?mask.p:nullptr,w*8,
            previous,features.p,h,w,nh,nw,w,h,w,h,flags[2]!=0,flags[1],flags[0]!=0,
            w,h,w,h,1,1,p[6],p[7],p[0],p[1],local,skin,automatic,stream.s));
        Check(hipEventRecord(events.e[1],stream.s));
        if(flags[3]>=2)Check(hipEventSynchronize(events.e[1]));
        std::array<double,71> blockWall{};
        NrV2::SpatialGraphProfile profile{};
        profile.events=events.e.data()+4;
        profile.isolateBlocks=flags[3]==2;
        profile.blockWallMs=blockWall.data();
        if(calibrate){profile.activationObserver=&ModelBObserver::Observe;profile.activationObserverContext=&observer;}
        if(stageProfile){profile.stageEvent=&StageEvents::Mark;profile.stageEventContext=&stages;}
        const bool blockProfile=flags[3]==1||flags[3]==2;
        if(!blockProfile)profile.events=nullptr;
        if(hipGraph)Check(hipGraphLaunch(captured.executable,stream.s));
        else graph.Enqueue(features.p,nh,nw,head.p,(blockProfile||stageProfile||calibrate)?&profile:nullptr);
        Check(hipEventRecord(events.e[2],stream.s));
        if(flags[3]>=2)Check(hipEventSynchronize(events.e[2]));
        Check(launch_v2_compose(color.p,w*8,flags[2]?mask.p:nullptr,w*8,head.p,features.p,
            model.TensorData(spec.post,"temporal_blend_scale",1),output.p,w*8,candidate,h,w,nw,w,h,w,h,
            flags[2]!=0,flags[0]!=0,p[5],stream.s));
        Check(hipEventRecord(events.e[3],stream.s));
        const auto submitted=std::chrono::steady_clock::now();
        Check(hipEventSynchronize(events.e[3]));graph.ValidateExperimental();const auto end=std::chrono::steady_clock::now();
        if(stageProfile)stages.Save(stageTimings,f);
        const auto stats=graph.Stats();Require(calibrate||stats.allocations==prepared.allocations,"workspace grew after preparation");
        timings<<f<<','<<flags[3]<<','<<events.Ms(0,1)<<','<<events.Ms(1,2)<<','<<events.Ms(2,3)<<','<<events.Ms(0,3)<<','
            <<std::chrono::duration<double,std::milli>(submitted-start).count()<<','<<std::chrono::duration<double,std::milli>(end-start).count();
        for(unsigned b=0;b<spec.blocks;++b)timings<<','<<(blockProfile?events.Ms(4+b,5+b):-1.f);
        for(unsigned b=0;b<spec.blocks;++b)timings<<','<<(flags[3]==2?blockWall[b]:-1.);
        timings<<'\n';timings.flush();
        const auto prefix=folder+"/"+std::to_string(f);
        if(!firstFrameOnly||f==0){
            if(!imagesOnly){features.Save(prefix+".features.f16");head.Save(prefix+".head.f16");}
            output.Save(prefix+".rgba.f16");
        }
        std::swap(previous,candidate);
        if(!imagesOnly&&(!firstFrameOnly||f==0)){if(previous==historyA.p)historyA.Save(prefix+".history.f32");else historyB.Save(prefix+".history.f32");}
        std::printf("FRAME %u/%u graph_ms=%.3f stream_ms=%.3f profile=%u\n",f+1,frames,events.Ms(1,2),events.Ms(0,3),flags[3]);
        std::fflush(stdout);
    }
    if(calibrate)observer.Save(folder+"/calibration.csv");
    char trailing;Require(!in.read(&trailing,1),"trailing corpus bytes");
    // Adversarial diagnostic: can composition conceal a nonfinite neural head?
    // This is outside the valid corpus and never committed or used as history.
    Check(hipMemsetAsync(head.p,0x7e,head.bytes,stream.s)); // 0x7e7e is FP16 NaN.
    Check(launch_v2_compose(color.p,w*8,nullptr,0,head.p,features.p,
        model.TensorData(spec.post,"temporal_blend_scale",1),output.p,w*8,candidate,h,w,nw,w,h,w,h,
        false,false,1.f,stream.s));
    Check(hipStreamSynchronize(stream.s));
    std::vector<std::uint16_t> adversarial(size_t(h)*w*4);
    Check(hipMemcpy(adversarial.data(),output.p,output.bytes,hipMemcpyDeviceToHost));
    size_t nonfinite=0;for(auto v:adversarial)if((v&0x7c00u)==0x7c00u)++nonfinite;
    Device status(sizeof(std::uint32_t));std::uint32_t observed=0;
    Check(hipMemsetAsync(status.p,0,status.bytes,stream.s));
    Check(launch_v2_validate_neural(features.p,head.p,candidate,nh*nw,h*w,
        static_cast<std::uint32_t*>(status.p),stream.s));
    Check(hipStreamSynchronize(stream.s));
    Check(hipMemcpy(&observed,status.p,status.bytes,hipMemcpyDeviceToHost));
    Require(observed==2,"head NaN was not caught by the strengthened gate");
    Check(hipMemsetAsync(head.p,0,head.bytes,stream.s));
    const std::uint16_t halfInf=0x7c00;
    Check(hipMemcpy(features.p,&halfInf,sizeof(halfInf),hipMemcpyHostToDevice));
    Check(hipMemsetAsync(status.p,0,status.bytes,stream.s));
    Check(launch_v2_validate_neural(features.p,head.p,candidate,nh*nw,h*w,
        static_cast<std::uint32_t*>(status.p),stream.s));
    Check(hipStreamSynchronize(stream.s));
    std::uint32_t featureBits=0;Check(hipMemcpy(&featureBits,status.p,status.bytes,hipMemcpyDeviceToHost));
    Require(featureBits==4,"feature infinity was not caught");
    Check(hipMemsetAsync(features.p,0,features.bytes,stream.s));
    const float historyInf=INFINITY;
    Check(hipMemcpy(candidate,&historyInf,sizeof(historyInf),hipMemcpyHostToDevice));
    Check(hipMemsetAsync(status.p,0,status.bytes,stream.s));
    Check(launch_v2_validate_neural(features.p,head.p,candidate,nh*nw,h*w,
        static_cast<std::uint32_t*>(status.p),stream.s));
    Check(hipStreamSynchronize(stream.s));
    std::uint32_t historyBits=0;Check(hipMemcpy(&historyBits,status.p,status.bytes,hipMemcpyDeviceToHost));
    Require(historyBits==8,"history infinity was not caught");
    std::ofstream masking(folder+"/nonfinite-masking.json");
    masking<<"{\"injected_head\":\"all FP16 NaN\",\"output_nonfinite_lanes\":"<<nonfinite
        <<",\"final_finite_gate_would_accept\":"<<(nonfinite==0?"true":"false")
        <<",\"strengthened_gate_bits\":"<<observed
        <<",\"feature_infinity_bits\":"<<featureBits<<",\"history_infinity_bits\":"<<historyBits<<"}\n";
    const auto stats=graph.Stats();std::ofstream memory(folder+"/memory.json");
    memory<<"{\"model_bytes\":"<<model.ByteLength()<<",\"workspace_reserved_bytes\":"<<stats.reservedBytes
        <<",\"workspace_peak_in_use_bytes\":"<<stats.peakInUseBytes<<",\"allocations\":"<<stats.allocations
        <<",\"int8_reserved_bytes\":"<<stats.int8ReservedBytes<<",\"int8_invocations\":"<<stats.int8Invocations
        <<",\"experimental_global_int8\":"<<(stats.experimentalGlobalInt8?"true":"false")
        <<",\"frame_buffer_bytes\":"<<color.bytes+motion.bytes+mask.bytes+features.bytes+head.bytes+output.bytes+historyA.bytes+historyB.bytes
        <<",\"model_uploads\":"<<model.UploadOperations()<<",\"frames\":"<<stats.frames
        <<",\"hip_graph_nodes\":"<<graphNodes<<",\"hip_graph_replays\":"<<(hipGraph?frames:0)<<"}\n";
    Check(hipStreamSynchronize(stream.s));std::puts("PASS quality corpus execution");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
