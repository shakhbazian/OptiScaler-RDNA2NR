#pragma once

// Calibration only. These synchronous downloads must never be used for timing.
// Keep channel histograms instead of archiving multi-gigabyte activation dumps.
struct ModelBObserver {
    static constexpr unsigned Bins=321;
    struct Entry {
        size_t columns=0, samples=0;
        std::vector<unsigned> histogram;
        std::vector<float> maximum;
    };
    std::map<std::pair<unsigned,std::string>,Entry> entries;
    static float HalfMagnitude(std::uint16_t bits){
        const unsigned exponent=(bits>>10)&31, fraction=bits&1023;
        if(exponent==31)throw std::runtime_error("nonfinite calibration activation");
        return exponent?std::ldexp(float(1024+fraction),int(exponent)-25):std::ldexp(float(fraction),-24);
    }
    static void Observe(void* context,std::uint32_t block,const char* role,
                        const void* values,size_t rows,size_t columns,hipStream_t stream){
        auto& self=*static_cast<ModelBObserver*>(context);
        auto& entry=self.entries[{block,role}];
        if(!entry.columns){entry.columns=columns;entry.histogram.resize(columns*Bins);entry.maximum.resize(columns);}
        if(entry.columns!=columns)throw std::runtime_error("calibration channel count changed");
        std::vector<std::uint16_t> host(rows*columns);
        auto error=hipMemcpyAsync(host.data(),values,host.size()*2,hipMemcpyDeviceToHost,stream);
        if(error!=hipSuccess||hipStreamSynchronize(stream)!=hipSuccess)throw std::runtime_error("calibration download failed");
        const size_t samples=std::min(rows,size_t(2048));
        for(size_t r=0;r<samples;++r){
            const size_t row=r*rows/samples;
            for(size_t c=0;c<columns;++c){
                const float v=HalfMagnitude(host[row*columns+c]);
                entry.maximum[c]=std::max(entry.maximum[c],v);
                const unsigned bin=v>0?unsigned(std::clamp(int(std::ceil((std::log2(v)+24)*8)),0,320)):0;
                ++entry.histogram[c*Bins+bin];
            }
        }
        entry.samples+=samples;
    }
    void Save(const std::string& path){
        std::ofstream out(path);out<<"block,role,channel,samples,q999,q9999,maximum\n";
        for(const auto& item:entries){const auto& e=item.second;
            for(size_t c=0;c<e.columns;++c){
                auto q=[&](double fraction){size_t count=0;const size_t target=size_t(std::ceil(e.samples*fraction));
                    for(unsigned bin=0;bin<Bins;++bin){count+=e.histogram[c*Bins+bin];if(count>=target)return std::exp2(float(bin)/8-24);}
                    return e.maximum[c];};
                out<<item.first.first<<','<<item.first.second<<','<<c<<','<<e.samples<<','<<q(.999)<<','<<q(.9999)<<','<<e.maximum[c]<<'\n';
            }
        }
        if(!out)throw std::runtime_error("calibration write failed");
    }
};
