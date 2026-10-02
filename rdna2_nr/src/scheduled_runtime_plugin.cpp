#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "../include/nr_scheduled_runtime_contract.h"
#include "../include/nr_submission_runtime_contract.h"
#include "../include/nr_runtime_v3_dx12.h"
#include "../include/nr_runtime_v3_validation.h"
#include "../include/nr_pinned_runtime_contract.h"

namespace {
using namespace NrV3;
using Runtime=ScheduledRuntimeDx12;
Runtime* live=nullptr;HMODULE selfModule=nullptr;bool poisoned=false;
Runtime* As(void* p){return static_cast<Runtime*>(p);}
Status Track(Status s){if(s==Status::Quarantined&&!poisoned){poisoned=true;HMODULE pin=nullptr;
    static_cast<void>(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|
        GET_MODULE_HANDLE_EX_FLAG_PIN,reinterpret_cast<LPCWSTR>(&Track),&pin));}return s;}
Status __cdecl Create(ID3D12Device* d,ID3D12CommandQueue* q,void** out){
    if(!out)return Status::InvalidArgument;*out=nullptr;if(poisoned)return Status::Quarantined;
    if(live)return Status::Busy;HMODULE self=nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,reinterpret_cast<LPCWSTR>(&Create),&self))return Status::Failed;
    try{live=new Runtime(d,q);selfModule=self;*out=live;return Status::Ok;}
    catch(...){FreeLibrary(self);return Status::Unsupported;}
}
Status __cdecl Configure(void* p,const Config* data,std::uint32_t bytes,const void* pkg,std::uint64_t size){
    if(!p)return Status::InvalidArgument;Config value{};const auto s=Decode(data,bytes,value);
    return Track(s==Status::Ok?As(p)->Configure(value,pkg,size):s);
}
Status Prepare(RecordResult* out,std::uint32_t bytes){if(!out||bytes<sizeof(*out))return Status::InvalidSize;
    *out={};out->prefix={sizeof(*out),NrV3::Version};return Status::Ok;}
Status __cdecl RecordInput(void* p,const Input* data,std::uint32_t bytes,RecordResult* out,std::uint32_t outBytes){
    auto s=Prepare(out,outBytes);if(s!=Status::Ok)return s;if(!p)return out->status=Status::InvalidArgument;
    Input value{};s=Decode(data,bytes,value);return out->status=Track(s==Status::Ok?As(p)->RecordInput(value,*out):s);
}
Status TokenOf(const TokenRequest* data,std::uint32_t bytes,Token& token){TokenRequest value{};
    auto s=Decode(data,bytes,value);if(s==Status::Ok)s=Validate(value);if(s==Status::Ok)token=value.token;return s;}
Status __cdecl InputSubmitted(void* p,const TokenRequest* data,std::uint32_t bytes){if(!p)return Status::InvalidArgument;
    Token t{};const auto s=TokenOf(data,bytes,t);return Track(s==Status::Ok?As(p)->NotifyInputSubmitted(t):s);}
Status __cdecl Poll(void* p,Snapshot* out,std::uint32_t bytes){if(!out||bytes<sizeof(*out))return Status::InvalidSize;
    *out={};out->prefix={sizeof(*out),NrV3::Version};return p?Track(As(p)->Poll(*out)):Status::InvalidArgument;}
Status __cdecl Enqueue(void* p,const TokenRequest* data,std::uint32_t bytes){if(!p)return Status::InvalidArgument;
    Token t{};const auto s=TokenOf(data,bytes,t);return Track(s==Status::Ok?As(p)->Enqueue(t):s);}
Status __cdecl RecordOutput(void* p,const Output* data,std::uint32_t bytes,RecordResult* out,std::uint32_t outBytes){
    auto s=Prepare(out,outBytes);if(s!=Status::Ok)return s;if(!p)return out->status=Status::InvalidArgument;
    Output value{};s=Decode(data,bytes,value);return out->status=Track(s==Status::Ok?As(p)->RecordOutput(value,*out):s);}
Status __cdecl RecordOutputDeferred(void* p,const Output* data,std::uint32_t bytes,RecordResult* out,std::uint32_t outBytes){
    auto s=Prepare(out,outBytes);if(s!=Status::Ok)return s;if(!p)return out->status=Status::InvalidArgument;
    Output value{};s=Decode(data,bytes,value);return out->status=Track(s==Status::Ok?As(p)->RecordOutputDeferred(value,*out):s);}
Status __cdecl ArmOutput(void* p,const TokenRequest* data,std::uint32_t bytes){if(!p)return Status::InvalidArgument;
    Token t{};const auto s=TokenOf(data,bytes,t);return Track(s==Status::Ok?As(p)->ArmOutput(t):s);}
Status __cdecl OutputSubmitted(void* p,const TokenRequest* data,std::uint32_t bytes){if(!p)return Status::InvalidArgument;
    Token t{};const auto s=TokenOf(data,bytes,t);return Track(s==Status::Ok?As(p)->NotifyOutputSubmitted(t):s);}
Status __cdecl Drop(void* p,const TokenRequest* data,std::uint32_t bytes){if(!p)return Status::InvalidArgument;
    Token t{};const auto s=TokenOf(data,bytes,t);return Track(s==Status::Ok?As(p)->Drop(t):s);}
Status __cdecl Ack(void* p,const TokenRequest* data,std::uint32_t bytes){if(!p)return Status::InvalidArgument;
    Token t{};const auto s=TokenOf(data,bytes,t);return Track(s==Status::Ok?As(p)->AcknowledgeTerminal(t):s);}
Status __cdecl Drain(void* p){return p?Track(As(p)->BeginDrain()):Status::InvalidArgument;}
Status __cdecl Shutdown(void* p){return p?Track(As(p)->Shutdown()):Status::InvalidArgument;}
Status __cdecl Quarantine(void* p){return p?Track(As(p)->Quarantine()):Status::InvalidArgument;}
Status __cdecl Destroy(void** p){if(!p||!*p)return Status::InvalidArgument;auto* runtime=As(*p);
    const auto s=Track(runtime->Destroy());if(s==Status::Ok){delete runtime;*p=nullptr;live=nullptr;
        auto module=selfModule;selfModule=nullptr;if(module)FreeLibrary(module);}return s;}
} // namespace

extern "C" __declspec(dllexport) NrV3::Status __cdecl
DlssNrHipBackendGetScheduledApiV1(std::uint32_t profileId,std::uint32_t requested,
    NrScheduledApi::Api* table,std::uint32_t bytes){
    if(profileId!=NrExecution::CompiledProfileId||requested!=NrScheduledApi::Version)return NrV3::Status::Unsupported;
    if(!table||bytes<sizeof(*table))return NrV3::Status::InvalidSize;
    *table={{sizeof(*table),NrScheduledApi::Version},Create,Configure,RecordInput,InputSubmitted,
        Poll,Enqueue,RecordOutput,OutputSubmitted,Drop,Ack,Drain,Shutdown,Quarantine,Destroy};
    return NrV3::Status::Ok;
}

extern "C" __declspec(dllexport) NrV3::Status __cdecl
DlssNrHipBackendGetScheduledApiV2(std::uint32_t profileId,std::uint32_t requested,
    NrSubmissionApi::Api* table,std::uint32_t bytes){
    if(profileId!=NrExecution::CompiledProfileId||requested!=NrSubmissionApi::Version)return NrV3::Status::Unsupported;
    if(!table||bytes<sizeof(*table))return NrV3::Status::InvalidSize;
    NrScheduledApi::Api base{};
    const auto s=DlssNrHipBackendGetScheduledApiV1(profileId,NrScheduledApi::Version,&base,sizeof(base));
    if(s!=NrV3::Status::Ok)return s;
    *table={};static_cast<NrScheduledApi::Api&>(*table)=base;
    table->prefix={sizeof(*table),NrSubmissionApi::Version};
    table->recordOutputDeferred=RecordOutputDeferred;table->armOutput=ArmOutput;
    return NrV3::Status::Ok;
}
