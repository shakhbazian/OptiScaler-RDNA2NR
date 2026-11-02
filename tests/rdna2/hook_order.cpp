#include <dlssnr/native/NativeHookHub.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
using namespace DlssNr::NativeHooks;
using Fn=void(STDMETHODCALLTYPE*)(void*,unsigned);
using Draw=Method<9101,Fn>;
using Copy=Method<9102,Fn>;
using Wrapped=Method<9103,Fn>;
using Late=Method<9104,Fn>;
using SameEndpoint=Method<9105,Fn>;
using ResultFn=HRESULT(STDMETHODCALLTYPE*)(void*,unsigned);
using Result=Method<9106,ResultFn>;
std::vector<unsigned> actual,observed;
unsigned behavior=0,legacyCalls=0;
int wrapperSelf=0,nativeSelf=0;
__declspec(noinline) void STDMETHODCALLTYPE DrawTarget(void*,unsigned value){actual.push_back(1000+value);}
__declspec(noinline) void STDMETHODCALLTYPE CopyTarget(void*,unsigned value){actual.push_back(2000+value);}
__declspec(noinline) void STDMETHODCALLTYPE NativeTarget(void*,unsigned value){actual.push_back(3000+value);}
__declspec(noinline) void STDMETHODCALLTYPE LateTarget(void*,unsigned value){actual.push_back(4000+value);}
__declspec(noinline) void STDMETHODCALLTYPE SameEndpointTarget(void* self,unsigned value){
    if(self==&wrapperSelf){Fn volatile call=SameEndpointTarget;call(&nativeSelf,value+4);}
    else actual.push_back(5000+value);
}
__declspec(noinline) HRESULT STDMETHODCALLTYPE ResultTarget(void*,unsigned value){actual.push_back(6000+value);return E_FAIL;}
Fn originalDraw=DrawTarget,originalLate=LateTarget,originalNative=NativeTarget;
ResultFn originalResult=ResultTarget;
HRESULT observedResult=S_OK;
void ObserveDraw(void*,unsigned value){observed.push_back(1000+value);}
void ObserveCopy(void*,unsigned value){observed.push_back(2000+value);}
void ObserveWrapped(void*,unsigned value){observed.push_back(3000+value);}
void ObserveLate(void*,unsigned value){observed.push_back(4000+value);}
void ObserveSame(void*,unsigned value){observed.push_back(5000+value);}
void ObserveResult(void*,unsigned value,HRESULT result){observed.push_back(6000+value);observedResult=result;}
HRESULT STDMETHODCALLTYPE ResultListener(void* self,unsigned value){
    if(behavior==2)return S_OK;
    return originalResult(self,value+1);
}
void STDMETHODCALLTYPE Listener(void* self,unsigned value){
    if(behavior==2)return;
    originalDraw(self,value+10);
    if(behavior==0){Fn volatile call=CopyTarget;call(self,value+20);}
    if(behavior==3)originalDraw(self,value+11);
}
void STDMETHODCALLTYPE LateListener(void* self,unsigned value){++legacyCalls;originalLate(self,value);}
void STDMETHODCALLTYPE InnerListener(void* self,unsigned value){
    if(behavior==4)return;
    if(behavior==5){Fn volatile copy=CopyTarget;copy(self,value+90);return;}
    originalNative(self,value);
}
__declspec(noinline) void STDMETHODCALLTYPE WrapperTarget(void*,unsigned value){
    Fn volatile call=NativeTarget;call(&nativeSelf,value+7);
    if(behavior==3)call(&nativeSelf,value+8);
}
bool SameAlias(const void* a,const void* b)noexcept{
    return (a==&wrapperSelf||a==&nativeSelf)&&(b==&wrapperSelf||b==&nativeSelf);
}
void Require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
bool Check(const char* name,bool ok){
    std::printf("%s %s actual=",ok?"PASS":"FAIL",name);
    for(auto x:actual)std::printf("%u,",x);
    std::printf(" observed=");for(auto x:observed)std::printf("%u,",x);
    std::printf("\n");actual.clear();observed.clear();return ok;
}
}
int main(int argc,char**)try{
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    const bool raw=argc>1;
    Require(Begin()==NO_ERROR,"begin");
    Require(Draw::Attach(&originalDraw,Listener)==NO_ERROR,"draw listener");
    Require(Copy::Ensure(reinterpret_cast<void*>(CopyTarget))==NO_ERROR,"copy observer");
    Require(Wrapped::Ensure(reinterpret_cast<void*>(WrapperTarget))==NO_ERROR,"wrapper observer");
    Require(Wrapped::Ensure(reinterpret_cast<void*>(NativeTarget))==NO_ERROR,"native observer");
    Require(Late::Ensure(reinterpret_cast<void*>(LateTarget))==NO_ERROR,"late observer");
    Require(SameEndpoint::Ensure(reinterpret_cast<void*>(SameEndpointTarget))==NO_ERROR,"same endpoint observer");
    Require(Result::Attach(&originalResult,ResultListener)==NO_ERROR,"HRESULT listener");
    Require(Commit()==NO_ERROR,"commit");
    Draw::observer=ObserveDraw;Copy::observer=ObserveCopy;Wrapped::observer=ObserveWrapped;Late::observer=ObserveLate;
    SameEndpoint::observer=ObserveSame;Result::observer=ObserveResult;
    bool good=true;
    Fn volatile draw=DrawTarget;
    behavior=0;draw(nullptr,1);good=Check("post_listener_order",actual==observed)&&good;
    behavior=1;draw(nullptr,1);good=Check("rewritten_arguments",actual==observed)&&good;
    behavior=2;draw(nullptr,1);good=Check("suppressed_call",actual==observed)&&good;
    behavior=3;draw(nullptr,1);good=Check("two_continuations",actual==observed)&&good;
    originalDraw(nullptr,9);good=Check("direct_restore",actual==observed&&actual.size()==1)&&good;
    sameObject=SameAlias;Fn volatile wrapped=WrapperTarget;
    behavior=0;wrapped(&wrapperSelf,1);good=Check("wrapper_arguments",actual==observed&&actual.size()==1)&&good;
    behavior=3;wrapped(&wrapperSelf,1);good=Check("wrapper_two_calls",actual==observed&&actual.size()==2)&&good;
    Require(Begin()==NO_ERROR,"inner listener begin");
    Require(Wrapped::Attach(&originalNative,InnerListener)==NO_ERROR,"inner listener attach");
    Require(Commit()==NO_ERROR,"inner listener commit");
    behavior=4;wrapped(&wrapperSelf,1);good=Check("wrapped_suppression",actual==observed&&actual.empty())&&good;
    behavior=5;wrapped(&wrapperSelf,1);good=Check("wrapped_translation",actual==observed&&actual.size()==1)&&good;
    Fn volatile same=SameEndpointTarget;
    same(&wrapperSelf,1);good=Check("same_endpoint_alias",actual==observed&&actual.size()==1)&&good;
    ResultFn volatile result=ResultTarget;
    behavior=1;const auto hr=result(nullptr,1);
    good=Check("HRESULT_forwarded",actual==observed&&actual.size()==1&&hr==E_FAIL&&observedResult==E_FAIL)&&good;
    behavior=2;const auto suppressed=result(nullptr,1);
    good=Check("HRESULT_suppressed",actual==observed&&actual.empty()&&suppressed==S_OK)&&good;
    if(raw){
        Require(DetourTransactionBegin()==NO_ERROR,"raw begin");DetourUpdateThread(GetCurrentThread());
        Require(DetourAttach(reinterpret_cast<PVOID*>(&originalLate),LateListener)==NO_ERROR,"raw attach");
        Require(DetourTransactionCommit()==NO_ERROR,"raw commit");
    }else{
        Require(Begin()==NO_ERROR,"typed begin");
        Require(Late::Attach(&originalLate,LateListener)==NO_ERROR,"typed late attach");
        Require(Commit()==NO_ERROR,"typed commit");
    }
    Fn volatile late=LateTarget;
    const auto oldLegacy=legacyCalls;
    {OriginScope scope(Origin::Private);late(nullptr,1);}
    good=Check("private_replay_bypasses_legacy",legacyCalls==oldLegacy&&observed.empty()&&actual.size()==1)&&good;
    late(nullptr,2);good=Check("game_observed_once",actual==observed&&actual.size()==1)&&good;
    return good?0:1;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL hook_order: %s\n",e.what());return 2;}
