#include <dlssnr/native/NativeHookHub.h>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {
using Fn=void(STDMETHODCALLTYPE*)(void*,unsigned);
using Hub=DlssNr::NativeHooks::Method<9001,Fn>;
std::atomic<unsigned> originals{0},first{0},second{0},observed{0},values{0};
__declspec(noinline) void STDMETHODCALLTYPE Target(void*,unsigned value) {
    ++originals; values+=value;
}
Fn one=Target,two=Target;
void STDMETHODCALLTYPE First(void* self,unsigned value){++first;one(self,value);}
void STDMETHODCALLTYPE Second(void* self,unsigned value){++second;two(self,value);}
void Observe(void*,unsigned){++observed;}
void Require(bool value,const char* what){if(!value)throw std::runtime_error(what);}
}
int main() try {
    using namespace DlssNr::NativeHooks;
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    Require(Begin()==NO_ERROR,"begin");
    Require(Hub::Attach(&one,First)==NO_ERROR&&Hub::Attach(&two,Second)==NO_ERROR,"attach listeners");
    Require(Commit()==NO_ERROR,"commit");
    Hub::observer.store(Observe);
    Fn volatile call=Target;
    call(nullptr,7);
    Require(originals==1&&first==1&&second==1&&observed==1&&values==7,"one physical invocation");
    one(nullptr,3);
    Require(originals==2&&first==1&&second==1&&observed==2&&values==10,"direct restore observation");
    {OriginScope scope(Origin::Private);call(nullptr,2);}
    Require(originals==3&&first==1&&second==1&&observed==2,"private origin");
    Require(Begin()==NO_ERROR,"begin detach");
    Require(Hub::Detach(&two,Second)==NO_ERROR,"detach listener");
    Require(Commit()==NO_ERROR,"commit detach");
    call(nullptr,1);
    Require(originals==4&&first==2&&second==1&&observed==3,"remaining listener");
    Require(Begin()==NO_ERROR,"begin rejected registration");
    Require(Hub::Detach(&one,First)==NO_ERROR,"pending detach");
    Require(Method<9002,Fn>::Ensure(reinterpret_cast<void*>(Target))!=NO_ERROR,"signature collision");
    Require(Commit()!=NO_ERROR,"reject entire conflicting transaction");
    call(nullptr,1);Require(first==3&&originals==5&&observed==4,"rollback keeps legacy listener");
    std::thread a([&]{for(unsigned i=0;i<2000;++i)call(nullptr,1);});
    std::thread b([&]{for(unsigned i=0;i<2000;++i)call(nullptr,1);});
    a.join();b.join();
    Require(originals==4005&&first==4003&&observed==4004,"thread-local invocation isolation");
    std::puts("PASS native_hook_hub listeners=2 physical_original_once=1 restore_observed=1 private_bypass=1 detach=1 rollback=1 concurrent_calls=4000");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL native_hook_hub %s\n",e.what());return 1;}
