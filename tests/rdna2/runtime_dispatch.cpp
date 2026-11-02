#include <dlssnr/native/NativeRuntimeDispatch.h>
#include <cstdio>
#include <stdexcept>

namespace {
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
__declspec(noinline) int First(){return 1;}
__declspec(noinline) int Second(){return 2;}
struct Header {void** table;};
std::array<void*,86> baseline,changed;
}
int main()try{
    using DlssNr::Native::RuntimeDispatch;
    baseline.fill(reinterpret_cast<void*>(First));changed=baseline;
    Header object{baseline.data()},alias{changed.data()};
    auto* list=reinterpret_cast<ID3D12GraphicsCommandList*>(&object);
    auto* other=reinterpret_cast<ID3D12GraphicsCommandList*>(&alias);
    RuntimeDispatch guard;guard.owner=list;guard.table=baseline.data();guard.slots=86;guard.prefix=baseline;
    bool restored=false;unsigned rejected=~0u;
    Require(guard.Restore(list,restored,rejected)&&!restored&&guard.Unchanged(list),"unchanged owner");
    Require(guard.Restore(other,restored,rejected)&&!restored&&alias.table==changed.data(),"never modify another self");
    Require(!RuntimeDispatch::Capture(list).table,"never adopt an application-owned table");
    object.table=changed.data();changed[26]=reinterpret_cast<void*>(Second);
    Require(!guard.Restore(list,restored,rejected)&&!restored&&rejected==26&&object.table==changed.data(),"reject changed barriers");
    changed=baseline;changed[12]=reinterpret_cast<void*>(Second);
    Require(!guard.Restore(list,restored,rejected)&&!restored&&rejected==12,"reject a foreign Draw implementation");
    changed=baseline;changed[85]=reinterpret_cast<void*>(Second);
    Require(!guard.Restore(list,restored,rejected)&&!restored&&rejected==85,"reject unknown Work Graphs capability");
    guard.workGraphsUnsupported=true;
    Require(!guard.Restore(list,restored,rejected)&&!restored&&rejected==~0u&&object.table==changed.data(),"Core-only graph change is not AMD bypass");
    object.table=baseline.data();baseline[12]=reinterpret_cast<void*>(Second);
    Require(!guard.Unchanged(list)&&!guard.Restore(list,restored,rejected)&&!restored,"detect mutation at the same table address");
    baseline=guard.prefix;object.table=reinterpret_cast<void**>(1);
    Require(!guard.Restore(list,restored,rejected)&&!restored&&object.table==reinterpret_cast<void**>(1),"unreadable replacement stays untouched");
    std::puts("PASS runtime_dispatch guards=9 owner=1 foreign_table=1 changed_slot=1 work_graphs=1 immutable_prefix=1 unreadable=1");
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL runtime_dispatch %s\n",error.what());return 1;}
