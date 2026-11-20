// Revised CA LI runner: BF16, BSND query, BSND/PA_BSND key, N2=1.
// Confirmed original CANN paths retained. Compile on the user's server:
/*
g++ -O2 -std=c++17 li_aclnn_bench.cpp -o li_aclnn_bench \
  -I/home/Ascend/cann-9.2.0/include \
  -L/home/Ascend/cann-9.2.0/aarch64-linux/lib64 \
  -lopapi -lascendcl -lruntime -lpthread
*/
// Isolate one logical LI call: ./li_aclnn_bench --dir a2 --iters 1 --warmup 0
// Total tick is emitted externally by CA, never synthesized by this application.
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_lightning_indexer.h"

namespace {
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void check(int64_t code,const char* what){if(code)throw std::runtime_error(std::string(what)+" status="+std::to_string(code));}
int64_t integer(const std::string& text){size_t n=0;int64_t v=std::stoll(text,&n);require(n==text.size(),"Invalid integer: "+text);return v;}
std::string trim(const std::string& s){auto a=s.find_first_not_of(" \t\n\r");if(a==std::string::npos)return {};return s.substr(a,s.find_last_not_of(" \t\n\r")-a+1);}
// The existing generator's metadata uses flat literal keys. Parse each exact key once,
// accept all JSON whitespace, reject missing/duplicate/wrong-type consumed fields.
struct Meta {
    std::string text;
    bool has(const std::string& key)const{return text.find("\""+key+"\"")!=std::string::npos;}
    std::string token(const std::string& key)const {
        const std::string quoted="\""+key+"\"";size_t pos=text.find(quoted);
        require(pos!=std::string::npos,"Missing metadata key: "+key);
        require(text.find(quoted,pos+quoted.size())==std::string::npos,"Duplicate metadata key: "+key);
        size_t i=pos+quoted.size();while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;
        require(i<text.size()&&text[i++]==':',"Missing colon: "+key);
        while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;
        require(i<text.size(),"Empty metadata value: "+key);size_t begin=i;
        if(text[i]=='"') {++i;while(i<text.size()&&text[i]!='"'){require(text[i]!='\\',"Escaped strings not supported for consumed metadata fields");++i;}require(i<text.size(),"Unclosed string");++i;}
        else if(text[i]=='['){i=text.find(']',i);require(i!=std::string::npos,"Unclosed shape array");++i;}
        else {while(i<text.size()&&text[i]!=','&&text[i]!='}'&&!std::isspace(static_cast<unsigned char>(text[i])))++i;}
        auto result=text.substr(begin,i-begin);
        while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;
        require(i<text.size()&&(text[i]==','||text[i]=='}'),"Invalid delimiter: "+key);
        return result;
    }
    std::string string(const std::string& key)const{auto s=token(key);require(s.size()>=2&&s.front()=='"'&&s.back()=='"',"Expected string: "+key);return s.substr(1,s.size()-2);}
    int64_t number(const std::string& key)const{return integer(token(key));}
    std::vector<int64_t> shape(const std::string& key,bool optional=false)const {
        if(optional&&!has(key))return {};
        auto s=token(key);
        require(s.size()>=2&&s.front()=='['&&s.back()==']',"Expected shape array: "+key);
        std::vector<int64_t> v;std::string body=trim(s.substr(1,s.size()-2));if(body.empty())return v;
        require(body.back()!=',',"Trailing comma in shape: "+key);std::stringstream ss(body);std::string item;
        while(std::getline(ss,item,','))v.push_back(integer(trim(item)));
        return v;
    }
};
size_t count(const std::vector<int64_t>& shape) {
    if(shape.empty())return 0;
    size_t n=1;
    for(auto d:shape){require(d>0,"Storage dimensions must be positive");require(uint64_t(d)<=std::numeric_limits<size_t>::max()/n,"Shape overflow");n*=size_t(d);}return n;
}
size_t bytes(size_t n,size_t width){require(n<=std::numeric_limits<size_t>::max()/width,"Byte-size overflow");return n*width;}
std::vector<int64_t> strides(const std::vector<int64_t>& shape){std::vector<int64_t> s(shape.size(),1);for(size_t i=shape.size();i>1;--i){require(shape[i-1]<=std::numeric_limits<int64_t>::max()/s[i-1],"Stride overflow");s[i-2]=s[i-1]*shape[i-1];}return s;}
template<class T>std::vector<T> read(const std::string& path,size_t expected) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"Cannot open "+path);
    auto length=f.tellg();require(length>=0&&uint64_t(length)==bytes(expected,sizeof(T)),"File size disagrees with metadata: "+path);
    std::vector<T> v(expected);f.seekg(0);if(expected){f.read(reinterpret_cast<char*>(v.data()),bytes(expected,sizeof(T)));require(bool(f),"Short read: "+path);}return v;
}
template<class T>void write(const std::string& path,const std::vector<T>& v){std::ofstream f(path,std::ios::binary);require(bool(f),"Cannot write "+path);f.write(reinterpret_cast<const char*>(v.data()),bytes(v.size(),sizeof(T)));f.close();require(bool(f),"Write/close failed: "+path);}
uint16_t bf16(float f){uint32_t u;std::memcpy(&u,&f,4);return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);}
std::vector<uint16_t> convert(const std::vector<float>& src,const char* name,bool allow) {
    std::vector<uint16_t> out(src.size());size_t changed=0;
    for(size_t i=0;i<src.size();++i){require(std::isfinite(src[i]),std::string(name)+": nonfinite input");out[i]=bf16(src[i]);require((out[i]&0x7f80u)!=0x7f80u,std::string(name)+": BF16 overflow");uint32_t u;std::memcpy(&u,&src[i],4);changed+=(u!=(uint32_t(out[i])<<16));}
    printf("INPUT %s elements=%zu rounded_elements=%zu\n",name,src.size(),changed);
    require(allow||!changed,std::string(name)+": input is not already BF16-valued; export CPU BF16->float32 or explicitly use --allow-bf16-rounding");return out;
}
struct Resources {
    bool initialized=false,device=false;aclrtStream stream=nullptr;
    std::vector<void*> memory;std::vector<aclTensor*> tensors;
    ~Resources(){for(auto t:tensors)if(t)aclDestroyTensor(t);for(auto p:memory)if(p)aclrtFree(p);if(stream)aclrtDestroyStream(stream);if(device)aclrtResetDevice(0);if(initialized)aclFinalize();}
    void start(){check(aclInit(nullptr),"aclInit");initialized=true;check(aclrtSetDevice(0),"aclrtSetDevice");device=true;check(aclrtCreateStream(&stream),"aclrtCreateStream");}
    void* alloc(size_t n){void* p=nullptr;check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_NORMAL_ONLY),"aclrtMalloc");require(p!=nullptr,"aclrtMalloc returned null");memory.push_back(p);return p;}
    template<class T>void* upload(const std::vector<T>& v){if(v.empty())return nullptr;auto p=alloc(bytes(v.size(),sizeof(T)));check(aclrtMemcpy(p,bytes(v.size(),sizeof(T)),v.data(),bytes(v.size(),sizeof(T)),ACL_MEMCPY_HOST_TO_DEVICE),"H2D");return p;}
    aclTensor* tensor(const std::vector<int64_t>& shape,aclDataType type,void* data){if(shape.empty())return nullptr;auto st=strides(shape);auto t=aclCreateTensor(shape.data(),shape.size(),type,st.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),data);require(t!=nullptr,"aclCreateTensor returned null");tensors.push_back(t);return t;}
};
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        std::string dir=".";int iters=1,warmup=0;bool allow=false,validate_only=false;
        for(int i=1;i<argc;++i){std::string arg=argv[i];
            if(arg=="--help"){puts("Usage: li_aclnn_bench [--dir DIR] [--iters N>=1] [--warmup N>=0] [--allow-bf16-rounding] [--validate-only]");return 0;}
            if(arg=="--allow-bf16-rounding")allow=true;
            else if(arg=="--validate-only")validate_only=true;
            else if(arg=="--dir"||arg=="--iters"||arg=="--warmup"){
                require(i+1<argc,"Missing value for "+arg);std::string value=argv[++i];
                if(arg=="--dir")dir=value;
                else{int64_t n=integer(value);require(n>=0&&n<=std::numeric_limits<int>::max(),"Invalid count");if(arg=="--iters")iters=int(n);else warmup=int(n);}
            }else throw std::runtime_error("Unknown argument: "+arg);
        }
        require(iters>0&&int64_t(iters)+warmup<=std::numeric_limits<int>::max(),"iters must be positive and call count must fit int");
        uint16_t endian=1;require(*reinterpret_cast<uint8_t*>(&endian)==1,"Input format is little endian");
        std::ifstream file(dir+"/li_a2_meta.json");require(bool(file),"Missing li_a2_meta.json");std::stringstream text;text<<file.rdbuf();Meta meta{text.str()};
        require(!trim(meta.text).empty()&&trim(meta.text).front()=='{'&&trim(meta.text).back()=='}',"Expected metadata object");
        auto qs=meta.shape("query_shape"),ks=meta.shape("key_shape"),ws=meta.shape("weights_shape");
        auto aqs=meta.shape("aq_shape",true),aks=meta.shape("ak_shape",true),bts=meta.shape("bt_shape",true);
        std::string lq=meta.string("layout_query"),lk=meta.string("layout_key");int64_t K=meta.number("sparse_count"),mode=meta.number("sparse_mode");
        require(lq=="BSND"&&(lk=="PA_BSND"||lk=="BSND"),"This runner supports BSND query and PA_BSND/BSND key; TND is not implemented");
        require(qs.size()==4&&ks.size()==4&&ws.size()==3,"Invalid Q/K/W ranks");count(qs);count(ks);count(ws);
        int64_t B=qs[0],S=qs[1],H=qs[2],D=qs[3],N=ks[2];
        require(N==1&&D==128&&ks[3]==D,"Expected N2=1 and D=128");
        require(H==8||H==16||H==24||H==32||H==64,"950PR N1 must be 8/16/24/32/64");
        require(ws==std::vector<int64_t>({B,S,H}),"Weights shape mismatch");
        require(K>=1&&(K<=2048||(K<=8192&&K%1024==0)),"K outside documented original LI range");
        require(mode==0||mode==3,"Expected sparse_mode 0 or 3");
        for(const char* key:{"dtype","qk_dtype","query_dtype","key_dtype","weight_dtype","weights_dtype"})if(meta.has(key)){
            auto dt=meta.string(key);require(dt=="bf16"||dt=="bfloat16"||dt=="torch.bfloat16","This runner is BF16-only; metadata requested "+dt);}
        if(meta.has("return_values"))require(meta.token("return_values")=="false","This runner uses return_values=False");
        require(aqs.empty()||aqs==std::vector<int64_t>({B}),"aq_shape must be [B] or omitted");
        require(aks.empty()||aks==std::vector<int64_t>({B}),"ak_shape must be [B] or omitted");
        require(S<=std::numeric_limits<int32_t>::max(),"S1 exceeds int32 lengths");
        auto aq=aqs.empty()?std::vector<int32_t>(B,int32_t(S)):read<int32_t>(dir+"/li_a2_aq.i32",count(aqs));
        std::vector<int32_t> ak,bt;
        if(lk=="PA_BSND"){
            require(ks[1]%16==0&&ks[1]<=1024,"PA block_size must be a multiple of16 and <=1024");
            require(!aks.empty()&&bts.size()==2&&bts[0]==B,"PA requires ak_shape=[B] and bt_shape=[B,max_blocks]");count(bts);
            ak=read<int32_t>(dir+"/li_a2_ak.i32",count(aks));bt=read<int32_t>(dir+"/li_a2_bt.i32",count(bts));
        }else{
            require(ks[0]==B&&bts.empty(),"BSND key requires same B and no block table");require(ks[1]<=std::numeric_limits<int32_t>::max(),"S2 exceeds int32 lengths");
            ak=aks.empty()?std::vector<int32_t>(B,int32_t(ks[1])):read<int32_t>(dir+"/li_a2_ak.i32",count(aks));
        }
        for(int64_t b=0;b<B;++b){require(aq[b]>=0&&aq[b]<=S&&ak[b]>=0,"Invalid effective lengths");
            if(lk=="PA_BSND"){int64_t nb=(int64_t(ak[b])+ks[1]-1)/ks[1];require(nb<=bts[1],"Block table too short");for(int64_t j=0;j<nb;++j)require(bt[size_t(b)*bts[1]+j]>=0&&bt[size_t(b)*bts[1]+j]<ks[0],"Invalid active PA block ID");}
            else require(ak[b]<=ks[1],"Key length exceeds BSND shape");}
        auto q=convert(read<float>(dir+"/li_a2_query.f32",count(qs)),"query",allow);
        auto k=convert(read<float>(dir+"/li_a2_key.f32",count(ks)),"key",allow);
        auto w=convert(read<float>(dir+"/li_a2_weights.f32",count(ws)),"weights",allow);
        printf("CONFIG B=%lld S1=%lld H=%lld D=%lld K=%lld mode=%lld dtype=bf16 layoutQ=%s layoutK=%s\n",(long long)B,(long long)S,(long long)H,(long long)D,(long long)K,(long long)mode,lq.c_str(),lk.c_str());
        printf("EXECUTION requested_li_calls=%d warmup=%d iters=%d\n",warmup+iters,warmup,iters);
        if(warmup||iters!=1)puts("TIMING_NOTE multiple LI calls: external CA tick scope must be resolved per invocation; never divide one raw Total tick blindly");
        if(validate_only){puts("VALIDATE_INPUTS PASS (no ACL initialization, no kernel execution)");return 0;}
        // Export exact bits ACTUALLY passed to CA; CPU reference must consume these.
        write(dir+"/li_query.bf16",q);write(dir+"/li_key.bf16",k);write(dir+"/li_weights.bf16",w);
        std::vector<int64_t> os={B,S,N,K};size_t out_num=count(os);std::vector<int32_t> result(out_num,-2);
        {
            Resources r;r.start();
            auto tq=r.tensor(qs,ACL_BF16,r.upload(q)),tk=r.tensor(ks,ACL_BF16,r.upload(k)),tw=r.tensor(ws,ACL_BF16,r.upload(w));
            auto taq=aqs.empty()?nullptr:r.tensor(aqs,ACL_INT32,r.upload(aq));
            auto tak=aks.empty()?nullptr:r.tensor(aks,ACL_INT32,r.upload(ak));
            auto tbt=bts.empty()?nullptr:r.tensor(bts,ACL_INT32,r.upload(bt));
            // One H2D initialization catches unwritten output slots; it is outside the LI call.
            void* dout=r.upload(result);auto tout=r.tensor(os,ACL_INT32,dout);
            std::vector<char> lqb(lq.begin(),lq.end()),lkb(lk.begin(),lk.end());lqb.push_back(0);lkb.push_back(0);
            void* workspace=nullptr;uint64_t capacity=0;std::vector<double> times;times.reserve(iters);
            for(int call=0;call<warmup+iters;++call){bool is_warmup=call<warmup;const char* phase=is_warmup?"warmup":"measure";
                printf("PHASE begin logical_li_call=%d phase=%s\n",call,phase);
                uint64_t need=0;aclOpExecutor* executor=nullptr;auto p0=Clock::now();
                check(aclnnLightningIndexerGetWorkspaceSize(tq,tk,tw,taq,tak,tbt,lqb.data(),lkb.data(),K,mode,
                    std::numeric_limits<int64_t>::max(),std::numeric_limits<int64_t>::max(),false,tout,nullptr,&need,&executor),"GetWorkspaceSize");
                require(executor!=nullptr,"GetWorkspaceSize returned null executor");double prepare_ms=ms(p0,Clock::now());
                if(need>capacity){workspace=r.alloc(size_t(need));capacity=need;}
                // Executor belongs to this one two-phase invocation, never reused.
                auto t0=Clock::now();check(aclnnLightningIndexer(need?workspace:nullptr,need,executor,r.stream),"aclnnLightningIndexer");
                check(aclrtSynchronizeStream(r.stream),"aclrtSynchronizeStream");double wall=ms(t0,Clock::now());
                printf("PHASE end logical_li_call=%d phase=%s host_prepare_ms=%.6f host_execute_sync_ms=%.6f workspace=%llu\n",call,phase,prepare_ms,wall,(unsigned long long)need);
                if(!is_warmup)times.push_back(wall);
            }
            check(aclrtMemcpy(result.data(),bytes(out_num,4),dout,bytes(out_num,4),ACL_MEMCPY_DEVICE_TO_HOST),"D2H output");
            auto sorted_times=times;std::sort(sorted_times.begin(),sorted_times.end());double sum=0;for(auto t:times)sum+=t;
            double med=sorted_times.size()%2?sorted_times[sorted_times.size()/2]:(sorted_times[sorted_times.size()/2-1]+sorted_times[sorted_times.size()/2])/2;
            printf("RESULT timing_kind=host_wall_simulator_overhead NOT_A5_MODEL_LATENCY\n");
            printf("RESULT avg_ms=%.6f min_ms=%.6f median_ms=%.6f max_ms=%.6f iters=%d warmup=%d\n",sum/times.size(),sorted_times.front(),med,sorted_times.back(),iters,warmup);
            printf("EXECUTION completed_li_calls=%d\n",warmup+iters);
        }
        write(dir+"/li_out.i32",result);
        size_t valid=0,pad=0,expected=0,bad_count=0,bad_range=0,duplicate_rows=0;int32_t first=-1;
        for(int64_t b=0;b<B;++b)for(int64_t i=0;i<S;++i){
            int64_t v=i<aq[b]?(mode==3?std::max<int64_t>(0,int64_t(ak[b])-aq[b]+i+1):ak[b]):0;
            size_t n=0;std::vector<int32_t> seen;seen.reserve(size_t(std::min<int64_t>(K,v)));
            const size_t start=(size_t(b)*size_t(S)+size_t(i))*size_t(K);
            for(int64_t j=0;j<K;++j){auto idx=result[start+size_t(j)];if(idx==-1){++pad;continue;}++valid;++n;if(first<0&&idx>=0)first=idx;
                if(idx<0||idx>=v)++bad_range;else seen.push_back(idx);}
            size_t want=size_t(std::min<int64_t>(K,v));expected+=want;if(n!=want)++bad_count;
            std::sort(seen.begin(),seen.end());if(std::adjacent_find(seen.begin(),seen.end())!=seen.end())++duplicate_rows;
        }
        printf("SANITY outNum=%zu valid=%zu neg1=%zu expected_valid=%zu expected_neg1=%zu first_nonneg=%d\n",out_num,valid,pad,expected,out_num-expected,first);
        printf("STRUCTURAL bad_count_rows=%zu bad_range_elements=%zu duplicate_rows=%zu golden_checked=false\n",bad_count,bad_range,duplicate_rows);
        require(!bad_count&&!bad_range&&!duplicate_rows&&valid==expected,"Output structural check failed; li_out.i32 saved for diagnosis");
        puts("DONE status=PASS_STRUCTURAL_ONLY CA_TICK_SCOPE_NOT_VERIFIED_BY_PROGRAM");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"[ERROR] %s\n",e.what());return 1;}
}
