#include "kernels/mimo_attn.hpp"
#include "models/mimo/snapshot_copy.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <vector>
namespace {
using dgpp::kda_test::DevBuf;
float decode(uint8_t x) {
  int e=(x>>3)&15, m=x&7;
  float v=e ? std::ldexp(1.f+m/8.f,e-7) : std::ldexp(float(m),-9);
  return x&128 ? -v:v;
}
uint8_t quant(float x) {
  unsigned best=0; float error=std::numeric_limits<float>::infinity();
  for(unsigned i=0;i<127;++i) {
    float d=std::abs(std::abs(x)-decode(i));
    if(d<error || (d==error && !(i&1))) {error=d;best=i;}
  }
  return uint8_t(best | (std::signbit(x)?128:0));
}
void require(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
}
DGPP_TEST(mimo_fp8_append_attention_snapshot_and_graph) {
  auto stream=dgpp::kda_test::test_stream();
  for(int window:{0,128}) {
    dgpp::MimoAttentionShape s{1,2,1, window?137:640,window,true};
    auto bf=s;bf.fp8_cache=false;
    const size_t nk=size_t(s.capacity)*192,nv=size_t(s.capacity)*128;
    DevBuf k(nk),v(nv),rk(nk*2),rv(nv*2),q(384*2),f(704*2),pos(8),freq(128),sinks(4),status(4),out(256*2),ref(256*2),scores(size_t(s.capacity)*2*6),snap(size_t(window?128:s.capacity)*320);
    const uint16_t sink_values[]{dgpp::float_to_bf16_bits(-4.f),dgpp::float_to_bf16_bits(4.f)};
    sinks.upload(sink_values,4);
    std::vector<uint8_t> hk(nk,0),hv(nv,0); k.upload(hk.data(),nk);v.upload(hv.data(),nv);
    auto frequencies=dgpp::mimo_ref::inv_freq(window?1e4:1e7);freq.upload(frequencies.data(),128);
    std::vector<uint16_t> fused(704),rq(384),bk(nk,0),bv(nv,0),got(256),expected(256);
    uint64_t clipped=0,changed=0;double squared=0;float maxerr=0;
    for(int64_t p=0;p<600;++p) {
      for(int i=0;i<704;++i) fused[i]=dgpp::float_to_bf16_bits(float((i*7+p*13)%101-50)/32);
      // Include sign, nearest-even ties, subnormals and overflow without poisoning Q.
      if(p==0) {float edge[]={0.f,-0.f,1.0625f,1.1875f,0.0009765625f,448.f,480.f,-512.f};for(int i=0;i<8;++i)fused[384+64+i]=dgpp::float_to_bf16_bits(edge[i]);}
      f.upload(fused.data(),1408);pos.upload(&p,8);
      dgpp::mimo_ref::qkv_append(bf,fused.data(),frequencies.data(),&p,rq.data(),bk.data(),bv.data());
      const int slot=window?p%s.capacity:p;
      auto convert=[&](auto& raw,auto& decoded,int width) {for(int i=0;i<width;++i){size_t j=size_t(slot)*width+i;float x=dgpp::bf16_bits_to_float(decoded[j]);raw[j]=quant(x);float y=decode(raw[j]);clipped+=std::abs(x)>448;changed+=x!=y;squared+=double(x-y)*(x-y);maxerr=std::max(maxerr,std::abs(x-y));decoded[j]=dgpp::float_to_bf16_bits(y);}};
      convert(hk,bk,192);convert(hv,bv,128);
      auto append=[&]{dgpp::mimo_qkv_append(s,f.as<uint16_t>(),freq.as<float>(),pos.as<int64_t>(),q.as<uint16_t>(),k.as<uint8_t>(),v.as<uint8_t>(),status.as<int32_t>(),stream);};
      if(p==0) {
        cudaGraph_t graph;cudaGraphExec_t exec;
        DGPP_CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));append();DGPP_CUDA_OK(cudaStreamEndCapture(stream,&graph));
        DGPP_CUDA_OK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));DGPP_CUDA_OK(cudaGraphLaunch(exec,stream));DGPP_CUDA_OK(cudaStreamSynchronize(stream));cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
      } else append();
      if(p%127==0||p==599) {
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));std::vector<uint8_t> ak(nk),av(nv);k.download(ak.data(),nk);v.download(av.data(),nv);require(ak==hk&&av==hv,"FP8 append differs from independent nearest-even reference");
        rk.upload(bk.data(),nk*2);rv.upload(bv.data(),nv*2);
        for(int mode=0;mode<3;++mode) {
          auto attention=[&](auto shape,void* keys,void* values,DevBuf& output){
            if(mode==0) dgpp::mimo_attention(shape,q.as<uint16_t>(),keys,values,pos.as<int64_t>(),window?sinks.as<uint16_t>():nullptr,output.as<uint16_t>(),stream);
            if(mode==1) dgpp::mimo_attention_decode(shape,q.as<uint16_t>(),keys,values,pos.as<int64_t>(),window?sinks.as<uint16_t>():nullptr,output.as<uint16_t>(),scores.as<float>(),stream);
            if(mode==2) dgpp::mimo_attention_prefill(shape,q.as<uint16_t>(),keys,values,pos.as<int64_t>(),window?sinks.as<uint16_t>():nullptr,output.as<uint16_t>(),scores.as<float>(),p+1,stream,true,true,true);
          };
          if (mode==2) {
            cudaGraph_t graph;cudaGraphExec_t exec;
            DGPP_CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            attention(s,k.as<uint8_t>(),v.as<uint8_t>(),out);
            DGPP_CUDA_OK(cudaStreamEndCapture(stream,&graph));
            DGPP_CUDA_OK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            DGPP_CUDA_OK(cudaGraphLaunch(exec,stream));
            DGPP_CUDA_OK(cudaStreamSynchronize(stream));
            cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
          } else attention(s,k.as<uint8_t>(),v.as<uint8_t>(),out);
          attention(bf,rk.as<uint16_t>(),rv.as<uint16_t>(),ref);
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));out.download(got.data(),512);ref.download(expected.data(),512);require(got==expected,"FP8 read differs from identically quantized BF16 cache");
        }
      }
      if(p==256||p==259) {
        const size_t copied=dgpp::mimo_write_snapshot_layer(s,p+1,p==259?257:0,k.as<uint8_t>(),v.as<uint8_t>(),snap.as<uint8_t>(),stream);
        require(copied==size_t(window?128:(p==259?3:257))*320,"snapshot byte count not halved/incremental");
        DGPP_CUDA_OK(cudaMemsetAsync(k.as<uint8_t>(),0xa5,nk,stream));DGPP_CUDA_OK(cudaMemsetAsync(v.as<uint8_t>(),0xa5,nv,stream));
        dgpp::mimo_read_snapshot_layer(s,p+1,snap.as<uint8_t>(),k.as<uint8_t>(),v.as<uint8_t>(),stream);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));std::vector<uint8_t> ak(nk),av(nv);k.download(ak.data(),nk);v.download(av.data(),nv);
        for(auto span:dgpp::mimo_snapshot_spans(s,p+1))for(int64_t t=span.first;t<span.first+span.count;++t){for(int d=0;d<192;++d)require(ak[t*192+d]==hk[t*192+d],"restored K mismatch");for(int d=0;d<128;++d)require(av[t*128+d]==hv[t*128+d],"restored V mismatch");}
        // Restore untouched bytes too, so full-allocation sentinels remain meaningful.
        k.upload(hk.data(),nk);v.upload(hv.data(),nv);
      }
    }
    require(clipped==2,"overflow diagnostic missing");
    std::cout<<"FP8 window="<<window<<" clipped="<<clipped<<" changed="<<changed<<" rms="<<std::sqrt(squared/(600*320))<<" max="<<maxerr<<'\n';
  }
}
int main() {
  int devices=0;
  if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices) return 2;
  return dgpp::test::run_all();
}
