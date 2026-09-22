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

// Isolate storage integration from the online algorithm's intentional changes
// to softmax rounding: compare each algorithm with itself on dequantized data.
DGPP_TEST(mimo_fp8_bounded_online_same_algorithm_graph_and_mapping) {
  auto stream = dgpp::kda_test::test_stream();
  for (int window : {0, 128}) {
    for (bool mapped : {false, true}) {
      for (int rows : (mapped ? std::vector<int>{4} : std::vector<int>{1, 17, 33})) {
        const int capacity = window ? 263 : 640;
        const int planes = mapped ? 3 : 1;
        dgpp::MimoAttentionShape shape{rows, 4, 2, capacity, window, true};
        auto bf = shape;
        bf.fp8_cache = false;
        const size_t nk = size_t(planes) * capacity * shape.k_width();
        const size_t nv = size_t(planes) * capacity * shape.v_width();
        const size_t nq = size_t(rows) * shape.q_width();
        const size_t no = size_t(rows) * shape.q_heads * 128;
        DevBuf k(nk), v(nv), rk(nk * 2), rv(nv * 2), q(nq * 2), positions(rows * 8),
            mapping(rows * 4), sinks(shape.q_heads * 2), actual(no * 2), reference(no * 2);
        std::vector<uint8_t> packed_k(nk), packed_v(nv);
        std::vector<uint16_t> expanded_k(nk), expanded_v(nv), queries(nq), got(no), expected(no);
        std::vector<uint16_t> sink_values(shape.q_heads);
        std::vector<int64_t> pos(rows);
        std::vector<int32_t> ids(rows);
        uint32_t rng = 0xabcdef01;
        auto fill = [&](auto& packed, auto& expanded) {
          for (size_t i = 0; i < packed.size(); ++i) {
            rng = rng * 1664525u + 1013904223u;
            // Finite E4M3 payloads include zero, subnormals and +/-448.
            packed[i] = uint8_t((rng % 127) | ((rng >> 16) & 128));
            expanded[i] = dgpp::float_to_bf16_bits(decode(packed[i]));
          }
        };
        const int end_key = window ? capacity * 3 + 17 : capacity;
        for (bool online : {false, true}) {
          for (bool with_sink : {false, true}) {
            auto launch = [&](auto s, void* keys, void* values, DevBuf& output) {
              const auto* sink = with_sink ? sinks.as<uint16_t>() : nullptr;
              if (mapped && !online)
                dgpp::mimo_attention(s, q.as<uint16_t>(), keys, values,
                    positions.as<int64_t>(), sink, output.as<uint16_t>(), stream,
                    false, nullptr, mapping.as<int32_t>());
              else if (mapped)
                dgpp::mimo_attention_online_decode(s, q.as<uint16_t>(), keys, values,
                    positions.as<int64_t>(), sink, output.as<uint16_t>(), stream,
                    mapping.as<int32_t>());
              else
                dgpp::mimo_attention_bounded_prefill(s, q.as<uint16_t>(), keys, values,
                    positions.as<int64_t>(), sink, output.as<uint16_t>(), end_key, stream, online);
            };
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launch(shape, k.as<uint8_t>(), v.as<uint8_t>(), actual);
            DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
            size_t count = 0;
            DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &count));
            require(count > 0, "empty FP8 attention graph");
            std::vector<cudaGraphNode_t> nodes(count);
            DGPP_CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &count));
            for (auto node : nodes) {
              cudaGraphNodeType type;
              DGPP_CUDA_OK(cudaGraphNodeGetType(node, &type));
              require(type == cudaGraphNodeTypeKernel, "FP8 attention graph has non-kernel node");
            }
            DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 3; ++replay) {
              fill(packed_k, expanded_k);
              fill(packed_v, expanded_v);
              for (auto& value : queries) {
                rng = rng * 1664525u + 1013904223u;
                value = dgpp::float_to_bf16_bits(float(int(rng >> 16) - 32768) / 16384.f);
              }
              for (int h = 0; h < shape.q_heads; ++h)
                sink_values[h] = dgpp::float_to_bf16_bits(float((h % 2 ? 1 : -1) * (4 + replay)));
              for (int row = 0; row < rows; ++row) {
                if (mapped) {
                  ids[row] = row == 1 ? -1 : (row + replay) % planes;
                  pos[row] = row == 1 ? -1 : (row == 2 ? 2 : end_key - 1 - row * 17);
                } else {
                  pos[row] = end_key - rows + row;
                }
              }
              // An invalid global row must zero output before dereferencing -1's cache plane.
              if (mapped && !window && replay == 2) {
                ids[3] = -1;
                pos[3] = capacity;
              }
              k.upload(packed_k.data(), nk);
              v.upload(packed_v.data(), nv);
              rk.upload(expanded_k.data(), nk * 2);
              rv.upload(expanded_v.data(), nv * 2);
              q.upload(queries.data(), nq * 2);
              sinks.upload(sink_values.data(), sink_values.size() * 2);
              positions.upload(pos.data(), rows * 8);
              mapping.upload(ids.data(), rows * 4);
              DGPP_CUDA_OK(cudaMemsetAsync(actual.as<uint16_t>(), 0xa5, no * 2, stream));
              DGPP_CUDA_OK(cudaMemsetAsync(reference.as<uint16_t>(), 0x5a, no * 2, stream));
              launch(bf, rk.as<uint16_t>(), rv.as<uint16_t>(), reference);
              DGPP_CUDA_OK(cudaGraphLaunch(executable, stream));
              DGPP_CUDA_OK(cudaStreamSynchronize(stream));
              actual.download(got.data(), no * 2);
              reference.download(expected.data(), no * 2);
              if (got != expected) {
                const auto first = std::mismatch(got.begin(), got.end(), expected.begin());
                throw std::runtime_error("FP8/dequantized BF16 same-algorithm mismatch: window=" +
                    std::to_string(window) + " mapped=" + std::to_string(mapped) +
                    " rows=" + std::to_string(rows) + " online=" + std::to_string(online) +
                    " sink=" + std::to_string(with_sink) + " replay=" + std::to_string(replay) +
                    " element=" + std::to_string(first.first - got.begin()));
              }
              for (int row = 0; row < rows; ++row) {
                const bool padded = mapped && (pos[row] < 0 || (!window && pos[row] >= capacity));
                for (int i = 0; i < shape.q_heads * 128; ++i) {
                  const auto value = got[size_t(row) * shape.q_heads * 128 + i];
                  require(std::isfinite(dgpp::bf16_bits_to_float(value)), "nonfinite FP8 attention output");
                  if (padded) require(value == 0, "padded/invalid FP8 decode output is nonzero");
                }
              }
            }
            DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
            DGPP_CUDA_OK(cudaGraphDestroy(graph));
          }
        }
      }
    }
  }
}
