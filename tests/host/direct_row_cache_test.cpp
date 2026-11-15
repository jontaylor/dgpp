#include "loaders/direct_row_cache.hpp"
#include "common/test.hpp"
#include <filesystem>
#include <fstream>

namespace {
void require(bool v, const char* why) { if (!v) throw std::runtime_error(why); }
struct Fixture {
  std::string path = "/tmp/dgpp-direct-rows-XXXXXX";
  static constexpr size_t rows=5000, pb=128, sb=8, payload=1234, scales=payload+rows*pb+19;
  Fixture() {
    int fd=mkstemp(path.data()); require(fd>=0,"temporary file"); close(fd);
    std::vector<uint8_t> data(scales+rows*sb);
    for (size_t row=0;row<rows;++row) {
      for (size_t c=0;c<pb;++c)data[payload+row*pb+c]=(row*37+c*13)%251;
      for (size_t c=0;c<sb;++c)data[scales+row*sb+c]=(row*11+c*3)%251;
    }
    std::ofstream out(path,std::ios::binary); out.write(reinterpret_cast<char*>(data.data()),data.size());
  }
  ~Fixture(){std::filesystem::remove(path);}
  void check(const std::vector<int32_t>& ids,const std::vector<uint8_t>& out) {
    for(size_t i=0;i<ids.size();++i) {
      for(size_t c=0;c<pb;++c)require(out[i*(pb+sb)+c]==(ids[i]*37+c*13)%251,"payload bytes and row order");
      for(size_t c=0;c<sb;++c)require(out[i*(pb+sb)+pb+c]==(ids[i]*11+c*3)%251,"scale bytes and row order");
    }
  }
};
}
DGPP_TEST(direct_rows_preserve_cross_block_rows_cache_and_concurrent_gathers) {
  Fixture f;
  dgpp::DirectRowCache cache(f.path,f.payload,f.scales,f.rows,f.pb,f.sb);
  std::vector<int32_t> ids(512);
  for(size_t i=0;i<ids.size();++i)ids[i]=i%7 ? (i*37)%f.rows : 17;
  ids.back()=f.rows-1; // final short block at an unaligned EOF
  std::vector<uint8_t> out(ids.size()*(f.pb+f.sb));
  const int32_t select=0;
  cache.gather(ids.data(),1,&select,1,ids.size(),out.data()); f.check(ids,out);
  auto before=cache.stats(); require(before.misses>0 && before.hits>0,"duplicates hit cached compact rows");
  std::vector<uint8_t> second(out.size());
  std::jthread a([&]{cache.gather(ids.data(),1,&select,1,ids.size(),out.data());});
  std::jthread b([&]{cache.gather(ids.data(),1,&select,1,ids.size(),second.data());});
  a.join(); b.join(); f.check(ids,out); f.check(ids,second);
  require(cache.stats().reads==before.reads,"warm row cache performs no file reads");
  require(before.bytes<=before.reads*4096,"reads are bounded storage blocks rather than host pages");
  bool failed=false;int32_t invalid=f.rows;
  try{cache.gather(&invalid,1,&select,1,1,out.data());}catch(const std::out_of_range&){failed=true;}
  require(failed,"invalid rows are rejected");
}
DGPP_TEST(direct_rows_propagate_short_read_without_publishing_a_cache_hit) {
  Fixture f;
  dgpp::DirectRowCache cache(f.path,f.payload,f.scales,f.rows,f.pb,f.sb);
  std::filesystem::resize_file(f.path,0);
  int32_t id=42,select=0;std::array<uint8_t,136> out{};
  for(int i=0;i<2;++i){bool failed=false;
    try{cache.gather(&id,1,&select,1,1,out.data());}catch(const std::runtime_error&){failed=true;}
    require(failed,"truncated input throws on every attempted row read");
  }
  require(cache.stats().hits==0,"failed row never becomes a cache hit");
}

int main() { return dgpp::test::run_all(); }
