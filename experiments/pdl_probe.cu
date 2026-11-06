#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do {auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__global__ void norm(const float*x,float*y,int K,unsigned long long*t,int early){
 __shared__ float s[256]; int r=blockIdx.x,i=threadIdx.x;
 if(i==0)t[2*r]=timer();
 if(early)cudaTriggerProgrammaticLaunchCompletion();
 float v=0;for(int k=i;k<K;k+=256)v+=x[r*K+k]*x[r*K+k];
 s[i]=v;__syncthreads();
 if(!early)cudaTriggerProgrammaticLaunchCompletion();
 for(int d=128;d;d/=2){if(i<d)s[i]+=s[i+d];__syncthreads();}
 float inv=rsqrtf(s[0]/K+1e-6f);for(int k=i;k<K;k+=256)y[r*K+k]=x[r*K+k]*inv;
 __syncthreads();if(i==0)t[2*r+1]=timer();
}
// Correct tiled FP32 GEMM. Optional L2 cache hints are independent of y.
__global__ void mm(const float*y,const float*w,float*z,int M,int K,int N,int pre,unsigned long long*t){
 int tx=threadIdx.x%16,ty=threadIdx.x/16,r=blockIdx.x*16+ty,c=blockIdx.y*16+tx;
 int bid=blockIdx.x*gridDim.y+blockIdx.y;
 if(threadIdx.x==0)t[3*bid]=timer();
 // Only one M tile issues hints per N tile; 16 floats per row (64 bytes).
 if(blockIdx.x==0)for(int k=ty;k<pre;k+=16){const float*p=w+k*N+blockIdx.y*16+tx; if(c<N)asm volatile("prefetch.global.L2 [%0];"::"l"(p));}
 cudaGridDependencySynchronize();
 if(threadIdx.x==0)t[3*bid+1]=timer();
 __shared__ float a[16][16],b[16][16];float v=0;
 for(int k=0;k<K;k+=16){a[ty][tx]=(r<M&&k+tx<K)?y[r*K+k+tx]:0;b[ty][tx]=(k+ty<K&&c<N)?w[(k+ty)*N+c]:0;__syncthreads();
 for(int j=0;j<16;j++)v+=a[ty][j]*b[j][tx];__syncthreads();}
 if(r<M&&c<N)z[r*N+c]=v;__syncthreads();if(threadIdx.x==0)t[3*bid+2]=timer();
}
__global__ void chain(int*in,int*out,int stage,unsigned long long*t){
 if(threadIdx.x==0)t[3*stage]=timer();
 if(stage)cudaGridDependencySynchronize();
 if(threadIdx.x==0)t[3*stage+1]=timer();
 cudaTriggerProgrammaticLaunchCompletion();
 if(threadIdx.x==0){*out=stage?*in+1:1;t[3*stage+2]=timer();}
}
template<class F,class... A>void launch(F f,dim3 grid,cudaStream_t stream,bool pdl,A...args){cudaLaunchConfig_t c{};c.gridDim=grid;c.blockDim=256;c.stream=stream;cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=pdl;c.attrs=&a;c.numAttrs=1;CK(cudaLaunchKernelEx(&c,f,args...));}
int main(int argc,char**argv){
 int M=argc>1?atoi(argv[1]):32,K=argc>2?atoi(argv[2]):512,N=argc>3?atoi(argv[3]):128,R=argc>4?atoi(argv[4]):20;
 if(M<1||K<1||N<1||R<1)return 2;
 cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));fprintf(stderr,"GPU=%s CC=%d.%d SM=%d L2=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount,prop.l2CacheSize);if(prop.major<9)return 2;
 std::vector<float>x(M*K),w(K*N),ref(M*N),out(M*N);for(size_t i=0;i<x.size();i++)x[i]=(int(i%31)-15)*.01f;for(size_t i=0;i<w.size();i++)w[i]=(int(i%23)-11)*.01f;
 for(int r=0;r<M;r++){double ss=0;for(int k=0;k<K;k++)ss+=double(x[r*K+k])*x[r*K+k];double inv=1/sqrt(ss/K+1e-6);for(int c=0;c<N;c++){double v=0;for(int k=0;k<K;k++)v+=x[r*K+k]*inv*w[k*N+c];ref[r*N+c]=v;}}
 float *dx,*dy,*dw,*dz;unsigned long long *nt,*mt;int blocks=((M+15)/16)*((N+15)/16);
 CK(cudaMalloc(&dx,x.size()*4));CK(cudaMalloc(&dy,x.size()*4));CK(cudaMalloc(&dw,w.size()*4));CK(cudaMalloc(&dz,out.size()*4));CK(cudaMalloc(&nt,2*M*8));CK(cudaMalloc(&mt,3*blocks*8));CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));std::vector<unsigned long long>hn(2*M),hm(3*blocks);
 puts("kind,M,K,N,pdl,prefetch_rows,early,rep,e2e_us,norm_span_us,overlap_window_us,max_abs_error");
 for(int early:{0,1})for(int pre:{0,std::min(K,16),K})for(int rep=-5;rep<R;rep++)for(int order=0;order<2;order++){
 int pdl=(order+(rep&1))&1;CK(cudaEventRecord(start,stream));norm<<<M,256,0,stream>>>(dx,dy,K,nt,early);CK(cudaGetLastError());launch(mm,dim3((M+15)/16,(N+15)/16),stream,pdl,dy,dw,dz,M,K,N,pre,mt);CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));
 CK(cudaMemcpy(out.data(),dz,out.size()*4,cudaMemcpyDeviceToHost));float err=0;for(size_t i=0;i<out.size();i++){if(!std::isfinite(out[i])||fabs(out[i]-ref[i])>1e-3+1e-3*fabs(ref[i])){fprintf(stderr,"FAIL %zu %.9g %.9g\n",i,out[i],ref[i]);return 3;}err=std::max(err,fabsf(out[i]-ref[i]));}
 if(rep<0)continue;CK(cudaMemcpy(hn.data(),nt,hn.size()*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hm.data(),mt,hm.size()*8,cudaMemcpyDeviceToHost));auto ns=hn[0],ne=hn[1],ms=hm[0];for(int i=0;i<M;i++){ns=std::min(ns,hn[2*i]);ne=std::max(ne,hn[2*i+1]);}for(int i=0;i<blocks;i++)ms=std::min(ms,hm[3*i]);float elapsed;CK(cudaEventElapsedTime(&elapsed,start,end));printf("pair,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.9g\n",M,K,N,pdl,pre,early,rep,elapsed*1000,(ne-ns)/1000.,ms<ne?(ne-ms)/1000.:0.,err);
 }
 for(int count:{1,2,4,8,16,32,64,128})for(int pdl:{0,1}){
 std::vector<cudaStream_t>streams(count);int*vals;unsigned long long*times;CK(cudaMalloc(&vals,count*3*sizeof(int)));CK(cudaMalloc(&times,count*9*8));
 for(int i=0;i<count;i++)CK(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));
 for(int i=0;i<count;i++)for(int stage=0;stage<3;stage++)launch(chain,dim3(1),streams[i],pdl&&stage>0,vals+i*3+std::max(stage-1,0),vals+i*3+stage,stage,times+i*9);
 CK(cudaDeviceSynchronize());std::vector<int>v(count*3);CK(cudaMemcpy(v.data(),vals,v.size()*sizeof(int),cudaMemcpyDeviceToHost));for(int i=0;i<count;i++)if(v[3*i]!=1||v[3*i+1]!=2||v[3*i+2]!=3)return 4;
 fprintf(stderr,"chain streams=%d pdl=%d correct=1\n",count,pdl);for(auto s:streams)CK(cudaStreamDestroy(s));CK(cudaFree(vals));CK(cudaFree(times));
 }
 CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));CK(cudaStreamDestroy(stream));CK(cudaFree(dx));CK(cudaFree(dy));CK(cudaFree(dw));CK(cudaFree(dz));CK(cudaFree(nt));CK(cudaFree(mt));
}
