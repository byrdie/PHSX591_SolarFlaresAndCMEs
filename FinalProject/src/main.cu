////////////////////////////////////////////////////////////////////
//                                                                //
// standard headers plus new one defining tridiagonal solvers     //
//                                                                //
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <trid.h>

////////////////////////////////////////////////////////////////////
//                                                                //
// explicit Black-Scholes finite difference kernels               //
//                                                                //
////////////////////////////////////////////////////////////////////

template <typename REAL>
__global__ void BS_explicit1_old(int N, REAL c1, REAL c2, REAL c3,
                             REAL dS, REAL K, REAL *u_d) {

  __shared__ REAL u[258];
  REAL  S, lambda, gamma, a, b, c, utmp;

  S      = threadIdx.x * dS;
  lambda = c1*S*S;
  gamma  = c2*S;

  if (threadIdx.x==blockDim.x-1) {
    a = - 2.0f*gamma;
    b = + 2.0f*gamma - c3;
    c =   0.0f;
  }
  else {
    a =        lambda - gamma;
    b = - 2.0f*lambda - c3;
    c =        lambda + gamma;
  }

  int i = threadIdx.x + 1;

  u[i] = 0.0f;
  if (S>K) u[i] = S-K;

  if (threadIdx.x==0) {
    u[0]            = 0.0f; // need to be zeroed out
    u[blockDim.x+1] = 0.0f; //
  }
  __syncthreads();

// main time-marching loop

  utmp = u[i];

  for (int n=0; n<N; n++) {
    utmp = utmp + a*u[i-1] + b*utmp + c*u[i+1];
    __syncthreads();
    u[i] = utmp;
    __syncthreads();
  }

// output to ensure compiler does force execution

  if (threadIdx.x==0)
    u_d[blockIdx.x] = 0.5f*(u[blockDim.x/2]
                           +u[blockDim.x/2+1]);
}



template <typename REAL>
__global__ void BS_explicit1(int N, REAL c1, REAL c2, REAL c3,
                             REAL dS, REAL K, REAL *u_d) {

  __shared__ REAL u[258], u2[258];
  REAL  S, lambda, gamma, a, b, c, utmp;

  S      = threadIdx.x * dS;
  lambda = c1*S*S;
  gamma  = c2*S;

  if (threadIdx.x==blockDim.x-1) {
    a = - 2.0f*gamma;
    b = + 2.0f*gamma - c3;
    c =   0.0f;
  }
  else {
    a =        lambda - gamma;
    b = - 2.0f*lambda - c3;
    c =        lambda + gamma;
  }

  int i = threadIdx.x + 1;

  u[i] = 0.0f;
  if (S>K) u[i] = S-K;

  if (threadIdx.x==0) {
    u[0]            = 0.0f; // need to be zeroed out
    u[blockDim.x+1] = 0.0f; //
  }
  __syncthreads();

// main time-marching loop, double-buffered for performance

  utmp = u[i];

  for (int n=0; n<N; n+=2) {
    utmp  = utmp + a*u[i-1] + b*utmp + c*u[i+1];
    u2[i] = utmp;
    __syncthreads();
    utmp  = utmp + a*u2[i-1] + b*utmp + c*u2[i+1];
    u[i]  = utmp;
    __syncthreads();
  }

// output to ensure compiler does force execution

  if (threadIdx.x==0)
    u_d[blockIdx.x] = 0.5f*(u[blockDim.x/2]
                           +u[blockDim.x/2+1]);
}


template <typename REAL>
__global__ void BS_explicit2_old(int N, REAL c1, REAL c2, REAL c3,
                             REAL dS, REAL K, REAL *u_d) {

// volatile keyword required because of "non-safe" use
  volatile __shared__ REAL u_shared[33*8];

  REAL a[8], b[8], c[8], u[8], S, lambda, gamma, um, up, u0;
  int  t;

// initialise lots of things

  t = threadIdx.x%32;

  for (int i=0; i<8; i++) {
    S      = (8*t+i) * dS;
    lambda = c1*S*S;
    gamma  = c2*S;

    if (i==7 && t==31) {
      a[i] = - 2.0f*gamma;
      b[i] = + 2.0f*gamma - c3;
      c[i] = 0.0;
    }
    else {
      a[i] =        lambda - gamma;
      b[i] = - 2.0f*lambda - c3;
      c[i] =        lambda + gamma;
    }

    u[i] = 0.0f;
    if (S>K) u[i] = S-K;
  }

// offset so that each warp has its own piece of shared memory

  t = t + 33 * (threadIdx.x/32);
  u_shared[t] = 0.0f;  // necessary to ensure not NaN

// main time-marching loop

  for (int n=0; n<N; n++) {
    if (sizeof(REAL) == 4) {
      um = __shfl_up((float)u[7], 1);
      up = __shfl_down((float)u[0], 1);
    }
    else {
      u_shared[t+1] = u[7];
      um            = u_shared[t];
      u_shared[t]   = u[0];
      up            = u_shared[t+1];
    }

    for (int i=0; i<7; i++) {
      u0   = u[i];
      u[i] = u[i] + a[i]*um + b[i]*u0 + c[i]*u[i+1];
      um   = u0;
    }

    u[7] = u[7] + a[7]*um + b[7]*u[7] + c[7]*up;
  }

// output to ensure compiler does force execution

  u_shared[t] = u[0];

  if (threadIdx.x%32 == 15)
    u_d[(threadIdx.x+blockIdx.x*blockDim.x)/32] =
                        0.5f*(u[7]+u_shared[t+1]);
}



template <typename REAL>
__global__ void BS_explicit2(int N, REAL c1, REAL c2, REAL c3,
                             REAL dS, REAL K, REAL *u_d) {

// volatile keyword required because of "non-safe" use
  volatile __shared__ REAL u_shared[33*8];

  REAL a[8], b[8], c[8], u[8], u2[8], S, lambda, gamma, um, up;
  int  t;

// initialise lots of things

  t = threadIdx.x%32;

  for (int i=0; i<8; i++) {
    S      = (8*t+i) * dS;
    lambda = c1*S*S;
    gamma  = c2*S;

    if (i==7 && t==31) {
      a[i] = - 2.0f*gamma;
      b[i] = + 2.0f*gamma - c3;
      c[i] = 0.0;
    }
    else {
      a[i] =        lambda - gamma;
      b[i] = - 2.0f*lambda - c3;
      c[i] =        lambda + gamma;
    }

    u[i] = 0.0f;
    if (S>K) u[i] = S-K;
  }

// offset so that each warp has its own piece of shared memory

  t = t + 33 * (threadIdx.x/32);
  u_shared[t] = 0.0f;  // necessary to ensure not NaN

// main time-marching loop, double-buffered for performance

  for (int n=0; n<N; n+=2) {

    // first pass: u -> u2

    if (sizeof(REAL) == 4) {
      um = __shfl_up((float)u[7], 1);
      up = __shfl_down((float)u[0], 1);
    }
    else {
      u_shared[t+1] = u[7];
      um            = u_shared[t];
      u_shared[t]   = u[0];
      up            = u_shared[t+1];
    }

    for (int i=1; i<4; i++)
      u2[i] = u[i] + a[i]*u[i-1] + b[i]*u[i] + c[i]*u[i+1];
    u2[0] = u[0] + a[0]*um + b[0]*u[0] + c[0]*u[1];
    u2[7] = u[7] + a[7]*u[6] + b[7]*u[7] + c[7]*up;
    for (int i=4; i<7; i++)
      u2[i] = u[i] + a[i]*u[i-1] + b[i]*u[i] + c[i]*u[i+1];

    // second pass: u2 -> u

    if (sizeof(REAL) == 4) {
      um = __shfl_up((float)u2[7], 1);
      up = __shfl_down((float)u2[0], 1);
    }
    else {
      u_shared[t+1] = u2[7];
      um            = u_shared[t];
      u_shared[t]   = u2[0];
      up            = u_shared[t+1];
    }

    for (int i=1; i<4; i++)
      u[i] = u2[i] + a[i]*u2[i-1] + b[i]*u2[i] + c[i]*u2[i+1];
    u[0] = u2[0] + a[0]*um + b[0]*u2[0] + c[0]*u2[1];
    u[7] = u2[7] + a[7]*u2[6] + b[7]*u2[7] + c[7]*up;
    for (int i=4; i<7; i++)
      u[i] = u2[i] + a[i]*u2[i-1] + b[i]*u2[i] + c[i]*u2[i+1];
  }

// output to ensure compiler does force execution

  u_shared[t] = u[0];

  if (threadIdx.x%32 == 15)
    u_d[(threadIdx.x+blockIdx.x*blockDim.x)/32] =
                        0.5f*(u[7]+u_shared[t+1]);
}


////////////////////////////////////////////////////////////////////
//                                                                //
// implicit Black-Scholes finite difference kernels               //
//                                                                //
////////////////////////////////////////////////////////////////////

template <typename REAL>
__launch_bounds__(256, 2) // use "only" 128 registers per thread
__global__ void BS_implicit1(int N, REAL c1, REAL c2, REAL c3,
                             REAL c4, REAL dS, REAL K, REAL *u_d) {

  REAL a[8], b[8], c[8], u[8], S, lambda, gamma;
  int  t = threadIdx.x%32;

// initialise payoff

  for (int i=0; i<8; i++) {
    S    = (8*t+i) * dS;
    u[i] = 0.0f;
    if (S>K) u[i] = S-K;
  }

// main time-marching loop

  for (int n=0; n<N; n++) {
    for (int i=0; i<8; i++) {
      S      = (8*t+i) * dS;
      lambda = c1*S*S;
      gamma  = c2*S;

      a[i] =              - ( lambda - gamma );
      b[i] = 1.0f + c3 + 2.0f*lambda + c4*n;
      c[i] =              - ( lambda + gamma );
    }

    if (t==31) {
      a[7] =           + 2.0f*gamma;
      b[7] = 1.0f + c3 - 2.0f*gamma;
      c[7] = 0.0f;
    }

    trid_warp<8>(a,b,c,u);
  }

// output to ensure compiler does force execution

  u[0] = __shfl_down(u[0],1);

  if (t == 15)
    u_d[(threadIdx.x+blockIdx.x*blockDim.x)/32] = 0.5f*(u[7]+u[0]);
}


template <typename REAL>
__launch_bounds__(256, 2) // use "only" 128 registers per thread
__global__ void BS_implicit2(int N, REAL c1, REAL c2, REAL c3,
                             REAL c4, REAL dS, REAL K, REAL *u_d) {

  REAL a[8], b[8], c[8], d[8], u[8], S, lambda, gamma;
  int  t = threadIdx.x%32;

// initialise payoff

  for (int i=0; i<8; i++) {
    S    = (8*t+i) * dS;
    u[i] = 0.0f;
    if (S>K) u[i] = S-K;
  }

// main time-marching loop

  for (int n=0; n<N; n++) {
    for (int i=0; i<8; i++) {
      S      = (8*t+i) * dS;
      lambda = c1*S*S;
      gamma  = c2*S;

      a[i] =       - ( lambda - gamma );
      b[i] = c3 + 2.0f*lambda + c4*n;
      c[i] =       - ( lambda + gamma );

      if (i==0)
        d[0] = - a[0]*__shfl_up(u[7],1) - b[0]*u[0] - c[0]*u[1];
      else if (i<7)
        d[i] = - a[i]*u[i-1] - b[i]*u[i] - c[i]*u[i+1];
    }

    if (t==31) {
      a[7] =    + 2.0f*gamma;
      b[7] = c3 - 2.0f*gamma;
      c[7] = 0.0f;
    }
    d[7]  = - a[7]*u[6] - b[7]*u[7] - c[7]*__shfl_down(u[0],1);

    for (int i=0; i<8; i++) b[i] += 1.0f;

    trid_warp<8>(a,b,c,d);

    for (int i=0; i<8; i++) u[i] += d[i];
  }

// output to ensure compiler does force execution

  u[0] = __shfl_down(u[0],1);

  if (t == 15)
    u_d[(threadIdx.x+blockIdx.x*blockDim.x)/32] = 0.5f*(u[7]+u[0]);
}


template <typename REAL>
__launch_bounds__(256, 2) // use "only" 128 registers per thread
__global__ void BS_implicit3(int N, REAL c1, REAL c2, REAL c3,
                             REAL dS, REAL K, REAL *u_d) {

  REAL a[8], b[8], c[8], u[8], a2[8], ap[5], bp[5], cp[5],
       S, lambda, gamma, bbi,bb0,a0,c0;
  int  t = threadIdx.x%32;

// initialise lots of things

#pragma unroll 1
  for (int i=0; i<8; i++) {
    S      = (8*t+i) * dS;
    lambda = c1*S*S;
    gamma  = c2*S;

    a[i] =              - ( lambda - gamma );
    b[i] = 1.0f + c3 + 2.0f*lambda;
    c[i] =              - ( lambda + gamma );

    u[i] = 0.0f;
    if (S>K) u[i] = S-K;
  }

  if (t==31) {
    a[7] =           + 2.0f*gamma;
    b[7] = 1.0f + c3 - 2.0f*gamma;
    c[7] = 0.0f;
  }

  trid_warp_setup<8>(a,b,c,a2,ap,bp,cp, bbi,bb0,a0,c0);

// main time-marching loop

  for (int n=0; n<N; n++) {
    trid_warp_solve<8>(a,b,c,a2,ap,bp,cp,u, bbi,bb0,a0,c0);
  }

// output to ensure compiler does force execution

  u[0] = __shfl_down(u[0],1);

  if (threadIdx.x%32 == 15)
    u_d[(threadIdx.x+blockIdx.x*blockDim.x)/32] = 0.5f*(u[7]+u[0]);
}


////////////////////////////////////////////////////////////////////
//                                                                //
// main code to test all solvers for single & double precision    //
//                                                                //
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  int    N, noptions, nthreads, nblocks;
  float  *u_h, *u_d, finsts, flops;
  double *U_h, *U_d, val;

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // set number of options to be calculated

  noptions = 1;

  // allocate memory for answers

  u_h = (float *)malloc(noptions*sizeof(float));
  U_h = (double *)malloc(noptions*sizeof(double));
  cudaMalloc((void **)&u_d, noptions*sizeof(float));
  cudaMalloc((void **)&U_d, noptions*sizeof(double));

  // execute kernels

  for (int prec=0; prec<2; prec++) {
    if (prec==0) {
      printf("\nsingle precision performance tests \n");
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    }
    else {
      printf("\ndouble precision performance tests \n");
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    }
    printf("---------------------------------- \n");
    printf(" method    exec time   GFinsts    GFlops       value at strike \n");

    for (int pass=0; pass<5; pass++) {
      if (pass==0)
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
      else
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

      cudaEventRecord(start);

      if (pass<2)
        N = 50000; // number of timesteps for explicit solvers
      else if (pass>=2)
        N = 2500;  // number of timesteps for implicit solvers

  // set parameters for BS simulation

      double Smax=200.0, K=100.0, r=0.05, sigma=0.2, T=1.0;

      double dS = Smax / 255.0;
      double dt = T / ( (double) N);
      double C1 = 0.5*dt*sigma*sigma / (dS*dS);
      double C2 = 0.5*dt*r / dS;
      double C3 = r*dt;

      float c1=C1, c2=C2, c3=C3, ds=dS, k=K;

      if (pass==0) {
        nthreads = 256;
        nblocks  = noptions;
      }
      else {
        nthreads = 128;
        nblocks  = noptions / (nthreads/32);
      }

      if (pass==0 && prec==0)
        BS_explicit1<<<nblocks,nthreads>>>(N,c1,c2,c3,ds,k, u_d);
      else if (pass==0 && prec==1)
        BS_explicit1<<<nblocks,nthreads>>>(N,C1,C2,C3,dS,K, U_d);
      else if (pass==1 && prec==0)
        BS_explicit2<<<nblocks,nthreads>>>(N,c1,c2,c3,ds,k, u_d);
      else if (pass==1 && prec==1)
        BS_explicit2<<<nblocks,nthreads>>>(N,C1,C2,C3,dS,K, U_d);
      else if (pass==2 && prec==0)
        BS_implicit1<<<nblocks,nthreads>>>(N,c1,c2,c3,0.0f,ds,k, u_d);
      else if (pass==2 && prec==1)
        BS_implicit1<<<nblocks,nthreads>>>(N,C1,C2,C3,0.0, dS,K, U_d);
      else if (pass==3 && prec==0)
        BS_implicit2<<<nblocks,nthreads>>>(N,c1,c2,c3,0.0f,ds,k, u_d);
      else if (pass==3 && prec==1)
        BS_implicit2<<<nblocks,nthreads>>>(N,C1,C2,C3,0.0, dS,K, U_d);
      else if (pass==4 && prec==0)
        BS_implicit3<<<nblocks,nthreads>>>(N,c1,c2,c3,ds,k, u_d);
      else if (pass==4 && prec==1)
        BS_implicit3<<<nblocks,nthreads>>>(N,C1,C2,C3,dS,K, U_d);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milli, start, stop);

      if (prec==0) {
        cudaMemcpy(u_h,u_d,noptions*sizeof(float),
                   cudaMemcpyDeviceToHost);
        val = (double) u_h[noptions-1];
        for (int n=0; n<noptions; n++) {
          if (val != (double) u_h[n] )
            printf(" n = %d, val1 = %f, val2 = %f \n",n,val,u_h[n]);
        }
      }
      else {
        cudaMemcpy(U_h,U_d,noptions*sizeof(double),
                   cudaMemcpyDeviceToHost);
        val = U_h[noptions-1];
        for (int n=0; n<noptions; n++) {
          if (val != U_h[n] )
            printf(" n = %d, val1 = %f, val2 = %f \n",n,val,U_h[n]);
        }
      }

      float factor = 256.0f*noptions*N / (1.0e6f*milli);

      if (pass<2) {
        printf("explicit%1d   %7.2f    %7.2f   %7.2f   %20.14f  \n",
                pass+1,milli,3.0f*factor,6.0f*factor,val);
      }
      else if (pass>=2) {
        if (pass==2) {
          finsts = (150 + 46*prec) / 8.0f;
          flops  = (220 + 91*prec) / 8.0f;
        }
        if (pass==3) {
          finsts = (190 + 46*prec) / 8.0f;
          flops  = (276 + 91*prec) / 8.0f;
        }
        if (pass==4) {
          finsts =  53             / 8.0f;
          flops  =  91             / 8.0f;
        }
        printf("implicit%1d   %7.2f    %7.2f   %7.2f   %20.14f  \n",
                pass-1,milli,finsts*factor,flops*factor,val);
      }
    }
  }

// CUDA exit -- needed to flush printf write buffer

  cudaThreadSynchronize();
  cudaDeviceReset();
  return 0;
}
