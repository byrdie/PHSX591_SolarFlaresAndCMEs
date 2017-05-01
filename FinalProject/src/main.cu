////////////////////////////////////////////////////////////////////
//                                                                //
// standard headers plus new one defining tridiagonal solvers     //
//                                                                //
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <trid.h>

/* fundamental physical constants */
const double k_b = 1.385e-16; // boltzmann constant (erg/K)
const double mbar = 0.593 * 1.6726218e-24; // m-bar (mean mass per particle = 0.593 m_p) (g)
const double c_v = 3 * k_b / (2 * mbar);	// specific heat
const double kappa_0 = 1e-6;// Thermal conductivity constant (erg cm^-1 s^-1 K^(-7/2))
const double Pr = 0.012;		// Prandtl number in fully ionized plasma
const double mu_0 = Pr * kappa_0 / c_v;		// Dynamic viscosity constant

/* Adjustable physical parameters */
const double L = 53e8;							// Length of half flux tube (cm)
const double T = 1e-3;							// Length of the simulation (s)
const double p_init = 1.0;						// Initial pressure (erg cm^-3)
const double T_init = 2e4;					// Initial temperture of the plasma
const double rho_init = mbar * p_init / (k_b * T_init);	// Initial density of the plasma
const double u_init = 0.0;				// Initial speed of the plasma
const double F = 3.5e10;				// Flare energy flux (erg cm^-2 s^-1)
const double Delta_fl = L;					// extent of heat flux function
const float h = 2 * F / Delta_fl;		// heating function

/* Adjustable simulation parameters */
const uint N_s = 16;		// Number of spatial points
const uint N_t = 5;		// Number of temporal points
const float ds = L / N_s;	// distance between spatial points
const float dt = T / N_t;	// distance between temporal points
const uint N_loops = 1;		// Number of parallel loops to simulate

/* velocity update constants */
/* c2 = k/m */
/* c3 = (4/3)*mu_0*/
const float u_c2 = k_b / mbar;
const float u_c3 = 4 * mu_0 / 3;

/* Temperture update constants */
/* c2 = 1 / c_v 				*/
/* c3 = 4 mu_0 / (3 c_v) 		*/
/* c4 = K_0 / c_v				*/
/* c5 = h 						*/
const float T_c2 = 1 / c_v;
const float T_c3 = 4 * mu_0 / (3 * c_v);
const float T_c4 = kappa_0 / c_v;

/* Finite Difference Derivative */
/* dx/ds */
template<typename REAL> __device__ REAL D(REAL x0, REAL x1, REAL ds) {
	return (x1 - x0) / ds;
}

/* Branchless upwind differencing scheme */
template<typename REAL> __device__ REAL uD(REAL u, REAL x9, REAL x0, REAL x1,
		REAL ds) {

	REAL t1 = x0 * abs(u) / ds;
	REAL t2 = -(x9 + x1) * abs(u) / (2 * ds);
	REAL t3 = (x9 - x1) * u / (2 * ds);

	return t1 + t2 + t3;
}

template<typename REAL> __device__ REAL ave(REAL x0, REAL x1) {
	return (x0 + x1) / 2;
}

/* d rho / dt  = - d/ds(rho u) 					*/
/* 			   = -rho du/ds - u (d rho / ds)	*/
template<typename REAL> __device__ REAL density_update(REAL rho_9, REAL rho_0,
		REAL rho_1, REAL u_9, REAL u_0, REAL ds, REAL dt) {

	/* -rho du/ds */
	REAL t1 = -rho_0 * D(u_9, u_0, ds);

	/*  - u (d rho / ds) */
	REAL t2 = -uD(ave(u_9, u_0), rho_9, rho_0, rho_1, ds);

	return rho_0 + dt * (t1 + t2);

}

/* du/dt = -u du/ds - (1/rho) dp/ds + (1/rho) d/ds((4/3) mu du/ds) */
/* c2 = k/m */
/* c3 = (4/3)*mu_0*/
template<typename REAL> __device__ REAL velocity_update(REAL rho_0, REAL rho_1,
		REAL u_9, REAL u_0, REAL u_1, REAL T_9, REAL T_0, REAL T_1, REAL ds,
		REAL dt) {

	/*-u du/ds*/
	REAL t1 = -uD(u_0, u_9, u_0, u_1, ds);

	/* -(1/rho) dp/ds */
	REAL p_0 = rho_0 * T_0;
	REAL p_1 = rho_1 * T_1;
	REAL t2 = -u_c2 * D(p_0, p_1, ds) / ave(rho_0, rho_1);

	/* (1/rho) d/ds((4/3) mu du/ds) */
	REAL mudu_0 = T_0 * T_0 * sqrt(T_0) * D(u_9, u_0, ds);
	REAL mudu_1 = T_1 * T_1 * sqrt(T_1) * D(u_0, u_1, ds);
	REAL t3 = u_c3 * D(mudu_0, mudu_1, ds) / ave(rho_0, rho_1);

	return u_0 + dt * (t1 + t2 + t3);

}

/* dT/dt = -u dT/ds - (1/c_v) (p/rho) du/ds + (4 mu_0 / (3 c_v)) (T^(5/2)/rho) (du/ds)^2 + (kappa_0/c_v)(1/rho) d/ds(T^5/2 dT/ds) + h / rho */
/* c2 = 1 / c_v 				*/
/* c3 = 4 mu_0 / (3 c_v) 		*/
/* c4 = K_0 / c_v				*/
template<typename REAL> __device__ REAL energy_update(REAL rho_9, REAL rho_0,
		REAL rho_1, REAL u_9, REAL u_0, REAL u_1, REAL T_9, REAL T_0, REAL T_1,
		REAL h_0, REAL ds, REAL dt) {

	/* -u dT/ds */
	REAL t1 = -uD(ave(u_9, u_0), T_9, T_0, T_1, ds);

	/* - (1/c_v) (p/rho) du/ds */
	REAL p_0 = rho_0 * T_0;
	REAL duds = D(u_9, u_0, ds);
	REAL t2 = -T_c2 * (p_0 / rho_0) * duds;

	/* (4 mu_0 / (3 c_v)) (T^(5/2)/rho) (du/ds)^2 */
	REAL T52_0 = T_0 * T_0 * sqrt(T_0);
	REAL t3 = T_c3 * (T52_0 / rho_0) * duds * duds;

	/* (kappa_0/c_v)(1/rho) d/ds(T^5/2 dT/ds) */
	REAL T52_1 = T_1 * T_1 * sqrt(T_1);
	REAL T52_9 = T_9 * T_9 * sqrt(T_9);
	REAL KdT_0 = ave(T52_9, T52_0) * D(T_9, T_0, ds);
	REAL KdT_1 = ave(T52_0, T52_1) * D(T_0, T_1, ds);
	REAL t4 = T_c4 * D(KdT_0, KdT_1, ds);

	/* h / rho  */
	REAL t5 = h_0 / rho_0;

	return T_0 + dt * (t1 + t2 + t3 + t4 + t5);
}

template<typename REAL>
__global__ void hydro_explicit(REAL * rho, REAL * u, REAL * T, size_t pitch) {

	uint j = threadIdx.x;	// Each thread solves for its own spatial index

	__syncthreads();

	/* Main time-marching loop */
	for (int i = 0; i < N_t - 1; i++) {

		/* Branch to deal with boundary conditions */
		if (j == 0) {		// Left-hand boundary

			__syncthreads();

			printf("+++++++++++++++++++++++++++++++\n");

			/*Neumann BCs in density and temperature*/
			rho[(i + 1) * pitch + j] = rho[(i + 1) * pitch + (j + 1)];
			T[(i + 1) * pitch + j] = T[(i + 1) * pitch + (j + 1)];

		} else if (j == N_s - 1) {	// Right-hand boundary

			__syncthreads();

			/* Neumann BCs in density and temperature */
			rho[(i + 1) * pitch + j] = rho[(i + 1) * pitch + (j - 1)];
			T[(i + 1) * pitch + j] = T[(i + 1) * pitch + (j - 1)];

		} else {	// No boundary

			/* Access density elements */
			REAL rho_9 = rho[i * pitch + j - 1];
			REAL rho_0 = rho[i * pitch + j];
			REAL rho_1 = rho[i * pitch + j + 1];

			/* Access velocity elements */
			REAL u_9 = u[i * pitch + j - 1];
			REAL u_0 = u[i * pitch + j];
			REAL u_1 = u[i * pitch + j + 1];

			/* Access temperture elements */
			REAL T_9 = T[i * pitch + j - 1];
			REAL T_0 = T[i * pitch + j];
			REAL T_1 = T[i * pitch + j + 1];

			printf("%d %d %e %e %e\n", i, j, rho_0, u_0, T_0);

			/* Update values */
			rho[(i + 1) * pitch + j] = density_update(rho_9, rho_0, rho_1, u_9,
					u_0, ds, dt);
			u[(i + 1) * pitch + j] = velocity_update(rho_0, rho_1, u_9, u_0,
					u_1, T_9, T_0, T_1, ds, dt);
			T[(i + 1) * pitch + j] = energy_update(rho_9, rho_0, rho_1, u_9,
					u_0, u_1, T_9, T_0, T_1, h, ds, dt);

			__syncthreads();

		}

		__syncthreads();

	}
	//printf("%d\n", j);

}

////////////////////////////////////////////////////////////////////
//                                                                //
// main code to test all solvers for single & double precision    //
//                                                                //
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

	/* Initialize CUDA timing */
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* Configure shared memory configuration*/
	cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeFourByte);

	/* allocate memory for density field */
	size_t rho_pitch_d;
	float *rho_h, *rho_d;
	rho_h = (float *) malloc(N_s * N_t * sizeof(float));	// Host memory
	uint ret = cudaMallocPitch(&rho_d, &rho_pitch_d, N_s * sizeof(float), N_t); // Device memory

	/* allocate memory for velocity field */
	size_t u_pitch_d;
	float *u_h, *u_d;
	u_h = (float *) malloc(N_s * N_t * sizeof(float));	// Host memory
	ret = cudaMallocPitch(&u_d, &u_pitch_d, N_s * sizeof(float), N_t); // Device memory

	/* allocate memory for temperature field */
	size_t T_pitch_d;
	float *T_h, *T_d;
	T_h = (float *) malloc(N_s * N_t * sizeof(float));	// Host memory
	ret = cudaMallocPitch(&T_d, &T_pitch_d, N_s * sizeof(float), N_t); // Device memory

	/* Set the pitch of the host  and device memory */
	size_t pitch_h = N_s * sizeof(float);
	size_t pitch_d = u_pitch_d;

	/* Set up the initial conditions */
	for (uint i = 0; i < 1; i++) {
		for (uint j = 0; j < N_s; j++) {

			rho_h[i * N_s + j] = rho_init;	// Initial density
			u_h[i * N_s + j] = u_init;		// Initial velocity
			T_h[i * N_s + j] = T_init;		// Initial temperture
		}
	}

	/* Copy initial conditions to the device */
	cudaMemcpy2D(rho_d, pitch_d, rho_h, pitch_h, N_s * sizeof(float), 1,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(u_d, pitch_d, u_h, pitch_h, N_s * sizeof(float), 1,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(T_d, pitch_d, T_h, pitch_h, N_s * sizeof(float), 1,
			cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	uint nthreads = N_s;
	uint nblocks = N_loops;

	hydro_explicit<<<nblocks,nthreads>>>(rho_d, u_d, T_d, pitch_d);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);

	/* Copy initial conditions to the device */
	cudaMemcpy2D(rho_h, pitch_h, rho_d, pitch_d, N_s * sizeof(float), N_t,
			cudaMemcpyDeviceToHost);
	cudaMemcpy2D(u_h, pitch_h, u_d, pitch_d, N_s * sizeof(float), N_t,
			cudaMemcpyDeviceToHost);
	cudaMemcpy2D(T_h, pitch_h, T_d, pitch_d, N_s * sizeof(float), N_t,
			cudaMemcpyDeviceToHost);

	printf("%f\n", milli);

// CUDA exit -- needed to flush printf write buffer

	cudaThreadSynchronize();
	cudaDeviceReset();

	FILE * meta_f = fopen("meta.dat", "wb");
	FILE * rho_f = fopen("rho.dat", "wb");
	FILE * u_f = fopen("u.dat", "wb");
	FILE * T_f = fopen("T.dat", "wb");

	fwrite(&N_s, sizeof(uint), 1, meta_f);
	fwrite(&N_t, sizeof(uint), 1, meta_f);

	fwrite(rho_h, sizeof(float), N_s * N_t, rho_f);
	fwrite(u_h, sizeof(float), N_s * N_t, u_f);
	fwrite(T_h, sizeof(float), N_s * N_t, T_f);

	fclose(meta_f);
	fclose(rho_f);
	fclose(u_f);
	fclose(T_f);

	return 0;
}
