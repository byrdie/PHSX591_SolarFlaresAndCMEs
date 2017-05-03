////////////////////////////////////////////////////////////////////
//                                                                //
// standard headers plus new one defining tridiagonal solvers     //
//                                                                //
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

/* fundamental physical constants */
const double k_b = 1.38065e-16; // boltzmann constant (erg/K)
const double mbar = 0.593 * 1.6726218e-24; // m-bar (mean mass per particle = 0.593 m_p) (g)
const double c_v = 3 * k_b / (2 * mbar); // specific heat
const double kappa_0 = 1e-6; // Thermal conductivity constant (erg cm^-1 s^-1 K^(-7/2))
const double Pr = 1e-4; // Prandtl number in fully ionized plasma
const double mu_0 = Pr * kappa_0 / c_v; // Dynamic viscosity constant

/* Adjustable physical parameters */
const double L = 53e8; // Length of flux tube      (cm)
const double L_t = 4; // Length of simulation     (s)
const double p_init = 1.0; // Initial pressure      (erg cm^-3)
const double u_init = 0.0; // Initial speed of the plasma       (cm/s)
const double T_init = 2e4; // Initial temperature of the plasma     (K)
const double F = 3.5e10; // Flare energy flux   (erg cm^-2 s^-1)
const double Delta_fl = 10e8; // extent of flare heat flux function (cm)
const float h = 0 * 2 * F / Delta_fl; // flare heating function

/* Transition region parameters */
const double T_chr = 2e4; // Temperature of the chromosphere  (K)
const double T_cor = 2e6; // Temperature of the corona    (K)
const double Delta_tr = 3e8; // Width of the transition region (cm)
const double Delta_chr = 5e8; // Width of the chromosphere    (cm)
const double w = 0.75e8; // width of the equilibrium heating function (cm)
const double A = 1e-20; // Coefficient of equilibrium heating function (erg??)

/* Adjustable simulation parameters */
const uint N_s = 1024; // Number of spatial points
const uint N_k = 1e3; // Number of temporal points to keep
const float ds = L / N_s; // distance between spatial points
const float dt = 1e-6; // distance between temporal points
const uint N_loops = 1; // Number of parallel loops to simulate
const uint N_t = L_t / dt; // total number of temporal points
const uint K = N_t / N_k; // Number of points to skip to keep the correct amount

/* velocity update constants */
/* c2 = k/m */
/* c3 = (4/3)*mu_0*/
const float u_c2 = k_b / mbar;
const float u_c3 = 4 * mu_0 / 3;

/* Temperture update constants */
/* c2 = 1 / c_v*/
/* c3 = 4 mu_0 / (3 c_v)*/
/* c4 = K_0 / c_v*/
/* c5 = h*/
const float T_c2 = 1 / c_v;
const float T_c3 = 4 * mu_0 / (3 * c_v);
const float T_c4 = kappa_0 / c_v;
const float T_c5 = 1 / c_v;

/**
 * Simple model of the transition region
 * @param s, parameterized loop location
 * @return approximate temperature at equilbrium
 */
template<typename REAL> REAL transition_region(REAL s) {

    return T_chr + (T_cor - T_chr) * (tanh(4 * (s - Delta_chr - (Delta_tr - 2 * w)) / Delta_tr) + tanh(4 * (L - s - Delta_chr - (Delta_tr - 2 * w)) / Delta_tr)) / 2;

}

/**
 * Calculates the density as a function of temperature
 * according to the ideal gas law
 * @param T
 * @return 
 */
template<typename REAL> REAL ideal_gas(REAL T) {
    return mbar * p_init / (k_b * T);
}

/**
 * Shape function
 * @param s
 * @return 
 */
template<typename REAL> REAL S(REAL s) {

    if (0 < s && s < 2 * w) {
        REAL x = (s / w - 1);
        return 1 - s * s;
    } else {
        return 0;
    }

}

template<typename REAL> REAL heating_function(REAL s) {

    REAL H_eq = A * (S(s - Delta_tr - Delta_chr) - S(s - Delta_chr) + S(L - Delta_tr - Delta_chr - s) - S(L - s - Delta_chr));

    REAL H_fl;

    if (((L - Delta_fl) / 2 < s) && (s < (L + Delta_fl) / 2)) {
        H_fl = h;
    } else {
        H_fl = 0;
    }

    return H_eq + H_fl;

}

/* Finite Difference Derivative */

/* dx/ds */
template<typename REAL> REAL D(REAL x0, REAL x1, REAL ds) {
    return (x1 - x0) / ds;
}

/* Branchless upwind differencing scheme */
template<typename REAL> REAL uD(REAL u, REAL x9, REAL x0, REAL x1,
        REAL ds) {

    if (u > 0.0f) {
        return (x0 - x9) * u / ds;
    } else {
        return (x1 - x0) * u / ds;
    }

    //    REAL t1 = x0 * abs(u) / ds;
    //    REAL t2 = -(x9 + x1) * abs(u) / (2 * ds);
    //    REAL t3 = -(x9 - x1) * u / (2 * ds);

    //    return t1 + t2 + t3;
}

template<typename REAL> REAL cD(REAL x9, REAL x1, REAL ds) {

    return (x1 - x9) / (2 * ds);

}

template<typename REAL> REAL D2(REAL x9, REAL x0, REAL x1, REAL ds) {

    return (x9 - 2 * x0 + x1) / (ds * ds);

}

template<typename REAL> REAL ave(REAL x0, REAL x1) {
    return (x0 + x1) / 2;
}

/* d rho / dt  = - d/ds(rho u) 					*/

/* 			   = -rho du/ds - u (d rho / ds)	*/
template<typename REAL> REAL density_update(REAL rho_9, REAL rho_0,
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
template<typename REAL> REAL velocity_update(REAL rho_0, REAL rho_1,
        REAL u_9, REAL u_0, REAL u_1, REAL T_9, REAL T_0, REAL T_1, REAL ds,
        REAL dt) {

    /*-u du/ds*/
    REAL t1 = -uD(u_0, u_9, u_0, u_1, ds);

    /* -(1/rho) dp/ds */
    REAL p_0 = rho_0 * T_0;
    REAL p_1 = rho_1 * T_1;
    REAL t2 = -u_c2 * D(p_0, p_1, ds) / ave(rho_0, rho_1);

    /* (1/rho) d/ds((4/3) mu du/ds) */
    //    REAL duds_0 = D(u_9, u_0, ds);
    //    REAL duds_1 = D(u_0, u_1, ds);
    //    REAL t3_1 = (5/2) * T_0
    REAL mudu_0 = T_0 * T_0 * sqrt(T_0) * D(u_9, u_0, ds);
    REAL mudu_1 = T_1 * T_1 * sqrt(T_1) * D(u_0, u_1, ds);
    REAL t3 = u_c3 * D(mudu_0, mudu_1, ds) / ave(rho_0, rho_1);

    return u_0 + dt * (t1 + t2 + t3);

}

/* dT/dt = -u dT/ds - (1/c_v) (p/rho) du/ds + (4 mu_0 / (3 c_v)) (T^(5/2)/rho) (du/ds)^2 + (kappa_0/c_v)(1/rho) d/ds(T^5/2 dT/ds) + h / rho */
/* c2 = 1 / c_v 				*/
/* c3 = 4 mu_0 / (3 c_v) 		*/

/* c4 = K_0 / c_v				*/
template<typename REAL> REAL energy_update(REAL rho_0,
        REAL u_9, REAL u_0, REAL T_9, REAL T_0, REAL T_1,
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
    //    REAL T32_0 = T_0 * sqrt(T_0);
    //    REAL dTds = cD(T_9, T_1, ds);
    //    REAL d2Tds2 = D2(T_9, T_0, T_1, ds);
    //    REAL t4 = T_c4 * (5 * T32_0 * dTds * dTds / 2 + T52_0 * d2Tds2) / rho_0;

    /* (kappa_0/c_v)(1/rho) d/ds(T^5/2 dT/ds) */
    REAL T52_1 = T_1 * T_1 * sqrt(T_1);
    REAL T52_9 = T_9 * T_9 * sqrt(T_9);
    REAL KdT_0 = ave(T52_9, T52_0) * D(T_9, T_0, ds);
    REAL KdT_1 = ave(T52_0, T52_1) * D(T_0, T_1, ds);
    REAL t4 = T_c4 * D(KdT_0, KdT_1, ds) / rho_0;

    /* (1/c_v) h / rho  */
    REAL t5 = T_c5 * h_0 / rho_0;

    return T_0 + dt * (t1 + t2 + t3 + t4 + t5);
}

template<typename REAL> uint hydro_explicit(REAL * h, REAL * rho, REAL * u, REAL * T, size_t pitch) {

    /* Main time-marching loop */
    for (uint k = 0; k < N_k - 1; k++) {

        /* Secondary time-marching loop */
        for (uint l = 0; l < K; l++) {

            /*  */
            if (l != 0) {
                for (uint j = 0; j < N_s; j++) {
                    rho[k * pitch + j] = rho[(k + 1) * pitch + j];
                    u[k * pitch + j] = u[(k + 1) * pitch + j];
                    T[k * pitch + j] = T[(k + 1) * pitch + j];
                }
            }

            /* loop over space */
            for (uint j = 1; j < N_s - 1; j++) {

                /* Access density elements */
                REAL rho_9 = rho[k * pitch + j - 1];
                REAL rho_0 = rho[k * pitch + j];
                REAL rho_1 = rho[k * pitch + j + 1];

                /* Access velocity elements */
                REAL u_9 = u[k * pitch + j - 1];
                REAL u_0 = u[k * pitch + j];
                REAL u_1 = u[k * pitch + j + 1];

                /* Access temperture elements */
                REAL T_9 = T[k * pitch + j - 1];
                REAL T_0 = T[k * pitch + j];
                REAL T_1 = T[k * pitch + j + 1];

                if (j != N_s - 2) {
                    u_0 = velocity_update(rho_0, rho_1, u_9, u_0, u_1, T_9, T_0, T_1, ds, dt);
                    if (isnan(u_0) || isinf(u_0)) {
                        printf("NaN detected\n");
                        return k;
                    }
                    u[(k + 1) * pitch + j] = u_0;
                }


                /* Update values */
                rho_0 = density_update(rho_9, rho_0, rho_1, u_9, u_0, ds, dt);
                if (isnan(rho_0) || isinf(rho_0)) {
                    printf("NaN detected\n");
                    return k;
                }
                rho[(k + 1) * pitch + j] = rho_0;

                T_0 = energy_update(rho_0, u_9, u_0, T_9, T_0, T_1, h[j], ds, dt);
                if (isnan(T_0) || isinf(rho_0)) {
                    printf("NaN detected\n");
                    return k;
                }
                T[(k + 1) * pitch + j] = T_0;


            }

            /* LHS boundary condition */
            uint j = 0;

            /* Dirichlet BCs in density and temperature */
            //        rho[(i + 1) * pitch + j] = ideal_gas(T_init);
            //        T[(i + 1) * pitch + j] = T_init;

            /* Neumann BCs in density and temperature */
            rho[(k + 1) * pitch + j] = rho[(k + 1) * pitch + (j + 1)];
            T[(k + 1) * pitch + j] = T[(k + 1) * pitch + (j + 1)];

            /* RHS boundary condition */
            j = N_s - 1;

            /* Dirichlet BCs in density and temperature */
            //        rho[(i + 1) * pitch + j] = ideal_gas(T_init);
            //        T[(i + 1) * pitch + j] = T_init;

            /* Neumann BCs in density and temperature */
            rho[(k + 1) * pitch + j] = rho[(k + 1) * pitch + (j - 1)];
            T[(k + 1) * pitch + j] = T[(k + 1) * pitch + (j - 1)];
        }
        printf("%d / %d\n", k, N_k);
    }




}



////////////////////////////////////////////////////////////////////
//                                                                //
// main code to test all solvers for single & double precision    //
//                                                                //
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

    /* Initialize CUDA timing */
    float milli;

    /* allocate memory for density field */
    float *rho_h;
    rho_h = (float *) malloc(N_s * N_k * sizeof (float)); // Host memory

    /* allocate memory for velocity field */
    float *u_h;
    u_h = (float *) malloc(N_s * N_k * sizeof (float)); // Host memory

    /* allocate memory for temperature field */
    float *T_h;
    T_h = (float *) malloc(N_s * N_k * sizeof (float)); // Host memory

    /* allocate memory for heating function */
    float *h_h;
    h_h = (float *) malloc(N_s * sizeof (float)); // Host memory

    /* Set the pitch of the host  and device memory */
    size_t pitch_h = N_s;

    /* Set up the initial conditions */
    for (uint j = 0; j < N_s; j++) {

        float T = transition_region(j * ds);
        rho_h[j] = ideal_gas(T); // Initial density
        u_h[j] = u_init; // Initial velocity
        T_h[j] = T; // Initial temperture

        /* Heating function initialization */
        h_h[j] = heating_function(j * ds);
    }




    uint num = hydro_explicit(h_h, rho_h, u_h, T_h, pitch_h);


    printf("%f\n", milli);

    FILE * meta_f = fopen("meta.dat", "wb");
    FILE * rho_f = fopen("rho.dat", "wb");
    FILE * u_f = fopen("u.dat", "wb");
    FILE * T_f = fopen("T.dat", "wb");

    fwrite(&N_s, sizeof (uint), 1, meta_f);
    fwrite(&num, sizeof (uint), 1, meta_f);

    fwrite(rho_h, sizeof (float), N_s * N_k, rho_f);
    fwrite(u_h, sizeof (float), N_s * N_k, u_f);
    fwrite(T_h, sizeof (float), N_s * N_k, T_f);

    fclose(meta_f);
    fclose(rho_f);
    fclose(u_f);
    fclose(T_f);

    return 0;
}
