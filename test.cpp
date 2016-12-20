#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>
#include <omp.h>

struct PosM{
	float x, y, z;
	float m;
};

struct AccP{
	float ax, ay, az;
	float pot;
};

void gravity_simple(
		const int N,
		const float eps2,
		const PosM * __restrict posm,
		AccP * __restrict accp)
{
#pragma omp parallel for
#pragma simd
	for(int i=0; i<N; i++){
		float ax=0.0f;
		float ay=0.0f;
		float az=0.0f;
		float po=0.0f;

		float xi = posm[i].x;
		float yi = posm[i].y;
		float zi = posm[i].z;

		for(int j=0; j<N; j++){
			float dx = xi - posm[j].x;
			float dy = yi - posm[j].y;
			float dz = zi - posm[j].z;

			float r2 =  eps2 + dx*dx + dy*dy + dz*dz;
			float ri = 1.0f / sqrtf(r2);

			float mri  = posm[j].m * ri;
			float ri2  = ri * ri;
			float mri3 = mri * ri2;

			ax -= mri3 * dx;
			ay -= mri3 * dy;
			az -= mri3 * dz;
		}

		accp[i].ax = ax;
		accp[i].ay = ay;
		accp[i].az = az;
	}
}

void gravity_avx512(
		const int N,
		const float seps2,
		const PosM * __restrict posm,
		AccP * __restrict accp)
{
#pragma omp parallel for
	for(int i=0; i<N; i+=16){
		__m512 ax = _mm512_set1_ps(0.0f);
		__m512 ay = _mm512_set1_ps(0.0f);
		__m512 az = _mm512_set1_ps(0.0f);

		__m512i vindex = _mm512_set_epi32(
				60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0);
		__m512 xi = _mm512_i32gather_ps(vindex, &posm[i].x, 4);
		__m512 yi = _mm512_i32gather_ps(vindex, &posm[i].y, 4);
		__m512 zi = _mm512_i32gather_ps(vindex, &posm[i].z, 4);

		__m512 eps2 = _mm512_set1_ps(seps2);

		for(int j=0; j<N; j++){
			__m512 dx = _mm512_sub_ps(xi, _mm512_set1_ps(posm[j].x));
			__m512 dy = _mm512_sub_ps(yi, _mm512_set1_ps(posm[j].y));
			__m512 dz = _mm512_sub_ps(zi, _mm512_set1_ps(posm[j].z));

			__m512 r2 = _mm512_fmadd_ps(dx, dx, eps2);
			r2 = _mm512_fmadd_ps(dy, dy, r2);
			r2 = _mm512_fmadd_ps(dz, dz, r2);

			__m512 ri = _mm512_rsqrt28_ps(r2);

			__m512 mri  = _mm512_mul_ps(ri, _mm512_set1_ps(posm[j].m));
			__m512 ri2  = _mm512_mul_ps(ri, ri);
			__m512 mri3 = _mm512_mul_ps(mri, ri2);

			ax = _mm512_fnmadd_ps(mri3, dx, ax);
			ay = _mm512_fnmadd_ps(mri3, dy, ay);
			az = _mm512_fnmadd_ps(mri3, dz, az);
		}

		_mm512_i32scatter_ps(&accp[i].ax, vindex, ax, 4);
		_mm512_i32scatter_ps(&accp[i].ay, vindex, ay, 4);
		_mm512_i32scatter_ps(&accp[i].az, vindex, az, 4);
	}
}

int main(){
	enum{
		N = 16 * 1024,
	};

	static PosM posm[N];
	static AccP accp[N];

	srand48(20161220);

	for(int i=0; i<N; i++){
		posm[i].x = drand48() - 0.5;
		posm[i].y = drand48() - 0.5;
		posm[i].z = drand48() - 0.5;
		posm[i].m = drand48() * (1.0/N);
	}

#pragma omp parallel
	{
		// dry run;
	}
	// dry run;
	const float eps = 1./256.;
	{
		gravity_simple(N, eps*eps, posm, accp);
	}

	const double t0 = omp_get_wtime();
	// gravity_simple(N, eps*eps, posm, accp);
	gravity_avx512(N, eps*eps, posm, accp);
	const double t1 = omp_get_wtime();

	{
		double nintr = N/(t1-t0) * N;
		double gflops = 38.e-9 * nintr;
		printf("%e intr/s, %f Gflops\n", nintr, gflops);
	}

	double fx=0.0, fy=0.0, fz=0.0;
	for(int i=0; i<N; i++){
		double m = posm[i].m;
		fx += m * accp[i].ax;
		fy += m * accp[i].ay;
		fz += m * accp[i].az;
	}
	printf("(%e %e %e)\n", fx, fy, fz);

	return 0;
}
