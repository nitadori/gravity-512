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

	const float eps = 1./256.;
	const double t0 = omp_get_wtime();
	gravity_simple(N, eps*eps, posm, accp);
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
