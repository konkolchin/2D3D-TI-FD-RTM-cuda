# HIP on AMD, with a double-precision CPU reference

This fork runs the RTM programs with HIP on AMD and keeps the CUDA path. The same kernels also build for the CPU. On that CPU path every arithmetic step uses `double`: the wavefield update, the angle, the illumination, and the gather. The GPU kernels stay `float`. Model files and written results stay little-endian float32. A CPU output file is the `double` result rounded to float32 when it is written.

```bash
make -f Makefile.cpu
```

The binaries are `cpu_build/vti2d_FD_RTM` and `cpu_build/tti2d_acousticRTM`.

## Execution time

One shot each, wall-clock seconds from `/usr/bin/time`. The AMD machine is a Radeon RX 7900 XTX (gfx1100). The NVIDIA machine is an RTX 4080. These times are not MI300X times.

| Run | RX 7900 XTX | RTX 4080 |
| --- | ---: | ---: |
| `vti2d` `kind=1` `ns=1` | 0.18 s | 0.33 s |
| `vti2d` `kind=2` | 0.28 s | 0.43 s |
| `vti3Dfd` | 10.73 s | 18.78 s |
| `vti3Drtm` `ns=1` | 16.68 s | 27.80 s |
| `tti2d_acousticFD` | 0.18 s | 0.36 s |
| `tti2d_acousticRTM` `ns=1` | 3.97 s | 4.95 s |
| `tti2d_elasticFD` `ns=1` | 1.02 s | 1.00 s |
| **Total** | **33.04 s** | **53.65 s** |

The Radeon finishes the seven runs in 33.04 s. The RTX 4080 takes 53.65 s. The angle gathers are part of the `kind=2` run and are not timed separately. The elastic modeling run is the exception: 1.02 s on the Radeon and 1.00 s on the RTX 4080.

## Accuracy

The CPU file is the reference. The relative error of a sample is `|GPU − CPU| / |CPU|`. The numbers use samples with `|CPU| ≥ 0.001`. The comparison is the VTI `kind=2` one-shot run against the double-precision CPU result.

| | HIP average | HIP largest absolute difference | CUDA average | CUDA largest absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Angle gathers | 5.7% (0.000488) | 0.143 | 15.1% (0.00133) | 0.256 |
| Zero-lag image | 0.068% (4.92×10⁻⁶) | 0.00606 | 0.074% (5.60×10⁻⁶) | 0.00598 |

HIP stays closer to the CPU reference than CUDA does. On the angle gathers the average relative error is 5.7% for HIP and 15.1% for CUDA, and the largest absolute difference is 0.143 for HIP and 0.256 for CUDA. On the zero-lag image both absolute differences stay near 0.006, and the HIP average is slightly lower.

A relative error of several thousand percent occurs on a gather sample whose CPU value is 0.00119. The absolute difference there is about 0.078. That sample is not the largest absolute difference.

# TI media Finite Different Modeling and Reverse Time Migration

## Aspect

* (2-dimension and 3-dimension)
* anisotropy media (VTI, TTI)
* Reverse Time Migration
* Finite Different Modeling
* Angle Domain Common Imaging Gathers (ADCIGs)
* Poynting vector methods

## Author

[Rong Tao](https://github.com/Rtoax)

## wave equation

![](waveequation.png)

## Migration

![](migration.png)

## poynting vector

![](poyntingvector.png)

## use Laplacian factor tobe a filter decrease the low frequence signal(noise)

```c
/*< linear operator > Copyright(C) Madagascar */
void laplac2_lop(int adj, int nz, int nx, float *in, float *out)
{
    int iz,ix,j;

    for (ix=0; ix < nx; ix++) {
        for (iz=0; iz < nz; iz++) {
            j = iz+ix*nz;

            if (iz > 0) {
            if (adj) {
                out[j-1] -= in[j];
                out[j]   += in[j];
            } else {
                out[j] += in[j] - in[j-1];
            }
            }
            if (iz < nz-1) {
            if (adj) {
                out[j+1] -= in[j];
                out[j]   += in[j];
            } else {
                out[j] += in[j] - in[j+1];
            }
            }

            if (ix > 0) {
            if (adj) {
                out[j-nz] -= in[j];
                out[j]    += in[j];
            } else {
                out[j] += in[j] - in[j-nz];
            }
            }
            if (ix < nx-1) {
            if (adj) {
                out[j+nz] -= in[j];
                out[j]    += in[j];
            } else {
                out[j] += in[j] - in[j+nz];
            }
            }
        }
    }
}
```

## Poynting Vector ADCIGs

```cpp
/**
 *  poynting vector extraction ADCIGs
 *    "Copyright(C) Madagascar:user/pyang/Mrtmadcig.c"
 */
__global__ void Poynting_Adcigs(int nnz,
                                int nx,
                                int nz,
                                int npml,
                                int nangle,
                                int dangle,
                                float *adcigs,
                                float *s_P,
                                float *s_Q,
                                float *s_u,
                                float *s_w,
                                float *g_P,
                                float *g_Q,
                                float *g_u,
                                float *g_w)
{
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    int ix = id/nz;
    int iz = id%nz;

    int ia = 0;

    float Ssx = -s_P[iz+npml+nnz*(ix+npml)]*s_u[iz+npml+nnz*(ix+npml)];
    float Ssz = -s_Q[iz+npml+nnz*(ix+npml)]*s_w[iz+npml+nnz*(ix+npml)];
    float Sgx =  g_P[iz+npml+nnz*(ix+npml)]*g_u[iz+npml+nnz*(ix+npml)];
    float Sgz =  g_Q[iz+npml+nnz*(ix+npml)]*g_w[iz+npml+nnz*(ix+npml)];

    float b1 =  Ssx*Ssx + Ssz*Ssz;
    float b2 =  Sgx*Sgx + Sgz*Sgz;
    float  a = (Ssx*Sgx + Ssz*Sgz)/(sqrtf(b1*b2)*(1 - 0.1));

    if(id<nx*nz) {

        if(a>=-1&&a<=1) {

            a = 0.5*acosf(a)*180.0/pi;
            ia = (int)(a/(dangle*1.0));

            if(ia<nangle) {
                adcigs[iz+nz*ia+nz*nangle*(id/nz)]
                    += s_P[iz+npml+nnz*(ix+npml)]*g_P[iz+npml+nnz*(ix+npml)]
                      *cosf(ia*pi/180.0)*cosf(ia*pi/180.0)*cosf(ia*pi/180.0);
            }
        }
    }
}
```