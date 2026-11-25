# ROCm machine. Architecture comes from rocminfo, for example:
#   make -f Makefile.hip ARCH=--offload-arch=gfx1100 vti3Dfd
HIPCC ?= hipcc
ARCH ?=
HIPFLAGS := -O3 -Wno-unused-result $(ARCH)

vti2d_FD_RTM: Toa_gpu2DvtiFdRtmAdcigsLaplace.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

vti3Dfd: Toa_gpu_3dvti_fd_1orderfunciton.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

vti3Drtm: Toa_gpu_3dvti_rtm_adcigs_1orderfunction.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

tti2d_acousticFD: Toa_gpu_2dtti_fd_1orderfunction.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

tti2d_acousticRTM: Toa_gpu_2dtti_rtm_adcigs_1orderfunciton.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

tti2d_elasticFD: Toa_gpu_2dtti_fd_elastic.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

clean:
	rm -f vti2d_FD_RTM vti3Dfd vti3Drtm tti2d_acousticFD tti2d_acousticRTM tti2d_elasticFD
