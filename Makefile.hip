# ROCm machine. Architecture comes from rocminfo, for example:
#   make ARCH=--offload-arch=gfx1100
#   make ARCH=--offload-arch=gfx942
HIPCC ?= hipcc
ARCH ?=

vti2d_FD_RTM: Toa_gpu2DvtiFdRtmAdcigsLaplace.hip
	$(HIPCC) -O3 -Wno-unused-result $(ARCH) -o $@ $<

clean:
	rm -f vti2d_FD_RTM
