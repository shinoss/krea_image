# Krea 2 Turbo Metal engine.
#   make            -> build/krea.metallib, build/libkrea.dylib
#   make BUILD=dir  -> the same into another directory (development builds next to the installed one)
BUILD := build
KERNELS := $(wildcard kernels/*.metal)
AIRS := $(patsubst kernels/%.metal,$(BUILD)/%.air,$(KERNELS))
SRCS := src/metal.mm src/weights.mm src/model.mm src/vae.mm src/ane.mm src/pipeline.mm
OBJS := $(patsubst src/%.mm,$(BUILD)/%.o,$(SRCS))

CXX := clang++
CXXFLAGS := -O2 -std=c++17 -fobjc-arc -Wall -Wno-unused-function -Wno-missing-braces -DACCELERATE_NEW_LAPACK
FRAMEWORKS := -framework Metal -framework Foundation -framework CoreML -framework Accelerate -framework IOSurface
METALFLAGS := -std=metal3.1 -O3 -ffast-math

all: $(BUILD)/krea.metallib $(BUILD)/libkrea.dylib

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.air: kernels/%.metal kernels/common.h | $(BUILD)
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@

$(BUILD)/krea.metallib: $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@

$(BUILD)/%.o: src/%.mm src/*.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/libkrea.dylib: $(OBJS)
	$(CXX) $(CXXFLAGS) -dynamiclib -install_name @rpath/libkrea.dylib $^ -o $@ $(FRAMEWORKS)

# stage-by-stage validation harness (sources in dev/, not part of the app)
$(BUILD)/krea_test: ../dev/engine/src/test_main.mm ../dev/engine/src/*.h src/*.h $(BUILD)/libkrea.dylib
	$(CXX) $(CXXFLAGS) -Isrc ../dev/engine/src/test_main.mm -o $@ -L$(BUILD) -lkrea -Wl,-rpath,@loader_path $(FRAMEWORKS)

test: $(BUILD)/krea_test

clean:
	rm -rf $(BUILD)

.PHONY: all clean test
