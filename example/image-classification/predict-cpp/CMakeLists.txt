
# Check OpenCV
if(NOT USE_OPENCV OR NOT OpenCV_FOUND)
  message(WARNING "\
OpenCV should be enabled and found to build image classification example, skipping...android arm64 build: modify not use opencv")
  #return()
endif()


if(NOT MSVC)
  set(IMG_CLASSIFICATION_EXAMPLE_STATIC_LINK ON CACHE BOOL "\
Link mxnet library statically in the c++ image classification example")
else()
  # disable static linking on Windows
  set(IMG_CLASSIFICATION_EXAMPLE_STATIC_LINK OFF)
endif()

# works:/usr/aarch64-linux-android/bin/aarch64-linux-android-clang++ /work/mxnet/example/image-classification/predict-cpp/mxnet-android-test.cpp -fPIE -pie
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pie -fPIE")
add_executable(image-classification-predict mxnet-android-test.cpp)

include_directories(SYSTEM)

if(IMG_CLASSIFICATION_EXAMPLE_STATIC_LINK)
  message(STATUS "android arm64 build: static link mxnet")
  target_link_libraries(image-classification-predict PUBLIC "-static"
                        ${BEGIN_WHOLE_ARCHIVE} mxnet_static ${END_WHOLE_ARCHIVE}
                        dmlc
                        ${mxnet_LINKER_LIBS}
                        )
  add_dependencies(image-classification-predict mxnet_static)
else()
  target_link_libraries(image-classification-predict
                        dmlc
                        ${nnvm_LINKER_LIBS}
                        ${mxnet_LINKER_LIBS}
                        mxnet
                        )
  add_dependencies(image-classification-predict mxnet)
endif()


