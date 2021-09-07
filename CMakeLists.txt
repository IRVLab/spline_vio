PROJECT(spline_vio)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

set(BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 
find_package(Pangolin 0.2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
	geometry_msgs
	roscpp
	rosbag
	sensor_msgs
	cv_bridge
	image_transport
)

catkin_package()

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
	"${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
)

include_directories(
	src
	thirdparty/Sophus
	thirdparty/sse2neon
	${EIGEN3_INCLUDE_DIR}
	${catkin_INCLUDE_DIRS}
	${Pangolin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${CSPARSE_INCLUDE_DIR} 
	${CHOLMOD_INCLUDE_DIR}
) 

add_library(spline_vio_lib
	src/FullSystem/FullSystem.cpp
	src/FullSystem/FullSystemOptimize.cpp
	src/FullSystem/FullSystemOptPoint.cpp
	src/FullSystem/FullSystemDebugStuff.cpp
	src/FullSystem/FullSystemMarginalize.cpp
	src/FullSystem/Residuals.cpp
	src/FullSystem/CoarseInitializer.cpp
	src/FullSystem/CoarseTracker.cpp
	src/FullSystem/ImmaturePoint.cpp
	src/FullSystem/HessianBlocks.cpp
	src/FullSystem/PixelSelector2.cpp
	src/OptimizationBackend/EnergyFunctional.cpp
	src/OptimizationBackend/AccumulatedTopHessian.cpp
	src/OptimizationBackend/AccumulatedSCHessian.cpp
	src/OptimizationBackend/EnergyFunctionalStructs.cpp
	src/util/settings.cpp
	src/util/Undistort.cpp
	src/util/globalCalib.cpp
	src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
	src/IOWrapper/Pangolin/PangolinDSOViewer.cpp
	src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp
)

add_executable(spline_vio_node src/main.cpp)

target_link_libraries(spline_vio_node
	spline_vio_lib 
	${catkin_LIBRARIES} 
	${BOOST_THREAD_LIBRARY} 
	${Pangolin_LIBRARIES} 
	${OpenCV_LIBS} 
	boost_system boost_thread cxsparse)

