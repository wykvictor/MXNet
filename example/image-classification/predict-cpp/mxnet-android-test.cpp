
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <sys/time.h>
#include <unistd.h>

// #include <opencv2/opencv.hpp>
// Path for c_predict_api
#include <mxnet/c_predict_api.h>

inline double tic() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1e9;
}

double KSGetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double mt1 = tv.tv_sec * 1000 + tv.tv_usec / 1000.0;
    return mt1;
}

// Read file to buffer
class BufferFile {
 public :
  std::string file_path_;
  std::size_t length_ = 0;
  std::unique_ptr<char[]> buffer_;

  explicit BufferFile(const std::string& file_path)
    : file_path_(file_path) {

    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      std::cerr << "Can't open the file. Please check " << file_path << ". \n";
      return;
    }

    ifs.seekg(0, std::ios::end);
    length_ = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

    buffer_.reset(new char[length_]);
    ifs.read(buffer_.get(), length_);
    ifs.close();
  }

  std::size_t GetLength() {
    return length_;
  }

  char* GetBuffer() {
    return buffer_.get();
  }
};

int main(int argc, char* argv[]) {
  std::cout << "Test begin." << std::endl;

  // Models path for your model, you have to modify it
  if (argc < 3) {
    std::cout << "Please input the 2 model files: json and params." << std::endl;
    return -1;
  }
  std::string json_file(argv[1]);
  std::string param_file(argv[2]);

  BufferFile json_data(json_file);
  BufferFile param_data(param_file);
  if (json_data.GetLength() == 0 || param_data.GetLength() == 0) {
    return EXIT_FAILURE;
  }

  // Image size and channels
  int width = 224;
  int height = 224;
  int channels = 3;
  if(argc == 4) {
    width = height = atoi(argv[3]);
  }

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(height),
                                        static_cast<mx_uint>(width) };
  PredictorHandle pred_hnd = nullptr;
  // Create Predictor
  // Parameters
  int dev_type = 1;  // 1: cpu, 2: gpu
  int dev_id = 0;  // arbitrary.
  mx_uint num_input_nodes = 1;  // 1 for feedforward
  const char* input_key[1] = { "data" }; // The name of input argument. For feedforward net, this is {"data"}
  const char** input_keys = input_key;
  MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
               static_cast<const char*>(param_data.GetBuffer()),
               static_cast<int>(param_data.GetLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               &pred_hnd);
  if(pred_hnd == NULL) {
    std::cerr << "Error creating pred hnd." << std::endl;
    return 1;
  }


  // Input Data
  auto image_size = static_cast<std::size_t>(width * height * channels);
  std::vector<mx_float> image_data(image_size, 1);
  // Set Input Image
  MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));

  // Do Predict Forward
  if(MXPredForward(pred_hnd) != 0) {
    std::cerr << "Mxnet Forward error" << std::endl;
    return 1;
  }

  // Get Output Result
  mx_uint output_index = 0;
  mx_uint* shape = nullptr;
  mx_uint shape_len;
  MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

  std::size_t size = 1;
  std::cout << "MXPredGetOutputShape:" << std::endl;
  for (mx_uint i = 0; i < shape_len; ++i) { 
    size *= shape[i];
    std::cout << shape[i] << " ";
  }
  std::cout << std::endl;
  std::vector<float> data(size);

  // std::cout << "\nMXPredGetOutput:" << std::endl;
  // MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));
  // for (std::size_t i = 0; i < data.size(); ++i) {
  //   if(i < 32 || i >= data.size() - 32) {
  //     std::cout << " " << std::setprecision(8) << data[i];
  //     if(i % 6 == 0)   std::cout << std::endl;
  //   }
  // }

  int num_warmup = 30, num_test = 80;
  for(int i=0; i<num_warmup; i++) {
    MXPredForward(pred_hnd);
  }
  MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));
  double start = KSGetTime();
  std::vector<double> times(num_test);
  for(int i=0; i<num_test; i++) {
    start = KSGetTime();
    MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));
    MXPredForward(pred_hnd);
    // need to sync the forward
    MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));
    times[i] = KSGetTime() - start;
    usleep(20000);
  }
  for(int i=0; i<num_test; i++)
    std::cout << times[i] << " ";
  std::sort(times.begin(), times.end());
  double sum = std::accumulate(times.begin()+5, times.end()-5, 0);  // 掐头去尾
  std::cout << "\nmin: " << times[0] << " max: " << times[num_test-1] << " avg: " << sum / (num_test-10) << " ms" << std::endl;


  // Release Predictor
  MXPredFree(pred_hnd);

  return 0;
}
