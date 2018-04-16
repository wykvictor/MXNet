
adb shell mkdir -p /data/local/tmp/model

adb push example/image-classification/predict-cpp/mobilenet1_0-symbol.json /data/local/tmp/model/
adb push example/image-classification/predict-cpp/mobilenet1_0-0000.params  /data/local/tmp/model/

adb push build/example/image-classification/predict-cpp/image-classification-predict /data/local/tmp/model/

for i in {1..3}; do
	sleep 3
	echo ""
	echo "MXnet mobilenet 2 threads=========="
	adb shell "export OPENBLAS_NUM_THREADS=2 && cd /data/local/tmp/model/ && ./image-classification-predict mobilenet1_0-symbol.json mobilenet1_0-0000.params 224"
	sleep 3
	echo ""
	echo "MXnet mobilenet 1 threads=========="
	adb shell "export OPENBLAS_NUM_THREADS=2 && cd /data/local/tmp/model/ && ./image-classification-predict mobilenet1_0-symbol.json mobilenet1_0-0000.params 224"
done
