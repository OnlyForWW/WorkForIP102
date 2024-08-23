#include <iostream>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

int main()
{
    ov::Core core;
    vector<string> devices = core.get_available_devices();

    for (int i = 0; i < devices.size(); i++)
    {
        cout << "supported device name: " << devices[i].c_str() << endl;
    }

    Mat img = imread("/home/w/Desktop/install_sh/dog.jpeg");

    // 显示图像
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", img);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
