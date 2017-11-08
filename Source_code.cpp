

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"


#include "dlib\image_processing\frontal_face_detector.h"
#include "dlib\image_processing\render_face_detections.h"
#include "dlib\image_processing.h"
#include "dlib\gui_widgets.h"
#include "dlib\image_io.h"
#include "dlib\opencv\cv_image.h"

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <stdio.h>
#include "dirent.h"
#include "windows.h"
#include <fstream>
#include <math.h>


using namespace dlib;
using namespace std;
using namespace cv;
using namespace cv::ml;


int face_detection(string emotion)
{
	CascadeClassifier facedetector, facedetector2, facedetector3, facedetector4;
	facedetector.load("c:/opencv-3.2.0/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml");
	facedetector2.load("c:/opencv-3.2.0/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
	facedetector3.load("c:/opencv-3.2.0/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml");
	facedetector4.load("c:/opencv-3.2.0/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml");
	DIR *dir;
	Mat image, gray;
	int filenumber = 0;
	struct dirent *ent;
	string path = "D:\\univ\\kursovaja\\EmotionRecognition\\EmotionRecognition\\sorted_set\\" + emotion;
	
	if ((dir = opendir(path.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (ent->d_name[0] != '.') {
				cout << ent->d_name << endl;
				string full_path = path + "\\" + ent->d_name;
				image = imread(full_path, CV_LOAD_IMAGE_COLOR);
				if (image.rows == 0)
					continue;
				cvtColor(image, gray, CV_RGB2GRAY); //convert to grayscale
													//namedWindow("window1", 1);
													//imshow("window1", grays.back());
													// Detect faces
				std::vector<Rect> faces1, faces2, faces3, faces4, facefeatures;
				facedetector.detectMultiScale(gray, faces1, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
				facedetector2.detectMultiScale(gray, faces2, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
				facedetector3.detectMultiScale(gray, faces3, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
				facedetector4.detectMultiScale(gray, faces4, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));


				if (faces1.size() == 1)
					facefeatures = faces1;
				else if (faces2.size() == 1)
					facefeatures == faces2;
				else if (faces3.size() == 1)
					facefeatures = faces3;
				else if (faces4.size() == 1)
					facefeatures = faces4;

				if (!facefeatures.empty())
				{
					cv::Rect roi(facefeatures[0].x, facefeatures[0].y, facefeatures[0].width, facefeatures[0].height);
					gray = gray(roi);
					Size size(350, 350);
					resize(gray, gray, size);
					string pp = "dataset\\" + emotion;
					pp += "\\" + to_string(filenumber);
					pp += ".jpg";
					imwrite(pp, gray);//Write image
					 //	imshow("fhf", grays.back());
					filenumber++;
				}
			}

		}
		waitKey(0);
		//system("pause");
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return -1;
	}
	return 0;
}

int facial_landmarks(string emotion, std::vector<std::vector<float>> training_data, std::vector<string> &training_label)
{
	DIR *dir;
	int filenumber = 0;
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor sp;
	image_window win, win_faces;
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	ofstream fout(emotion + ".txt");
	struct dirent *ent;
	string path = "D:\\univ\\kursovaja\\EmotionRecognition\\EmotionRecognition\\dataset\\" + emotion;
	if ((dir = opendir(path.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			if (ent->d_name[0] != '.') {
				string full_path = path + "\\" + ent->d_name;
				cout << full_path << endl;
				array2d<rgb_pixel> img;
				load_image(img, full_path);
				std::vector<dlib::rectangle> dets = detector(img);
				cout << "Number of faces detected: " << dets.size() << endl;
				std::vector<full_object_detection> shapes;
				std::vector<int> xlist, ylist;
				std::vector<int> xcentral, ycentral;
				int xmean=0, ymean=0, anglenose;
				for (unsigned long j = 0; j < dets.size(); ++j)
				{
					full_object_detection shape = sp(img, dets[j]);
					cout << "number of parts: " << shape.num_parts() << endl;
					for (int i = 0; i < shape.num_parts(); i++)
					{
						xlist.push_back(int(shape.part(i).x()));
						ylist.push_back(int(shape.part(i).y()));
						xmean += shape.part(i).x();
						ymean += shape.part(i).y();
						
					};
					xmean /= xlist.size();
					ymean /= ylist.size();
					for (auto x : xlist)
						xcentral.push_back(x - xmean);
					for (auto y : ylist)
						ycentral.push_back(y - ymean);
					if (xlist[26] == xlist[29])
						anglenose = 0;
					else
						anglenose = int(atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * 180 / 3.14159265358979323846);
					if (anglenose < 0)
						anglenose += 90;
					else
						anglenose -= 90;
					std::vector <float> vec_landmarks;
					std::vector<std::vector<int>>  meannp, coornp;
					for (int k = 0; k < xlist.size(); k++)
					{
						vec_landmarks.push_back(xcentral[k]);
						vec_landmarks.push_back(ycentral[k]);

						int dist = sqrt(pow((ylist[k] - ymean), 2) + pow((xlist[k] - xmean), 2));
						vec_landmarks.push_back(dist);
						int anglerelative;
						if (xlist[k] == xmean)
							anglerelative = 0;
						else
						 anglerelative = (atan((ylist[k] - ymean) / (xlist[k] - xmean)) * 180 / 3.14159265358979323846) - anglenose;
						vec_landmarks.push_back(anglerelative);
						fout << xcentral[k] << " " << ycentral[k] << " " << dist << " " << anglerelative << " ";
					}
					training_data.push_back(vec_landmarks);
					training_label.push_back(emotion);
					shapes.push_back(shape);	
					fout << "\n";
				}
				// To view face poses on the screen.
				win.clear_overlay();
				win.set_image(img);
				win.add_overlay(render_face_detections(shapes));
				dlib::array<array2d<rgb_pixel> > face_chips;
				extract_image_chips(img, get_face_chip_details(shapes), face_chips);
				
				win_faces.set_image(tile_images(face_chips));
				cout << "Hit enter to process the next image..." << endl;
				cin.get();
			}
		}
		waitKey(0);
		fout.close();
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return -1;
	}
	return 0;
}

void getSVMParams(SVM *svm)
{
	cout << "Kernel type     : " << svm->getKernelType() << endl;
	cout << "Type            : " << svm->getType() << endl;
	cout << "C               : " << svm->getC() << endl;
	cout << "Degree          : " << svm->getDegree() << endl;
	cout << "Nu              : " << svm->getNu() << endl;
	cout << "Gamma           : " << svm->getGamma() << endl;
}

int main()
{
	string emotions[8] = { "neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise" };
	/*for (int i = 1; i < 8; i++)
	{
		face_detection(emotions[i]);
	}*/
	face_detection("test");
	std::vector<std::vector<float>> testing_data;//change to int
	std::vector<string> testing_labels, training_labels;
	std::vector<std::vector<float>>train;//change to int
	std::vector<int> labels;
	facial_landmarks("test", testing_data, testing_labels);
	//for (int i = 1; i < 8; i++)
	//{
	//	facial_landmarks(emotions[i], train, training_labels);
	//}
	
	for (int i = 0; i < 8; i++)
	{
		ifstream inputFile(emotions[i] + ".txt");
		if (inputFile) {
			string str;
			while (getline(inputFile, str))
			{
				stringstream iss(str);
				int number;
				std::vector<float> row;//change to int
				while (iss >> number)
					row.push_back(number);
				if (row.size() == 272)
				{
					train.push_back(row);
					labels.push_back(i);
				}

			}
			inputFile.close();
		}
	}

	ifstream inputFile( "test.txt");
	if (inputFile) {
		string str;
		while (getline(inputFile, str))
		{
			stringstream iss(str);
			int number;
			std::vector<float> row;//change to int
			while (iss >> number)
				row.push_back(number);
			testing_data.push_back(row);
		}
		inputFile.close();
	}
	float tr[552][272];
	float t[3][272];//	CHANGE
	int l[552];
	for (int i = 0; i <552; i++)
	{
		std::copy(train[i].begin(), train[i].end(), tr[i]);
	}

	std::copy(labels.begin(), labels.end(), l);
	//for (int i = 0; i < 552; i++)
	//	l[i] = 0;
	Mat labelsMat(labels.size(), 1, CV_32SC1, l);
	Mat trainingDataMat(train.size(), train[0].size(), CV_32FC1, tr);
	for (int i = 0; i < 3; i++)//CHANGE
	{
		std::copy(testing_data[i].begin(), testing_data[i].end(), t[i]);
	}
	Mat testingMat(testing_data.size(), testing_data[0].size(), CV_32FC1, t);


	Ptr<ml::SVM> svm = ml::SVM::create();

	svm->setC(12.5);
	// Set parameter Gamma
	svm->setGamma(0.50625);
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
	// Train SVM on training data 
	Ptr<ml::TrainData> td = ml::TrainData::create(trainingDataMat, ml::ROW_SAMPLE,labelsMat);
	svm->train(td);
	svm->save("model4.xml");

	Mat testResponse;
	std::vector <float> arr;

		svm->predict(testingMat, testResponse, true);		
		if (testResponse.isContinuous()) {
			arr.assign((float*)testResponse.datastart, (float*)testResponse.dataend);
		}
		else {
			for (int i = 0; i < testResponse.rows; ++i) {
				arr.insert(arr.end(), (float*)testResponse.ptr<uchar>(i), (float*)testResponse.ptr<uchar>(i) + testResponse.cols);
			}
		}

		DIR *dir;
		int filenumber = 0;
		struct dirent *ent;
		int i = 0;
		image_window win;
		string path = "D:\\univ\\kursovaja\\EmotionRecognition\\EmotionRecognition\\dataset\\test";
		if ((dir = opendir(path.c_str())) != NULL) {
			while ((ent = readdir(dir)) != NULL) {
				if (ent->d_name[0] != '.') {
					string full_path = path + "\\" + ent->d_name;
					array2d<rgb_pixel> image;
					//imread(full_path);
					load_image(image, full_path);
					//putText(image, emotions[static_cast<int>(arr[i])], cvPoint(30, 30),
					//	FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
					win.set_image(image);
					//imshow("window", image);
					cout << emotions[static_cast<int>(arr[i])];
					float confidence = 1.0 / (1.0 + exp(-arr[i]));
					i++;
					cout << "-------confidence-------->" << confidence << endl;

					cout << "Hit enter to process the next image..." << endl;
					cin.get();
					
				}

			}
			waitKey(0);
			closedir(dir);
			
		}
		//getSVMParams(svm);
		
		

		/*	std::vector <std::vector <float>> trainingData = { { 23, 10 },{ 2555555, 1000000 },{ 501, 255 },{ -1, -1} };
	float a[4][2];
	for (int i = 0; i < 4; i++)
	{
		std::copy(trainingData[i].begin(), trainingData[i].end(), a[i]);
	}
	Mat trainingDataMat(trainingData.size(), trainingData[0].size(), CV_32FC1, a);
	std::vector <int> labels = { 1, -1, -5, 3 };
	int b[4];
	std::copy(labels.begin(), labels.end(), b);
	Mat labelsMat(labels.size(), 1, CV_32SC1, b);
	std::vector <std::vector <float>> testingData= { {14565545, 45665456}, {-2, -1} };
	float c[2][2];
	for (int i = 0; i < 2; i++)
	{
		std::copy(testingData[i].begin(), testingData[i].end(), c[i]);
	}
	Mat testingMat(testingData.size(), testingData[0].size(), CV_32FC1, c);*/


	waitKey(0);
	system("pause");
	return 0;
}
