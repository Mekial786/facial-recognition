//
//  detect_face.cpp
//  Facial Recognition
//
//  Created by Mekial Khan on 17/01/2023.
//

#include "detect_face.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int person_no {0};
map<string, int> person_faces;

void detect_face (){
    
    string name;
    int name_label;
    
    ifstream infile;
    infile.open("faces.txt");
    if(!infile){
        std::cout<< "No Data"<<endl;
    }
    else{
        while (infile>>name>>name_label){
            person_faces.insert(pair<string, int>(name, name_label));
        }
        infile.close();
    }
    
    Ptr<cv::face::FaceRecognizer>  model = cv::face::LBPHFaceRecognizer::create();
    model->read("LBPHFaceRecognizer.xml");
    
    CascadeClassifier faceCascade;
    faceCascade.load("haarcascade_frontalface_default.xml");
    
    VideoCapture cap (0);
    
    Mat grayScale;
    Mat img;
    
    while(true){
        
        cap.read(img);
        cvtColor(img, grayScale, COLOR_BGR2GRAY);
        
        vector <Rect> faces;
        faceCascade.detectMultiScale(grayScale, faces, 1.5, 5);
        for (int i =0; i<faces.size(); i++){
            rectangle(img, faces [i].tl(), faces [i].br(), Scalar(0, 255, 0), 3 );
            
            Mat roi;
            Mat face_resized;
            
            roi = grayScale(Rect(faces[i].tl(), faces[i].br()));
            resize(roi, face_resized, Size(128, 128), 1.0, 1.0, INTER_CUBIC);
            
            int label {-1}; double confidence {0}; string identity;
            model->predict(face_resized, label, confidence);
            double confidence_percentage = (1 - (confidence / (128*128))) * 100;
            for(auto it = person_faces.begin(); it != person_faces.end(); it++){
                if (it->second == label) {
                    identity = it->first;
                    break;
                }else{
                    identity = "No identity found";
                }
            }
            putText(img,"Match Found: " + identity, Point(30, 60), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
            putText(img, "Accuracy: "  + to_string(confidence_percentage) + "%", Point(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
                            
        }
        imshow("Facial Recognition", img);
        
        char key = (char) waitKey(1);
        if (key == 'q'){
            cap.release();
            destroyAllWindows();
            break;
        }
        waitKey(1);
    }
}

void take_picture(){
    CascadeClassifier faceCascade;
    faceCascade.load("haarcascade_frontalface_default.xml");
    string name;
    int picture_no {1};
    
    ifstream infile;
    int name_label;
    infile.open("faces.txt");
    if(!infile){
        std::cout<< "No file found"<<endl;
    }
    else{
        while (infile>>name>>name_label){
            person_faces.insert(pair<string, int>(name, name_label));
            if (name_label>=person_no){
                person_no = name_label + 1;
            }
        }
        infile.close();
    }
    
    cout << "\nEnter Your First Name:  ";
    cin >> name;

fs::path dir("Images/" + name);

if (!fs::exists(dir)) {
    fs::create_directory("Images");
    fs::create_directory(dir);
    person_faces.insert(pair<string, int>(name, person_no));
    
    ofstream outfile;
    outfile.open("faces.txt");
    for (auto itr=person_faces.begin(); itr!=person_faces.end(); itr++){
        outfile<<itr->first<< " "<< itr->second<<endl;
    }
    outfile.close();
    person_no++;
} else {
    picture_no = count_files ("Images/" + name);
}
    Mat grayScale;
    Mat img;
    Mat roi;
    
    VideoCapture cap (0);
    
    while(true){
        cap.read(img);
        cvtColor(img, grayScale, COLOR_BGR2GRAY);
        
        vector <Rect> faces;
        faceCascade.detectMultiScale(grayScale, faces, 1.5, 5);
        for (int i =0; i<faces.size(); i++){
            rectangle(img, faces [i].tl(), faces [i].br(), Scalar(0, 255, 0), 3 );
            roi = grayScale(Rect(faces[i].tl(), faces[i].br()));
        }
        imshow(name, img);
    
        
        char key = (char) waitKey(1);
                if (key == 'c') {
                    imwrite("Images/" + name + "/" + to_string(picture_no) + ".png", roi);
                    picture_no++;
                    
                } else if (key == 'q'){
                    
                        vector<Mat> images;
                        vector<int> labels;
                        
                        for (int i = 1; i<picture_no; i++){
                            Mat res;
                            Mat image = imread("Images/" + name + "/" + to_string(i) + ".png", IMREAD_GRAYSCALE);
                            resize(image, res, Size(128, 128), 0, 0, INTER_LINEAR);
                            images.push_back(res);
                            auto it = person_faces.find(name);
                            labels.push_back(it ->second);
                    }
                    cout << "Number of images: " << images.size() << endl;
                    cout << "Training begins...." << endl;
                    
                    if(fs::exists("LBPHFaceRecognizer.xml")){
                        Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
                        model->read("LBPHFaceRecognizer.xml");
                        model->update(images, labels);
                        model->save("LBPHFaceRecognizer.xml");
                    }
                    else{
                        Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
                        model->train(images, labels);
                        model->save("LBPHFaceRecognizer.xml");
                        
                    }
                    cout << "Training finished...." << endl;
                    detect_face();
                    }
        waitKey(1);
    }
}

int count_files(string path) {
    int count {1};
    for (const auto &entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            count++;
        }
    }
    return count;
}
