#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include "detect_face.hpp"

 
  using namespace cv;
  using namespace std;
 
  int main() {
      
      int option;
      while(option !=3){
          cout<< "**** Facial Recognition System ****"<<endl;
          cout<<"\n Choose one of the following options:"<<endl;
          cout<<"1: Open cam and detect face"<<endl;
          cout<<"2: Add face"<<endl;
          cout<<"3: Quit"<<endl;
          cout<<"option: ";
          cin>>option;
      
          switch (option) {
              case 1:
                  detect_face();
                  break;
                  
              case 2:
                  take_picture();
                  break;
                  
              case 3:
                  cout<<"Successfully quit"<<endl;
                  break;
          }
      }
      
    return 0;
   }
