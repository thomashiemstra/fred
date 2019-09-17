#include <Servo.h>
Servo myservo;
int datafromUser=0;
int pos = 0;

void setup() {
  Serial.begin(9600); 
  myservo.attach(9, 720 , 1240);
  myservo.write(80);
}

void loop() {

  delay(15);
  if(Serial.available() > 0) {
    datafromUser=Serial.read();
  }

  if(datafromUser == '1') {
    myservo.write(80);
  }
  else if(datafromUser == '0') {
    myservo.write(180);
  }
}
