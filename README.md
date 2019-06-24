# Auto teacher v1.0

Welcome to Auto teacher v1.0! Auto teacher is an automatic feedback system that integrates computer graphics, machine learning and computer vision, it is designed for the Graphics course C317 at Imperial College London. There are three parts of Auto teacher: Graphics system, ML system and Feedback system.

![Auto teacher](https://github.com/TerrenceCKCHAN/Auto-teacher/blob/master/UIdisplay.png)

## How it works
When a student implements phong shading by writing the glsl code in the fragment shader, we capture the scene rendered in our graphics system and send the image to the server for a prediction. The prediction is done on a pre-trained ML model (A multi-label classification CNN network). Feedback is then generated and displayed based on the classification result of our ML model.


## Getting Started
The application is separated into the client, which contains the graphics system, and the server, which contains the machine learning system and the feedback system. They communicate with each other using HTTP protocol.

### Installing

First, you have to install all packages and library dependencies to run the server

```
cd server
npm install
```

After installing all necessary packages, we can run the server directly by calling:

```
npm start
```
This will automatically start the server on localhost:3001. Next we go to the client/src directory and do the following

```
http-server -o
```
This starts the client and run it on localhost:8080. Now we are ready to edit GLSL code, implement Phong shading and receive feedback!


## Built With

* [Three JS](https://threejs.org/) - The WebGL framework for graphics system
* [Tensorflow JS](https://www.tensorflow.org/js) - Machine Learning in JavaScript
* [Node JS](https://nodejs.org/en/) - Web framework for server

## Authors

* **TerrenceCKCHAN**

## Acknowledgments

Final Year Project - Imperial College London
* Supervisor: Bernhard Kainz
* Second marker: Abhijeet Ghosh
