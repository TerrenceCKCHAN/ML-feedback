const express = require('express')
const app = express()
const bodyParser = require('body-parser')   
const port = 3001
const fs = require('fs')
const jimp = require('jimp');
const tf = require('@tensorflow/tfjs-node');
const {createCanvas, loadImage} = require('canvas');
const obj = require('./teapot-claraio.json')

app.use(bodyParser.json())
app.use(bodyParser.urlencoded({
    extended: true
}));

app.get('/', (req, res) => res.send(obj))
app.get('/terrence', (req, res) => res.send("hi noel"))

app.post("/post", (req, res) => {
    storeTrainingData(req.body);
    res.send("Training data!");
}) 

var trainingSize = 1000;
var trainingId = 0;
var trainingClass = '7';
// Store all training data from front end
function storeTrainingData(body) {
    console.log(trainingId);
    if (trainingId == trainingSize - 1) {
        console.log("Generation of training data complete");
    }
    if (trainingId < trainingSize) {
        // parse and save photo to the server
        // handlePhoto(body);
        trainingId += 1;
    }
}
// Helper function - parse and write image data to files
function handlePhoto(body) {
    var stream = parsePhoto(Object.keys(body)[0]);
    // console.log(stream);
    var buf = new Buffer.from(stream, 'base64');
    fs.writeFile('./training_data/exp3/raw/class_' + trainingClass + '/training_' + trainingId + '.png', buf, (err) => {
        if (err) {console.log(err);} else {console.log("Success")}
    })
}
// Helper function - image parser
function parsePhoto(body) {
    var photoStr = body.substring(1, body.length - 1);
    photoStr     = photoStr.substring(photoStr.indexOf(":") + 2, photoStr.length - 1);
    photoStr     = photoStr.replace(/^data:image\/\w+\-\w+;base64,/, "");
    // Replaces all white spaces globally with +
    photoStr     = photoStr.replace(/\s/g, "+");
    return photoStr;
}




// Handling User GLSL shader post request and return corresponding feedback
app.post("/glsl", (req, res) => {
    saveGLSL(req.body)
    .then((data) => modelPredict(data.loadedModel, data.image))
    .then((arr) => feedbackMap(arr))
    .then((feedback) => {
        console.log(feedback);
        res.setHeader("Access-Control-Allow-Origin", "*");
        var json = { 
            feedback: feedback, 
        };
        res.json(json);
    })
});


// Save and preprocess user GLSL
async function saveGLSL(body) {

    var stream = parsePhoto(Object.keys(body)[0]);
    var buf = new Buffer.from(stream, 'base64');
    await fs.writeFile('./glsl/raw/' + 'user' + '.png', buf, (err) => {
        if (err) {console.log(err);} else {console.log("Success")}
    })

    await preprocessGLSL();
    const loadedModel = await tf.loadLayersModel('file://./models/finalmodel/model.json');
    // loadedModel.summary();
    // console.log(loadedModel.summary());
    const image = await loadImage('./glsl/export/user.png');

    return {loadedModel, image};
}
// Predict User GLSL shading using pre-trained model
async function modelPredict(loadedModel, image) {
    const dataArray = [];
    const canvas = createCanvas(28,28);
    const cx = canvas.getContext('2d');
    cx.drawImage(image, 0, 0);
    const tensorObj = tf.browser.fromPixels(canvas, 3);
    const values = tensorObj.dataSync();
    const arr = Array.from(values);
    dataArray.push(arr);

    const xDims = dataArray[0].length;
    const xs = tf.tensor2d(dataArray, [1, xDims]);
    console.log(xs);
    const xsre = xs.reshape([1,28,28,3]);
    console.log(xsre);

    // console.log(arr);
    const pred = await loadedModel.predict(xsre).dataSync();
    const predArr = Array.from(pred);
    console.log(predArr);
    const predOnes = [0,0,0,0,0,0,0,0,0,0,0,0];
    for (let i = 0; i < predArr.length; ++i) {
        if (predArr[i] > 0.5) {
            predOnes[i] = 1;
        }
    }
    console.log(predOnes);
    return predOnes;
}
// Create arrays of feedback
async function feedbackMap(arr) {
    const labelArr = [];
    const sugArr = [];
    for (let i = 0; i < arr.length; ++i) {
        if (arr[i] == 1) {
            const {label, suggestion} = findMapping(i);
            labelArr.push(label);
            sugArr.push(suggestion); 
        }
    }
    console.log(labelArr);
    console.log(sugArr);
    return {labelArr,sugArr};
}
// Function for feedback mapping
function findMapping(num) {
    var label;
    var suggestion;
    switch (num) {
        case 0:
            label = "correct phong shading";
            suggestion = "Your implementation is correct, congratulations!";
            break;
        case 1:
            label = "normal vector not normalized";
            suggestion = "Seems like the normal vector in specular reflection is not normalized.";
            break;
        case 2:
            label = "incorrect specular lighting";
            suggestion = "Try to implement the specular lighting! Get started by looking at the Phong equation.";
            break;
        case 3:
            label = "incorrect diffuse lighting";
            suggestion = "Seems like diffuse lighting is missing, have a look at the phong equation and implement it!";
            break;
        case 4:
            label = "light vector not normalized";
            suggestion = "Remember to normalize the lighting when using the light vector!";
            break;
        case 5:
            label = "light direction vector is reversed";
            suggestion = "Light seems to be appearing from the back of the teapot, the light direction vector probably need a little fix";
            break;
        case 6:
            label = "viewing direction vector not normalized";
            suggestion = "The viewing vector doesn't seem to be normalized!";
            break;
        case 7:
            label = "diffuse and specular calculations are swapped";
            suggestion = "Did you accidentally put the diffuse calculation in the specular term and did the same the other way round?";
            break;
        case 8:
            label = "ambient term and specular term are swapped";
            suggestion = "Look, the specular lighting is coloured! I think you switched the ambient and specular lighting colour";
            break;
        case 9:
            label = "incorrect implementation of shininess";
            suggestion = "It is too shinny! Tune down the specular lighting by including the shininess term in the specular lighting";
            break;
        case 10:
            label = "diffuse and specular calculations are mixed up";
            suggestion = "Remember it is the normal and light direction that is involved in the diffuse term, and we use the viewing direction and reflection direction to calculate the specular lighting, don't mix them up!!";
            break;
        case 11:
            label = "ambient term is missing"
            suggestion = "Hmmm I don't see any ambient lighting, I am pretty sure the diffuse lighting is missing as well otherwise I won't be able to tell the ambient is missing";
    }
    return {label, suggestion};
}

// Helper function to preprocess user glsl image
async function preprocessGLSL() {

    let imgRaw = './glsl/raw/user.png';
    let imgActive = './glsl/active/user.png';
    let imgExported = './glsl/export/user.png';

    await jimp.read(imgRaw)
    .then(img => (img.clone().write(imgActive)))
    .then(() => jimp.read(imgActive))
    .then(img => {return img.crop(130,70,250,190);})
    .then(img => img.resize(28,28))
    .then(img => img.write(imgExported))
    .then(() => console.log('Exported file to: ' + imgExported))
    .catch((err) => {console.error(err); preprocessGLSL()});  

    
}

const preprocessor = require('./preprocess.js');
const mlUtils = require('./mlUtils.js');


async function loadModel() { 
    const loadedModel = await tf.loadLayersModel('file://./models/model-exp3/model.json');
    loadedModel.summary();
}

// Experiements on data preprocessing, MLP, CNN and multilabel CNN

// preprocessor.process();

// const mlp = require('./mlpModel.js');
// mlp.trainMLP();

// const cnn = require('./cnnModel.js');
// cnn.trainCNN();

// const multicnn = require('./multi-label-cnnModel.js');
// multicnn.trainMultiLabelCNN();

// const mlpf = require('./mlpModelF.js');
// mlpf.trainMLP();

// const cnnf = require('./cnnModelF.js');
// cnnf.trainCNN();

// const multicnnf = require('./multi-label-cnnModelF.js');
// multicnnf.trainMultiLabelCNN();

module.exports = app;



