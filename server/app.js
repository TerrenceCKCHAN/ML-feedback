const express = require('express')
const app = express()
const bodyParser = require('body-parser')   
const port = 3001
const fs = require('fs')

const obj = require('./teapot-claraio.json')

app.use(bodyParser.json())
app.use(bodyParser.urlencoded({
    extended: true
}));

app.get('/', (req, res) => res.send(obj))
app.get('/terrence', (req, res) => res.send("hi noel"))

app.post("/post", (req, res) => {storeTrainingData(req.body);res.send("Created training data: " + trainingId)}) 

var trainingSize = 1000;
var trainingId = 0;
var trainingClass = 3;
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

function handlePhoto(body) {
    var stream = parsePhoto(Object.keys(body)[0]);
    // console.log(stream);
    var buf = new Buffer.from(stream, 'base64');
    fs.writeFile('./training_data/class_' + trainingClass + '/training_' + trainingId + '.png', buf, (err) => {
        if (err) {console.log(err);} else {console.log("Success")}
    })
}

function parsePhoto(body) {
    var photoStr = body.substring(1, body.length - 1);
    photoStr     = photoStr.substring(photoStr.indexOf(":") + 2, photoStr.length - 1);
    photoStr     = photoStr.replace(/^data:image\/\w+\-\w+;base64,/, "");
    // Replaces all white spaces globally with +
    photoStr     = photoStr.replace(/\s/g, "+");
    return photoStr;
}


// Image Preprocessing
// Resize and apply normalization to training data
const jimp = require('jimp');


function preprocessPhoto(c, t_id) {
    let imgRaw = 'training_data/raw/class_' + c + '/training_' + t_id + '.png';
    let imgActive = 'training_data/active/class_' + c + '/training_' + t_id + '.png';
    let imgExported = 'training_data/export/class_' + c + '/training_' + t_id + '.png';

    jimp.read(imgRaw)
    .then(img => (img.clone().write(imgActive)))
    .then(() => jimp.read(imgActive))
    .then(img => {return img.resize(100,100);})
    .then(img => {return img.crop(20,25,62,62);})
    .then(img => img.resize(28,28))
    .then(img => img.normalize())
    .then(img => img.grayscale())
    .then(img => img.write(imgExported))
    .then(() => console.log('Exported file to: ' + imgExported))
    .catch(err => console.error(err));
    console.log(c+': '+t_id);    
}

function preprocessPhotos(c) {
    var t_id = 0;
    while(t_id < trainingSize) {
        preprocessPhoto(c, t_id);
        ++t_id;
    }
}

// preprocessPhotos(0)
// preprocessPhotos(1)
// preprocessPhotos(2)
// preprocessPhotos(3)


// Convert Image to Tensors so that we can feed it into our network
const tf = require('@tensorflow/tfjs-core');
const util = require('util')
const {createCanvas, loadImage} = require('canvas');
const CLASSES = ['correct-phong', 'normal-not-normalized', 'no-specular', 'no-diffuse'];
const NUM_CLASSES = CLASSES.length;


async function toTensor() {
    // const image = fs.readFileSync('training_data/export/class_0/training_0.png');
    // var buf = new Buffer.from(image, 'base64');
    // tf.fromPixels(image);
    // console.log(image);
    const canvas = createCanvas(100,100);
    const cx = canvas.getContext('2d');
    const image1 = await loadImage('training_data/export/class_0/training_0.png');
    console.log(image1);
    await cx.drawImage(image1,0,0);
    var tensor = tf.browser.fromPixels(canvas);


    // console.log(util.inspect(await tensor.data(), {maxArrayLength:10,compact:true,showHidden: true, depth: null}));
    


}



async function convertImageToData() {
    var dataArray = [];
    const canvas = createCanvas(28,28);
    const cx = canvas.getContext('2d');
    for (let c = 0; c < trainingClass; ++c) {
        for(let id = 0; id < trainingSize; ++id) {
            const image = await loadImage('training_data/export/class_'+c+'/training_'+id+'.png');
            cx.drawImage(image, 0, 0);
            const tensorObj = tf.browser.fromPixels(canvas, 1);
            const values = tensorObj.dataSync();
            const arr = Array.from(values);
            arr.push(c);
            dataArray.push(arr);
        }
    }
    return dataArray;
    // const canvas = createCanvas(28,28);
    // const cx = canvas.getContext('2d');
    // const image1 = await loadImage('training_data/export/class_0/training_0.png');
    // const image2 = await loadImage('training_data/export/class_0/training_1.png')
    // await cx.drawImage(image1,0,0);
    // var tensor = tf.browser.fromPixels(canvas, 1);
    // const values = tensor.dataSync();
    // const arr = Array.from(values);
    // arr.push(0);
    // // console.log(util.inspect(arr, {maxArrayLength:1000}));
    // dataArray.push(arr);
    // await cx.drawImage(image2,0,0);
    // tensor = tf.browser.fromPixels(canvas, 1);
    // const values1 = tensor.dataSync();
    // const arr1 = Array.from(values1);

    // dataArray.push(arr1);
    // console.log(util.inspect(dataArray, {maxArrayLength:155}));

}

async function gen_train_test_data(split) {
    
    const dataByClass = [];
    const tartgetByClass = [];
    for (let i = 0; i < NUM_CLASSES; ++i) {
        dataByClass.push([]);
        tartgetByClass.push([]);
    }
    const teapotData = await convertImageToData();
    for (const teapot of teapotData) {
        const target = teapot[teapot.length - 1];
        const data   = teapot.slice(0, teapot.length - 1);
        dataByClass[target].push(data);
        targetByClass[target].push(target);
    }

    const xTrains = [];
    const yTrains = [];
    const xTests  = [];
    const yTests  = [];
    for (let c = 0; c < NUM_CLASSES; ++i) {
        const [xTrain,yTrain,xTest,yTest] = 
            convertToTensors(dataByClass[c], targetsByClass[c], split);
        xTrains.push(xTrain);
        yTrains.push(yTrain);
        xTests.push(xTest);
        yTests.push(yTest);
    }


}




const data = [[1,2,3,4],
              [2,2,3,4],
              [3,2,3,4],
              [4,2,3,4],
              [6,2,3,4],
              [7,2,3,4],
              [8,2,3,4],
              [3,2,3,4],
              [5,3,4,2],
              [10,10,10,10]];

const targets = [0,0,0,0,1,1,2,2,3,3];
// const [xtr,ytr,xte,yte] = convertToTensors(data, targets, 0.2);
// console.log(ytr)
// console.log(yte);

function convertToTensors(data, targets, testSplit) {
    const numExamples = data.length;
    if (numExamples !== targets.length) {
        throw new Error('data and target mismatch');
    }

    const numTestExamples = Math.round(numExamples * testSplit);
    const numTrainExamples = numExamples - numTestExamples;

    const xDims = data[0].length;

    const xs = tf.tensor2d(data, [numExamples, xDims]);
    const ys = tf.oneHot(tf.tensor1d(targets).toInt(), NUM_CLASSES);

    // split the data into training and test sets
    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    const xTest  = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
    const yTrain = ys.slice([0, 0], [numTrainExamples, NUM_CLASSES]);
    const yTest  = ys.slice([numTrainExamples, 0], [numTestExamples, NUM_CLASSES]);

    return [xTrain, yTrain, xTest, yTest];

}







// app.listen(port, () => console.log(`Server is running on port ${port}!`))

module.exports = app;









































// var createError = require('http-errors');
// var express = require('express');
// var path = require('path');
// var cookieParser = require('cookie-parser');
// var logger = require('morgan');

// var indexRouter = require('./routes/index');
// var usersRouter = require('./routes/users');

// var app = express();

// // view engine setup
// app.set('views', path.join(__dirname, 'views'));
// app.set('view engine', 'ejs');

// app.use(logger('dev'));
// app.use(express.json());
// app.use(express.urlencoded({ extended: false }));
// app.use(cookieParser());
// app.use(express.static(path.join(__dirname, 'public')));

// app.use('/', indexRouter);
// app.use('/users', usersRouter);

// // catch 404 and forward to error handler
// app.use(function(req, res, next) {
//   next(createError(404));
// });

// // error handler
// app.use(function(err, req, res, next) {
//   // set locals, only providing error in development
//   res.locals.message = err.message;
//   res.locals.error = req.app.get('env') === 'development' ? err : {};

//   // render the error page
//   res.status(err.status || 500);
//   res.render('error');
// });


