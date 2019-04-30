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
    fs.writeFile('./training_data/class_' + trainingClass + '/training_' + trainingId + '.png', buf, (err) => {
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


// Image Preprocessing
// Resize and apply normalization to training data
const jimp = require('jimp');


async function preprocessPhoto(c, t_id) {
    let imgRaw = 'training_data/raw/class_' + c + '/training_' + t_id + '.png';
    let imgActive = 'training_data/active/class_' + c + '/training_' + t_id + '.png';
    let imgExported = 'training_data/export/class_' + c + '/training_' + t_id + '.png';

    await jimp.read(imgRaw)
    .then(img => (img.clone().write(imgActive)))
    .then(() => jimp.read(imgActive))
    .then(img => {return img.resize(100,100);})
    .then(img => {return img.crop(35,50,30,23);})
    .then(img => img.resize(28,28))
    // .then(img => img.normalize())
    .then(img => img.grayscale())
    .then(img => img.write(imgExported))
    .then(() => console.log('Exported file to: ' + imgExported))
    .catch(err => console.error(err));
    console.log(c+': '+t_id);    
}

async function preprocessPhotos(c) {
    var t_id = 0;
    while(t_id < trainingSize) {
        await preprocessPhoto(c, t_id);
        ++t_id;
    }
}

// preprocessPhoto(0,535);
// preprocessPhoto(0,533);
// preprocessAll();
async function preprocessAll() {
    await preprocessPhotos(0);
    await preprocessPhotos(1);
    await preprocessPhotos(2);
    await preprocessPhotos(3);
    await preprocessPhotos(0);
    await preprocessPhotos(1);
    await preprocessPhotos(2);
    await preprocessPhotos(3);
    console.log("Complete");
}

// ML code main: convertImageToData -> generate train test data by converting to tensors -> ML model
const tf = require('@tensorflow/tfjs');
const util = require('util');
const {createCanvas, loadImage} = require('canvas');
const CLASSES = ['correct-phong', 'normal-not-normalized', 'no-specular', 'no-diffuse'];
const NUM_CLASSES = 4;

convertImageToData()
.then((teapotData) => gen_train_test_data(0.2, teapotData))
.then(([xtr, ytr, xte, yte]) => do_teapot(xtr,ytr,xte,yte))
.catch((err)=> console.log(err));


async function convertImageToData() {
    var dataArray = [];
    const canvas = createCanvas(28,28);
    const cx = canvas.getContext('2d');
    const NC = [0,1,2,3];
    const TZ = Array.from(Array(100).keys());
    for (const c of NC) {
        for(const id of TZ) {
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
}



async function do_teapot(xtrain, ytrain, xtest, ytest) {
    model = await trainModel(xtrain, ytrain, xtest, ytest);
    
    // console.log(util.inspect(Array.from(xtrain.dataSync()), {maxArrayLength:1}));
    // console.log(util.inspect(Array.from(xtest.dataSync()), {maxArrayLength:1}));
    // const input = tf.tensor2d(
    //     [Array.from(xtrain.dataSync())[0], 
    //     Array.from(ytrain.dataSync())[0]]);
    // const prediction = model.predict(input);
    // console.log(prediction);
}

async function trainModel(xTrain, yTrain, xTest, yTest) {
    const model = tf.sequential();
    const learningRate = 0.0001;
    const epochs = 10;
    const optimizer = tf.train.adam(learningRate);
    // console.log(util.inspect(xTrain, {maxArrayLength:1}));
    model.add(tf.layers.dense(
        { units: 32, activation:'sigmoid', inputShape: [xTrain.shape[1]]}
    ));

    model.add(tf.layers.dense(
        { units: 4, activation: 'softmax'}
    ));

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const history = await model.fit(xTrain, yTrain,
        { epochs: epochs, validationData: [xTest, yTest],
            callbacks: {
                onEpochEnd: async (epochs, logs) => {
                    console.log("Epoch" + epochs + "Logs:" + logs.loss);
                    await tf.nextFrame();
                },
            }
        }
    );
    return model;

}

function gen_train_test_data(split, teapotData) {
    return tf.tidy(() => {
        const dataByClass = [[]];
        const targetsByClass = [[]];
        for (let i = 0; i < NUM_CLASSES - 1; ++i) {
            dataByClass.push([]);
            targetsByClass.push([]);
        }

        for (const teapot of teapotData) {
            const target = teapot[teapot.length - 1];
            const data   = teapot.slice(0, teapot.length - 1);
            dataByClass[target].push(data);
            targetsByClass[target].push(target);
        }

        const xTrains = [];
        const yTrains = [];
        const xTests  = [];
        const yTests  = [];
        for (let c = 0; c < NUM_CLASSES; ++c) {
            const [xTrain,yTrain,xTest,yTest] = 
                convertToTensors(dataByClass[c], targetsByClass[c], split);
            xTrains.push(xTrain);
            yTrains.push(yTrain);
            xTests.push(xTest);
            yTests.push(yTest);
        }

        const concatAxis = 0;

        return [
            tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
            tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
        ];
    });
}

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


