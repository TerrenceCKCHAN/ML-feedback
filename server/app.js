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

var trainingSize = 500;
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
    fs.writeFile('./training_data/exp4/raw/class_' + trainingClass + '/training_' + trainingId + '.png', buf, (err) => {
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
    let imgRaw = 'training_data/exp4/raw/class_' + c + '/training_' + t_id + '.png';
    let imgActive = 'training_data/exp4/active/class_' + c + '/training_' + t_id + '.png';
    let imgExported = 'training_data/exp4/export/class_' + c + '/training_' + t_id + '.png';

    await jimp.read(imgRaw)
    .then(img => (img.clone().write(imgActive)))
    .then(() => jimp.read(imgActive))
    .then(img => {return img.crop(370,230,470,280);})
    // .then(img => {return img.resize(100,100);})
    // .then(img => {return img.crop(20,35,60,45);})
    .then(img => img.resize(28,28))
    // .then(img => img.normalize())
    // .then(img => img.grayscale())
    .then(img => img.write(imgExported))
    .then(() => console.log('Exported file to: ' + imgExported))
    .catch((err) => {console.error(err); preprocessPhoto(c, t_id)});
    console.log(c+': '+t_id);    
}

async function preprocessPhotos(c) {
    var t_id = 0;
    while(t_id < trainingSize) {
        await preprocessPhoto(c, t_id);
        ++t_id;
    }
}

// preprocessAll();
async function preprocessAll() {
    await preprocessPhotos(0);
    await preprocessPhotos(1);
    await preprocessPhotos(2);
    await preprocessPhotos(3);
    await preprocessPhotos(4);
    await preprocessPhotos(5);
    await preprocessPhotos(6);
    await preprocessPhotos(7);
    await preprocessPhotos(8);
    await preprocessPhotos(9);
    await preprocessPhotos(10);
    await preprocessPhotos('1-3');
    await preprocessPhotos('1-3-10');
    await preprocessPhotos('1-10');
    await preprocessPhotos('2-6-10');
    await preprocessPhotos('3-4-5');
    await preprocessPhotos('3-4-5-11');
    await preprocessPhotos('3-10');
    await preprocessPhotos('4-5');
    await preprocessPhotos('5-10');
    await preprocessPhotos('6-10');
    console.log("Complete");
}

// Handling User GLSL shader data
app.post("/glsl", (req, res) => {saveGLSL(req.body);res.send("GLSL received!")});

function saveGLSL(body) {
    var stream = parsePhoto(Object.keys(body)[0]);
    var buf = new Buffer.from(stream, 'base64');
    fs.writeFile('./glsl/' + 'user' + '.png', buf, (err) => {
        if (err) {console.log(err);} else {console.log("Success")}
    })
}













// Abort websocket
// var http = require('http').Server(app);
// var socket_io = require('socket.io').listen(http, {origins: 'http://127.0.0.1:8080'});

// socket_io.on('connection', function(socket) {
//     console.log('a user connected');
// })






// ML code main: convertImageToData -> generate train test data by converting to tensors -> ML model
const tf = require('@tensorflow/tfjs-node');
const util = require('util');
const {createCanvas, loadImage} = require('canvas');
const CLASSES = ['correct-phong', 'normal-not-normalized', 'no-specular', 'no-diffuse'];
const NUM_CLASSES = 10;





// For Multi-label classification
train_model();
function train_model() {
    convertImageToDataMultiLabel()
    .then((teapotData) => gen_train_test_data_multi_label(0.4, teapotData))
    .then(([xtr, ytr, xva,yva,xte, yte]) => do_teapot_multi_label(xtr,ytr,xva,yva,xte,yte))
    .catch((err)=> console.log(err));
    
}

async function convertImageToDataMultiLabel() {
    var dataArray = [];
    console.log('START');
    const canvas = createCanvas(28,28);
    const cx = canvas.getContext('2d');
    const NC = [0,1,2,3,4,5,6,7,8,9,10,'1-3','1-3-10','1-10','2-6-10','3-4-5','3-4-5-11','3-10','4-5','5-10','6-10'];
    const TZ = Array.from(Array(500).keys());
    for (const c of NC) {
        for(const id of TZ) {
            const image = await loadImage('training_data/exp4/export/class_'+c+'/training_'+id+'.png');
            cx.drawImage(image, 0, 0);
            const tensorObj = tf.browser.fromPixels(canvas, 3);
            const values = tensorObj.dataSync();
            const arr = Array.from(values);
            arr.push(c);
            dataArray.push(arr);
        }
    }
    console.log('NEXT');
    return dataArray;
}



function gen_train_test_data_multi_label(split, teapotData) {
    return tf.tidy(() => {
        const dataByClass = [[]];
        const targetsByClass = [[]];
        for (let i = 0; i < 20; ++i) {
            dataByClass.push([]);
            targetsByClass.push([]);
        }

        for (const teapot of teapotData) {
            const target = label_to_class(teapot[teapot.length - 1]);
            const data   = teapot.slice(0, teapot.length - 1);
            dataByClass[target].push(data);
            targetsByClass[target].push(target);
        }

        const xTrains = [];
        const yTrains = [];
        const xValids  = [];
        const yValids  = []
        const xTests  = [];
        const yTests  = [];
        for (let c = 0; c < 21; ++c) {
            const [xTrain,yTrain,xValid,yValid,xTest,yTest] = 
                convertToMultiLabelTensors(dataByClass[c], targetsByClass[c], split);
            xTrains.push(xTrain);
            yTrains.push(yTrain);
            xValids.push(xValid);
            yValids.push(yValid);
            xTests.push(xTest);
            yTests.push(yTest);
        }

        const concatAxis = 0;

        return [
            tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
            tf.concat(xValids, concatAxis), tf.concat(yValids, concatAxis),
            tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
        ];
    });
}

async function do_teapot_multi_label(xtrain, ytrain, xvalid, yvalid, xtest, ytest) {
    console.log(ytrain);
    model = await trainModelMultiLabel(xtrain, ytrain, xvalid, yvalid);
    model.summary;
    // const saveModel = await model.save('file://./models/model-exp4');
}

async function trainModelMultiLabel(xTrain, yTrain, xValid, yValid) {
    const model = tf.sequential();
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const CHANNELS = 3;
    
    // First layer
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS],
        kernelSize:5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // MaxPooling Layer 
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Second Layer with Max Pooling
    model.add(tf.layers.conv2d({
        kernelSize:5,
        filters: 16,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Flatten output from the 2D filters into a 1D vector
    model.add(tf.layers.flatten());

    // output Dense
    const NUM_OUTPUT_CLASSES = 12;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'sigmoid'
    }));

    // Training hyperparameters
    const learningRate = 0.00009;
    const epochs = 50;
      
    const optimizer = tf.train.adam(learningRate);
    model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
    });

    const xt = xTrain.reshape([xTrain.shape[0], 28, 28, 3]);
    const xv = xValid.reshape([xValid.shape[0], 28, 28, 3]);

    const history = await model.fit(xt, yTrain,
        {   epochs: epochs, 
            validationData: [xv, yValid],
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epochs, logs) => {
                    console.log("Epoch" + epochs + "\nAccuracy:" + logs.acc);
                    await tf.nextFrame();
                },
            }
        }
    );
    return model;

}


//Helper to convert to tensors - multi label
function convertToMultiLabelTensors(data, targets, testSplit) {
    const NUM_MulLabelClasses = 12;
    const numExamples = data.length;
    if (numExamples !== targets.length) {
        throw new Error('data and target mismatch');
    }

    const numTestExamples = Math.round(numExamples * testSplit / 2);
    const numValidExamples = Math.round(numExamples * testSplit / 2);
    const numTrainExamples = numExamples - numValidExamples - numTestExamples;

    const xDims = data[0].length;

    const xs = tf.tensor2d(data, [numExamples, xDims]);
    const ys = tf.tensor2d(label_to_one_hot(targets));
    console.log(ys);

    // split the data into training and test sets
    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    const xValid  = xs.slice([numTrainExamples, 0], [numValidExamples, xDims]);
    const xTest  = xs.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, xDims]);
    const yTrain = ys.slice([0, 0], [numTrainExamples, NUM_MulLabelClasses]);
    const yValid  = ys.slice([numTrainExamples, 0], [numValidExamples, NUM_MulLabelClasses]);
    const yTest  = ys.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, NUM_MulLabelClasses]);
    return [xTrain, yTrain, xValid, yValid, xTest, yTest];
}


// Helper to get one hot encoding for multiclass labels
function label_to_one_hot(targets) {
    targetLabels = decodeLabels(targets);
    targetsArr = []
    for (const target of targetLabels) {
        var i = 0;
        var num;
        targetArr = [0,0,0,0,0,0,0,0,0,0,0,0];
        while (i < target.length) {
            if (target[i] != '-') {
                num = target[i];
                if ((i + 1) < target.length && target[i+1] != '-') {
                    num += target[i + 1]; 
                    ++i;           
                }
                targetArr[num] += 1;
            }
            ++i;
        }
        targetsArr.push(targetArr);
    }
    return targetsArr;
}

function decodeLabels(targets) {
    returnArr = []
    for (const target of targets) {
        if (target <= 10) {
            returnArr.push(target);
        }
        switch (target) {
            case 11:
            returnArr.push('1-3');
            case 12:
            returnArr.push('1-3-10');
            case 13:
            returnArr.push('1-10');
            case 14:
            returnArr.push('2-6-10');
            case 15:
            returnArr.push('3-4-5');
            case 16:
            returnArr.push('3-4-5-11');
            case 17:
            returnArr.push('3-10');
            case 18:
            returnArr.push('4-5');
            case 19:
            returnArr.push('5-10');
            case 20:
            returnArr.push('6-10');
        }
    }
    return returnArr;
}

function label_to_class(str) {
    if (str <= 10) {
        return str;
    }
    switch (str) {
        case '1-3':
            return 11;
        case '1-3-10':
            return 12;
        case '1-10':
            return 13;
        case '2-6-10':
            return 14;
        case '3-4-5':
            return 15;
        case '3-4-5-11':
            return 16;
        case '3-10':
            return 17;
        case '4-5':
            return 18;
        case '5-10':
            return 19;
        case '6-10':
            return 20;
    }
}


// For Multi-class classification
// model_and_predict();
function model_and_predict() {
    convertImageToData()
    .then((teapotData) => gen_train_test_data(0.4, teapotData))
    .then(([xtr, ytr, xva,yva,xte, yte]) => do_teapot(xtr,ytr,xva,yva,xte,yte))
    // .then((model) => {return model})
    .catch((err)=> console.log(err));
}


async function convertImageToData() {
    var dataArray = [];
    const canvas = createCanvas(28,28);
    const cx = canvas.getContext('2d');
    const NC = [0,1,2,3,4,5,6,7,8];
    const TZ = Array.from(Array(1000).keys());
    for (const c of NC) {
        for(const id of TZ) {
            const image = await loadImage('training_data/exp3/export/class_'+c+'/training_'+id+'.png');
            cx.drawImage(image, 0, 0);
            const tensorObj = tf.browser.fromPixels(canvas, 3);
            const values = tensorObj.dataSync();
            const arr = Array.from(values);
            arr.push(c);
            dataArray.push(arr);
        }
    }
    return dataArray;
}


// loadModel();
async function loadModel() { 
    const loadedModel = await tf.loadLayersModel('file://./models/model-1/model.json');
    loadedModel.summary();
}



async function do_teapot(xtrain, ytrain,xvalid,yvalid, xtest, ytest) {
    model = await trainModelCNN(xtrain, ytrain, xvalid, yvalid);
    model.summary;
    const saveModel = await model.save('file://./models/model-exp3');

    
    // // Generate predictions using test sets
    // // Tensors of predictions
    // const predictions = await model.predict(xtest).argMax(-1);
    // const predList    = predictions.dataSync();
    // // Encode prediction tensors to one hot representation
    // const predictionsOneHot = tf.oneHot(predictions, 4);
    // // Decode ytest from one hot encoding; Still a tensor
    // const yTruth = tf.argMax(ytest, axis=1);
    // const yTruthList = yTruth.dataSync();
    // // Custom code to generating the accuracy
    // var correct = 0;
    // var wrong   = 0;
    // for (var i = 0; i < yTruthList.length; ++i) {
    //     if (yTruthList[i] == predList[i]) {
    //         correct++;
    //     } else {
    //         wrong++;
    //     }
    // }
    // console.log('Accuracy: ' + correct / yTruthList.length);
    // console.log('Number of correct:' + correct);
    // console.log('Number of wrong: ' + wrong);
    
    // Using tf metrics
    // const precision = tf.metrics.precision(ytest, predictionsOneHot).dataSync()[0];
    // const recall = tf.metrics.recall(ytest, predictionsOneHot).dataSync()[0];
    // console.log('Classification results on Test set: ' +  
    //             '\nPrecision: ' + precision +
    //             '\nRecall: ' + recall);    
    
}

async function trainModelCNN(xTrain, yTrain, xValid, yValid) {
    const model = tf.sequential();
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const CHANNELS = 3;
    
    // First layer
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS],
        kernelSize:5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // MaxPooling Layer 
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Second Layer with Max Pooling
    model.add(tf.layers.conv2d({
        kernelSize:5,
        filters: 16,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Flatten output from the 2D filters into a 1D vector
    model.add(tf.layers.flatten());

    // output Dense
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    // Training hyperparameters
    const learningRate = 0.00009;
    const epochs = 50;
      
    const optimizer = tf.train.adam(learningRate);
    model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
    });

    // const BATCH_SIZE = 512;
    // const TRAIN_DATA_SIZE = 5500;
    // const TEST_DATA_SIZE = 1000;
  
    // const [trainXs, trainYs] = tf.tidy(() => {
    //   const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    //   return [
    //     d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
    //     d.labels
    //   ];
    // });
  
    // const [testXs, testYs] = tf.tidy(() => {
    //   const d = data.nextTestBatch(TEST_DATA_SIZE);
    //   return [
    //     d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
    //     d.labels
    //   ];
    // });

    const xt = xTrain.reshape([xTrain.shape[0], 28, 28, 3]);
    const xv = xValid.reshape([xValid.shape[0], 28, 28, 3]);

    const history = await model.fit(xt, yTrain,
        {   epochs: epochs, 
            validationData: [xv, yValid],
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epochs, logs) => {
                    console.log("Epoch" + epochs + "\nAccuracy:" + logs.acc);
                    await tf.nextFrame();
                },
            }
        }
    );
    return model;


}

async function trainModelMLP(xTrain, yTrain, xValid, yValid) {
    const model = tf.sequential();
    const learningRate = 0.00008;
    const epochs = 50;
    const optimizer = tf.train.adam(learningRate);
    // console.log(util.inspect(xTrain, {maxArrayLength:1}));
    model.add(tf.layers.dense(
        { units: 10, activation:'relu', inputShape: [xTrain.shape[1]]}
    ));

    model.add(tf.layers.dense(
        { units: 4, activation: 'softmax'}
    ));

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy', ]
    });

    const history = await model.fit(xTrain, yTrain,
        { epochs: epochs, validationData: [xValid, yValid],
            callbacks: {
                onEpochEnd: async (epochs, logs) => {
                    console.log("Epoch" + epochs + "\nAccuracy:" + logs.acc);
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
        const xValids  = [];
        const yValids  = []
        const xTests  = [];
        const yTests  = [];
        for (let c = 0; c < NUM_CLASSES; ++c) {
            const [xTrain,yTrain,xValid,yValid,xTest,yTest] = 
                convertToTensors(dataByClass[c], targetsByClass[c], split);
            xTrains.push(xTrain);
            yTrains.push(yTrain);
            xValids.push(xValid);
            yValids.push(yValid);
            xTests.push(xTest);
            yTests.push(yTest);
        }

        const concatAxis = 0;

        return [
            tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
            tf.concat(xValids, concatAxis), tf.concat(yValids, concatAxis),
            tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
        ];
    });
}

function convertToTensors(data, targets, testSplit) {
    const numExamples = data.length;
    if (numExamples !== targets.length) {
        throw new Error('data and target mismatch');
    }

    const numTestExamples = Math.round(numExamples * testSplit / 2);
    const numValidExamples = Math.round(numExamples * testSplit / 2);
    const numTrainExamples = numExamples - numValidExamples - numTestExamples;

    const xDims = data[0].length;

    const xs = tf.tensor2d(data, [numExamples, xDims]);
    const ys = tf.oneHot(tf.tensor1d(targets).toInt(), NUM_CLASSES);
    // console.log(ys);

    // split the data into training and test sets
    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    const xValid  = xs.slice([numTrainExamples, 0], [numValidExamples, xDims]);
    const xTest  = xs.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, xDims]);
    const yTrain = ys.slice([0, 0], [numTrainExamples, NUM_CLASSES]);
    const yValid  = ys.slice([numTrainExamples, 0], [numValidExamples, NUM_CLASSES]);
    const yTest  = ys.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, NUM_CLASSES]);

    return [xTrain, yTrain, xValid, yValid, xTest, yTest];

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


