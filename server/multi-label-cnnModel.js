const {createCanvas, loadImage} = require('canvas');
const tf = require('@tensorflow/tfjs-node');
const mlUtils = require('./mlUtils.js');
const util = require('util');


module.exports.trainMultiLabelCNN = function() {
    convertImageToDataMultiLabel()
    .then((teapotData) => gen_train_test_data_multi_label(teapotData))
    .then(([xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
        yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s]) => 
        do_teapot_multi_label(xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
            yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s))
    .catch((err)=> console.log(err));
    
}

async function convertImageToDataMultiLabel() {
    var dataArray = [];
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
    return dataArray;
}



function gen_train_test_data_multi_label(teapotData) {
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

        // const xTrains = [];
        // const yTrains = [];
        // const xValids  = [];
        // const yValids  = []
        // const xTests  = [];
        // const yTests  = [];

        const xb1s = [];
        const xb2s = [];
        const xb3s = [];
        const xb4s = [];
        const xb5s = [];
        const xb6s = [];
        const xb7s = [];
        const xb8s = [];
        const xb9s = [];
        const xb10s = [];

        const yb1s = [];
        const yb2s = [];
        const yb3s = [];
        const yb4s = [];
        const yb5s = [];
        const yb6s = [];
        const yb7s = [];
        const yb8s = [];
        const yb9s = [];
        const yb10s = [];

        for (let c = 0; c < 21; ++c) {
            const [xb1,xb2,xb3,xb4,xb5,xb6,xb7,xb8,xb9,xb10,yb1,yb2,yb3,yb4,yb5,yb6,yb7,yb8,yb9,yb10] = 
                convertToMultiLabelTensors(dataByClass[c], targetsByClass[c]);
            // xTrains.push(xTrain);
            // yTrains.push(yTrain);
            // xValids.push(xValid);
            // yValids.push(yValid);
            // xTests.push(xTest);
            // yTests.push(yTest);

            xb1s.push(xb1);
            xb2s.push(xb2);
            xb3s.push(xb3);
            xb4s.push(xb4);
            xb5s.push(xb5);
            xb6s.push(xb6);
            xb7s.push(xb7);
            xb8s.push(xb8);
            xb9s.push(xb9);
            xb10s.push(xb10);

            yb1s.push(yb1);
            yb2s.push(yb2);
            yb3s.push(yb3);
            yb4s.push(yb4);
            yb5s.push(yb5);
            yb6s.push(yb6);
            yb7s.push(yb7);
            yb8s.push(yb8);
            yb9s.push(yb9);
            yb10s.push(yb10);
        }

        const concatAxis = 0;

        // return [
        //     tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
        //     tf.concat(xValids, concatAxis), tf.concat(yValids, concatAxis),
        //     tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
        // ];

        return [
            tf.concat(xb1s, concatAxis), tf.concat(xb2s, concatAxis),
            tf.concat(xb3s, concatAxis), tf.concat(xb4s, concatAxis), 
            tf.concat(xb5s, concatAxis), tf.concat(xb6s, concatAxis), 
            tf.concat(xb7s, concatAxis), tf.concat(xb8s, concatAxis),  
            tf.concat(xb9s, concatAxis), tf.concat(xb10s, concatAxis), 
            tf.concat(yb1s, concatAxis), tf.concat(yb2s, concatAxis), 
            tf.concat(yb3s, concatAxis), tf.concat(yb4s, concatAxis), 
            tf.concat(yb5s, concatAxis), tf.concat(yb6s, concatAxis), 
            tf.concat(yb7s, concatAxis), tf.concat(yb8s, concatAxis), 
            tf.concat(yb9s, concatAxis), tf.concat(yb10s, concatAxis), 
        ];

    });
}

async function do_teapot_multi_label(xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
    yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s) {


    const hyperparameters = 
        [{kernelSize:3, strides: 1, filters: 4, poolSize:2, poolStrides:2, learningRate:0.00004},
        {kernelSize:3, strides: 1, filters: 4, poolSize:2, poolStrides:2, learningRate:0.00008},
        {kernelSize:3, strides: 1, filters: 4, poolSize:2, poolStrides:2, learningRate:0.00012},
        {kernelSize:3, strides: 1, filters: 8, poolSize:2, poolStrides:2, learningRate:0.00004},
        {kernelSize:3, strides: 1, filters: 8, poolSize:2, poolStrides:2, learningRate:0.00008},
        {kernelSize:3, strides: 1, filters: 8, poolSize:2, poolStrides:2, learningRate:0.00012},
        {kernelSize:5, strides: 1, filters: 4, poolSize:2, poolStrides:2, learningRate:0.00004},
        {kernelSize:5, strides: 1, filters: 4, poolSize:2, poolStrides:2, learningRate:0.00008},
        {kernelSize:5, strides: 1, filters: 4, poolSize:2, poolStrides:2, learningRate:0.00012},
        {kernelSize:5, strides: 1, filters: 8, poolSize:2, poolStrides:2, learningRate:0.00004},
        {kernelSize:5, strides: 1, filters: 8, poolSize:2, poolStrides:2, learningRate:0.00008},
        {kernelSize:5, strides: 1, filters: 8, poolSize:2, poolStrides:2, learningRate:0.00012},
    ]

    const table_of_results = [];
    const batchSelector = [xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
        yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s];
    
    for (var params = 0; params < hyperparameters.length; ++params) {
        console.log(hyperparameters[params]);
        var accumulate_metrics = [0, 0, 0, 0];
        for (var i = 0; i < 10; ++i) {
            const xtrain = [];
            const ytrain = [];
            const xvalid = [];
            const yvalid = [];
            const xtest  = [];
            const ytest  = [];
            for (var j = 0; j < 10; ++j) {
                if (i == 9 && j == 0) {
                    xtest.push(batchSelector[0]);
                    ytest.push(batchSelector[0 + 10]);
                    // console.log("Push test" + "i: " + i + "j: " + j);
                }

                if (i == j) {
                    xvalid.push(batchSelector[j]);
                    yvalid.push(batchSelector[j + 10]);
                    // console.log("Push valid" + "i: " + i + "j: " + j);
                    if (i != 9) {
                        xtest.push(batchSelector[j + 1]);
                        ytest.push(batchSelector[j + 1 + 10]);
                        // console.log("Push test" + "i: " + i + "j: " + j);
                        ++j;
                    }
                } 
                else {
                    if (!(i == 9 && j == 0)) {
                        xtrain.push(batchSelector[j]);
                        ytrain.push(batchSelector[j + 10]);
                        // console.log("Push train" + "i: " + i + "j: " + j);
                    }
                }
            }
            const xt = tf.concat(xtrain, 0);
            const yt = tf.concat(ytrain, 0);
            const xv = tf.concat(xvalid, 0);
            const yv = tf.concat(yvalid, 0);
            const xtecon = tf.concat(xtest, 0);
            const ytecon = tf.concat(ytest, 0);
            const xte = xtecon.reshape([xtecon.shape[0], 28, 28, 3]);

            model = await trainModelMultiLabel(xt, yt, xv, yv, hyperparameters[params]);
            console.log(model.summary());

            const score = await model.evaluate(xte, ytecon);
            console.log("Evaluation: LOSS" + score[0] + "\nACC " + score[1] + "\nPRECIS" + score[2]);
            const predictions = await model.predict(xte).argMax(-1);
            const yTruth = tf.argMax(ytecon, axis=1);
            const predictionsOneHot = tf.oneHot(predictions, 12);
            const precision = tf.metrics.precision(ytecon, predictionsOneHot).dataSync()[0];
            const recall = tf.metrics.recall(ytecon, predictionsOneHot).dataSync()[0];
            console.log('Classification results on Test set: ' +  
                        '\nPrecision: ' + precision +
                        '\nRecall: ' + recall);


            for (let k = 0; k <= 3; ++k) {
                // accumulate_metrics[k] +=  mlUtilsMetrics[k];
            }
            
        }
        const avg_metrics = {};
        table_of_results.push(avg_metrics); 
        console.log(table_of_results);
    }
    console.log(table_of_results); 

}

async function trainModelMultiLabel(xTrain, yTrain, xValid, yValid, hps) {
    const model = tf.sequential();
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const CHANNELS = 3;
    
    // First layer
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS],
        kernelSize: hps.kernelSize,
        filters: hps.filters,
        strides: hps.strides,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // MaxPooling Layer 
    model.add(tf.layers.maxPooling2d({poolSize: [hps.poolSize, hps.poolSize], strides: [hps.poolStrides, hps.poolStrides]}));

    // Second Layer with Max Pooling
    model.add(tf.layers.conv2d({
        kernelSize:hps.kernelSize,
        filters: hps.filters *2,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [hps.poolSize, hps.poolSize], strides: [hps.poolStrides, hps.poolStrides]}));
    
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
    const learningRate = hps.learningRate;
    const epochs = 30;
      
    const optimizer = tf.train.adam(learningRate);
    model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy', 'precision'],
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
function convertToMultiLabelTensors(data, targets) {
    const NUM_CLASSES = 12;
    const numExamples = data.length;
    if (numExamples !== targets.length) {
        throw new Error('data and target mismatch');
    }

    const bs = Math.round(numExamples / 10);
    const lbs = numExamples-(bs*9);
    // const numTestExamples = Math.round(numExamples * testSplit / 2);
    // const numValidExamples = Math.round(numExamples * testSplit / 2);
    // const numTrainExamples = numExamples - numValidExamples - numTestExamples;

    const xDims = data[0].length;

    const xs = tf.tensor2d(data, [numExamples, xDims]);
    const ys = tf.tensor2d(label_to_one_hot(targets));

    // split the data into training and test sets
    // const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    // const xValid  = xs.slice([numTrainExamples, 0], [numValidExamples, xDims]);
    // const xTest  = xs.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, xDims]);
    // const yTrain = ys.slice([0, 0], [numTrainExamples, NUM_MulLabelClasses]);
    // const yValid  = ys.slice([numTrainExamples, 0], [numValidExamples, NUM_MulLabelClasses]);
    // const yTest  = ys.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, NUM_MulLabelClasses]);
    // return [xTrain, yTrain, xValid, yValid, xTest, yTest];
    const xb1  = xs.slice([0,0],[bs,xDims]);
    const xb2  = xs.slice([bs,0],[bs,xDims])
    const xb3  = xs.slice([bs*2,0],[bs,xDims])
    const xb4  = xs.slice([bs*3,0],[bs,xDims])
    const xb5  = xs.slice([bs*4,0],[bs,xDims])
    const xb6  = xs.slice([bs*5,0],[bs,xDims])
    const xb7  = xs.slice([bs*6,0],[bs,xDims])
    const xb8  = xs.slice([bs*7,0],[bs,xDims])
    const xb9  = xs.slice([bs*8,0],[bs,xDims])
    const xb10 = xs.slice([bs*9,0],[lbs,xDims])

    const yb1  = ys.slice([0,0],[bs,NUM_CLASSES]);
    const yb2  = ys.slice([bs,0],[bs,NUM_CLASSES]);
    const yb3  = ys.slice([bs*2,0],[bs,NUM_CLASSES]);
    const yb4  = ys.slice([bs*3,0],[bs,NUM_CLASSES]);
    const yb5  = ys.slice([bs*4,0],[bs,NUM_CLASSES]);
    const yb6  = ys.slice([bs*5,0],[bs,NUM_CLASSES]);
    const yb7  = ys.slice([bs*6,0],[bs,NUM_CLASSES]);
    const yb8  = ys.slice([bs*7,0],[bs,NUM_CLASSES]);
    const yb9  = ys.slice([bs*8,0],[bs,NUM_CLASSES]);
    const yb10 = ys.slice([bs*9,0],[lbs,NUM_CLASSES]);

    return[xb1,xb2,xb3,xb4,xb5,xb6,xb7,xb8,xb9,xb10,yb1,yb2,yb3,yb4,yb5,yb6,yb7,yb8,yb9,yb10];
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