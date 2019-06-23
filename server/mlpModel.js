const {createCanvas, loadImage} = require('canvas');
const tf = require('@tensorflow/tfjs-node');
const NUM_CLASSES = 4;
const fs = require('fs')
const mlUtils = require('./mlUtils.js');

// MLP experiment cross validation
module.exports.trainMLP = function() {
    convertImageToData()
    .then((teapotData) => gen_train_test_data(teapotData))
    .then(([xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
        yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s]) => do_teapot(xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
            yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s))
    // .then((model) => {return model})
    .catch((err)=> console.log(err));
}



async function convertImageToData() {
    var dataArray = [];
    const canvas = createCanvas(28,28);
    const cx = canvas.getContext('2d');
    const NC = [0,1,2,3];
    const TZ = Array.from(Array(1000).keys());
    for (const c of NC) {
        for(const id of TZ) {
            const image = await loadImage('training_data/exp1/export/class_'+c+'/training_'+id+'.png');
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

async function do_teapot(xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
    yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s) {

    const hyperparameters = 
        [{units:8, learningRate:0.00006},
        {units:8, learningRate:0.00009},
        {units:8, learningRate:0.00012},
        {units:12, learningRate:0.00006},
        {units:12, learningRate:0.00009},
        {units:12, learningRate:0.00012},
        {units:16, learningRate:0.00006},
        {units:16, learningRate:0.00009},
        {units:16, learningRate:0.00012},
        ]

    const table_of_results = [];
    const batchSelector = [xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
        yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s];
    
    for (var params = 0; params < hyperparameters.length; ++params) {
        var accumulate_metrics = [0, 0, 0, 0,0,0,0,0];
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
            const xte = xtecon.reshape([xtecon.shape[0], 784]);

            model = await trainModelMLP(xt, yt, xv, yv, hyperparameters[params]);
            console.log(model.summary());
            // const predictions = await model.predict(xte).argMax(-1);
            console.log("NO")
            const score = await model.evaluate(xte, ytecon);
            console.log("YES")
            console.log("Evaluation: LOSS" + score[0] + "\nACC " + score[1] + "\nPRECIS" + score[2]);
            const predictions = await model.predict(xte).argMax(-1);
            // const yTruth = tf.argMax(ytecon, axis=1);
            const predictionsOneHot = tf.oneHot(predictions, 4);
            const precision = tf.metrics.precision(ytecon, predictionsOneHot).dataSync()[0];
            const recall = tf.metrics.recall(ytecon, predictionsOneHot).dataSync()[0];
            console.log('Classification results on Test set: ' +  
                        '\nPrecision: ' + precision +
                        '\nRecall: ' + recall);
            // console.log(predictions);
            // const predList    = predictions.dataSync();
            // console.log(predList);
            // Encode prediction tensors to one hot representation
            // const predictionsOneHot = tf.oneHot(predictions, 9);
            const yTruth = tf.argMax(ytecon, axis=1);
            // const yTruthList = yTruth.dataSync();

            const confusionMatrix = tf.math.confusionMatrix(yTruth, predictions, 9);
            const confuse1d = confusionMatrix.dataSync();
            const confuse2d = [];
            for (let i = 0; i < confuse1d.length / 9; ++i) {
                const currentRow = [];
                for (let j = 0; j < 9; ++j) {
                    currentRow.push(confuse1d[i * 9 + j])
                }
                confuse2d.push(currentRow);
            }
            
            // confusionMatrix.print();
            // console.log(confuse2d);
            

            const mlUtilsMetrics = mlUtils.calculate_metrics(confuse2d);
            console.log(mlUtilsMetrics[0]);
            console.log(mlUtilsMetrics[1]);
            console.log(mlUtilsMetrics[2]);
            console.log(mlUtilsMetrics[3]);

            accumulate_metrics[0] += score[0].dataSync()[0];
            accumulate_metrics[1] += score[1].dataSync()[0];
            // precision
            accumulate_metrics[2] += precision; 
            // precision ALT1
            accumulate_metrics[3] += score[2].dataSync()[0];
            // precision ALT2
            accumulate_metrics[4] += mlUtilsMetrics[1];
            // recall 
            accumulate_metrics[5] += recall;
            // recall ALT
            accumulate_metrics[6] += mlUtilsMetrics[2];
            // f1 ALT
            accumulate_metrics[7] += mlUtilsMetrics[3];
        }
        const final_precision = accumulate_metrics[2] / 10;
        const final_recall    = accumulate_metrics[5] / 10;
        const final_f1        = (2 * final_precision * final_recall) / (final_precision + final_recall);

        const avg_metrics = {Loss: accumulate_metrics[0] / 10,
                            accuracy: accumulate_metrics[1] / 10, 
                            precision: final_precision,
                            PresALT1: accumulate_metrics[3] / 10,
                            PresALT2: accumulate_metrics[4] / 10, 
                            recall: final_recall,
                            recallALT: accumulate_metrics[6] / 10,
                            f1: final_f1,
                            f1ALT: accumulate_metrics[7] / 10
        };
        table_of_results.push(avg_metrics);
        console.log(table_of_results);  
    }
    console.log(table_of_results);
    
}

async function trainModelMLP(xTrain, yTrain, xValid, yValid, hps) {
    const model = tf.sequential();
    const learningRate = hps.learningRate;
    const epochs = 50;
    const optimizer = tf.train.adam(learningRate);
    // console.log(util.inspect(xTrain, {maxArrayLength:1}));
    model.add(tf.layers.dense(
        { units: hps.units, activation:'relu', inputShape: [xTrain.shape[1]]}
    ));

    model.add(tf.layers.dense(
        { units: 4, activation: 'softmax'}
    ));

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy', 'precision']
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

function gen_train_test_data(teapotData) {
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


        for (let c = 0; c < NUM_CLASSES; ++c) {
            const [xb1,xb2,xb3,xb4,xb5,xb6,xb7,xb8,xb9,xb10,yb1,yb2,yb3,yb4,yb5,yb6,yb7,yb8,yb9,yb10] = 
                convertToTensors(dataByClass[c], targetsByClass[c]);
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
            tf.concat(yb9s, concatAxis), tf.concat(yb10s, concatAxis)
        ];
    });
}

function convertToTensors(data, targets, testSplit) {
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
    const ys = tf.oneHot(tf.tensor1d(targets).toInt(), NUM_CLASSES);
    // console.log(ys);


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

    // // split the data into training and test sets
    // const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    // const xValid  = xs.slice([numTrainExamples, 0], [numValidExamples, xDims]);
    // const xTest  = xs.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, xDims]);
    // const yTrain = ys.slice([0, 0], [numTrainExamples, NUM_CLASSES]);
    // const yValid  = ys.slice([numTrainExamples, 0], [numValidExamples, NUM_CLASSES]);
    // const yTest  = ys.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, NUM_CLASSES]);

    // return [xTrain, yTrain, xValid, yValid, xTest, yTest];
    return[xb1,xb2,xb3,xb4,xb5,xb6,xb7,xb8,xb9,xb10,yb1,yb2,yb3,yb4,yb5,yb6,yb7,yb8,yb9,yb10];

}