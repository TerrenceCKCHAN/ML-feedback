const {createCanvas, loadImage} = require('canvas');
const tf = require('@tensorflow/tfjs-node');
const NUM_CLASSES = 4;

// MLP experiement final
module.exports.trainMLP = function() {
    convertImageToData()
    .then((teapotData) => gen_train_test_data(0.4, teapotData))
    .then(([xtr, ytr, xva,yva,xte, yte]) => do_teapot(xtr,ytr,xva,yva,xte,yte))
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

async function do_teapot(xtrain, ytrain,xvalid,yvalid, xtest, ytest) {
    model = await trainModelMLP(xtrain, ytrain, xvalid, yvalid);
    model.summary;

    const xtecon = xtest;
    const ytecon = ytest;
    const xte = xtecon.reshape([xtecon.shape[0], 784]);

    console.log(model.summary());

    const score = await model.evaluate(xte, ytecon);
    console.log("Evaluation: LOSS" + score[0] + "\nACC " + score[1] + "\nPRECIS" + score[2]);
    const predictions = await model.predict(xte).argMax(-1);
    const yTruth = tf.argMax(ytecon, axis=1);
    const predictionsOneHot = tf.oneHot(predictions, 4);
    const precision = tf.metrics.precision(ytecon, predictionsOneHot).dataSync()[0];
    const recall = tf.metrics.recall(ytecon, predictionsOneHot).dataSync()[0];
    console.log('Classification results on Test set: ' +  
                '\nPrecision: ' + precision +
                '\nRecall: ' + recall);



    
}

async function trainModelMLP(xTrain, yTrain, xValid, yValid) {
    const model = tf.sequential();
    const learningRate = 0.00008;
    const epochs = 50;
    const optimizer = tf.train.adam(learningRate);
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

    // split the data into training and test sets
    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    const xValid  = xs.slice([numTrainExamples, 0], [numValidExamples, xDims]);
    const xTest  = xs.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, xDims]);
    const yTrain = ys.slice([0, 0], [numTrainExamples, NUM_CLASSES]);
    const yValid  = ys.slice([numTrainExamples, 0], [numValidExamples, NUM_CLASSES]);
    const yTest  = ys.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, NUM_CLASSES]);

    return [xTrain, yTrain, xValid, yValid, xTest, yTest];

}