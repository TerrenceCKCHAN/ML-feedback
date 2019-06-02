const {createCanvas, loadImage} = require('canvas');
const tf = require('@tensorflow/tfjs-node');
const NUM_CLASSES = 9;




module.exports.trainCNN = function() {
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
    // const saveModel = await model.save('file://./models/model-exp3');

    
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
    const NUM_OUTPUT_CLASSES = 9;
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