const {createCanvas, loadImage} = require('canvas');
const tf = require('@tensorflow/tfjs-node');
const NUM_CLASSES = 9;




module.exports.trainCNN = function() {
    convertImageToData()
    .then((teapotData) => gen_train_test_data(teapotData))
    .then(([xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
        yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s]) =>
        do_teapot(xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
        yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s))
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

async function do_teapot(xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
    yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s) {
    
    const hyperparameters = {kernelSize:3, strides: 1, filters: 8, poolSize:2, poolStrides:2, learningRate:0.00008}
    var xtrain, ytrain, xvalid, yvalid;
    xtrain = ytrain = xvalid = yvalid = [];
    const batchSelector = [xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
        yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s]
    for (var i = 1; i <= 1; ++i) {
        for (var j = 1; j <= 10; ++j) {
            if (i == j) {
                xvalid.push(batchSelector[i - 1]);
                yvalid.push(batchSelector[i - 1 + 10]);
            } else {
                xtrain.push(batchSelector[i]);
                xtrain.push(batchSelector[i]);
            }
        }
        console.log("Okay");
        model = await trainModelCNN(xtrain, ytrain, xvalid, yvalid, hyperparameters);
    }

    // model.summary;
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

async function trainModelCNN(xTrain, yTrain, xValid, yValid, hps) {
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
        filters: hps.filters * 2,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [hps.poolSize, hps.poolSize], strides: [hps.poolStrides, hps.poolStrides]}));
    
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
    const learningRate = hps.learningRate;
    const epochs = 1;
      
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

        var xb1s,xb2s,xb3s,xb4s,xb5s,xb6s,xb7s,xb8s,xb9s,xb10s,
            yb1s,yb2s,yb3s,yb4s,yb5s,yb6s,yb7s,yb8s,yb9s,yb10s;
        
        xb1s=xb2s=xb3s=xb4s=xb5s=xb6s=xb7s=xb8s=xb9s=xb10s=
        yb1s=yb2s=yb3s=yb4s=yb5s=yb6s=yb7s=yb8s=yb9s=yb10s=[];


        for (let c = 0; c < NUM_CLASSES; ++c) {
            const [xb1,xb2,xb3,xb4,xb5,xb6,xb7,xb8,xb9,xb10,yb1,yb2,yb3,yb4,yb5,yb6,yb7,yb8,yb9,yb10] = 
                convertToTensors(dataByClass[c], targetsByClass[c]);
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

        console.log(xb1s);
        console.log(tf.concat(xb1s, 0));
        
        
        const concatAxis = 1;


        // return [
        //     tf.concat(xb1s, concatAxis), tf.concat(xb2s, concatAxis),
        //     tf.concat(xb3s, concatAxis), tf.concat(xb4s, concatAxis), 
        //     tf.concat(xb5s, concatAxis), tf.concat(xb6s, concatAxis), 
        //     tf.concat(xb7s, concatAxis), tf.concat(xb8s, concatAxis),  
        //     tf.concat(xb9s, concatAxis), tf.concat(xb10s, concatAxis), 
        //     tf.concat(yb1s, concatAxis), tf.concat(yb2s, concatAxis), 
        //     tf.concat(yb3s, concatAxis), tf.concat(yb4s, concatAxis), 
        //     tf.concat(yb5s, concatAxis), tf.concat(yb6s, concatAxis), 
        //     tf.concat(yb7s, concatAxis), tf.concat(yb8s, concatAxis), 
        //     tf.concat(yb9s, concatAxis), tf.concat(yb10s, concatAxis), 
        // ];





        // const xTrains = [];
        // const yTrains = [];
        // const xValids  = [];
        // const yValids  = []
        // const xTests  = [];
        // const yTests  = [];
        // for (let c = 0; c < NUM_CLASSES; ++c) {
        //     const [xTrain,yTrain,xValid,yValid,xTest,yTest] = 
        //         convertToTensors(dataByClass[c], targetsByClass[c]);
        //     xTrains.push(xTrain);
        //     yTrains.push(yTrain);
        //     xValids.push(xValid);
        //     yValids.push(yValid);
        //     xTests.push(xTest);
        //     yTests.push(yTest);
        // }

        // const concatAxis = 0;

        // return [
        //     tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
        //     tf.concat(xValids, concatAxis), tf.concat(yValids, concatAxis),
        //     tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
        // ];
    });
}

function convertToTensors(data, targets) {
    const numExamples = data.length;
    if (numExamples !== targets.length) {
        throw new Error('data and target mismatch');
    }

    //Number of batches
    // bs for batch size, lbs for last batch size
    const bs = Math.round(numExamples / 10);
    const lbs = numExamples-bs*9;


    // const numTestExamples = Math.round(numExamples * testSplit / 2);
    // const numValidExamples = Math.round(numExamples * testSplit / 2);
    // const numTrainExamples = numExamples - numValidExamples - numTestExamples;

    const xDims = data[0].length;

    const xs = tf.tensor2d(data, [numExamples, xDims]);
    const ys = tf.oneHot(tf.tensor1d(targets).toInt(), NUM_CLASSES);
    // console.log(ys);

    // split the data into training and test sets
    // const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    // const xValid  = xs.slice([numTrainExamples, 0], [numValidExamples, xDims]);
    // const xTest  = xs.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, xDims]);
    // const yTrain = ys.slice([0, 0], [numTrainExamples, NUM_CLASSES]);
    // const yValid  = ys.slice([numTrainExamples, 0], [numValidExamples, NUM_CLASSES]);
    // const yTest  = ys.slice([numTrainExamples + numValidExamples, 0], [numTestExamples, NUM_CLASSES]);

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

    // return [xTrain, yTrain, xValid, yValid, xTest, yTest];
    return[xb1,xb2,xb3,xb4,xb5,xb6,xb7,xb8,xb9,xb10,yb1,yb2,yb3,yb4,yb5,yb6,yb7,yb8,yb9,yb10];

}