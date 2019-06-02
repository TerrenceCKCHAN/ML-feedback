const {createCanvas, loadImage} = require('canvas');
const tf = require('@tensorflow/tfjs-node');


module.exports.trainMultiLabelCNN = function() {
    convertImageToDataMultiLabel()
    .then((teapotData) => gen_train_test_data_multi_label(0.4, teapotData))
    .then(([xtr, ytr, xva,yva,xte, yte]) => do_teapot_multi_label(xtr,ytr,xva,yva,xte,yte))
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
    // console.log(ytrain);
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