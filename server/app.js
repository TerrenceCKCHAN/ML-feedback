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
    .then(img => img.resize(100,100))
    .then(img => img.normalize())
    // .then(img => img.grayscale())
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
toTensor();








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


