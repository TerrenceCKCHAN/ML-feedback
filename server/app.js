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




// Handling User GLSL shader data
app.post("/glsl", (req, res) => {saveGLSL(req.body);res.send("GLSL received!")});

function saveGLSL(body) {
    var stream = parsePhoto(Object.keys(body)[0]);
    var buf = new Buffer.from(stream, 'base64');
    fs.writeFile('./glsl/' + 'user' + '.png', buf, (err) => {
        if (err) {console.log(err);} else {console.log("Success")}
    })
}

const preprocessor = require('./preprocess.js');
const mlUtils = require('./mlUtils.js');

// const metrics = mlUtils.calculate_metrics([[7,3,1],[6,10,2],[3,3,9]]);
// console.log(metrics[0]);
// console.log(metrics[1]);
// console.log(metrics[2]);
// console.log(metrics[3]);
// preprocessor.process();

// const mlp = require('./mlpModel.js');
// mlp.trainMLP();

const cnn = require('./cnnModel.js');
cnn.trainCNN();

// const multicnn = require('./multi-label-cnnModel.js');
// multicnn.trainMultiLabelCNN();



/////////////////////////////JUNK

function tx() {
    var yy = []
    
    const xx = [2,3]
    yy.push(xx)
    yy.push(xx)
    console.log(yy);

    var q,w,e,r;
    q=w=e=r=Math.round(20/5); 
    const y = 20-q-w-e-r;
    console.log(y);
    console.log(q);
}
// tx();

// function hyper(hyperparams) {

//     console.log(hyperparams.x);
//     console.log(hyperparams.y);
//     console.log(hyperparams.z);
// }

// hyper({x:1,y:2,z:3});



////////////////////////////////////JUNK






// Abort websocket
// var http = require('http').Server(app);
// var socket_io = require('socket.io').listen(http, {origins: 'http://127.0.0.1:8080'});

// socket_io.on('connection', function(socket) {
//     console.log('a user connected');
// })







// loadModel();
async function loadModel() { 
    const loadedModel = await tf.loadLayersModel('file://./models/model-1/model.json');
    loadedModel.summary();
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


