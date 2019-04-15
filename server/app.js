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

var trainingSize = 100;
var trainingId = 0;
function storeTrainingData(body) {
    if (trainingId < trainingSize) {
        handlePhoto(body);
        trainingId += 1;
    }
}

function handlePhoto(body) {
    var stream = parsePhoto(Object.keys(body)[0]);
    // console.log(stream);
    var buf = new Buffer.from(stream, 'base64');
    fs.writeFile('./training_data/training_' + trainingId + '.png', buf, (err) => {
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


