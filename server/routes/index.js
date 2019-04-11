var express = require('express');
var router = express.Router();
var app = express();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

// app.use(myParser.urlencoded({extended: true}));
app.post("/post", function(req, res) {
  console.log(request.body);
}) 

module.exports = router;
