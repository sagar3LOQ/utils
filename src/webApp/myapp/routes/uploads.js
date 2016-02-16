var express = require('express');
var router = express.Router();
var util = require("util");
var fs = require("fs"); 

var multer = require('multer');

var upload = multer({dest:'uploads/'});

router.get('/', function(req, res) {
  res.render("uploadPage", {title: "Uploading files!"});
}); 




module.exports = router;

