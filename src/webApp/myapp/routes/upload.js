var express = require('express');
var path = require('path')
var router = express.Router();
var util = require("util");
var fs = require("fs"); 

var multer = require('multer');

var upload = multer({dest:'uploads/'});

router.get('/', function(req, res) {
  res.render("uploadPage", {title: "Uploading files!"});
}); 


router.post('/',upload.array('myf'), function(req, res, next){
	console.log('I am multi file version', util.inspect(req.files));
	var processed = 0;

	req.files.forEach(function(file) {
                console.log(util.inspect(file));
		var dir = file.destination;
		var filename = file.filename;
                console.log("dir " + dir + " filename " + filename);
		fs.renameSync(file.path, dir  + file.originalname);
		processed++;
	});

	if(processed == 0)
	{
                   res.end(" Provide valid files");            
        }
	else{
		res.statusCode = 302;
		res.setHeader("Location","/python");
		console.log(processed + " files processed");
        	res.end(processed + " files processed");
        }
});


module.exports = router;

