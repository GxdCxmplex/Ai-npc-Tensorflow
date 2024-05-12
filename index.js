const express = require("express");
const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const bodyParser= require("body-parser");
const fs = require('fs');

const app = express();
app.use(bodyParser.json())

let model;
let handler = tfn.io.fileSystem("./v1-js/model.json");
tf.loadLayersModel(handler).then(x => {
	model = x;
});

const special = /[\`|\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\+|\=|\[|\{|\]|\}|\||\\|\'|\<|\,|\.|\>|\?|\/|\""|\;|\:|]/g

let tokens;
fs.readFile('v1_tokens.json', (err, data) => {
	if (!err){
		tokens = JSON.parse(data);
	}
});

const emoji = {
	1: '😕',
	2: '😀',
	3: '😥',
	4: '😠',
	5: '😧',
	6: '🥰',
	7: '😮',
}

app.post('/api', (request, response) => {
	let data = request.body.input;
	data = data.toLowerCase();
	data = data.replace(special, '');
	data = data.split(' ');
	data = data.map(x => tokens[x] || 1)
	
	if (data.length > 32) {
		data = data.slice(0, 31);
	}
	else if (data.length < 32) {
		data = data.concat(Array(32 - data.length).fill(0))
	}

	let tensor = tf.tensor2d([data])

	let max;
	let prediction = model.predict(tensor);

	prediction.data().then(x => {
		max = x.indexOf(Math.max(...x));
		response.send(emoji[max]);
	}).catch(() => {
		response.send(emoji[1]);
	});
});

app.listen(3000);
console.log("Listening to port 3000");
console.log()
