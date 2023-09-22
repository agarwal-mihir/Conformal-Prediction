console.log("Hello TensorFlow");

import { MnistData } from "./data_2.js";

async function run() {
	const data = new MnistData();
	await data.load();
	await showExamples(data);

	const model = getModel();
	const container = document.getElementById("model-def");
	// tfvis.show.modelSummary(container, model);

	await train(model, data);

	await get_scores(model, data);

	await img_scores(model, data);

	// await showAccuracy(model, data);
	await showConfusion(model, data);
}

document.addEventListener("DOMContentLoaded", run);

async function showExamples(data) {
	const surface = document.getElementById("sample-img");

	// Get the examples
	const [x_train, y_train] = data.getTrainData();
	const numExamples = 20;

	// Create a canvas element to render each example
	for (let i = 0; i < numExamples; i++) {
		const imageTensor = tf.tidy(() => {
			// Reshape the image to 28x28 px
			return x_train.slice([i, 0], [1, x_train.shape[1]]).reshape([28, 28, 1]);
		});

		const canvas = document.createElement("canvas");
		canvas.width = 28;
		canvas.height = 28;
		canvas.style = "margin: 4px;";
		await tf.browser.toPixels(imageTensor, canvas);
		surface.appendChild(canvas);

		imageTensor.dispose();
	}
}

function getModel() {
    const model = tf.sequential();

    // Define the input shape (28x28 for a grayscale image)
    const INPUT_SHAPE = [28, 28, 1];

    // Flatten the input to 784
    model.add(tf.layers.flatten({ inputShape: INPUT_SHAPE }));

    // Add the first fully connected layer (fc1) with 32 units and a sigmoid activation function
    model.add(
        tf.layers.dense({
            units: 32,
            activation: "sigmoid",
        })
    );

    // Add the second fully connected layer (fc2) with 10 units (output classes)
    model.add(
        tf.layers.dense({
            units: 10,
            activation: "softmax", 
        })
    );

    // Choose an optimizer, loss function, and accuracy metric, then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });

    return model;
}


// function getModel() {
// 	const model = tf.sequential();

// 	const IMAGE_WIDTH = 28;
// 	const IMAGE_HEIGHT = 28;
// 	const IMAGE_CHANNELS = 1;

// 	// In the first layer of our convolutional neural network we have
// 	// to specify the input shape. Then we specify some parameters for
// 	// the convolution operation that takes place in this layer.
// 	model.add(
// 		tf.layers.conv2d({
// 			inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
// 			kernelSize: 5,
// 			filters: 8,
// 			strides: 1,
// 			activation: "relu",
// 			kernelInitializer: "varianceScaling",
// 		})
// 	);

// 	// The MaxPooling layer acts as a sort of downsampling using max values
// 	// in a region instead of averaging.
// 	model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

// 	// Repeat another conv2d + maxPooling stack.
// 	// Note that we have more filters in the convolution.
// 	model.add(
// 		tf.layers.conv2d({
// 			kernelSize: 5,
// 			filters: 16,
// 			strides: 1,
// 			activation: "relu",
// 			kernelInitializer: "varianceScaling",
// 		})
// 	);
// 	model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

// 	// Now we flatten the output from the 2D filters into a 1D vector to prepare
// 	// it for input into our last layer. This is common practice when feeding
// 	// higher dimensional data to a final classification output layer.
// 	model.add(tf.layers.flatten());

// 	// Our last layer is a dense layer which has 10 output units, one for each
// 	// output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
// 	const NUM_OUTPUT_CLASSES = 10;
// 	model.add(
// 		tf.layers.dense({
// 			units: NUM_OUTPUT_CLASSES,
// 			kernelInitializer: "varianceScaling",
// 			activation: "softmax",
// 		})
// 	);

// 	// Choose an optimizer, loss function and accuracy metric,
// 	// then compile and return the model
// 	const optimizer = tf.train.adam();
// 	model.compile({
// 		optimizer: optimizer,
// 		loss: "categoricalCrossentropy",
// 		metrics: ["accuracy"],
// 	});

// 	return model;
// }

async function train(model, data) {
	const metrics = ["loss", "val_loss", "acc", "val_acc"];
	const container = document.getElementById("train-plot");
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

	const BATCH_SIZE = 64;

	const [trainXs, trainYs] = data.getTrainData();

	const [testXs, testYs] = data.getTestData();

	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE,
		validationData: [testXs, testYs],
		epochs: 5,
		shuffle: true,
		callbacks: fitCallbacks,
	});
}

const classNames = [
	"Zero",
	"One",
	"Two",
	"Three",
	"Four",
	"Five",
	"Six",
	"Seven",
	"Eight",
	"Nine",
];

function histogramPlot(scores, q_val) {
	// Set up the chart dimensions
	const margin = { top: 20, right: 30, bottom: 70, left: 40 };
	const width = 600 - margin.left - margin.right;
	const height = 300 - margin.top - margin.bottom;

	// Create an SVG element
	const svg = d3.select("#hist")
		.append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	// Create a histogram
	const histogram = d3.histogram()
		.domain([d3.min(scores), d3.max(scores)]) // Set the domain of scores
		.thresholds(30); // Number of bins

	const bins = histogram(scores);

	// Calculate the cumulative frequency
	let cumulative = 0;
	for (const bin of bins) {
		cumulative += bin.length;
		bin.cumulative = cumulative;
	}

	// Create scales for x and y axes
	const xScale = d3.scaleLinear()
		.domain([d3.min(scores), d3.max(scores)])
		.range([0, width]);

	const yScale = d3.scaleLinear()
		.domain([0, d3.max(bins, d => d.cumulative)])
		.range([height, 0]);

	// Calculate the x-position of the vertical line based on q_val
	const q_val_x = xScale(q_val);

	// Create the bars for the cumulative histogram
	svg.selectAll("rect")
		.data(bins)
		.enter()
		.append("rect")
		.attr("x", d => xScale(d.x0))
		.attr("y", d => yScale(d.cumulative))
		.attr("width", d => xScale(d.x1) - xScale(d.x0) - 1)
		.attr("height", d => height - yScale(d.cumulative))
		.style("fill", "#E94B3CFF")
		.style("stroke", "black")
		.style("stroke-width", "1");

	// Add vertical line for q_val
	svg.append("line")
		.attr("x1", q_val_x)
		.attr("y1", 0)
		.attr("x2", q_val_x)
		.attr("y2", height)
		.style("stroke", "blue")
		.style("stroke-width", "2");

	// Add x-axis
	svg.append("g")
		.attr("transform", "translate(0," + height + ")")
		.call(d3.axisBottom(xScale));

	// Add y-axis
	svg.append("g")
		.call(d3.axisLeft(yScale));

	// Add labels
	svg.append("text")
		.attr("x", width / 2)
		.attr("y", height + margin.top + 20)
		.style("text-anchor", "middle")
		.text("Scores");

	svg.append("text")
		.attr("transform", "rotate(-90)")
		.attr("x", -height / 2)
		.attr("y", -margin.left)
		.attr("dy", "1em")
		.style("text-anchor", "middle")
		.text("Cumulative Frequency");

	// Title
	svg.append("text")
		.attr("x", width / 2)
		.attr("y", -margin.top / 2)
		.style("text-anchor", "middle")
		.text("Cumulative Histogram of Scores with Quantile Line (q=" + q_val + ")");
}


async function showAccuracy(model, data) {
	const [preds, labels] = await doPrediction(model, data);
	const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
	const container = document.getElementById("accuracy");
	tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

	labels.dispose();
}

async function showConfusion(model, data) {
	const [preds, labels] = await doPrediction(model, data);
	const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
	const container = document.getElementById("conf-matrix");
	tfvis.render.confusionMatrix(container, {
		values: confusionMatrix,
		tickLabels: classNames,
	});

	labels.dispose();
}

async function doPrediction(model, data) {
	const IMAGE_WIDTH = 28;
	const IMAGE_HEIGHT = 28;
	const [x_test, y_test] = data.getTestData();

	const labels = y_test.argMax(-1);
	const preds_vals = model.predict(x_test);
	const preds = preds_vals.argMax(-1);

	let temp = await preds_vals
		.slice([1, 0], [1, preds_vals.shape[1]])
		.reshape([10])
		.array();
	console.log(temp);

	// await barGraph(temp);

	x_test.dispose();
	return [preds, labels];
}

async function get_scores(model, data){
	const [XCalib, yCalib] = data.getCalibData();
	const yCalibArray = await yCalib.data();
	let n_calib = XCalib.shape[0];
	
	let logits = model.predict(XCalib);
	logits = await logits.array();
	
	const scores = Array.from({ length: XCalib.shape[0] }, (_, i) => {
	  return 1 - logits[i][yCalibArray[i]];
	});
	console.log(scores);
	
	let alpha = 0.05;
	let q = ((1-alpha)*(n_calib))/(n_calib+1);
	let q_val = calculatePercentile(scores, q);

	histogramPlot(scores, q_val);
	return scores;
}

function calculatePercentile(arr, q) {
	// Step 1: Sort the array in ascending order
	arr.sort((a, b) => a - b);
  
	// Step 2: Calculate the index k
	const n = arr.length;
	const k = (q / 100) * (n - 1);
  
	// Step 3: If k is an integer, return the element at index k
	if (Number.isInteger(k)) {
	  return arr[k];
	}
  
	// Step 4: Interpolate between elements at indices Math.floor(k) and Math.ceil(k)
	const lowerIndex = Math.floor(k);
	const upperIndex = Math.ceil(k);
	const lowerValue = arr[lowerIndex];
	const upperValue = arr[upperIndex];
	const fraction = k - lowerIndex;
	return lowerValue + (upperValue - lowerValue) * fraction;
  }

async function img_scores(model, data) {
	const [x_test, y_test] = await data.getTestData();
	const num_test_samples = 1;
	const surface = document.getElementById("img-scores");
	console.log(surface);
  
	for (let i = 0; i < num_test_samples; i++) {
	  const imageTensor = tf.tidy(() => {
		return x_test.slice([i, 0], [1, x_test.shape[1]]).reshape([1, 28, 28, 1]);
	  });
  
	  const pred = await model.predict(imageTensor).squeeze().array();
	  console.log(pred);
	  const predWithIndex = pred.map((value, index) => ({ index, value }))
  
	  
	  tfvis.render.barchart(surface, predWithIndex);
	  const canvas = document.createElement("canvas");
	  canvas.width = 56;
	  canvas.height = 56;
	  canvas.style.margin = "5px";
	  await tf.browser.toPixels(imageTensor.reshape([28, 28, 1]), canvas);
	  surface.appendChild(canvas);
	  imageTensor.dispose();
	}
  }
  