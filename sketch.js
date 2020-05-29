// TODO :  clean all this crap!
// TODO :  better naming!
// TODO :  try different loss funcitons 
// TODO :  come up with some interactive user experience (buttons, sliders)
// TODO :  
// TODO :  
// TODO :  


const x_vals = [];
const y_vals = [];

let m, b;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);


function setup(){
    createCanvas(400, 400);
    m = tf.scalar(random(1)).variable();
    b = tf.scalar(random(1)).variable();
}

function calculateLoss(pred, labels){
    return pred.sub(labels).square().mean();
}

function makePrediction(xList){
    // y = m * x + b;
    const xT = tf.tensor1d(xList);
    return xT.mul(m).add(b);
}

function mousePressed(){
    const x = map(mouseX, 0, width, 0, 1);
    const y = map(mouseY, 0, height, 1, 0);
    x_vals.push(x);
    y_vals.push(y);
}

function draw(){


    if(x_vals.length > 0){
        tf.tidy(() => {
            const yT = tf.tensor1d(y_vals);
            optimizer.minimize(() => calculateLoss(makePrediction(x_vals), yT));
        });
    }

    background(0);
    stroke(255);
    strokeWeight(4);
    for (let i = 0; i < x_vals.length; i++) {
        const px = map(x_vals[i], 0, 1, 0, width);
        const py = map(y_vals[i], 0, 1, height, 0);
        point(px, py, 10);   
    }


    let xLine = [0, 1];
    const yT = tf.tidy(() => makePrediction(xLine));
    let yLine = yT.dataSync();
    yT.dispose();

    xLine = xLine.map(val => map(val, 0, 1, 0, width));
    yLine = yLine.map(val => map(val, 0, 1, height, 0));
    // console.log(xLine.toString(), yLine.toString());

    line(xLine[0], yLine[0], xLine[1], yLine[1]);

    console.log(tf.memory().numTensors);
}