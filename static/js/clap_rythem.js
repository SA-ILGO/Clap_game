function animateFirstImage() {
  var firstImage = document.getElementById("up");
  firstImage.addEventListener("animationend", function() {
      animateSecondImage();
  });
}

function animateSecondImage() {
  var secondImage = document.getElementById("up1");
  // 두 번째 이미지의 애니메이션을 시작합니다.
  secondImage.style.animation = "moveLeft 1s linear forwards";
}

animateFirstImage();


let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');


let square = {
	x: 280,
  	width: 60,
  	height: 700,
}

// let boxImage = new Image();
let boxImage = document.getElementById('up');

boxImage.src = 'static/img/clap.png';
class Box {
	constructor() {
    	this.width = 40;
      	this.height = 40;
      	this.x = canvas.width - this.width;
      	this.y = 156;
    }
  	draw() {
    	ctx.drawImage(boxImage, this.x, this.y, this.width, this.height);
    }
}

function frameRun() {
	animatiom = requestAnimationFrame(frameRun)
  	timer++;
  	ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  	if (timer % 120 === 0) {
    	let box = new Box();
      	manyBoxes.push(box);
    }
  	manyBoxes.forEach((a, i, o) => {
      	a.x--;
    });
}
frameRun();

function crash(square, box) {
	let xCalculate = box.x - (square.x + square.width);
  
  	if (xCalculate < 0) {
    	print("attached!!!!!!!!!!!!!")
    }
}
