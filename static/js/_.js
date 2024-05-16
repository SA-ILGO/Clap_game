var score = 0;

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

canvas.width = window.innerWidth - 100;
canvas.height = window.innerHeight - 100;
  
var dino = {
    x: 10,
    y: 200, //공룡 등장 좌표
    width: 50,
    height: 50, //공룡 크기
    draw() {
      ctx.fillStyle = "green";
      ctx.fillRect(this.x, this.y, this.width, this.height);
    },
  };
  
  class Cactus {
    constructor() {
      this.x = 500;
      this.y = 200;
      this.width = 50;
      this.height = 50;
    }
    draw() {
      ctx.fillStyle = "red";
      ctx.fillRect(this.x, this.y, this.width, this.height);
    }
  }
  var timer = 0;
var cactuses = [];

function executePerFrame() {
  requestAnimationFrame(executePerFrame);
  
  timer++;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (timer % 120 === 0) {
    var cactus = new Cactus();
    cactuses.push(cactus);
  }

  dino.draw();

  if (timer % 240 === 0) {
    var cactus = new Cactus();
    cactuses.push(cactus);
  }

  cactuses.forEach((a, i, o) => {
    if (a.x < 0) {
      o.splice(i, 1);
      console.log("delete");
    }
    a.x -= 2;
    a.draw();
    isBumped(dino, a);
  });
  console.log(clap);
}
executePerFrame();

const filePath = 'static/js/clap_data_rhythm.txt';
var flag = 0;

function isBumped(dino, cactus) {
    var xDif = cactus.x - (dino.x + dino.width);
    var yDif = cactus.y - (dino.y + dino.height);
    if (xDif < 0 && yDif < 0) {
      fetch(filePath)
            .then(response => response.text())
            .then(data => {
                var lines = data.split('\n');
                var clap = lines[lines.length - 1];
                
                if(flag == 0 && clap == "Clap!!"){
                    score += 10;
                    flag = 1;
                    console.log(clap);
                }else if(flag == 1 && clap == "Ready..."){
                    flag = 0;
                    console.log(clap);
                }
                
            })
            .catch(error => console.error('file load error', error));
    }
  }