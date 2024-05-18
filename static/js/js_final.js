
var score = 0;

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

canvas.width = window.innerWidth - 100;
canvas.height = window.innerHeight - 100;

var clapImg = new Image();
clapImg.src = "static/img/clap_.png";

//박스
var dino = {
    x: 100,
    y: 0, //공룡 등장 좌표
    width: 100,
    height: 800, //공룡 크기
    draw() {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(this.x, this.y, this.width, this.height);
    },
  };

  //장애물
  class Cactus {
    constructor() {
      this.x = 1500;
      this.y = 250;
      this.width = 70;
      this.height = 300;
      this.speed = 3;
    }
    draw() {
      ctx.fillStyle = "red";
      ctx.drawImage(clapImg, this.x, this.y, this.width, this.height);
      this.x -= this.speed; // x 좌표를 감소시켜 왼쪽으로 이동

    }
  }

var timer = 0;
var cactuses = [];

const filePath = 'static/js/clap_data_rhythm.txt';
var flag = 0;

//cactus를 계속 생기도록 한다.
function executePerFrame() {
  requestAnimationFrame(executePerFrame);
  
  timer+=2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (timer % (Math.floor(Math.random() * 180) + 120) === 0) { // 등장 간격을 랜덤하게 설정
    var cactus = new Cactus();
    cactuses.push(cactus);
    }
  }

//장애물과 dino가 부딪히는지 확인 & txt에 clap!과, ready...를 읽어옴
function isBumped(dino, cactus) {
  var xDif = cactus.x - (dino.x + dino.width);

  if (xDif < 0) {
    fetch(filePath)
    .then(response => response.text())
    .then(data => {
      var lines = data.split('\n');
      try{
        var clap = lines[lines.length - 2].trim();
      }catch(error){}

      if(flag == 0 && clap == "Clap!!"){
          score += 10;
          flag = 1;
          updateScore(score); // score를 업데이트하는 함수 호출
      }

      else if(flag == 1 && clap == "Ready..."){
        flag = 0;
      }
        
    })
    .catch(error => console.error('file load error', error));
  }
}

//공룡 그림
dino.draw();

cactuses.forEach((a, i, o) => {
    if (a.x < 0) {
      o.splice(i, 1);
      // console.log("delete");
    }
    a.x -= 2;
    a.draw();
    isBumped(dino, a);
  });


executePerFrame();

function updateScore(score) {
    // HTML 요소에서 점수를 업데이트
    var scoreText = document.querySelector('.score-text');
    scoreText.textContent = "Score : " + score;
}

