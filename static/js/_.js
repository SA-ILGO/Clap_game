
var score = 0;

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

canvas.width = window.innerWidth - 100;
canvas.height = window.innerHeight - 100;
  
// var backgroundImg = new Image();
// backgroundImg.src = "static/img/background.png";

var clapImg = new Image();
clapImg.src = "static/img/clap.png";

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
      this.x = 2000;
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

function executePerFrame() {
  requestAnimationFrame(executePerFrame);
  
  timer+=2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (timer % (Math.floor(Math.random() * 180) + 120) === 0) { // 등장 간격을 랜덤하게 설정
    var cactus = new Cactus();
    cactuses.push(cactus);
    // if ((timer + 5) % interval === 0) { // 두 번째 장애물을 생성하기 전에 현재 timer 값이 interval에 근접한지 확인합니다.
    //   var cactus = new Cactus();
    //   cactuses.push(cactus);
    // }
  }

  dino.draw();

  // if (timer % 360 === 0) {
  //   var cactus = new Cactus();
  //   cactuses.push(cactus);
  // }

  cactuses.forEach((a, i, o) => {
    if (a.x < 0) {
      o.splice(i, 1);
      // console.log("delete");
    }
    a.x -= 2;
    a.draw();
    isBumped(dino, a);

  });
}
executePerFrame();

const filePath = 'static/js/clap_data_rhythm.txt';
var flag = 0;

function isBumped(dino, cactus) {
    console.log("bumped");

    var xDif = cactus.x - (dino.x + dino.width);
    if (xDif < 0) {
      fetch(filePath)
            .then(response => response.text())
            .then(data => {
                var lines = data.split('\n');
                var clap = lines[lines.length - 1];

                // console.log(clap);

                if(flag == 0 && clap == "Clap!!"){
                    score += 10;
                    flag = 1;
                    updateScore(score); // score를 업데이트하는 함수 호출
                    console.log(score);

                }
                else if(flag == 1 && clap == "Ready..."){
                    flag = 0;
                    console.log(clap);
                }
                
            })
            .catch(error => console.error('file load error', error));
    }
  }

function updateScore(score) {
    // HTML 요소에서 점수를 업데이트
    var scoreText = document.querySelector('.score-text');
    scoreText.textContent = "Score : " + score;
}