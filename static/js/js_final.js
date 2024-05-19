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
    }
}

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


function executePerFrame() {
  requestAnimationFrame(executePerFrame);

  timer += 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 캐릭터 그리기
  dino.draw();

  // 장애물 생성 및 그리기
  if (timer % (Math.floor(Math.random() * 180) + 120) === 0 && cactuses.length < 15) { // 15번만 생성하도록 수정
      var cactus = new Cactus();
      cactuses.push(cactus);
  }

  // 장애물 그리고 충돌 여부 확인
  cactuses.forEach((a, i, o) => {
      if (a.x < 0) {
          o.splice(i, 1);
      }
      a.x -= 2;
      a.draw();
      isBumped(dino, a);
  });

  // 점수가 100점을 넘지 않고 15번 생성했을 때만 게임 종료 버튼을 표시
  if (score < 100 && cactuses.length >= 15) {
      drawGameOverButton();
  }
}


function drawGameOverButton() {
  // 게임 종료 버튼 그리기
  ctx.fillStyle = "red";
  ctx.fillRect(canvas.width - 150, 50, 100, 50);
  ctx.fillStyle = "white";
  ctx.font = "20px Arial";
  ctx.fillText("게임 종료", canvas.width - 140, 85);

  // 게임 종료 모달 표시
  var endGameModal = document.getElementById("endGameModal");
  endGameModal.style.display = "block";
}

// 게임 종료 버튼 클릭 이벤트 처리
canvas.addEventListener('click', function(event) {
  var rect = canvas.getBoundingClientRect();
  var mouseX = event.clientX - rect.left;
  var mouseY = event.clientY - rect.top;

  if (mouseX > canvas.width - 150 && mouseX < canvas.width - 50 &&
      mouseY > 50 && mouseY < 100) {
      endGame();
  }
});

function endGame() {
  // 게임 종료 처리
  alert("게임 종료!");
  // 여기에 추가적인 종료 처리를 할 수 있습니다.
}

function goToMainPage() {
  // 메인 페이지로 이동하는 함수
  location.href = 'templates/main.html'; // 메인 페이지의 URL로 이동합니다.
}

function restartGame() {
  // 게임 재시작 처리
  // 여기에 추가적인 재시작 처리를 할 수 있습니다.
  location.reload(); // 페이지 새로고침을 통해 게임을 재시작합니다.
}