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
    y: 0, //박스 등장 좌표
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

//부딪히는지 확인하는 부분
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


var roundClapCount = 0; // 라운드 당 그릴 clap_.png 이미지 수 카운트
round = 0;

//계속 장애물 생성
function executePerFrame() {
    requestAnimationFrame(executePerFrame);

    timer += 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 캐릭터 그리기
    dino.draw();

    // 장애물 생성 및 그리기
    if (timer % (Math.floor(Math.random() * 180) + 120) === 0) { // 등장 간격을 랜덤하게 설정
         // 라운드 당 clap_.png 이미지 수가 15개가 될 때까지 그리기
                    // 라운드 당 clap_.png 이미지 수가 15개가 될 때까지 그리기
                    if (roundClapCount < 11) {
                        var cactus = new Cactus();
                        cactuses.push(cactus);
                        roundClapCount++;
                        endGame();
                    }
        }

    // 장애물 그리고 충돌 여부 확인
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


// 게임 종료 함수
function endGame() {
    if (score < 100 && roundClapCount > 10) {
        const endGameModal = document.querySelector('.end-game-modal');
        endGameModal.style.display = "flex";
        cancelAnimationFrame(animationFrameId);

    }
}

// 종료 버튼 클릭 시
const endGameBtn = document.getElementById('end-game-btn');
endGameBtn.addEventListener("click", () => {
    const endGameModal = document.querySelector('.end-game-modal');
    endGameModal.style.display = "none"; // 모달을 숨기는 코드
});
const restartGameBtn = document.getElementById('end-restart-game-btn');
restartGameBtn.addEventListener("click", () => {
    // 게임을 재시작하는 코드 추가
    const endGameModal = document.querySelector('.end-game-modal');
    endGameModal.style.display = "none"; // 모달을 숨기는 코드
});


//점수 계산
function updateScore(score) {
    // HTML 요소에서 점수를 업데이트
    var scoreText = document.querySelector('.score-text');
    scoreText.textContent = "Score : " + score;
}


