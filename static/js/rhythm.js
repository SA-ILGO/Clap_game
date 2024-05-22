//before game start
var round = 0;
var roundClapCount = 0; // 라운드 당 그릴 clap_.png 이미지 수 카운트

//game start
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
var animationFrameId;
var bumpedCactusesCount = 0; // 특정 박스에 도달한 장애물의 수

//부딪히는지 확인하는 부분
function isBumped(dino, cactus) {
    var xDif = cactus.x - (dino.x + dino.width);
    if (xDif < 0) {
        fetch(filePath)
        .then(response => response.text())
        .then(data => {
            var lines = data.split('\n');
            try {
                var clap = lines[lines.length - 2].trim();
            } catch (error) {}
            if (flag == 0 && clap == "Clap!!") {
                score += 10;
                flag = 1;
                updateScore(score); // score를 업데이트하는 함수 호출
            } else if (flag == 1 && clap == "Ready...") {
                flag = 0;
            }
        })
        .catch(error => console.error('file load error', error));

    }
    if(xDif === 0){
        bumpedCactusesCount++;
        console.log(bumpedCactusesCount);
        // 특정 박스에 도달한 장애물의 수가 10이 되면 게임 종료
        if (bumpedCactusesCount >= 10) {
            endGame();
        }
    }
}

function endGame() {
    cancelAnimationFrame(animationFrameId);
    var gameOverModal = document.querySelector('.game-over');
    var gameOverText = document.getElementById('game-over-text');
    if (score >= 10) {
        gameOverText.textContent = "게임 종료! 성공!";
    } else {
        gameOverText.textContent = "게임 종료! 실패!";
    }
    gameOverModal.style.display = 'flex';
}

function executePerFrame() {
    animationFrameId = requestAnimationFrame(executePerFrame);

    timer += 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 박수 아이콘 생성
    dino.draw();
    if (timer % (Math.floor(Math.random() * 180) + 120) === 0) {
        var cactus = new Cactus();
        cactuses.push(cactus);
        // roundClapCount++;
        // console.log(roundClapCount);
    }

    // 박수 아이콘 충돌 여부 확인
    cactuses.forEach((a, i, o) => {
        if (a.x < 0) {
            o.splice(i, 1);
        }
        a.x -= 2;
        a.draw();
        isBumped(dino, a);
    });
}

function resetGame() {
    var gameOverModal = document.querySelector('.game-over');
    gameOverModal.style.display = 'none';
    score = 0;
    roundClapCount = 0;
    bumpedCactusesCount = 0; // 특정 박스에 도달한 장애물의 수 초기화
    updateScore(score);
    cactuses = [];
    timer = 0; // 타이머 초기화
    flag = 0; // 플래그 초기화
    executePerFrame();
    round++;
}

//game start
const modal = document.querySelector('.modal');
document.addEventListener("DOMContentLoaded", () => {
    round += 1;
    modalText.textContent = round;
    modal.style.display = "flex";
});
const btnCloseModal = document.getElementById('modal_btn');
const modalText = document.getElementById('modal_text');
btnCloseModal.addEventListener("click", () => {
    modal.style.display = "none";
    console.log("게임 시작");
    executePerFrame(); // 모달이 닫힐 때마다 게임이 시작되도록 함
});

const restartBtn = document.getElementById('restart_btn');
restartBtn.addEventListener('click', resetGame);

//점수 계산
function updateScore(score) {
    // HTML 요소에서 점수를 업데이트
    var scoreText = document.querySelector('.score-text');
    scoreText.innerHTML = "점수 : " + score;
}