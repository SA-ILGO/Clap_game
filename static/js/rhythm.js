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

// const answerButton = document.getElementById('answerButton'); // 수정된 부분

// const successButton = document.getElementById('successButton');
// const failureButton = document.getElementById('failureButton');

// const result_modal = document.querySelector('.result_modal');
// const result = document.getElementById('result_text');

// function finish() {
//     console.log("finissssssh")
//     answerButton.addEventListener("click", () => { // 수정된 부분
//         if (score > 100 && roundClapCount > 10) {
//             // 성공한 경우
//             round += 1;
//             modalText.textContent = round + " 단 계";
//             modal.style.display = 'flex';
//             successButton.style.display = 'block'; // 성공 버튼 표시
//             failureButton.style.display = 'none'; // 실패 버튼 숨기기
//         } else if (score < 100 && roundClapCount > 10) {
//             // 실패한 경우
//             round_result = round;
//             result.textContent = "결과:" + round_res + "단계";
//             result_modal.style.display = "flex";
//             successButton.style.display = 'none'; // 성공 버튼 숨기기
//             failureButton.style.display = 'block'; // 실패 버튼 표시
//         }
//     });
// }

//계속 장애물 생성
function executePerFrame() {
    requestAnimationFrame(executePerFrame);

    timer += 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 캐릭터 그리기
    dino.draw();
    if (roundClapCount < 11) {
        if (timer % (Math.floor(Math.random() * 180) + 120) === 0) {
            var cactus = new Cactus();
            cactuses.push(cactus);
            roundClapCount++;
            console.log(roundClapCount);
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

//game start
const modal = document.querySelector('.modal');
document.addEventListener("DOMContentLoaded", ()=>{
    round += 1;
    modalText.textContent = round + " 단 계";
    modal.style.display="flex";
    // finish();
});
const btnCloseModal=document.getElementById('modal_btn');
const modal_btn = document.getElementById('modal_btn');
const modalText=document.getElementById('modal_text');
btnCloseModal.addEventListener("click", ()=>{
    // round_text.textContent = round;
    modal.style.display="none";
    // reset.click();
    console.log("게임 시작");
    executePerFrame(); // 모달이 닫힐 때마다 게임이 시작되도록 함
});


//점수 계산
function updateScore(score) {
    // HTML 요소에서 점수를 업데이트
    var scoreText = document.querySelector('.score-text');
    scoreText.textContent = "Score : " + score;
}


