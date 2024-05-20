var round = 0;
var level = 1;

const round_text = document.getElementById("round");
function cal_level(){
    if(round % 3 == 0){
        if(level > 8) {
            level = level
        } else {
            level++;
        }
    }
    else {
        level = level;
    }   
}

//print clap
const clapImgList = ["static/img/yellow_flower.png", "static/img/egg_clap.png", "static/img/wrist_clap.png", "static/img/fist_clap.png", "static/img/edge_clap.png", "static/img/red_flower.png",];
const clapNameList = ["", "egg clap", "wrist clap", "fist clap", "edge clap"];
const clapNameList_kor = ["", "달걀 박수", "손목 박수", "주먹 박수", "손날 박수"];
const randClap = document.getElementById('sample_img');
const randClapName = document.getElementById('sample_text');
var clapAnswer = [];

function assign_clap_img() {
    cal_level();
    let count_level = 2*level-1;
    let count = 0;

    function repeat_assign() {

        var odd = count % 2;
        if (count < count_level && odd == 0) {
            const randNum = Math.floor(Math.random() * 4) + 1;
            clapAnswer.push(clapNameList_kor[randNum]);
            randClap.src = clapImgList[randNum];
            randClapName.textContent = clapNameList_kor[randNum];

            setTimeout(repeat_assign, 1500); 
            
        } else if(count < count_level && odd == 1){
            randClap.src = clapImgList[5];
            randClapName.textContent = clapNameList_kor[5];
            setTimeout(repeat_assign, 500); 

        }else {
            randClap.src = clapImgList[0];
            randClapName.textContent = clapNameList_kor[0];
        }

        count++;
    }
    setTimeout(repeat_assign, 1500);
}



// game start
const modal = document.querySelector('.modal');
document.addEventListener("DOMContentLoaded", ()=>{
    round += 1;
    modalText.textContent = round + " 단 계";
    modal.style.display="flex";
});
const btnCloseModal=document.getElementById('modal_btn');
const modalText=document.getElementById('modal_text');
btnCloseModal.addEventListener("click", ()=>{
    round_text.textContent = round;
    modal.style.display="none";
    clapAnswer = [];
    correct = 0;
    reset.click();
    watch_dataFile();
    assign_clap_img();
});


const filePath = 'static/js/clap_data.txt';
// clap_data 
const cognition1 = document.getElementById("cognition1");
const cognition2 = document.getElementById("cognition2");
const cognition3 = document.getElementById("cognition3");
const cognition4 = document.getElementById("cognition4");
const cognition5 = document.getElementById("cognition5");
const cognition6 = document.getElementById("cognition6");
const cognition7 = document.getElementById("cognition7");
const cognition8 = document.getElementById("cognition8");

const cognitionList = ["cognition1", "cognition2", "cognition3", "cognition4",
            "cognition5", "cognition6", "cognition7", "cognition8"];

function watch_dataFile() {
    var intervalId;
    function load_dataFile() {
        fetch(filePath)
            .then(response => response.text())
            .then(data => {
                var lines = data.split('\n');

                if(lines.length > level) clearInterval(intervalId);

                lines.forEach((line, index) => {
                    cognitionList.forEach((cognition, i) => {
                        try{
                            var clap_kor = eng_to_kor(lines[i]);
                            document.getElementById(cognition).textContent = clap_kor; 
                        }catch(error){
                            
                        }
                    });
                });
            })
            .catch(error => console.error('file load error', error));
    }
    intervalId = setInterval(load_dataFile, 500); 

    window.addEventListener('unload', () => {
        clearInterval(intervalId);
    });
  };

function eng_to_kor(clap){
    try{
        for(let i = 0; i < 5; i++){
            if(clapNameList[i] == clap.trim()){
                return clapNameList_kor[i];
            }
        }
    }catch(error){

    }
}

//press correct button
var correct = 0;
const answer = document.getElementById('answer');
const result = document.getElementById("result_text");
const result_modal = document.querySelector('.result_modal');
answer.addEventListener("click", ()=>{
    for(let i = 0; i < level; i++){
        var compare = document.getElementById(cognitionList[i]).textContent;
        if (clapAnswer[i] === compare) {
            correct = 1;
        }
        else {
            correct = 0;
        }
    }
    if(correct == 1) {
        round += 1;
        modalText.textContent = round + " 단 계";
        modal.style.display='flex';
    }
    else {
        round_res = round ;
        result.textContent = "결과:" + round_res + "단계";
        result_modal.style.display="flex";
    }
})

//reset button
const reset = document.getElementById('reset');
reset.addEventListener("click", () => {
    fetch('/reset')
        .catch(error => console.error("Error:", error));
    watch_dataFile();
});


