function randomAnimation() {
    var up = document.getElementById('up');
    var random = Math.random();
  
    if (random < 0.5) {
      // 50% 확률로 1이 나오면 이동
      up.style.animationPlayState = 'running';
    } else {
      // 50% 확률로 0이 나오면 제자리에 있음
      up.style.animationPlayState = 'paused';
    }
  
    setTimeout(randomAnimation, 10); // 0.01초 후에 다시 실행
}
  
  randomAnimation(); // 처음에 한번 실행
  